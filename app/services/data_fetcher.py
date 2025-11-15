# app/services/data_fetcher.py
import time
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import yfinance as yf
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.database import SessionLocal, MarketData, engine
from sqlalchemy import text

# canonical field names and common synonyms
_CANONICAL = {
    'open': ['open'],
    'high': ['high'],
    'low': ['low'],
    'close': ['close'],
    'adj_close': ['adj close', 'adj_close', 'adjclose', 'adj'],
    'volume': ['volume', 'vol']
}

class DataFetcher:
    """Robust data fetcher tailored to yfinance MultiIndex (Price,Ticker) output."""

    def __init__(self, batch_size: int = 1000, retry_attempts: int = 3, retry_backoff: float = 1.0):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff

    def fetch_ohlcv(self, symbols: List[str], start: str, end: str,
                    save_to_db: bool = True, pause_between: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV using yfinance for each symbol, robust to the MultiIndex output described in inspection.
        Returns dict: symbol -> DataFrame indexed by UTC-aware datetimes with canonical columns:
        ['open','high','low','close','adj_close','volume'] (adj_close may equal close if not provided).
        """
        results: Dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            sym = symbol.strip().upper()
            print(f"Fetching {sym}...")
            df_raw = self._download_with_retries(sym, start, end)
            if df_raw is None or df_raw.empty:
                print(f"Warning: no data for {sym}")
                continue

            df = self._normalize_yf_df(df_raw, sym)
            if df is None or df.empty:
                print(f"Warning: normalization produced no usable rows for {sym}; skipping.")
                continue

            # Final cleaning: drop rows missing essential values
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            if df.empty:
                print(f"Warning: after dropping NaNs no rows remain for {sym}")
                continue

            # Ensure dtypes
            for col in ['open', 'high', 'low', 'close', 'adj_close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64')

            # keep canonical column order if present
            ordered_cols = [c for c in ['open','high','low','close','adj_close','volume'] if c in df.columns]
            df = df[ordered_cols]

            # cache and optionally save to DB
            self.cache[sym] = df
            results[sym] = df

            if save_to_db:
                try:
                    self._save_to_db(sym, df)
                except Exception as e:
                    print(f"Error saving {sym} to DB: {e}")

            time.sleep(pause_between)

        return results

    def _normalize_yf_df(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """
        Normalize a raw DataFrame from yfinance into canonical single-symbol format.
        Handles:
        - MultiIndex columns with (Price, Ticker) ordering (observed in your inspection).
        - Flattened single-level columns like 'aapl_open' or 'open'.
        Returns DataFrame with canonical column names or None if normalization fails.
        """
        # Helper to canonicalize a name string
        def _clean(s: str) -> str:
            return str(s).strip().lower().replace(' ', '_')

        # If MultiIndex (Price, Ticker) as in outputs, pick level0 where level1 == requested symbol
        if isinstance(df.columns, pd.MultiIndex):
            # determine which level corresponds to price/ticker by inspecting names if present
            # Observed ordering: level 0 = Price (Adj Close/Close/Open/...), level 1 = Ticker
            cols_for_symbol = {}
            for lvl0, lvl1 in df.columns:
                try:
                    tick = str(lvl1).strip().upper()
                except Exception:
                    tick = str(lvl1).strip().upper()
                if tick == symbol:
                    cols_for_symbol[_clean(lvl0)] = (lvl0, lvl1)
            if not cols_for_symbol:
                # No columns for this symbol in MultiIndex — fallback to flattening
                # flatten all (join with underscore)
                flat = ['_'.join(map(str, c)).lower().replace(' ', '_') for c in df.columns]
                df_flat = df.copy()
                df_flat.columns = flat
                df = df_flat
            else:
                # Build DataFrame from columns selected for this symbol using the level 0 names
                sel = []
                remap = {}
                for lvl0_clean, pair in cols_for_symbol.items():
                    # map e.g. 'adj_close' or 'close' etc
                    sel.append(pair)
                # select columns by MultiIndex tuples
                df_symbol = df.loc[:, sel].copy()
                # rename columns to level0 cleaned strings (e.g., 'Adj Close' -> 'adj_close')
                df_symbol.columns = [_clean(c[0]) for c in df_symbol.columns]
                df = df_symbol

        else:
            # single-level columns: clean them
            df = df.copy()
            df.columns = [_clean(c) for c in df.columns]

        # Now at this point, df should have columns like 'adj_close', 'close', 'open', etc. Or flattened 'aapl_open'
        # If flattened columns exist like 'aapl_open', attempt to extract symbol-prefixed columns
        cols = list(df.columns)
        # detect prefix patterns: e.g., 'aapl_open' (symbol prefix)
        prefix = symbol.lower() + '_'
        prefixed = [c for c in cols if c.startswith(prefix)]
        if prefixed:
            # map prefixed columns into canonical names by removing prefix and using cleaned suffix
            renamed = {}
            for c in prefixed:
                new = c[len(prefix):]
                renamed[c] = new
            df = df.rename(columns=renamed)

        # Now try to find canonical columns using synonyms
        found = {}
        for can, synonyms in _CANONICAL.items():
            for syn in synonyms:
                syn_clean = syn.replace(' ', '_')
                # direct match
                if syn_clean in df.columns:
                    found[can] = syn_clean
                    break
            if can in found:
                continue
            # fallback: look for columns that contain the synonym as substring
            for col in df.columns:
                if any(syn.replace(' ', '_') in col for syn in synonyms):
                    found[can] = col
                    break

        # If no close but adj_close present, use adj_close as close
        if 'close' not in found and 'adj_close' in found:
            found['close'] = found['adj_close']
        # If adj_close missing but close present, use close
        if 'adj_close' not in found and 'close' in found:
            found['adj_close'] = found['close']

        # Ensure we have minimal required set
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [r for r in required if r not in found]
        if missing:
            print(f"Normalization: missing required columns for {symbol}: {missing}")
            print("Normalization: available columns after cleaning:", list(df.columns))
            return None

        # Create reduced frame with canonical names
        df_canon = pd.DataFrame(index=df.index)
        for canon, colname in found.items():
            df_canon[canon] = df[colname]

        # ensure adj_close exists
        if 'adj_close' not in df_canon.columns:
            df_canon['adj_close'] = df_canon['close']

        # Normalize index to UTC-aware datetimes
        idx = pd.to_datetime(df_canon.index)
        if idx.tz is None:
            idx = idx.tz_localize('UTC')
        else:
            idx = idx.tz_convert('UTC')
        df_canon.index = idx
        df_canon.index.name = 'date'

        return df_canon

    def _download_with_retries(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        attempt = 0
        while attempt < self.retry_attempts:
            try:
                # explicit auto_adjust=False to keep raw Adj Close column
                df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
                return df
            except Exception as e:
                attempt += 1
                wait = self.retry_backoff * (2 ** (attempt - 1))
                print(f"Error fetching {symbol} (attempt {attempt}): {e}. Retrying in {wait}s.")
                time.sleep(wait)
        print(f"Failed to fetch {symbol} after {self.retry_attempts} attempts.")
        return None

    def _save_to_db(self, symbol: str, df: pd.DataFrame):
        """Batch insert with ON CONFLICT DO NOTHING. Assumes uix_symbol_date exists."""
        rows = []
        for ts, r in df.iterrows():
            dt = ts.to_pydatetime()
            rows.append({
                'symbol': symbol,
                'date': dt,
                'open': float(r['open']),
                'high': float(r['high']),
                'low': float(r['low']),
                'close': float(r['close']),
                'adj_close': float(r.get('adj_close', r['close'])),
                'volume': int(r['volume'])
            })

        if not rows:
            return

        market_table = MarketData.__table__
        with engine.begin() as conn:
            for i in range(0, len(rows), self.batch_size):
                batch = rows[i:i + self.batch_size]
                stmt = pg_insert(market_table).values(batch)
                stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'date'])
                conn.execute(stmt)

    def load_from_db(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Load market_data into DataFrames (index tz-aware UTC)."""
        result = {}
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        # localize to UTC if naive
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize('UTC')
        else:
            start_ts = start_ts.tz_convert('UTC')
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize('UTC')
        else:
            end_ts = end_ts.tz_convert('UTC')

        for sym in symbols:
            query = text(
                "SELECT date, open, high, low, close, adj_close, volume "
                "FROM market_data WHERE symbol = :sym AND date >= :start AND date <= :end "
                "ORDER BY date"
            )
            params = {'sym': sym, 'start': start_ts, 'end': end_ts}
            df = pd.read_sql_query(query, con=engine, params=params, parse_dates=['date'])
            if not df.empty:
                df = df.set_index('date')
                idx = pd.to_datetime(df.index)
                if idx.tz is None:
                    idx = idx.tz_localize('UTC')
                else:
                    idx = idx.tz_convert('UTC')
                df.index = idx
                result[sym] = df
        return result
