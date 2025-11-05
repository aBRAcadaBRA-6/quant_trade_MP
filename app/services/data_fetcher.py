# app/services/data_fetcher.py
import time
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import yfinance as yf
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.database import SessionLocal, MarketData, engine
from sqlalchemy import text

class DataFetcher:
    """Downloads and caches historical market data with safe, idempotent DB writes."""

    def __init__(self, batch_size: int = 1000, retry_attempts: int = 3, retry_backoff: float = 1.0):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff

    def fetch_ohlcv(self, symbols: List[str], start: str, end: str,
                    save_to_db: bool = True, threads: Optional[int] = None,
                    pause_between: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV using yfinance for each symbol (safely). Returns dict of DataFrames.
        """
        data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            sym = symbol.strip().upper()
            print(f"Fetching {sym}...")
            df = self._download_with_retries(sym, start, end)
            if df is None or df.empty:
                print(f"Warning: no data for {sym}")
                continue

            # Normalize columns: lower-case and rename 'adj close' -> 'adj_close'
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            if 'adj_close' not in df.columns and 'adjclose' in df.columns:
                df = df.rename(columns={'adjclose': 'adj_close'})

            # Ensure index is tz-aware UTC or naive UTC normalized to midnight
            df.index = pd.to_datetime(df.index).tz_convert(None).tz_localize(None)
            df.index.name = 'date'

            # Drop rows with essential NaNs
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

            self.cache[sym] = df
            data[sym] = df

            if save_to_db:
                self._save_to_db(sym, df)

            # small pause to be polite / avoid rate limits
            time.sleep(pause_between)

        return data

    def _download_with_retries(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        attempt = 0
        while attempt < self.retry_attempts:
            try:
                df = yf.download(symbol, start=start, end=end, progress=False)
                return df
            except Exception as e:
                attempt += 1
                wait = self.retry_backoff * (2 ** (attempt - 1))
                print(f"Error fetching {symbol} (attempt {attempt}): {e}. Retrying in {wait}s.")
                time.sleep(wait)
        print(f"Failed to fetch {symbol} after {self.retry_attempts} attempts.")
        return None

    def _save_to_db(self, symbol: str, df: pd.DataFrame):
        """
        Bulk insert with ON CONFLICT DO NOTHING (idempotent). Uses SQLAlchemy Core insert for PostgreSQL.
        """
        # Prepare list of dicts
        rows = []
        for ts, r in df.iterrows():
            rows.append({
                'symbol': symbol,
                'date': ts.to_pydatetime(),   # timezone-naive Python datetime
                'open': float(r['open']),
                'high': float(r['high']),
                'low': float(r['low']),
                'close': float(r['close']),
                'adj_close': float(r.get('adj_close', r.get('adjclose', r['close']))),
                'volume': int(r['volume'])
            })

        if not rows:
            return

        # Insert in batches
        from sqlalchemy import Table, MetaData
        meta = MetaData()
        market_table = MarketData.__table__

        with engine.begin() as conn:
            for i in range(0, len(rows), self.batch_size):
                batch = rows[i:i + self.batch_size]
                stmt = pg_insert(market_table).values(batch)
                # Do nothing on conflict of unique (symbol, date)
                stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'date'])
                conn.execute(stmt)

    def load_from_db(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Load cached data from the market_data table into DataFrames using pandas.read_sql."""
        data = {}
        # Parse start/end to datetimes
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
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
                data[sym] = df
        return data
