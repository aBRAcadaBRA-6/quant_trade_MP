# app/services/feature_engineer.py
from __future__ import annotations
import os
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from app.services.data_fetcher import DataFetcher
from app.models.database import engine

# Persistence folders
PROCESSED_DIR = "data/processed"
DIAGNOSTICS_DIR = Path("data/processed/diagnostics")
VAR_OUTPUTS_DIR = Path("data/processed/var_outputs")

os.makedirs(PROCESSED_DIR, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
VAR_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class FeatureConfig:
    lookback_short: int = 5
    lookback_medium: int = 20
    lookback_long: int = 60
    vol_window: int = 20
    z_window: int = 60
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_window: int = 14
    atr_window: int = 14
    min_history_ratio: float = 0.5  # min fraction of non-null rows to keep a column

class FeatureEngineer:
    """Feature Engineer that computes time-series features and a compact VAR pipeline.

    Main methods:
    - compute_features_for_symbol(symbol, start, end) -> DataFrame (features saved to parquet by default)
    - compute_features_bulk(symbols, start, end) -> dict(symbol -> DataFrame)
    - pipeline_var_cov(data_dict, persist_outputs=False, save_db_record=False) -> (standardized_returns_df, A_matrix, cov_matrix, diagnostics)

    Persistence:
    - Per-symbol features -> data/processed/{SYMBOL}_features.parquet
    - Diagnostics -> data/processed/diagnostics/diag_<ts>.json (always saved)
    - Optional VAR outputs -> data/processed/var_outputs/<ts>/* (A/cov/standardized_returns/diagnostics)
    - Optional metadata table var_runs in Postgres (only if save_db_record=True)
    """

    def __init__(self, cfg: FeatureConfig = FeatureConfig()):
        self.cfg = cfg
        self.fetcher = DataFetcher()

    # -----------------------
    # Data loading + feature compute
    # -----------------------
    def compute_features_for_symbol(self, symbol: str, start: str, end: str,
                                    save: bool = True) -> pd.DataFrame:
        data_map = self.fetcher.load_from_db([symbol], start, end)
        if symbol not in data_map:
            raise ValueError(f"No market data for {symbol} in DB for range {start} - {end}")

        df = data_map[symbol].sort_index()
        features = self._compute_all_features(df)
        if save:
            path = os.path.join(PROCESSED_DIR, f"{symbol}_features.parquet")
            features.to_parquet(path)
        return features

    def compute_features_bulk(self, symbols: List[str], start: str, end: str, save: bool = True) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                out[s] = self.compute_features_for_symbol(s, start, end, save=save)
                print(f"Computed features for {s}: {out[s].shape}")
            except Exception as e:
                print(f"Error computing {s}: {e}")
        return out

    # -----------------------
    # Core features
    # -----------------------
    def _compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input df must contain: 'open','high','low','close' OR 'adj_close' and 'volume'.
        Works on adj_close primarily.
        """
        X = df.copy()
        # Prefer adj_close if present
        if 'adj_close' not in X.columns and 'close' in X.columns:
            X['adj_close'] = X['close']

        # Ensure sorted
        X = X.sort_index()

        # Basic returns
        X['return'] = X['adj_close'].pct_change()
        # Log returns (useful for volatility)
        X['log_return'] = np.log(X['adj_close']).diff()

        # Lagged returns
        X['return_1'] = X['return'].shift(1)
        X['return_2'] = X['return'].shift(2)
        X['return_3'] = X['return'].shift(3)

        # Moving averages (shifted to avoid lookahead)
        X['sma_short'] = X['adj_close'].rolling(self.cfg.lookback_short, min_periods=1).mean().shift(1)
        X['sma_medium'] = X['adj_close'].rolling(self.cfg.lookback_medium, min_periods=1).mean().shift(1)
        X['sma_long'] = X['adj_close'].rolling(self.cfg.lookback_long, min_periods=1).mean().shift(1)

        X['ema_short'] = X['adj_close'].ewm(span=self.cfg.lookback_short, adjust=False).mean().shift(1)
        X['ema_medium'] = X['adj_close'].ewm(span=self.cfg.lookback_medium, adjust=False).mean().shift(1)

        # Momentum (shifted)
        X['mom_5'] = (X['adj_close'] / X['adj_close'].shift(self.cfg.lookback_short) - 1).shift(1)
        X['mom_20'] = (X['adj_close'] / X['adj_close'].shift(self.cfg.lookback_medium) - 1).shift(1)

        # Volatility (std of log returns), shifted
        X['vol_20'] = X['log_return'].rolling(self.cfg.vol_window, min_periods=1).std().shift(1)

        # Price vs SMA z-score
        X['price_minus_sma20'] = X['adj_close'] - X['sma_medium'].shift(0)  # sma already shifted
        rolling_mean = X['price_minus_sma20'].rolling(self.cfg.z_window, min_periods=1).mean().shift(1)
        rolling_std = X['price_minus_sma20'].rolling(self.cfg.z_window, min_periods=1).std().shift(1).replace(0, np.nan)
        X['zscore_60'] = (X['price_minus_sma20'].shift(1) - rolling_mean) / rolling_std

        # MACD
        ema_fast = X['adj_close'].ewm(span=self.cfg.macd_fast, adjust=False).mean()
        ema_slow = X['adj_close'].ewm(span=self.cfg.macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.cfg.macd_signal, adjust=False).mean()
        X['macd'] = macd.shift(1)
        X['macd_signal'] = signal.shift(1)
        X['macd_hist'] = (X['macd'] - X['macd_signal']).shift(0)

        # RSI
        X['rsi_14'] = self._rsi(X['adj_close'], window=self.cfg.rsi_window).shift(1)

        # ATR
        X['atr_14'] = self._atr(X[['high', 'low', 'close']], window=self.cfg.atr_window).shift(1)

        # Volume features
        X['vol_mean_20'] = X['volume'].rolling(self.cfg.vol_window, min_periods=1).mean().shift(1)
        vol_std = X['volume'].rolling(self.cfg.vol_window, min_periods=1).std().shift(1).replace(0, np.nan)
        X['vol_zscore'] = (X['volume'] - X['vol_mean_20']) / vol_std

        # Drop rows where adj_close missing
        X = X.dropna(subset=['adj_close'])

        # Optionally drop rows that are completely NaN in features
        X = X.dropna(axis=0, how='all')

        return X

    # -----------------------
    # VAR & covariance pipeline (compact)
    # -----------------------
    def _build_price_panel(self, data: Dict[str, pd.DataFrame], price_candidates: Tuple[str, ...] = ('adj_close', 'close')) -> pd.DataFrame:
        """Return aligned prices (index=date UTC, columns=symbols)."""
        price_frames = {}
        for sym, df in data.items():
            # choose price column
            col = next((c for c in price_candidates if c in df.columns), None)
            if col is None:
                continue
            ser = pd.Series(df[col].values, index=pd.to_datetime(df.index), name=sym).sort_index()
            # ensure timezone-aware UTC
            if ser.index.tz is None:
                ser.index = ser.index.tz_localize('UTC')
            else:
                ser.index = ser.index.tz_convert('UTC')
            price_frames[sym] = ser
        if not price_frames:
            raise ValueError("No valid price columns found in input data.")
        prices = pd.concat(price_frames.values(), axis=1, keys=price_frames.keys()).sort_index()
        # drop rows where all are NaN
        prices = prices.dropna(how='all')
        return prices

    def compute_log_returns(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return aligned log returns DataFrame (dates x symbols)."""
        prices = self._build_price_panel(data)
        returns = np.log(prices).diff().iloc[1:]  # drop first NaN row
        # drop columns that are mostly NaN
        min_nonnull = int(self.cfg.min_history_ratio * len(returns))
        returns = returns.dropna(axis=1, thresh=min_nonnull)
        # drop near-zero variance columns
        variances = returns.var()
        keep_cols = variances[variances > 1e-12].index.tolist()
        returns = returns[keep_cols]
        return returns

    def standardize_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Z-score each column; returns DataFrame with same index/columns."""
        scaler = StandardScaler()
        X = returns.fillna(0.0).values
        Xs = scaler.fit_transform(X)
        return pd.DataFrame(Xs, index=returns.index, columns=returns.columns)

    def _preclean_assets(self, returns: pd.DataFrame, min_history_ratio: Optional[float] = None):
        """Drop assets with too many NaNs or near-zero variance before VAR estimation."""
        if min_history_ratio is None:
            min_history_ratio = self.cfg.min_history_ratio
        n_obs = len(returns)
        thresh = int(min_history_ratio * n_obs)
        kept = returns.dropna(axis=1, thresh=thresh)
        dropped_by_history = [c for c in returns.columns if c not in kept.columns]
        # drop near-zero variance columns
        variances = kept.var()
        low_var = variances[variances <= 1e-12].index.tolist()
        kept = kept.drop(columns=low_var)
        dropped = dropped_by_history + low_var
        return kept, dropped

    def _adaptive_ridge(self, XtX_cond: float, n_obs: int, n_assets: int, base_lambda: float):
        """Heuristic: increase lambda when condition number or sample shortage is severe."""
        lam = float(base_lambda)
        # if too few observations relative to assets
        if n_obs <= n_assets:
            lam = max(lam, 1e-1)
        # scale by cond
        if np.isfinite(XtX_cond):
            if XtX_cond > 1e6:
                lam = max(lam, 10.0)
            elif XtX_cond > 1e4:
                lam = max(lam, 1.0)
            elif XtX_cond > 1e3:
                lam = max(lam, 1e-2)
        return lam

    def estimate_var1_ridge(self, returns: pd.DataFrame, ridge_lambda: float = 1e-3, auto_ridge: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Estimate VAR(1) with ridge regularization and return diagnostics.
        Returns (A_matrix, diagnostics) where A is n_assets x n_assets.
        diagnostics contains n_obs, n_assets, cond_XtX, used_ridge_lambda, eigenvalues, asset_order.
        """
        # Preclean
        R_clean, dropped_cols = self._preclean_assets(returns, min_history_ratio=self.cfg.min_history_ratio)
        n_obs, n_assets = R_clean.shape
        diagnostics = {
            'original_assets': list(returns.columns),
            'dropped_assets': dropped_cols,
            'n_obs_original': len(returns),
            'timestamp': int(time.time())
        }

        if n_assets == 0 or n_obs < 2:
            diagnostics.update({'n_assets': n_assets, 'n_obs': n_obs, 'note': 'insufficient data after cleaning'})
            return np.zeros((0, 0)), diagnostics

        X = R_clean.values[:-1, :]
        Y = R_clean.values[1:, :]

        XtX = X.T.dot(X)
        cond = np.linalg.cond(XtX) if XtX.size else np.nan
        diagnostics.update({'n_obs': n_obs, 'n_assets': n_assets, 'cond_XtX': cond})

        lam = float(ridge_lambda)
        if auto_ridge:
            lam = self._adaptive_ridge(cond, n_obs, n_assets, ridge_lambda)
        diagnostics['used_ridge_lambda'] = lam

        reg = lam * np.eye(XtX.shape[0])
        try:
            A_T = np.linalg.solve(XtX + reg, X.T.dot(Y))
        except np.linalg.LinAlgError:
            A_T = np.linalg.lstsq(XtX + reg, X.T.dot(Y), rcond=None)[0]
        A = A_T.T

        eigs = np.linalg.eigvals(A) if A.size else np.array([])
        diagnostics['eigenvalues'] = eigs.tolist()
        diagnostics['asset_order'] = list(R_clean.columns)

        return A, diagnostics

    def compute_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Return covariance matrix (numpy array) computed from aligned returns (pandas)."""
        return returns.cov().values

    # -----------------------
    # Persistence helpers for VAR outputs
    # -----------------------
    def _ensure_var_runs_table(self):
        """Create a small var_runs metadata table if it doesn't exist."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS var_runs (
            id SERIAL PRIMARY KEY,
            run_name TEXT,
            symbols TEXT[],
            a_path TEXT,
            cov_path TEXT,
            std_path TEXT,
            diag_path TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        """
        with engine.begin() as conn:
            conn.execute(text(create_sql))

    def _persist_outputs(self,
                         standardized: pd.DataFrame,
                         A: np.ndarray,
                         cov: np.ndarray,
                         diagnostics: dict,
                         output_dir: Optional[str] = None) -> dict:
        """
        Save A, cov, standardized returns, and diagnostics to disk.
        Returns dict of file paths.
        """
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if output_dir is None:
            base = VAR_OUTPUTS_DIR / ts
        else:
            base = Path(output_dir) / ts
        base.mkdir(parents=True, exist_ok=True)

        # Paths
        a_npy = base / "A_matrix.npy"
        a_csv = base / "A_matrix.csv"
        cov_npy = base / "cov_matrix.npy"
        cov_csv = base / "cov_matrix.csv"
        std_parquet = base / "standardized_returns.parquet"
        diag_json = base / "diagnostics.json"

        # Save matrices
        np.save(a_npy, A)
        np.save(cov_npy, cov)

        # Save csv with labels if diagnostics contains asset_order
        asset_order = diagnostics.get("asset_order", list(standardized.columns if standardized is not None else []))
        try:
            pd.DataFrame(A, index=asset_order, columns=asset_order).to_csv(a_csv)
            pd.DataFrame(cov, index=asset_order, columns=asset_order).to_csv(cov_csv)
        except Exception:
            pd.DataFrame(A).to_csv(a_csv, index=False)
            pd.DataFrame(cov).to_csv(cov_csv, index=False)

        # Save standardized returns parquet
        if standardized is not None:
            standardized.to_parquet(std_parquet)

        # Save diagnostics JSON
        with open(diag_json, "w") as f:
            json.dump(diagnostics, f, default=lambda o: (o.tolist() if isinstance(o, np.ndarray) else str(o)), indent=2)

        return {
            "a_npy": str(a_npy),
            "a_csv": str(a_csv),
            "cov_npy": str(cov_npy),
            "cov_csv": str(cov_csv),
            "std_parquet": str(std_parquet),
            "diag_json": str(diag_json)
        }

    # -----------------------
    # VAR & covariance pipeline (with persistence option)
    # -----------------------
    def pipeline_var_cov(self,
                         data: Dict[str, pd.DataFrame],
                         ridge_lambda: float = 1e-3,
                         auto_ridge: bool = True,
                         rolling_window: Optional[int] = None,
                         persist_outputs: bool = False,
                         output_dir: Optional[str] = None,
                         save_db_record: bool = False,
                         run_name: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
        """
        End-to-end compact pipeline:
        - builds log returns DataFrame
        - standardizes returns
        - estimates VAR(1) with ridge
        - computes covariance matrix

        Additional optional args:
        - persist_outputs: save A/cov/standardized_returns/diagnostics to disk
        - output_dir: custom output base dir
        - save_db_record: if True, create a row in var_runs pointing to saved files
        - run_name: descriptive name for DB record
        """
        returns = self.compute_log_returns(data)
        if returns.empty:
            raise ValueError("No returns available for VAR pipeline.")

        diagnostics: dict = {'input_symbols': list(returns.columns), 'n_input_obs': len(returns)}

        if rolling_window is None:
            standardized = self.standardize_returns(returns)
            A, diag = self.estimate_var1_ridge(standardized, ridge_lambda=ridge_lambda, auto_ridge=auto_ridge)
            cov = self.compute_covariance(standardized)
            diagnostics.update(diag)
        else:
            A_last, diag_agg = self.rolling_var_pipeline(returns, rolling_window, ridge_lambda, auto_ridge)
            standardized = self.standardize_returns(returns)
            cov = self.compute_covariance(standardized)
            diagnostics.update(diag_agg)
            A = A_last

        # Persist outputs if requested
        if persist_outputs:
            paths = self._persist_outputs(standardized, A, cov, diagnostics, output_dir=output_dir)
            diagnostics['persisted_paths'] = paths

            if save_db_record:
                # ensure table exists
                self._ensure_var_runs_table()
                run = run_name or f"var_run_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
                insert_sql = text("""
                    INSERT INTO var_runs (run_name, symbols, a_path, cov_path, std_path, diag_path)
                    VALUES (:run_name, :symbols, :a_path, :cov_path, :std_path, :diag_path)
                    RETURNING id, created_at;
                """)
                params = {
                    "run_name": run,
                    "symbols": list(returns.columns),
                    "a_path": paths.get('a_csv') or paths.get('a_npy'),
                    "cov_path": paths.get('cov_csv') or paths.get('cov_npy'),
                    "std_path": paths.get('std_parquet'),
                    "diag_path": paths.get('diag_json')
                }
                with engine.begin() as conn:
                    res = conn.execute(insert_sql, params)
                    row = res.fetchone()
                diagnostics['db_record'] = {"id": row[0], "created_at": str(row[1])}

        # Save diagnostics JSON (always)
        diag_path = self.save_diagnostics(diagnostics)
        diagnostics['diag_saved_path'] = diag_path

        return standardized, A, cov, diagnostics

    def rolling_var_pipeline(self, returns: pd.DataFrame, window: int, ridge_lambda: float = 1e-3, auto_ridge: bool = True):
        """
        Compute VAR estimates on sliding windows. Returns the most-recent A and aggregated diagnostics.
        """
        n = len(returns)
        if n < window:
            raise ValueError("Not enough observations for rolling window")

        As = []
        diags = []
        for start in range(0, n - window + 1):
            wnd = returns.iloc[start:start + window]
            std_wnd = self.standardize_returns(wnd)
            A_w, d_w = self.estimate_var1_ridge(std_wnd, ridge_lambda=ridge_lambda, auto_ridge=auto_ridge)
            As.append(A_w)
            diags.append(d_w)

        # choose last non-empty A
        A_last = None
        for A in reversed(As):
            if A.size:
                A_last = A
                break
        agg_diag = {
            'rolling_window': window,
            'n_estimations': len(As),
            'last_diag': diags[-1] if diags else {}
        }
        return A_last, agg_diag

    # -----------------------
    # Diagnostics persistence
    # -----------------------
    def save_diagnostics(self, diag: dict) -> str:
        """Save diagnostics JSON with timestamp and return the path."""
        ts = int(time.time())
        fname = DIAGNOSTICS_DIR / f"diag_{ts}.json"
        with open(fname, "w") as f:
            json.dump(diag, f, default=lambda o: (o.tolist() if isinstance(o, np.ndarray) else str(o)), indent=2)
        print(f"Saved diagnostics to {fname}")
        return str(fname)

    # -----------------------
    # Helpers: RSI, ATR
    # -----------------------
    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0.0)
        down = -1 * delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1/window, adjust=False).mean()
        roll_down = down.ewm(alpha=1/window, adjust=False).mean()
        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _atr(df_hlc: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df_hlc['high']
        low = df_hlc['low']
        prev_close = df_hlc['close'].shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window, min_periods=1).mean()
        return atr

# -----------------------
# CLI helper
# -----------------------
def cli_compute(symbols_csv: str, start: str, end: str):
    syms = [s.strip().upper() for s in symbols_csv.split(",")]
    fe = FeatureEngineer()
    out = fe.compute_features_bulk(syms, start, end, save=True)
    print("Completed:", list(out.keys()))
    return out

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python -m app.services.feature_engineer SYMBOL1,SYMBOL2 START END")
        print("Example: python -m app.services.feature_engineer AAPL,MSFT 2023-01-01 2024-01-01")
    else:
        cli_compute(sys.argv[1], sys.argv[2], sys.argv[3])
