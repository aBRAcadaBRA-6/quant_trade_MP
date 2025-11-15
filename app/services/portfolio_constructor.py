# app/services/portfolio_constructor.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize
from sklearn.covariance import GraphicalLassoCV
from sqlalchemy import text
from pathlib import Path

from app.models.database import engine

# ensure data folder exists
Path("data/processed").mkdir(parents=True, exist_ok=True)

PORTFOLIO_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS portfolio_runs (
    id SERIAL PRIMARY KEY,
    run_name TEXT,
    symbols TEXT[],
    weights_json TEXT,
    method TEXT,
    link_run_id INTEGER,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
"""


@dataclass
class PCOptions:
    # optimization / portfolio params
    max_weight: float = 0.25
    min_weight: float = 0.0
    allow_short: bool = False
    method: Literal["mean_variance", "minvar", "sparse_mean_reverting"] = "sparse_mean_reverting"
    risk_aversion: float = 1.0  # used for mean-variance (gamma)
    sparsity_k: int = 10        # for sparse_mean_reverting
    sparsity_keep_signed: bool = False  # keep sign when picking top-k (True) or by abs (False)
    cov_ridge: float = 1e-6     # ridge added to diagonal of covariance for numerical stability
    use_graphical_lasso: bool = False  # whether to estimate precision via GraphicalLassoCV
    gl_alphas: Optional[List[float]] = None  # optional alpha grid for GraphicalLassoCV
    persist: bool = True        # persist portfolio run to DB
    run_name: Optional[str] = None
    verbose: bool = True


def _ensure_portfolio_runs_table():
    with engine.begin() as conn:
        conn.execute(text(PORTFOLIO_RUNS_TABLE_SQL))


def _normalize_weights(w: np.ndarray, min_w: float, max_w: float, allow_short: bool) -> np.ndarray:
    """Clip then renormalize to sum=1 (if possible)."""
    if allow_short:
        # permit negative weights, clip symmetric by max magnitude
        lb, ub = -max_w, max_w
    else:
        lb, ub = min_w, max_w
    w_clipped = np.clip(w, lb, ub)
    s = np.sum(w_clipped)
    if abs(s) < 1e-12:
        # fallback equal weights
        n = len(w_clipped)
        return np.repeat(1.0 / n, n)
    return w_clipped / s


def _mv_objective(w: np.ndarray, cov: np.ndarray, mu: np.ndarray, gamma: float) -> float:
    return 0.5 * w.dot(cov).dot(w) - gamma * w.dot(mu)


def _mv_jac(w: np.ndarray, cov: np.ndarray, mu: np.ndarray, gamma: float) -> np.ndarray:
    return cov.dot(w) - gamma * mu


def mean_variance_weights(mu: pd.Series,
                          cov: np.ndarray,
                          opts: PCOptions,
                          w_prev: Optional[pd.Series] = None) -> pd.Series:
    assets = list(mu.index)
    n = len(assets)
    x0 = np.repeat(1.0 / n, n)

    lb = opts.min_weight
    ub = opts.max_weight if not opts.allow_short else max(opts.max_weight, 1.0)
    bounds = [(lb, ub) for _ in range(n)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    res = minimize(fun=_mv_objective,
                   x0=x0,
                   jac=_mv_jac,
                   args=(cov, mu.values, opts.risk_aversion),
                   bounds=bounds,
                   constraints=cons,
                   method='SLSQP',
                   options={'ftol': 1e-9, 'maxiter': 500})
    if opts.verbose and not res.success:
        print("MV optimization warning:", res.message)
    w = _normalize_weights(res.x, opts.min_weight, opts.max_weight, opts.allow_short)
    return pd.Series(w, index=assets)


def minimum_variance_weights(cov: np.ndarray, assets: List[str], opts: PCOptions) -> pd.Series:
    n = cov.shape[0]
    x0 = np.repeat(1.0 / n, n)
    lb = opts.min_weight
    ub = opts.max_weight if not opts.allow_short else max(opts.max_weight, 1.0)
    bounds = [(lb, ub) for _ in range(n)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def obj(w): return 0.5 * w.dot(cov).dot(w)
    def jac(w): return cov.dot(w)

    res = minimize(fun=obj, x0=x0, jac=jac, bounds=bounds, constraints=cons, method='SLSQP')
    if opts.verbose and not res.success:
        print("MinVar optimization warning:", res.message)
    w = _normalize_weights(res.x, opts.min_weight, opts.max_weight, opts.allow_short)
    return pd.Series(w, index=assets)


def _safely_symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def sparse_precision_via_glasso(returns: pd.DataFrame, opts: PCOptions) -> np.ndarray:
    if opts.gl_alphas is not None:
        model = GraphicalLassoCV(alphas=opts.gl_alphas)
    else:
        model = GraphicalLassoCV()
    model.fit(returns)
    return model.precision_


def box_tiao_decomposition(A: np.ndarray, cov: np.ndarray, ridge: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve generalized eigenproblem A v = lambda cov v robustly.
    If cov is singular, add ridge to diagonal.
    Returns (eigenvalues, eigenvectors) real-valued.
    """
    cov_reg = cov + ridge * np.eye(cov.shape[0])
    # Convert generalized eigenproblem to standard: inv(cov_reg) @ A v = lambda v
    try:
        inv_cov = np.linalg.inv(cov_reg)
        M = inv_cov.dot(A)
        eigvals, eigvecs = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        # fallback to scipy generalized eig
        eigvals, eigvecs = linalg.eig(A, cov_reg)
    # ensure real
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    return eigvals, eigvecs


def select_sparse_portfolio_from_eigen(eigvecs: np.ndarray,
                                       eigvals: np.ndarray,
                                       symbols: List[str],
                                       k: int,
                                       keep_signed: bool = False,
                                       normalize_sum: bool = True) -> pd.Series:
    """
    Pick the eigenvector corresponding to smallest |eig| (most mean-reverting),
    then keep top-k entries by abs (or signed) and normalize to sum=1.
    Returns pandas Series indexed by symbols (zeros for dropped assets).
    """
    if eigvals.size == 0:
        raise ValueError("Empty eigenvalues")
    idx = int(np.argmin(np.abs(eigvals)))
    v = eigvecs[:, idx]
    # convert nan/infs
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    abs_v = np.abs(v)
    if k >= len(v):
        selected = np.arange(len(v))
    else:
        if keep_signed:
            # pick top-k by signed magnitude (keep sign)
            order = np.argsort(-abs_v)
            selected = order[:k]
        else:
            selected = np.argsort(abs_v)[-k:]
    sparse = np.zeros_like(v, dtype=float)
    sparse[selected] = v[selected]
    if normalize_sum:
        s = np.sum(sparse)
        if abs(s) < 1e-12:
            # normalize by abs sum
            s = np.sum(np.abs(sparse))
            if abs(s) < 1e-12:
                # fallback uniform on selected
                cnt = max(1, len(selected))
                sparse[selected] = 1.0 / cnt
            else:
                sparse = sparse / s
        else:
            sparse = sparse / s
    return pd.Series(sparse, index=symbols)


def construct_sparse_mean_reverting(returns: pd.DataFrame,
                                    A: np.ndarray,
                                    cov: np.ndarray,
                                    symbols: List[str],
                                    opts: PCOptions) -> pd.Series:
    """
    Full pipeline for sparse mean-reverting portfolio:
    - Optional use of GraphicalLasso to get precision (unused directly here, but helpful if you want to invert)
    - Box-Tiao decomposition
    - Select top-k and normalize
    """
    # ensure shapes align
    n_assets = len(symbols)
    if cov.shape != (n_assets, n_assets):
        raise ValueError("cov shape mismatch")

    cov_reg = cov + opts.cov_ridge * np.eye(n_assets)
    eigvals, eigvecs = box_tiao_decomposition(A, cov_reg, ridge=opts.cov_ridge)
    weights = select_sparse_portfolio_from_eigen(eigvecs, eigvals, symbols, k=opts.sparsity_k,
                                                keep_signed=opts.sparsity_keep_signed)
    return weights


def persist_portfolio(weights: pd.Series,
                      method: str,
                      metrics: Dict,
                      link_run_id: Optional[int] = None,
                      run_name: Optional[str] = None) -> Dict:
    _ensure_portfolio_runs_table()
    run_name = run_name or f"port_{int(time.time())}"
    symbols = list(weights.index)
    weights_json = json.dumps(weights.to_dict())
    metrics_json = json.dumps(metrics)
    insert_sql = text("""
        INSERT INTO portfolio_runs (run_name, symbols, weights_json, method, link_run_id, metrics)
        VALUES (:run_name, :symbols, :weights_json, :method, :link_run_id, :metrics)
        RETURNING id, created_at;
    """)
    params = {
        "run_name": run_name,
        "symbols": symbols,
        "weights_json": weights_json,
        "method": method,
        "link_run_id": link_run_id,
        "metrics": metrics_json
    }
    with engine.begin() as conn:
        res = conn.execute(insert_sql, params)
        row = res.fetchone()
    return {"id": row[0], "created_at": str(row[1])}


def construct_portfolio_from_var_and_cov(standardized: pd.DataFrame,
                                         A: np.ndarray,
                                         cov: np.ndarray,
                                         raw_returns: Optional[pd.DataFrame],
                                         symbols: List[str],
                                         opts: PCOptions,
                                         w_prev: Optional[pd.Series] = None,
                                         link_run_id: Optional[int] = None) -> Tuple[pd.Series, Dict]:
    """
    Unified entrypoint:
    - generate expected returns from VAR (simple scaling by sigma)
    - choose method according to opts and produce weights
    - persist optionally and return metrics
    """
    # compute expected returns as simple scaled forecast: r_pred_std * sigma_raw
    r_last = standardized.iloc[-1]
    r_pred_std = A.dot(r_last.values)
    mu_std = pd.Series(r_pred_std, index=standardized.columns)

    if raw_returns is not None:
        # align columns
        raw = raw_returns[standardized.columns].dropna(how="all")
        sigma = raw.std(ddof=1)
        mu = mu_std * sigma
    else:
        # fallback: use standardized scores (ranking only, small magnitudes)
        mu = mu_std

    # Ensure cov is regularized
    cov_reg = cov + opts.cov_ridge * np.eye(cov.shape[0])

    if opts.method == "mean_variance":
        w = mean_variance_weights(mu, cov_reg, opts, w_prev=w_prev)
    elif opts.method == "minvar":
        w = minimum_variance_weights(cov_reg, list(standardized.columns), opts)
    elif opts.method == "sparse_mean_reverting":
        w = construct_sparse_mean_reverting(raw if raw_returns is not None else standardized,
                                            A, cov_reg, list(standardized.columns), opts)
        # after sparse selection, clip/normalize according to long-only constraints
        w = _normalize_weights(w.values, opts.min_weight, opts.max_weight, opts.allow_short)
        w = pd.Series(w, index=standardized.columns)
    else:
        raise ValueError("unknown method")

    # compute metrics
    portfolio_var = float(w.values.dot(cov_reg).dot(w.values))
    portfolio_std = float(np.sqrt(portfolio_var))
    expected_return = float(w.dot(mu))
    metrics = {"expected_return": expected_return, "portfolio_std": portfolio_std, "n_assets": int((w != 0).sum())}

    if opts.persist:
        row = persist_portfolio(w, opts.method, metrics, link_run_id=link_run_id, run_name=opts.run_name)
        metrics["db_row"] = row

    return pd.Series(w, index=standardized.columns), metrics


# -----------------------
# CLI / quick test
# -----------------------
if __name__ == "__main__":
    # quick demo: requires your FeatureEngineer + DataFetcher
    try:
        from app.services.data_fetcher import DataFetcher
        from app.services.feature_engineer import FeatureEngineer
    except Exception as e:
        print("Imports failed (ensure project is on PYTHONPATH).", e)
        raise

    syms = ["AAPL", "MSFT", "GOOGL"]
    fetcher = DataFetcher()
    data = fetcher.load_from_db(syms, "2023-01-01", "2024-01-01")
    fe = FeatureEngineer()
    standardized, A, cov, diag = fe.pipeline_var_cov(data, ridge_lambda=1e-3, persist_outputs=False)
    raw_returns = fe.compute_log_returns(data)

    opts = PCOptions(
        method="sparse_mean_reverting",
        sparsity_k=3,
        max_weight=0.5,
        min_weight=0.0,
        allow_short=False,
        persist=True,
        run_name="demo_sparse_mr"
    )

    weights, metrics = construct_portfolio_from_var_and_cov(standardized, A, cov, raw_returns, syms, opts)
    print("Weights:\n", weights)
    print("Metrics:\n", metrics)
