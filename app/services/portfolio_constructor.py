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


# -------------------------
# DB helpers
# -------------------------
def _ensure_portfolio_runs_table():
    with engine.begin() as conn:
        conn.execute(text(PORTFOLIO_RUNS_TABLE_SQL))


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


# -------------------------
# Math / utility helpers
# -------------------------
def _l1_normalize_preserve_sign(w: np.ndarray) -> np.ndarray:
    """Normalize vector so sum(abs(w)) == 1 while preserving sign."""
    w = np.asarray(w, dtype=float)
    denom = np.sum(np.abs(w))
    if denom == 0:
        return w
    return w / denom


def _project_to_long_only(w: np.ndarray) -> np.ndarray:
    """Project weights vector to long-only simplex (non-negative, sum to 1)."""
    w = np.asarray(w, dtype=float)
    w_pos = np.maximum(w, 0.0)
    s = w_pos.sum()
    if s <= 0:
        # fallback to equal weights
        n = len(w)
        return np.ones(n) / float(n)
    return w_pos / s


def _clamp_max_weight(w: np.ndarray, max_abs: float) -> np.ndarray:
    """Clamp absolute weight per asset then renormalize to sum to 1 (preserving signs)."""
    if max_abs is None or max_abs <= 0:
        return w
    w = np.array(w, dtype=float)
    # clamp by magnitude
    w = np.sign(w) * np.minimum(np.abs(w), max_abs)
    s = np.sum(np.abs(w))
    if s == 0:
        return w
    return w / s


def _normalize_weights_for_output(w: np.ndarray, min_w: float, max_w: float, allow_short: bool) -> np.ndarray:
    """Clip then renormalize to sum=1 (if possible)."""
    if allow_short:
        lb, ub = -max_w, max_w
    else:
        lb, ub = min_w, max_w
    w_clipped = np.clip(w, lb, ub)
    s = np.sum(w_clipped)
    if abs(s) < 1e-12:
        n = len(w_clipped)
        return np.repeat(1.0 / n, n)
    return w_clipped / s


# -------------------------
# Optimization helpers
# -------------------------
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
    w = _normalize_weights_for_output(res.x, opts.min_weight, opts.max_weight, opts.allow_short)
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
    w = _normalize_weights_for_output(res.x, opts.min_weight, opts.max_weight, opts.allow_short)
    return pd.Series(w, index=assets)


# -------------------------
# Sparse MR (Box-Tiao) helpers
# -------------------------
def box_tiao_decomposition(A: np.ndarray, cov: np.ndarray, ridge: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve generalized eigenproblem A v = lambda cov v robustly.
    If cov is singular, add ridge to diagonal.
    Returns (eigenvalues, eigenvectors) real-valued.
    """
    cov_reg = cov + ridge * np.eye(cov.shape[0])
    try:
        inv_cov = np.linalg.inv(cov_reg)
        M = inv_cov.dot(A)
        eigvals, eigvecs = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = linalg.eig(A, cov_reg)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    return eigvals, eigvecs


def select_sparse_portfolio_from_eigen(eigvecs: np.ndarray,
                                       eigvals: np.ndarray,
                                       symbols: List[str],
                                       k: int,
                                       keep_signed: bool = False,
                                       long_only: bool = True,
                                       max_weight: Optional[float] = None) -> Tuple[pd.Series, dict]:
    """
    Improved selection:
    - choose eigenvector (smallest |eig|)
    - choose top-k by abs (or signed if keep_signed)
    - If long_only: use absolute values of selected entries and normalize among selected assets
    - If long-short allowed: preserve sign and L1-normalize (sum abs = 1)
    - clamp max weight then final renormalize
    """
    if eigvals.size == 0:
        raise ValueError("Empty eigenvalues")
    idx = int(np.argmin(np.abs(eigvals)))
    v = eigvecs[:, idx].astype(float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(v)
    k = max(1, min(k, n))
    abs_v = np.abs(v)

    # pick indices
    if k >= n:
        selected = np.arange(n)
    else:
        selected = np.argsort(abs_v)[-k:]

    sparse = np.zeros_like(v, dtype=float)

    if long_only:
        # For long-only, use absolute magnitudes of the selected entries (no sign)
        selected_vals = np.abs(v[selected])
        # If all zeros (rare), fallback to uniform on selected
        if selected_vals.sum() == 0:
            selected_vals = np.ones_like(selected_vals) / float(len(selected_vals))
        # fill only selected positions with positive magnitudes
        for i, idx_sel in enumerate(selected):
            sparse[idx_sel] = float(selected_vals[i])
        # normalize among selected by sum (so sum(abs)=1 among selected)
        denom = np.sum(np.abs(sparse))
        if denom == 0:
            # safety: equal weights across selected
            cnt = max(1, len(selected))
            for idx_sel in selected:
                sparse[idx_sel] = 1.0 / cnt
        else:
            sparse = sparse / np.sum(np.abs(sparse))
    else:
        # preserve signed values for long-short portfolio (or when keep_signed True)
        sparse[selected] = v[selected]
        # normalize by L1 to preserve sign structure (sum(abs)=1)
        denom = np.sum(np.abs(sparse))
        if denom == 0:
            cnt = max(1, len(selected))
            sparse[selected] = 1.0 / cnt
            sparse = sparse / np.sum(np.abs(sparse))
        else:
            sparse = sparse / denom

    # clamp per-asset absolute weight (preserve sign for long-short)
    if max_weight is not None and max_weight > 0:
        sparse = _clamp_max_weight(sparse, max_weight)

    # final formatting
    weights = pd.Series(sparse, index=symbols)
    diagnostics = {
        "chosen_eig_index": int(idx),
        "chosen_eig_value": float(eigvals[idx]),
        "selected_indices": [int(i) for i in selected],
        "selected_symbols": [symbols[int(i)] for i in selected],
        "sparsity_k": int(k),
        "long_only": bool(long_only),
        "max_weight": float(max_weight) if max_weight is not None else None
    }
    return weights, diagnostics



def construct_sparse_mean_reverting(returns: pd.DataFrame,
                                    A: np.ndarray,
                                    cov: np.ndarray,
                                    symbols: List[str],
                                    opts: PCOptions) -> Tuple[pd.Series, dict]:
    """
    Construct a sparse mean-reverting portfolio:
    - regularize cov
    - compute Box-Tiao decomposition
    - select sparse vector
    """
    n_assets = len(symbols)
    if cov.shape != (n_assets, n_assets):
        raise ValueError("cov shape mismatch")

    cov_reg = cov + opts.cov_ridge * np.eye(n_assets)
    eigvals, eigvecs = box_tiao_decomposition(A, cov_reg, ridge=opts.cov_ridge)
    weights, diag = select_sparse_portfolio_from_eigen(
        eigvecs, eigvals, symbols,
        k=opts.sparsity_k,
        keep_signed=opts.sparsity_keep_signed,
        long_only=not opts.allow_short,
        max_weight=opts.max_weight
    )
    return weights, diag


# -------------------------
# Public entrypoint
# -------------------------
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
    - standardized: standardized returns DataFrame (columns match symbols)
    - A: VAR(1) coefficient matrix
    - cov: covariance matrix
    - raw_returns: raw log returns DataFrame (optional) used to scale predictions
    - symbols: list of symbols in same order as columns
    - opts: PCOptions
    Returns: (weights Series indexed by symbols, metrics dict)
    """
    # basic checks
    n = A.shape[0]
    if cov.shape[0] != n:
        raise ValueError("var_matrix and cov_matrix dimension mismatch")
    if len(symbols) != n:
        raise ValueError("symbols length must match matrix dimension")

    # compute simple expected returns from VAR forecast
    r_last = standardized.iloc[-1].values
    r_pred_std = A.dot(r_last)  # standardized forecast
    mu_std = pd.Series(r_pred_std, index=standardized.columns)

    if raw_returns is not None:
        raw = raw_returns.reindex(columns=standardized.columns).dropna(how="all")
        sigma = raw.std(ddof=1)
        mu = mu_std * sigma
    else:
        mu = mu_std

    cov_reg = cov + opts.cov_ridge * np.eye(cov.shape[0])

    metrics: Dict = {"method": opts.method}

    if opts.method == "mean_variance":
        w = mean_variance_weights(mu, cov_reg, opts, w_prev=w_prev)
        metrics["submethod"] = "mean_variance"

    elif opts.method == "minvar" or opts.method == "minimum_variance":
        w = minimum_variance_weights(cov_reg, list(standardized.columns), opts)
        metrics["submethod"] = "minvar"

    elif opts.method == "sparse_mean_reverting":
        w, diag = construct_sparse_mean_reverting(raw if raw_returns is not None else standardized, A, cov_reg, list(standardized.columns), opts)
        metrics.update(diag)
        # final ensure clipping and formatting per opts
        w = _normalize_weights_for_output(w.values, opts.min_weight, opts.max_weight, opts.allow_short)
        w = pd.Series(w, index=standardized.columns)

    else:
        raise ValueError(f"unknown method {opts.method}")

    # diagnostics
    portfolio_var = float(w.values.dot(cov_reg).dot(w.values))
    portfolio_std = float(np.sqrt(max(portfolio_var, 0.0)))
    expected_return = float(w.dot(mu))
    metrics.update({
        "expected_return": expected_return,
        "portfolio_std": portfolio_std,
        "n_assets": int((w != 0).sum())
    })

    # persist if requested
    if opts.persist:
        try:
            row = persist_portfolio(w, opts.method, metrics, link_run_id=link_run_id, run_name=opts.run_name)
            metrics["db_row"] = row
        except Exception as e:
            if opts.verbose:
                print("Warning: failed to persist portfolio run:", e)

    return pd.Series(w, index=standardized.columns), metrics
