# app/services/performance_evaluator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class PerfConfig:
    trading_days: int = 252
    risk_free_rate: float = 0.02  # annual risk-free default

class PerformanceEvaluator:
    """
    Performance utilities.

    Backwards-compatible helpers kept:
      - daily_returns(equity: Series) -> Series
      - cagr(equity: Series) -> float
      - annualized_vol(returns) -> float
      - sharpe, sortino, max_drawdown, tail_var, summarize

    Test-friendly API (recommended):
      - calculate_returns(equity_curve) -> pd.Series
      - sharpe_ratio(returns, risk_free_rate=None) -> float
      - sortino_ratio(returns, risk_free_rate=None) -> float
      - max_drawdown(equity_curve) -> float  # returns positive fraction
      - summary_metrics(equity_curve) -> Dict[str,float]
    """

    def __init__(self, cfg: PerfConfig = PerfConfig()):
        self.cfg = cfg

    # ----------------------
    # Basic helpers (your old API)
    # ----------------------
    @staticmethod
    def daily_returns(equity: pd.Series) -> pd.Series:
        """Percent-change returns (drop first NaN)."""
        return equity.pct_change().dropna()

    @staticmethod
    def cagr(equity: pd.Series) -> float:
        """Compound annual growth rate from equity series (index must be datetime)."""
        if equity.empty:
            return 0.0
        start, end = equity.index[0], equity.index[-1]
        total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
        days = (end - start).days
        if days <= 0:
            return 0.0
        years = days / 365.25
        if years <= 0:
            return 0.0
        return float((1 + total_ret) ** (1.0 / years) - 1.0)

    @staticmethod
    def annualized_vol(returns: pd.Series, trading_days: int = 252) -> float:
        """Annualized volatility from period returns."""
        if returns.empty:
            return 0.0
        return float(returns.std(ddof=1) * np.sqrt(trading_days))

    @staticmethod
    def tail_var(returns: pd.Series, q: float = 0.05) -> float:
        """Return the q-quantile (left tail) of returns."""
        if returns.empty:
            return 0.0
        return float(returns.quantile(q))

    # ----------------------
    # Test-friendly API (aliases + robust implementations)
    # ----------------------
    def calculate_returns(self, equity_curve: pd.DataFrame | pd.Series) -> pd.Series:
        """
        Accepts:
          - pandas Series of equity values
          - pandas DataFrame with 'equity' column

        Returns percent-change series (dropna).
        """
        if isinstance(equity_curve, pd.DataFrame):
            if "equity" not in equity_curve.columns:
                raise ValueError("equity_curve DataFrame must contain an 'equity' column")
            series = equity_curve["equity"]
        elif isinstance(equity_curve, pd.Series):
            series = equity_curve
        else:
            raise TypeError("equity_curve must be a pandas Series or DataFrame")

        return self.daily_returns(series)

    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """Annualized Sharpe: sqrt(T) * (mean_ret - rf/T) / std_ret"""
        if risk_free_rate is None:
            risk_free_rate = self.cfg.risk_free_rate
        if returns.empty:
            return float("nan")
        rf_per_period = risk_free_rate / self.cfg.trading_days
        excess = returns - rf_per_period
        std = returns.std(ddof=1)
        if std == 0 or np.isnan(std):
            return float("nan")
        return float(np.sqrt(self.cfg.trading_days) * (excess.mean() / std))

    def sortino_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """Annualized Sortino ratio (penalizes downside volatility only)."""
        if risk_free_rate is None:
            risk_free_rate = self.cfg.risk_free_rate
        if returns.empty:
            return float("nan")
        rf_per_period = risk_free_rate / self.cfg.trading_days
        excess = returns - rf_per_period
        downside = excess[excess < 0]
        if downside.empty:
            return float("inf")
        downside_std = downside.std(ddof=1)
        if downside_std == 0 or np.isnan(downside_std):
            return float("inf")
        return float(np.sqrt(self.cfg.trading_days) * (excess.mean() / downside_std))

    def max_drawdown(self, equity_curve: pd.DataFrame | pd.Series) -> float:
        """
        Compute maximum drawdown as a positive fraction (e.g. 0.25 for 25%).
        Works with Series or DataFrame with 'equity'.
        """
        if isinstance(equity_curve, pd.DataFrame):
            if "equity" not in equity_curve.columns:
                raise ValueError("equity_curve DataFrame must contain an 'equity' column")
            eq = equity_curve["equity"]
        elif isinstance(equity_curve, pd.Series):
            eq = equity_curve
        else:
            raise TypeError("equity_curve must be a pandas Series or DataFrame")

        running_max = eq.cummax()
        drawdowns = (running_max - eq) / running_max
        max_dd = drawdowns.max()
        return float(max_dd if not np.isnan(max_dd) else 0.0)

    def summary_metrics(self, equity_curve: pd.DataFrame | pd.Series, risk_free_rate: Optional[float] = None) -> Dict:
        """
        Convenience aggregator producing:
          {sharpe, sortino, max_drawdown, annual_return, annual_vol, n_periods}
        """
        # convert to Series if DataFrame
        if isinstance(equity_curve, pd.DataFrame):
            if "equity" not in equity_curve.columns:
                raise ValueError("equity_curve DataFrame must contain an 'equity' column")
            eq = equity_curve["equity"]
        elif isinstance(equity_curve, pd.Series):
            eq = equity_curve
        else:
            raise TypeError("equity_curve must be a pandas Series or DataFrame")

        rets = self.daily_returns(eq)
        if rets.empty:
            return {
                "sharpe": float("nan"),
                "sortino": float("nan"),
                "max_drawdown": 0.0,
                "annual_return": float("nan"),
                "annual_vol": float("nan"),
                "n_periods": 0
            }

        sharpe = self.sharpe_ratio(rets, risk_free_rate=risk_free_rate)
        sortino = self.sortino_ratio(rets, risk_free_rate=risk_free_rate)
        max_dd = self.max_drawdown(eq)

        # cumulative and annualized return (geometric)
        total_periods = len(rets)
        cumulative_return = (1.0 + rets).prod() - 1.0
        annual_return = float((1.0 + cumulative_return) ** (self.cfg.trading_days / total_periods) - 1.0) if total_periods > 0 else float("nan")
        annual_vol = float(rets.std(ddof=1) * np.sqrt(self.cfg.trading_days)) if total_periods > 0 else float("nan")

        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_dd),
            "annual_return": float(annual_return),
            "annual_vol": float(annual_vol),
            "n_periods": int(total_periods)
        }

    # ----------------------
    # Backwards-compatible convenience wrapper matching your old summarize()
    # ----------------------
    def summarize(self, equity_df: pd.DataFrame) -> Dict:
        """Compat wrapper (keeps previous API name)."""
        # previous implementation expected a DataFrame with 'equity' column
        return self.summary_metrics(equity_df)
