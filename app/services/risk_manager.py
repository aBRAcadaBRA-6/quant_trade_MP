# app/services/risk_manager.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from app.models.database import User  # your SQLAlchemy User model

@dataclass
class RiskConfig:
    max_position_fraction: float = 0.20  # fraction of capital per position (default 20%)
    max_portfolio_exposure: float = 1.0  # total exposure (longs) <= this (1.0 = fully invested)
    min_cash_buffer: float = 0.0         # keep at least this fraction of capital in cash
    use_half_kelly: bool = True

class RiskManager:
    """
    Validate and (optionally) adjust signals/weights to satisfy user's risk profile.

    High-level method:
      validate_signal(signal_or_weights, current_equity, peak_equity) -> {approved, reason, adjusted_signal}
    """

    def __init__(self, user: User, cfg: RiskConfig = RiskConfig()):
        self.user = user
        self.cfg = cfg
        # use user.capital if present, otherwise require caller to pass capital
        self.capital = float(user.capital) if getattr(user, "capital", None) is not None else None

    # ------------------------
    # Basic checks & utilities
    # ------------------------
    def check_position_sizes(self, weights: Dict[str, float], capital: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check each position notional <= max_position_fraction * capital.
        weights are fractional (sum to 1). capital in USD.
        """
        if capital is None:
            capital = self.capital
        if capital is None:
            return False, "Capital not provided"

        max_notional = self.cfg.max_position_fraction * capital
        for s, w in (weights or {}).items():
            notional = abs(w) * capital
            if notional > max_notional + 1e-9:
                return False, f"Position {s} notional {notional:.2f} exceeds max {max_notional:.2f}"
        return True, "Position sizes OK"

    def check_portfolio_exposure(self, weights: Dict[str, float]) -> Tuple[bool, str]:
        """
        Ensure total long exposure <= limit and total short exposure <= limit (if any).
        """
        longs = sum([w for w in (weights or {}).values() if w > 0])
        shorts = -sum([w for w in (weights or {}).values() if w < 0])
        if longs > self.cfg.max_portfolio_exposure + 1e-9:
            return False, f"Long exposure {longs:.3f} > allowed {self.cfg.max_portfolio_exposure}"
        # we don't enforce short limit here unless you configure it
        return True, "Exposure OK"

    def check_drawdown(self, current_equity: float, peak_equity: float) -> Tuple[bool, str]:
        """
        Check drawdown against user.drawdown_limit
        """
        if peak_equity <= 0:
            return True, "No peak equity recorded"
        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown > getattr(self.user, "drawdown_limit", 1.0):
            return False, f"Drawdown {drawdown*100:.2f}% exceeds user limit {getattr(self.user, 'drawdown_limit', 0.0)*100:.2f}%"
        return True, "Drawdown OK"

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Compute Kelly fraction; return half-Kelly if config requests it.
        """
        if avg_loss == 0:
            return 0.0
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        kelly = (p * b - q) / b
        if np.isnan(kelly) or kelly <= 0:
            return 0.0
        if self.cfg.use_half_kelly:
            return max(0.0, kelly / 2.0)
        return max(0.0, kelly)

    def enforce_limits_on_weights(self, weights: Dict[str, float], capital: Optional[float] = None) -> Dict[str, float]:
        """
        Enforce max position fraction and min cash buffer by rescaling weights.
        Returns adjusted weights (still sum to <= 1 - min_cash_buffer).
        """
        if capital is None:
            capital = self.capital
        if capital is None:
            raise ValueError("Capital required to enforce limits")

        max_pos_notional = self.cfg.max_position_fraction * capital
        # first clamp individual weights by max fraction
        max_w = self.cfg.max_position_fraction
        adj = {s: np.sign(w) * min(abs(w), max_w) for s, w in (weights or {}).items()}

        # now rescale to ensure total long exposure <= (1 - min_cash_buffer)
        total_long = sum([w for w in adj.values() if w > 0])
        allowed_long = min(self.cfg.max_portfolio_exposure, 1.0 - self.cfg.min_cash_buffer)
        if total_long > allowed_long and total_long > 0:
            scale = allowed_long / total_long
            for s in adj:
                if adj[s] > 0:
                    adj[s] = adj[s] * scale

        # final normalization so weights sum to <= 1 (respect sign)
        total = sum(adj.values())
        if abs(total) > 1e-12 and abs(total - 1.0) > 1e-9:
            adj = {s: w / total for s, w in adj.items()}

        return adj

    # ------------------------
    # High-level validator (compatibility)
    # ------------------------
    def validate_signal(self,
                        signal_or_weights: Any,
                        current_equity: float,
                        peak_equity: float,
                        capital: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate a trading signal or weight dict.

        Args:
          - signal_or_weights: either the `signal` dict produced by TradeSignalEngine.generate_signal()
                               (expected to contain 'portfolio' key with weights) or a raw weights dict {sym: w}
          - current_equity: current portfolio equity (USD)
          - peak_equity: historical peak equity (USD)
          - capital: optional capital to use for notional checks (defaults to self.capital or current_equity)

        Returns:
          {
            'approved': bool,
            'reason': str,
            'adjusted_signal': dict (contains 'portfolio' key with possibly-adjusted weights)
          }
        """
        # extract weights
        if isinstance(signal_or_weights, dict) and 'portfolio' in signal_or_weights:
            weights = signal_or_weights.get('portfolio') or {}
        elif isinstance(signal_or_weights, dict):
            weights = signal_or_weights
            signal_or_weights = {'portfolio': dict(weights)}
        else:
            # unsupported type
            return {'approved': False, 'reason': 'Unsupported signal type', 'adjusted_signal': None}

        if capital is None:
            capital = self.capital if self.capital is not None else float(current_equity)

        # 1) drawdown check
        dd_ok, dd_msg = self.check_drawdown(current_equity, peak_equity)
        if not dd_ok:
            return {'approved': False, 'reason': dd_msg, 'adjusted_signal': {'portfolio': weights}}

        # 2) position size check
        size_ok, size_msg = self.check_position_sizes(weights, capital=capital)
        if size_ok:
            # also check exposure
            exp_ok, exp_msg = self.check_portfolio_exposure(weights)
            if not exp_ok:
                return {'approved': False, 'reason': exp_msg, 'adjusted_signal': {'portfolio': weights}}
            return {'approved': True, 'reason': 'All risk checks passed', 'adjusted_signal': {'portfolio': weights}}
        else:
            # attempt to fix by enforcing per-position limits and re-check
            try:
                adjusted = self.enforce_limits_on_weights(weights, capital=capital)
            except Exception as e:
                return {'approved': False, 'reason': f"Position size violation and automatic adjustment failed: {e}", 'adjusted_signal': {'portfolio': weights}}

            # re-check sizes & exposure on adjusted
            size_ok2, size_msg2 = self.check_position_sizes(adjusted, capital=capital)
            exp_ok2, exp_msg2 = self.check_portfolio_exposure(adjusted)
            if size_ok2 and exp_ok2:
                return {'approved': True, 'reason': 'Adjusted to satisfy limits', 'adjusted_signal': {'portfolio': adjusted}}
            else:
                # cannot satisfy constraints automatically
                reason = f"Adjusted but still failing: sizes: {size_msg2}; exposure: {exp_msg2}"
                return {'approved': False, 'reason': reason, 'adjusted_signal': {'portfolio': adjusted}}
