# app/services/trade_signal_engine.py
from __future__ import annotations
from enum import Enum
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

# Backwards-compatibility alias: some code/tests import `Signal`
Signal = SignalType

class TradeSignalEngine:
    """
    Produce actionable trading signals given:
     - target portfolio weights (weights sum to 1)
     - current holdings (weights or quantities)
     - current prices

    Provides:
     - generate_portfolio_rebalance(...)  # your detailed rebalance planner
     - calculate_portfolio_value(...)     # compatibility helper for the master test
     - generate_signal(...)               # simple mean-reversion wrapper used by master test
    """

    def __init__(self, deviation_threshold: float = 0.02, min_trade_notional: float = 50.0):
        """
        deviation_threshold: fraction difference to trigger trade (e.g. 0.02 = 2%).
        min_trade_notional: don't trade very tiny amounts (USD) in rebalance planner.
        """
        self.deviation_threshold = float(deviation_threshold)
        self.min_trade_notional = float(min_trade_notional)

    # ----------------------
    # Compatibility helpers
    # ----------------------
    def calculate_portfolio_value(self, portfolio: Dict[str, float], current_prices: Dict[str, float]) -> float:
        """
        Calculate a value-like scalar from portfolio weights and current prices.

        - If `portfolio` contains fractional weights that sum to 1, this returns a weighted
          sum of prices (useful as a proxy to compute deviation in the test harness).
        - If `portfolio` contains position quantities, this will compute notional value
          (quantity * price) for each symbol.
        This function is intentionally generic to support both uses.
        """
        if not portfolio:
            return 0.0

        # Heuristic: if weights sum to ~1, treat as fractional weights
        s = sum(abs(v) for v in portfolio.values())
        if 0.9 <= s <= 1.1:
            # fractional weights -> weighted sum of prices
            total = 0.0
            for sym, w in portfolio.items():
                price = current_prices.get(sym)
                if price is None:
                    continue
                total += float(w) * float(price)
            return float(total)
        else:
            # treat values as quantities -> compute notional
            total = 0.0
            for sym, qty in portfolio.items():
                price = current_prices.get(sym)
                if price is None:
                    continue
                total += float(qty) * float(price)
            return float(total)

    def generate_signal(self, current_value: float, target_value: float, portfolio: Dict[str, float]) -> Dict[str, Any]:
        """
        Simple mean-reversion signal generator (compatibility wrapper).
        Uses the same threshold logic used across the project:
          - current > target * (1+threshold) -> SELL
          - current < target * (1-threshold) -> BUY
          - otherwise -> HOLD

        Returns a dict:
          {
            "signal": SignalType,
            "deviation": float,
            "current_value": float,
            "target_value": float,
            "message": str,
            "portfolio": dict
          }
        """
        if target_value == 0:
            deviation = 0.0
        else:
            deviation = (current_value - target_value) / float(target_value)

        if deviation > self.deviation_threshold:
            sig = SignalType.SELL
            msg = f"Portfolio overvalued by {deviation*100:.2f}%"
        elif deviation < -self.deviation_threshold:
            sig = SignalType.BUY
            msg = f"Portfolio undervalued by {abs(deviation)*100:.2f}%"
        else:
            sig = SignalType.HOLD
            msg = f"Within threshold ({abs(deviation)*100:.2f}%)"

        return {
            "signal": sig,
            "deviation": float(deviation),
            "current_value": float(current_value),
            "target_value": float(target_value),
            "message": msg,
            "portfolio": dict(portfolio or {})
        }

    # ----------------------
    # Your existing rebalance logic (kept intact)
    # ----------------------
    # helper to compute portfolio notional from prices and holdings
    @staticmethod
    def portfolio_notional_from_qty(holdings_qty: Dict[str, float], prices: Dict[str, float]) -> float:
        return float(sum(holdings_qty.get(s, 0.0) * prices.get(s, 0.0) for s in holdings_qty))

    @staticmethod
    def weights_from_qty(holdings_qty: Dict[str, float], prices: Dict[str, float]) -> Dict[str, float]:
        tot = TradeSignalEngine.portfolio_notional_from_qty(holdings_qty, prices)
        if tot == 0:
            return {s: 0.0 for s in holdings_qty}
        return {s: (holdings_qty.get(s, 0.0) * prices.get(s, 0.0)) / tot for s in holdings_qty}

    def generate_portfolio_rebalance(self,
                                     target_weights: Dict[str, float],
                                     current_qty: Dict[str, int],
                                     prices: Dict[str, float],
                                     cash: float = 0.0,
                                     capital: Optional[float] = None,
                                     prefer_notional: bool = True
                                     ) -> Dict:
        """
        Compute trade plan to move from current_qty -> target_weights.
        Returns dictionary:
          {
            'trades': {symbol: {'side': SignalType, 'target_notional':..., 'current_notional':..., 'notional_diff':..., 'quantity':...}},
            'summary': {'current_value':..., 'target_value':..., 'l1_deviation':..., 'action': SignalType or HOLD}
          }

        If `capital` supplied, target notional = capital * target_weight. Otherwise, total current value (holdings + cash) used.
        prefer_notional: if True, compute target in USD notional and then convert to integer quantity.
        """
        # compute current positions notional
        current_notional = {}
        for s, q in current_qty.items():
            price = prices.get(s)
            current_notional[s] = float(q * price) if price is not None else 0.0

        current_total = sum(current_notional.values()) + float(cash)
        if capital is None:
            capital = current_total

        # compute target notional per symbol
        target_notional = {s: float(target_weights.get(s, 0.0)) * capital for s in target_weights}
        # compute diffs
        trades = {}
        any_action = False
        for s, targ_n in target_notional.items():
            curr_n = current_notional.get(s, 0.0)
            diff = targ_n - curr_n
            deviation = 0.0 if targ_n == 0 else (curr_n - targ_n) / targ_n
            # Only trade if difference is meaningful and cross threshold relative to target notional
            trade_needed = abs(diff) >= max(self.deviation_threshold * max(abs(targ_n), 1.0), self.min_trade_notional)
            if not trade_needed:
                side = SignalType.HOLD
                qty = 0
            else:
                side = SignalType.BUY if diff > 0 else SignalType.SELL
                any_action = True
                # compute integer quantity to trade
                price = prices.get(s, 0.0) or 0.0
                if price <= 0:
                    qty = 0
                else:
                    qty = int(np.floor(abs(diff) / price))
            trades[s] = {
                "side": side,
                "target_notional": float(targ_n),
                "current_notional": float(curr_n),
                "notional_diff": float(diff),
                "deviation": float(deviation),
                "quantity": int(qty)
            }

        summary_deviation = 0.0
        # for a simple overall deviation, measure L1 difference relative to target total
        target_total = sum(abs(v) for v in target_notional.values())
        if target_total > 0:
            l1 = sum(abs(trades[s]['notional_diff']) for s in trades)
            summary_deviation = l1 / target_total

        overall_action = SignalType.BUY if any(t['notional_diff'] > 0 for t in trades.values()) and any_action else (
                         SignalType.SELL if any(t['notional_diff'] < 0 for t in trades.values()) and any_action else SignalType.HOLD)

        return {
            "trades": trades,
            "summary": {
                "current_value": float(current_total),
                "target_value": float(capital),
                "l1_deviation": float(summary_deviation),
                "action": overall_action.value
            }
        }
