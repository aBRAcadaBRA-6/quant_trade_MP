# app/services/backtester.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import time

from app.services.trade_signal_engine import TradeSignalEngine
from app.services.risk_manager import RiskManager
from app.models.database import SessionLocal, Trade


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission_rate: float = 0.0005
    slippage_pct: float = 0.0005
    rebalance_freq_days: int = 7
    min_trade_notional: float = 20.0
    persist_trades: bool = False
    results_dir: str = "results"


# ---------------------------------------------------------
# BACKTESTER
# ---------------------------------------------------------
class Backtester:
    """
    Simulates applying target portfolio weights across time.
    Compatible with both:
      Backtester(BacktestConfig())
      Backtester(initial_capital=100000)
    """

    def __init__(self,
                 cfg: Optional[BacktestConfig] = None,
                 *,
                 initial_capital: Optional[float] = None,
                 commission_rate: Optional[float] = None,
                 slippage_pct: Optional[float] = None,
                 rebalance_freq_days: Optional[int] = None,
                 min_trade_notional: Optional[float] = None,
                 persist_trades: Optional[bool] = None,
                 results_dir: Optional[str] = None):

        if cfg is None:
            cfg = BacktestConfig()

        if initial_capital is not None:
            cfg.initial_capital = float(initial_capital)
        if commission_rate is not None:
            cfg.commission_rate = float(commission_rate)
        if slippage_pct is not None:
            cfg.slippage_pct = float(slippage_pct)
        if rebalance_freq_days is not None:
            cfg.rebalance_freq_days = int(rebalance_freq_days)
        if min_trade_notional is not None:
            cfg.min_trade_notional = float(min_trade_notional)
        if persist_trades is not None:
            cfg.persist_trades = bool(persist_trades)
        if results_dir is not None:
            cfg.results_dir = str(results_dir)

        self.cfg = cfg
        Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
        self.ts = int(time.time())

    # -----------------------------------------------------

    def _apply_trade(self, symbol: str, qty: int, price: float) -> Tuple[float, float]:
        """
        Returns:
            notional,
            total fee (slippage+commission)
        """
        notional = qty * price
        slippage = abs(notional) * self.cfg.slippage_pct
        commission = abs(notional) * self.cfg.commission_rate
        return notional, float(slippage + commission)

    # -----------------------------------------------------

    def run_from_weights_history(self,
                                 price_panel: pd.DataFrame,
                                 weights_history: Dict[pd.Timestamp, pd.Series],
                                 initial_cash: Optional[float] = None,
                                 risk_mgr: Optional[RiskManager] = None
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if initial_cash is None:
            initial_cash = self.cfg.initial_capital

        # Risk manager validation
        if risk_mgr is not None:
            required = ("check_position_sizes", "enforce_limits_on_weights")
            if not all(hasattr(risk_mgr, m) for m in required):
                raise TypeError(
                    f"risk_mgr must implement: {required}. Got {type(risk_mgr)}"
                )

        dates = price_panel.index
        symbols = list(price_panel.columns)
        cash = float(initial_cash)
        holdings_qty = {s: 0 for s in symbols}
        equity_records = []
        trades = []

        # -------------------------------------------------
        #       FIXED: SAFE REBALANCE DATE ALIGNMENT
        # -------------------------------------------------
        orig_reb_dates = sorted(weights_history.keys())

        # tz-normalized search index
        if getattr(dates, "tz", None) is not None:
            dates_for_search = dates.tz_convert("UTC").tz_localize(None)
        else:
            dates_for_search = dates

        def _to_search_ts(ts):
            ts = pd.Timestamp(ts)
            if getattr(dates, "tz", None) is not None:
                if ts.tz is None:
                    return ts.tz_localize("UTC").tz_convert("UTC").tz_localize(None)
                return ts.tz_convert("UTC").tz_localize(None)
            else:
                if ts.tz is not None:
                    return ts.tz_convert("UTC").tz_localize(None)
                return ts

        # Build aligned mapping
        weights_history_aligned = {}

        for ts in orig_reb_dates:
            # exact match?
            try:
                if ts in dates:
                    aligned = ts
                else:
                    raise KeyError
            except Exception:
                # normalized search
                t_norm = _to_search_ts(ts)
                loc = dates_for_search.searchsorted(t_norm)
                if loc >= len(dates_for_search):
                    continue
                aligned = dates[loc]

            # store the mapping ONCE
            if aligned not in weights_history_aligned:
                weights_history_aligned[aligned] = weights_history[ts]

        reb_dates = sorted(weights_history_aligned.keys())

        if not reb_dates:
            raise ValueError("No valid rebalance dates after alignment.")

        # -----------------------------------------------------
        #                MAIN BACKTEST LOOP
        # -----------------------------------------------------
        for i, reb in enumerate(reb_dates):

            target_weights = weights_history_aligned[reb].reindex(symbols).fillna(0.0)

            # Portfolio value at rebalance
            prices_now = price_panel.loc[reb].to_dict()
            pv_now = sum(holdings_qty[s] * prices_now[s] for s in symbols)
            current_value = pv_now + cash

            # Generate trade instructions
            tse = TradeSignalEngine(deviation_threshold=0.01,
                                    min_trade_notional=self.cfg.min_trade_notional)

            plan = tse.generate_portfolio_rebalance(
                target_weights.to_dict(),
                holdings_qty,
                prices_now,
                cash=current_value,
                capital=current_value
            )

            trades_spec = plan["trades"]

            # Risk enforcement
            if risk_mgr is not None:
                ok, msg = risk_mgr.check_position_sizes(target_weights.to_dict(), capital=current_value)
                if not ok:
                    target_weights = pd.Series(
                        risk_mgr.enforce_limits_on_weights(
                            target_weights.to_dict(),
                            capital=current_value
                        )
                    ).reindex(symbols).fillna(0.0)

            # Execute trades
            for s, info in trades_spec.items():
                qty = int(info["quantity"])
                if qty == 0:
                    continue

                price = float(prices_now[s])
                if price <= 0:
                    continue

                side = info["side"].value
                sign = 1 if side == "BUY" else -1
                qty_signed = sign * qty

                notional, fee = self._apply_trade(s, qty_signed, price)
                cash -= (notional + fee)
                holdings_qty[s] += qty_signed

                trades.append({
                    "date": reb,
                    "symbol": s,
                    "side": side,
                    "quantity": qty,
                    "price": price,
                    "notional": notional,
                    "fees_and_slippage": fee,
                    "cash_after": cash
                })

                # Optional DB save
                if self.cfg.persist_trades:
                    try:
                        with SessionLocal() as db:
                            tr = Trade(
                                timestamp=reb.to_pydatetime(),
                                symbol=s,
                                side=side,
                                quantity=qty,
                                price=price,
                                slippage=fee,
                                commission=0.0,
                                pnl=0.0,
                                user_id=None,
                                portfolio_id=None
                            )
                            db.add(tr)
                            db.commit()
                    except Exception:
                        pass

            # Apply returns until next rebalance
            next_reb = reb_dates[i + 1] if i + 1 < len(reb_dates) else dates[-1]

            start_idx = dates.get_loc(reb)
            end_idx = dates.get_loc(next_reb) if next_reb in dates else len(dates) - 1

            for di in range(start_idx, end_idx + 1):
                day = dates[di]
                prices_day = price_panel.iloc[di].to_dict()
                pv = sum(holdings_qty[s] * prices_day[s] for s in symbols)
                total = pv + cash

                equity_records.append({
                    "date": day,
                    "rebalance_date": reb,
                    "equity": float(total),
                    "cash": float(cash),
                    "positions_value": float(pv),
                })

        equity_df = pd.DataFrame(equity_records).set_index("date")
        trades_df = pd.DataFrame(trades)

        equity_df.to_csv(Path(self.cfg.results_dir) / f"equity_{self.ts}.csv")
        trades_df.to_csv(Path(self.cfg.results_dir) / f"trades_{self.ts}.csv", index=False)

        return equity_df, trades_df

    # ---------------------------------------------------------
    # Convenience API
    # ---------------------------------------------------------
    def simulate(self,
                 weights: Dict[str, float],
                 price_data: Dict[str, pd.DataFrame],
                 risk_mgr: Optional[RiskManager] = None,
                 rebalance_dates: Optional[List[pd.Timestamp]] = None,
                 initial_cash: Optional[float] = None):

        # Build price panel
        dfs = {}
        for s, df in price_data.items():
            if "close" in df.columns:
                dfs[s] = df["close"].rename(s)
            elif "adj_close" in df.columns:
                dfs[s] = df["adj_close"].rename(s)
            else:
                raise ValueError(f"No close/adj_close column for {s}")

        price_panel = pd.concat(dfs.values(), axis=1).sort_index().ffill().bfill()

        idx = price_panel.index
        if rebalance_dates is None:
            if getattr(idx, "tz", None) is not None:
                idx2 = idx.tz_convert("UTC").tz_localize(None)
            else:
                idx2 = idx

            periods = idx2.to_series().dt.to_period("M")
            unique_months = periods.drop_duplicates().tolist()

            rebalance_dates = []
            for m in unique_months:
                mask = periods == m
                first_pos = mask.values.nonzero()[0]
                if len(first_pos):
                    rebalance_dates.append(idx2[first_pos[0]])

        weights_series = pd.Series(weights).reindex(price_panel.columns).fillna(0.0)
        weights_history = {pd.Timestamp(d): weights_series for d in rebalance_dates}

        return self.run_from_weights_history(
            price_panel,
            weights_history,
            initial_cash=initial_cash,
            risk_mgr=risk_mgr
        )
