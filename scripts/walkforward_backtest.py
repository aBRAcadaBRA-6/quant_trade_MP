# scripts/walkforward_backtest.py
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from app.services.data_fetcher import DataFetcher
from app.services.feature_engineer import FeatureEngineer
from app.services.portfolio_constructor import construct_portfolio_from_var_and_cov, PCOptions

# --- CONFIG ---
SYMS = ["AAPL","MSFT","GOOGL"]                 # universe
START = "2023-01-01"
END = "2024-01-01"
REBALANCE = "7D"   # options: '1D', '7D', '30D' or integer days - frequency of recompute
RISK_AVERSION = 1.0
MAX_WEIGHT = 0.4
SPARSITY_K = 3
TURNOVER_COST_PCT = 0.0005   # 0.05% per turnover unit (adjust)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def date_range_indexed(dates):
    # expects pandas DatetimeIndex
    return [d for d in dates]

def summary_stats(returns):
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * (252 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + returns).prod() - 1
    dd = (1 + returns).cumprod()
    running_max = dd.cummax()
    drawdown = (dd - running_max) / running_max
    max_dd = drawdown.min()
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "cum_return": cum, "max_drawdown": float(max_dd)}

def main():
    fetcher = DataFetcher()
    fe = FeatureEngineer()

    # Load raw aligned prices from DB once (for dates)
    data = fetcher.load_from_db(SYMS, START, END)
    if not data:
        raise SystemExit("No data loaded for given symbols/date range")

    # Build price panel (reuse feature_engineer's helper concept)
    prices = {}
    for s, df in data.items():
        # ensure tz naive index
        ser = pd.Series(df['adj_close'].values, index=pd.to_datetime(df.index), name=s)
        prices[s] = ser
    price_panel = pd.concat(prices.values(), axis=1, keys=prices.keys()).sort_index()
    price_panel = price_panel.dropna(how="all")

    # compute daily log returns (raw)
    logrets = np.log(price_panel).diff().dropna(how="all")

    # rebalancing dates - align to available dates
    all_dates = logrets.index
    if isinstance(REBALANCE, str) and REBALANCE.endswith("D"):
        freq_days = int(REBALANCE[:-1])
    else:
        freq_days = int(REBALANCE)

    rebal_dates = [all_dates[0]]
    for i in range(0, len(all_dates), freq_days):
        rebal_dates.append(all_dates[i])
    rebal_dates = sorted(set(rebal_dates))
    # ensure last date is included
    rebal_dates = [d for d in rebal_dates if d >= all_dates[0] and d <= all_dates[-2]]

    # state
    w_prev = None
    rows = []

    for i, reb_date in enumerate(rebal_dates):
        # cutoff: use data up to reb_date (inclusive)
        cutoff = reb_date
        data_cut = {}
        for s in SYMS:
            df = data[s]
            data_cut[s] = df.loc[df.index <= cutoff]

        # must have at least 20 obs for stability
        try:
            std, A, cov, diag = fe.pipeline_var_cov(data_cut, ridge_lambda=1e-3, persist_outputs=False)
        except Exception as e:
            print("VAR pipeline failed at", cutoff, e)
            continue

        raw_returns = fe.compute_log_returns(data_cut)

        opts = PCOptions(method="sparse_mean_reverting",
                         sparsity_k=SPARSITY_K,
                         max_weight=MAX_WEIGHT,
                         min_weight=0.0,
                         allow_short=False,
                         persist=False,
                         run_name=f"bt_{int(time.time())}")

        # get weights (for universe order = std.columns)
        w, metrics = construct_portfolio_from_var_and_cov(std, A, cov, raw_returns, SYMS, opts, w_prev=w_prev, link_run_id=None)

        # next day returns window: find next day index
        # apply weights to returns from next day until next rebalance (exclusive)
        # find index of cutoff in all_dates
        idx = all_dates.get_loc(cutoff)
        next_idx = idx + 1
        if next_idx >= len(all_dates):
            break
        # period end before next rebalance
        next_reb_point = rebal_dates[i+1] if i+1 < len(rebal_dates) else all_dates[-1]
        # indices between next_idx and index(next_reb_point) inclusive of next_reb_point - 1
        end_idx = all_dates.get_loc(next_reb_point) if next_reb_point in all_dates else min(len(all_dates)-1, idx + freq_days)
        period_indices = list(range(next_idx, end_idx + 1))

        for pi in period_indices:
            day = all_dates[pi]
            ret_vec = logrets.iloc[pi].reindex(std.columns).fillna(0.0).values  # aligned to assets
            pnl = float((w.reindex(std.columns).fillna(0.0).values * np.exp(ret_vec) - w.reindex(std.columns).fillna(0.0).values).sum())
            # simpler: daily portfolio return approx = w @ raw_return (log returns): exp vs approx; use w.dot(np.exp(ret)-1)
            daily_ret = float((w.reindex(std.columns).fillna(0.0).values * (np.exp(ret_vec) - 1)).sum())
            # turnover cost when we rebalance (charge once on first day of period)
            tcost = 0.0
            if pi == next_idx and w_prev is not None:
                turnover = float(np.sum(np.abs(w.reindex(std.columns).fillna(0.0).values - w_prev.reindex(std.columns).fillna(0.0).values)))
                tcost = turnover * TURNOVER_COST_PCT
                daily_ret = daily_ret - tcost

            rows.append({
                "date": str(day),
                "rebalance_date": str(cutoff),
                "daily_return": daily_ret,
                "pnl": daily_ret, 
                "exp_return": metrics.get("expected_return"),
                "portfolio_std": metrics.get("portfolio_std"),
                "n_assets": metrics.get("n_assets"),
                "turnover_cost": tcost
            })
        w_prev = w.copy()

    # compile results
    df_res = pd.DataFrame(rows)
    if df_res.empty:
        print("No backtest rows - check dates / data")
        return
    df_res['cum_ret'] = (1 + df_res['daily_return']).cumprod() - 1
    df_res.to_csv(RESULTS_DIR / "backtest_daily.csv", index=False)

    # summary stats
    stats = summary_stats(df_res['daily_return'])
    with open(RESULTS_DIR / "backtest_report.json","w") as f:
        json.dump({"stats": stats}, f, indent=2)
    print("Backtest saved to", RESULTS_DIR)
    print("Summary:", stats)

if __name__ == "__main__":
    main()
