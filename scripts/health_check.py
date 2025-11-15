#!/usr/bin/env python3
# scripts/health_check.py
"""
Comprehensive health-check for the quant-trade project.
Checks:
  - core Python packages versions
  - DB connectivity using app.models.database.engine
  - runs scripts/init_db.py (will create tables / test user)
  - runs data fetcher for a few tickers and saves to DB
  - runs feature engineer single-symbol and VAR pipeline

Run:
  conda activate quant-trade
  python scripts/health_check.py
"""

import sys
import traceback
import subprocess
import time
from pathlib import Path

def safe_import(name):
    try:
        module = __import__(name)
        return module, None
    except Exception as e:
        return None, str(e)

def print_section(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def show_package_versions():
    print_section("PACKAGE / ENV CHECK")
    pkgs = [
        ("python", lambda: sys.version.replace("\n"," ")),
        ("numpy", lambda: __import__("numpy").__version__),
        ("pandas", lambda: __import__("pandas").__version__),
        ("sqlalchemy", lambda: __import__("sqlalchemy").__version__),
        ("psycopg2", lambda: __import__("psycopg2").__version__),
        ("yfinance", lambda: __import__("yfinance").__version__),
        ("sklearn", lambda: __import__("sklearn").__version__),
        ("pydantic_settings", lambda: __import__("pydantic_settings").__version__ if safe_import("pydantic_settings")[0] else "not-installed"),
    ]
    for name, getver in pkgs:
        try:
            print(f"{name:16s} : {getver()}")
        except Exception as e:
            print(f"{name:16s} : ERROR ({e})")

def run_init_db():
    print_section("RUNNING scripts/init_db.py (initializes DB tables and creates test user)")
    script = Path("scripts/init_db.py")
    if not script.exists():
        print("scripts/init_db.py not found — skipping init_db run")
        return False, "no-script"
    try:
        # run as subprocess so it behaves exactly like your normal run
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
        print("=== STDOUT ===")
        print(p.stdout.strip())
        print("=== STDERR ===")
        print(p.stderr.strip())
        if p.returncode == 0:
            return True, None
        else:
            return False, f"returncode={p.returncode}"
    except Exception as e:
        print("Exception running init_db:", e)
        return False, str(e)

def check_db_connection():
    print_section("DB CONNECTION CHECK")
    try:
        from sqlalchemy import text
        from app.models import database
        engine = database.engine
        print("Engine URL:", engine.url)
        with engine.connect() as conn:
            res = conn.execute(text("SELECT current_user, session_user, current_schema();"))
            rows = res.fetchall()
            print("DB Response:", rows)
            # list tables in public schema (if access)
            res2 = conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"
            ))
            tables = [r[0] for r in res2.fetchall()]
            print("Public tables:", tables)
        return True, None
    except Exception as e:
        tb = traceback.format_exc()
        print("DB connection failed:\n", tb)
        return False, str(e)

def run_data_fetcher_test(symbols, start, end):
    print_section("DATA FETCHER TEST")
    try:
        from app.services.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        print("Fetching:", symbols, "from", start, "to", end)
        # call fetch_ohlcv (signature may vary), attempt common variants
        data = None
        try:
            data = fetcher.fetch_ohlcv(symbols, start, end)
        except TypeError:
            # older/newer signature? try save_to_db explicitly
            data = fetcher.fetch_ohlcv(symbols, start, end, save_to_db=True)
        except Exception as e:
            print("fetch_ohlcv raised:", e)
            # fallback: try per-symbol download using yf inside fetcher if available
            raise

        if not data:
            print("No data returned by fetcher (empty dict).")
        else:
            for s, df in data.items():
                try:
                    print(f"{s}: type={type(df)}, shape={getattr(df,'shape', None)}, cols={list(df.columns)[:10]}")
                except Exception:
                    print(f"{s}: (couldn't introspect df)")
        # attempt to save to DB if fetcher provides _save_to_db
        saved_counts = {}
        if hasattr(fetcher, "_save_to_db"):
            try:
                for s, df in data.items():
                    fetcher._save_to_db(s, df)
                # load back from DB
                loaded = fetcher.load_from_db(symbols, start, end)
                for s, df in loaded.items():
                    saved_counts[s] = getattr(df, "shape", None)
            except Exception as e:
                print("Saving/loading via fetcher._save_to_db / load_from_db failed:", e)
        else:
            print("fetcher._save_to_db not present — assuming fetcher saved to DB internally or no DB save required.")
        return True, {"fetched": {s: getattr(df,'shape',None) for s,df in (data or {}).items()}, "loaded": saved_counts}
    except Exception as e:
        tb = traceback.format_exc()
        print("Data fetcher test failed:\n", tb)
        return False, str(e)

def run_feature_engineer_tests(symbols, start, end):
    print_section("FEATURE ENGINEER TEST")
    try:
        from app.services.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        # single symbol features
        single = symbols[0]
        print(f"Compute features for single symbol: {single}")
        df = fe.compute_features_for_symbol(single, start, end, save=False)
        print(f"{single} features shape: {df.shape}, columns sample: {list(df.columns)[:10]}")
        # pipeline_var_cov
        print("Running pipeline_var_cov for symbols:", symbols)
        data = fe.fetcher.load_from_db(symbols, start, end)
        if not data:
            print("No data loaded from DB for VAR pipeline. Aborting FE pipeline test.")
            return False, "no-data"
        std, A, cov, diag = fe.pipeline_var_cov(data, ridge_lambda=1e-3, persist_outputs=False)
        print("standardized returns shape:", getattr(std,'shape',None))
        print("A shape:", getattr(A,'shape',None))
        print("cov shape:", getattr(cov,'shape',None))
        # show a few diagnostics
        diag_summary = {k: diag[k] for k in diag if k not in ("eigenvalues",)}
        print("Diagnostics summary:", diag_summary)
        return True, {"features_shape": df.shape, "std_shape": getattr(std,'shape',None), "A_shape": getattr(A,'shape',None)}
    except Exception as e:
        tb = traceback.format_exc()
        print("Feature engineer test failed:\n", tb)
        return False, str(e)

def main():
    start_time = time.time()
    report = {}
    show_package_versions()

    ok, info = check_db_connection()
    report["db_connection"] = (ok, info)

    ok_init, info_init = run_init_db()
    report["init_db_run"] = (ok_init, info_init)

    # wait a bit to let DB settle (if init created tables)
    time.sleep(1.0)

    # Data fetcher test
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start, end = "2023-01-01", "2024-01-01"
    ok_fetch, info_fetch = run_data_fetcher_test(symbols, start, end)
    report["fetcher"] = (ok_fetch, info_fetch)

    # Feature engineer tests
    ok_fe, info_fe = run_feature_engineer_tests(symbols, start, end)
    report["feature_engineer"] = (ok_fe, info_fe)

    print_section("SUMMARY")
    for k, v in report.items():
        status = "OK" if v[0] else "FAIL"
        print(f"{k:20s} : {status} : {v[1]}")

    total = time.time() - start_time
    print(f"\nHealth-check finished in {total:.1f}s")
    if not report["db_connection"][0] or not report["feature_engineer"][0]:
        print("\nOne or more checks failed. See output above for tracebacks.")
        sys.exit(2)
    print("\nAll core checks passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
