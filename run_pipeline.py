import sys
import time
from pathlib import Path
from datetime import datetime

CLEAN_DIR = Path("data/database/clean")


def step(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")


def run_merge() -> Path:
    step("STEP 1 — merge CSVs → parquet")
    import merge_to_parquet as m2p

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    output = CLEAN_DIR / f"diputados_{timestamp}.parquet"
    m2p.OUTPUT_FILE = output

    m2p.main()
    return output


def run_eda(parquet_path: Path):
    step("STEP 2 — SQL EDA report")
    import sqleda

    sqleda.DB_DIR = CLEAN_DIR
    sqleda.PARQUET = str(parquet_path)
    sqleda.main()


def main():
    t0 = time.perf_counter()
    print(f"Pipeline start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        parquet_path = run_merge()
    except Exception as e:
        print(f"\n[FAIL] merge_to_parquet: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        run_eda(parquet_path)
    except Exception as e:
        print(f"\n[FAIL] sqleda: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
