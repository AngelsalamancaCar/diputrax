import pandas as pd
from pathlib import Path
from datetime import datetime

SOURCE_DIR = Path("data/source")
OUTPUT_FILE = Path(f"data/database/diputados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")

def main():
    csv_files = sorted(SOURCE_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {SOURCE_DIR}")

    print(f"Found {len(csv_files)} files:")
    for f in csv_files:
        print(f"  {f.name}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df.assign(source_file=f.name))
        print(f"  {f.name}: {len(df)} rows")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(merged)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
