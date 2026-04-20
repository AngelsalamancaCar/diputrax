import pandas as pd
from pathlib import Path
from datetime import datetime
from io import StringIO

DB_DIR = Path("data/database/raw")
REPORTS_DIR = Path("reports/eda")


def latest_parquet() -> Path:
    files = sorted(DB_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DB_DIR}")
    return files[-1]


def main():
    path = latest_parquet()
    df = pd.read_parquet(path)
    pd.set_option("display.float_format", "{:.4f}".format)

    buf = StringIO()

    def w(text=""):
        buf.write(text + "\n")

    def section(title):
        w(f"## {title}")
        w()

    timestamp = datetime.now().strftime("%Y%m%d_%H")
    w(f"# EDA Report — {timestamp}")
    w(f"**Source:** `{path}`  ")
    w(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w()

    section("Shape")
    w(f"- Rows: **{df.shape[0]:,}**")
    w(f"- Columns: **{df.shape[1]:,}**")
    w()

    section("Schema")
    schema = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "non_null": df.notna().sum(),
            "null": df.isna().sum(),
            "null_%": (df.isna().mean() * 100).round(2),
        }
    )
    w(schema.to_markdown())
    w()

    section("Unique Values per Column")
    unique = pd.DataFrame(
        {
            "unique_count": df.nunique(),
            "sample_values": {
                col: str(df[col].dropna().unique()[:5].tolist()) for col in df.columns
            },
        }
    )
    w(unique.to_markdown())
    w()

    section("Descriptive Statistics — Numeric")
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        stats = numeric.describe().T
        stats.columns = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        w(stats.to_markdown())
    else:
        w("_No numeric columns._")
    w()

    section("Descriptive Statistics — Categorical")
    categorical = df.select_dtypes(exclude="number")
    if not categorical.empty:
        w(categorical.describe().T.to_markdown())
    else:
        w("_No categorical columns._")
    w()

    pd.reset_option("display.float_format")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"eda_{timestamp}.md"
    out.write_text(buf.getvalue())
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
