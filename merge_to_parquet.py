import pandas as pd
from pathlib import Path
from datetime import datetime

SOURCE_DIR = Path("data/source")
OUTPUT_FILE = Path(f"data/database/diputados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")

# Partido con más escaños en cada legislatura (pluralidad de la cámara).
# Fuente: distribución observada en los datos + registros históricos CAMARA.
PARTIDO_MAYORIA = {
    57: "PRI",     # LVII  1997-2000 — PRI 48 %
    58: "PRI",     # LVIII 2000-2003 — PRI 42 %, PAN 41 % (Fox presidencia, PRI pluralidad)
    59: "PRI",     # LIX   2003-2006 — PRI 40 %
    60: "PAN",     # LX    2006-2009 — PAN 41 % (Calderón)
    61: "PRI",     # LXI   2009-2012 — PRI 48 %
    62: "PRI",     # LXII  2012-2015 — PRI 43 % (Peña Nieto)
    63: "PRI",     # LXIII 2015-2018 — PRI 40 %
    64: "MORENA",  # LXIV  2018-2021 — MORENA 50 % (AMLO)
    65: "MORENA",  # LXV   2021-2024 — MORENA 40 %
    66: "MORENA",  # LXVI  2024-2027 — MORENA 51 % (Sheinbaum)
}


def add_partido_mayoria(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["partido_mayoria"] = df["legislatura_num"].map(PARTIDO_MAYORIA)
    df["es_partido_mayoria"] = (df["partido"] == df["partido_mayoria"]).astype(int)
    return df


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
    merged = add_partido_mayoria(merged)
    print(f"\nTotal rows: {len(merged)}")
    print(f"es_partido_mayoria — rate: {merged['es_partido_mayoria'].mean():.3f}  nulls: {merged['partido_mayoria'].isna().sum()}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
