import duckdb
from pathlib import Path
from datetime import datetime
from io import StringIO

DB_DIR = Path("data/database/clean")
REPORTS_DIR = Path("reports/eda")
PARQUET = "data/database/clean/diputados_20260419_193955.parquet"


def latest_parquet() -> str:
    files = sorted(DB_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DB_DIR}")
    return str(files[-1])


def main():
    path = latest_parquet()
    con = duckdb.connect()

    buf = StringIO()

    def w(text=""):
        buf.write(text + "\n")

    def section(title):
        w(f"## {title}")
        w()

    def table(df):
        w(df.to_markdown(index=False))
        w()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    w(f"# EDA Report (SQL/DuckDB) — {timestamp}")
    w(f"**Source:** `{path}`  ")
    w(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w()

    # ── Shape ────────────────────────────────────────────────────────────────
    section("Shape")
    shape = con.execute(f"""
        SELECT
            COUNT(*)                                          AS rows,
            (SELECT COUNT(*) FROM (DESCRIBE SELECT * FROM '{path}')) AS cols
        FROM '{path}'
    """).fetchdf()
    table(shape)

    # ── Schema / null audit ───────────────────────────────────────────────────
    section("Schema — dtypes & null audit")
    schema = con.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchdf()
    schema = schema[["column_name", "column_type"]]

    rows_n = con.execute(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]

    null_rows = []
    for _, row in schema.iterrows():
        col = row["column_name"]
        null_count = con.execute(
            f"SELECT COUNT(*) FROM '{path}' WHERE \"{col}\" IS NULL"
        ).fetchone()[0]
        null_rows.append({
            "column": col,
            "type": row["column_type"],
            "non_null": rows_n - null_count,
            "null": null_count,
            "null_%": round(null_count / rows_n * 100, 2),
        })

    import pandas as pd
    null_df = pd.DataFrame(null_rows)
    table(null_df)

    # ── Unique values ─────────────────────────────────────────────────────────
    section("Unique Values per Column")
    uniq_rows = []
    for col in schema["column_name"]:
        uniq = con.execute(
            f"SELECT COUNT(DISTINCT \"{col}\") FROM '{path}'"
        ).fetchone()[0]
        samples = con.execute(
            f"SELECT \"{col}\" FROM '{path}' WHERE \"{col}\" IS NOT NULL LIMIT 5"
        ).fetchdf()[col].tolist()
        uniq_rows.append({"column": col, "unique_count": uniq, "sample_values": samples})
    table(pd.DataFrame(uniq_rows))

    # ── Numeric descriptive stats ─────────────────────────────────────────────
    section("Descriptive Statistics — Numeric")
    numeric_cols = schema[schema["column_type"].isin(["BIGINT", "DOUBLE", "INTEGER", "FLOAT"])]["column_name"].tolist()

    if numeric_cols:
        stat_rows = []
        for col in numeric_cols:
            r = con.execute(f"""
                SELECT
                    COUNT("{col}")                   AS count,
                    AVG("{col}")                     AS mean,
                    STDDEV("{col}")                  AS std,
                    MIN("{col}")                     AS min,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{col}") AS p25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY "{col}") AS p50,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{col}") AS p75,
                    MAX("{col}")                     AS max
                FROM '{path}'
            """).fetchone()
            stat_rows.append({
                "column": col,
                "count": r[0], "mean": round(r[1], 4) if r[1] else None,
                "std": round(r[2], 4) if r[2] else None,
                "min": r[3], "25%": r[4], "50%": r[5], "75%": r[6], "max": r[7],
            })
        table(pd.DataFrame(stat_rows))
    else:
        w("_No numeric columns._\n")

    # ── Categorical descriptive stats ─────────────────────────────────────────
    section("Descriptive Statistics — Categorical")
    cat_cols = schema[schema["column_type"] == "VARCHAR"]["column_name"].tolist()

    if cat_cols:
        cat_rows = []
        for col in cat_cols:
            r = con.execute(f"""
                SELECT
                    COUNT("{col}")         AS count,
                    COUNT(DISTINCT "{col}") AS unique,
                    MODE("{col}")          AS top
                FROM '{path}'
            """).fetchone()
            top_freq = con.execute(
                f"SELECT COUNT(*) FROM '{path}' WHERE \"{col}\" = '{r[2]}'"
            ).fetchone()[0] if r[2] else None
            cat_rows.append({
                "column": col, "count": r[0], "unique": r[1],
                "top": r[2], "top_freq": top_freq,
            })
        table(pd.DataFrame(cat_rows))
    else:
        w("_No categorical columns._\n")

    # ── Domain: partido distribution ──────────────────────────────────────────
    section("Party Distribution (partido_nombre)")
    partido = con.execute(f"""
        SELECT
            partido_nombre,
            COUNT(*)                              AS n,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
        FROM '{path}'
        GROUP BY partido_nombre
        ORDER BY n DESC
        LIMIT 20
    """).fetchdf()
    table(partido)

    # ── Domain: legislatura distribution ─────────────────────────────────────
    section("Legislature Distribution (legislatura_nombre)")
    leg = con.execute(f"""
        SELECT
            legislatura_nombre,
            legislatura_num,
            COUNT(*) AS n
        FROM '{path}'
        GROUP BY legislatura_nombre, legislatura_num
        ORDER BY legislatura_num
    """).fetchdf()
    table(leg)

    # ── Domain: education level ───────────────────────────────────────────────
    section("Education Level (grado_estudios_ord)")
    edu = con.execute(f"""
        SELECT
            grado_estudios_ord,
            COUNT(*)                              AS n,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
        FROM '{path}'
        GROUP BY grado_estudios_ord
        ORDER BY grado_estudios_ord
    """).fetchdf()
    table(edu)

    # ── Domain: area_formacion ────────────────────────────────────────────────
    section("Academic Field (area_formacion)")
    area = con.execute(f"""
        SELECT
            area_formacion,
            COUNT(*) AS n,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
        FROM '{path}'
        GROUP BY area_formacion
        ORDER BY n DESC
        LIMIT 20
    """).fetchdf()
    table(area)

    # ── Domain: prior legislative experience ─────────────────────────────────
    section("Prior Legislative Experience")
    exp = con.execute(f"""
        SELECT
            n_cargos_legislativos_prev,
            COUNT(*) AS n,
            ROUND(AVG(n_comisiones), 2)           AS avg_comisiones,
            ROUND(AVG(n_trayectoria_politica), 2) AS avg_trayectoria_pol
        FROM '{path}'
        GROUP BY n_cargos_legislativos_prev
        ORDER BY n_cargos_legislativos_prev
    """).fetchdf()
    table(exp)

    # ── Domain: age at office ─────────────────────────────────────────────────
    section("Age at Office (edad_al_tomar_cargo) — by party")
    age = con.execute(f"""
        SELECT
            partido_nombre,
            ROUND(AVG(edad_al_tomar_cargo), 1)    AS avg_age,
            ROUND(MIN(edad_al_tomar_cargo), 1)    AS min_age,
            ROUND(MAX(edad_al_tomar_cargo), 1)    AS max_age,
            COUNT(*)                               AS n
        FROM '{path}'
        WHERE edad_al_tomar_cargo IS NOT NULL
        GROUP BY partido_nombre
        ORDER BY avg_age DESC
        LIMIT 15
    """).fetchdf()
    table(age)

    # ── Domain: commission leadership ────────────────────────────────────────
    section("Commission Leadership")
    comm = con.execute(f"""
        SELECT
            SUM(presidente_comision)                         AS total_presidentes,
            SUM(lider_comision)                              AS total_lideres,
            ROUND(AVG(n_comisiones), 2)                     AS avg_comisiones,
            ROUND(AVG(n_comisiones_especiales), 2)          AS avg_especiales,
            ROUND(AVG(n_presidencias), 2)                   AS avg_presidencias,
            ROUND(AVG(n_secretarias), 2)                    AS avg_secretarias
        FROM '{path}'
    """).fetchdf()
    table(comm)

    # ── Domain: university prestige ───────────────────────────────────────────
    section("University Prestige Flags")
    uni = con.execute(f"""
        SELECT
            SUM(acad_unam)       AS unam,
            SUM(acad_itesm)      AS itesm,
            SUM(acad_itam)       AS itam,
            SUM(acad_ibero)      AS ibero,
            SUM(acad_udg)        AS udg,
            SUM(acad_ipn)        AS ipn,
            SUM(acad_uam)        AS uam,
            SUM(acad_anahuac)    AS anahuac,
            SUM(acad_uanl)       AS uanl,
            SUM(acad_uv)         AS uv,
            SUM(univ_publica)    AS publica,
            SUM(univ_privada)    AS privada,
            SUM(univ_extranjera) AS extranjera
        FROM '{path}'
    """).fetchdf()
    table(uni)

    # ── Domain: administrative background ────────────────────────────────────
    section("Administrative Background")
    admin = con.execute(f"""
        SELECT
            SUM(admin_en_partido)       AS partido,
            SUM(admin_en_sindicato)     AS sindicato,
            SUM(admin_en_universidad)   AS universidad,
            SUM(admin_en_gobierno_fed)  AS gobierno_fed,
            SUM(admin_en_gobierno_est)  AS gobierno_est,
            SUM(admin_en_gobierno_mun)  AS gobierno_mun
        FROM '{path}'
    """).fetchdf()
    table(admin)

    # ── Domain: correlations between career features & commission count ───────
    section("Correlation: career features vs n_comisiones")
    corr_cols = [
        "n_cargos_legislativos_prev", "n_trayectoria_admin",
        "n_trayectoria_politica", "n_trayectoria_empresarial",
        "nivel_cargo_max", "edad_al_tomar_cargo",
    ]
    corr_rows = []
    for col in corr_cols:
        r = con.execute(f"""
            SELECT CORR(n_comisiones, "{col}") FROM '{path}'
        """).fetchone()[0]
        corr_rows.append({"feature": col, "corr_with_n_comisiones": round(r, 4) if r else None})
    table(pd.DataFrame(corr_rows))

    # ── Save ──────────────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"sqleda_{timestamp}.md"
    out.write_text(buf.getvalue())
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
