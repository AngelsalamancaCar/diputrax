import pandas as pd
from pathlib import Path
from datetime import datetime
from unidecode import unidecode

try:
    import gender_guesser.detector as _gg_mod
    _GG = _gg_mod.Detector()
except ImportError:
    _GG = None

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


_GENDER_SUPPLEMENT: dict[str, str] = {
    # Male
    "JOAQUIN": "M", "CUAUHTEMOC": "M", "LEOBARDO": "M", "RAYMUNDO": "M",
    "ECTOR": "M", "TERESO": "M", "YERICO": "M", "HERNAN": "M", "ROMULO": "M",
    "FERMIN": "M", "FAVIO": "M", "ISAEL": "M", "BALDEMAR": "M", "ERUBIEL": "M",
    "WENCESLAO": "M", "CUITLAHUAC": "M", "ZEUS": "M", "NABOR": "M",
    "BRASIL": "M", "AZAEL": "M", "HIREPAN": "M", "IRINEO": "M",
    "EUKID": "M", "ERUVIEL": "M", "INELVO": "M", "RIULT": "M",
    "LIEV": "M", "ESDRAS": "M", "GIBRAN": "M", "RAUDEL": "M",
    "OTNIEL": "M", "OTONIEL": "M", "ASAEL": "M", "RANULFO": "M",
    "RACIEL": "M", "ELMAR": "M", "DANNER": "M", "NEFTALI": "M",
    "ERACLIO": "M", "OSIEL": "M", "OSSIEL": "M", "DELVIM": "M",
    "LIMBERT": "M", "BALFRE": "M", "ABDALLAN": "M", "BARUCH": "M",
    "ENOCH": "M", "FERNEL": "M", "RODIMIRO": "M", "DOMITILO": "M",
    "GUMERCINDO": "M", "ARQUIMIDES": "M", "ABENAMAR": "M", "FAUZI": "M",
    "ELIHER": "M", "CUAUHTLI": "M", "ITZCOATL": "M", "TECUTLI": "M",
    "ZENYAZEN": "M", "AMARANTE": "M", "WBLESTER": "M", "DELBER": "M",
    "SILBESTRE": "M", "UBERLY": "M", "ABDIES": "M", "ELEAZAR": "M",
    "CHARBEL": "M", "HIRAM": "M", "INTI": "M", "CRISTOBAL": "M",
    "ORESTES": "M", "ERANDI": "M", "WILLIAMS": "M", "SANTY": "M",
    "PALACIOS": "M", "UUC-KIB": "M",
    # Female
    "ROCIO": "F", "CONCEPCION": "F", "ARELI": "F", "ARACELY": "F",
    "ANNIA": "F", "ESTHELA": "F", "DIONICIA": "F", "ABRIL": "F",
    "ELBA": "F", "ROSELIA": "F", "MAIELLA": "F", "CIRIA": "F",
    "GUILLERMINA": "F", "MERARY": "F", "FRINNE": "F", "ENOE": "F",
    "LORENIA": "F", "XITLALIC": "F", "ARLET": "F", "ZACIL": "F",
    "IVETH": "F", "LUCELY": "F", "AZUL": "F", "LEYDI": "F",
    "YULMA": "F", "YARITH": "F", "YARET": "F", "EVELYNG": "F",
    "SHANTALL": "F", "EMILSE": "F", "LILIAM": "F", "LANDY": "F",
    "YATZIRI": "F", "ALLIET": "F", "ZULEYMA": "F", "DORHENY": "F",
    "EDILTRUDIS": "F", "ANILU": "F", "IRASEMA": "F", "CRISTABELL": "F",
    "NELY": "F", "MARICARMEN": "F", "ARANZAZU": "F", "YULENNY": "F",
    "ANAY": "F", "ANAIS": "F", "HAIDYD": "F", "ANTARES": "F",
    "DELHI": "F", "LEIDE": "F", "ANY": "F", "TEY": "F",
    "AREMY": "F", "DENISSE": "F", "GREYCY": "F", "IRAIS": "F",
    "ANAYELI": "F", "AMANCAY": "F", "YEIDCKOL": "F", "JANICIE": "F",
    "BENNELLY": "F", "ANABEY": "F", "YEIMI": "F", "TAYGETE": "F",
    "YARY": "F", "AMALIN": "F", "ANGELES": "F",
}

_GG_TO_BINARY = {
    "male": "M", "mostly_male": "M",
    "female": "F", "mostly_female": "F",
}

_SKIP_PREPOSITIONS = {"DE", "DEL", "LOS", "LAS", "LA", "EL"}


def _parse_first_name(nombre: str) -> str:
    parts = unidecode(nombre.strip()).split()
    if not parts:
        return ""
    first = parts[0].upper()
    if first in ("MA.", "MA", "M."):
        rest = parts[1:]
        while rest and rest[0].upper() in _SKIP_PREPOSITIONS:
            rest = rest[1:]
        return rest[0].upper() if rest else "MA"
    if first in ("J.", "J"):
        return parts[1].upper() if len(parts) > 1 else "J"
    return first


def _infer_one(nombre: str) -> str | None:
    parts = unidecode(nombre.strip()).split()
    first = _parse_first_name(nombre)

    if first in _GENDER_SUPPLEMENT:
        return _GENDER_SUPPLEMENT[first]

    if _GG is not None:
        gg = _GG.get_gender(first.capitalize())
        if gg in _GG_TO_BINARY:
            return _GG_TO_BINARY[gg]

    # Try second given name for ambiguous/unknown first names
    if len(parts) > 1:
        second = unidecode(parts[1]).upper()
        if second in _GENDER_SUPPLEMENT:
            return _GENDER_SUPPLEMENT[second]
        if _GG is not None:
            gg2 = _GG.get_gender(second.capitalize())
            if gg2 in _GG_TO_BINARY:
                return _GG_TO_BINARY[gg2]

    # Spanish morphological fallback
    if first.endswith("A"):
        return "F"
    if first.endswith(("O", "OR", "AN", "ON", "EL", "IN")):
        return "M"

    return None


def add_sexo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sexo"] = df["nombre"].apply(_infer_one)
    return df


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
    merged = add_sexo(merged)
    print(f"\nTotal rows: {len(merged)}")
    print(f"es_partido_mayoria — rate: {merged['es_partido_mayoria'].mean():.3f}  nulls: {merged['partido_mayoria'].isna().sum()}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
