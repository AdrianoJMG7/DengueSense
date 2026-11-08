# -*- coding: utf-8 -*-
"""
01_importar.py — Paso 2 de la guia: importar con tipos correctos y estandarizar texto.

Lee los CSV de data/raw/ (nunca se modifican), normaliza nombres de columnas,
texto a MAYUSCULAS sin acentos, ubigeo con padding a 6 digitos, y guarda
versiones tipadas en data/interim/ (parquet).
"""
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)


def ascii_upper(s: pd.Series) -> pd.Series:
    """Mayusculas, sin acentos, espacios colapsados."""
    s = s.fillna("").astype(str).str.strip()
    s = s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    return s.str.upper().str.replace(r"\s+", " ", regex=True)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in df.columns:
        # Con pandas 3 el texto se lee como dtype "str", que ya no es igual a
        # object; is_string_dtype cubre ambos casos.
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = ascii_upper(df[col])
    return df


# El CSV de casos ya trae la letra perdida como U+FFFD ("FERRE�AFE"); el
# dano viene de la fuente. En provincia y distrito siempre era una N con tilde
# (verificado: CANETE, FERRENAFE, PARINAS, ...). En localidad no siempre (por
# ejemplo "N� 11" es el simbolo de numero), asi que ahi no se reemplaza.
COLS_CON_ENE = ["provincia", "distrito"]


# ---------------------------------------------------------------- casos ----
print("Importando casos (108 MB, puede tardar)...")
casos = pd.read_csv(
    RAW / "datos_abiertos_vigilancia_dengue_2000_2024.csv",
    sep=";",
    dtype=str,
    keep_default_na=False,
)
n_fffd = casos[COLS_CON_ENE].apply(lambda s: s.str.contains("�")).sum().to_dict()
print(f"  letras perdidas en la fuente (U+FFFD), reemplazadas por N: {n_fffd}")
for col in COLS_CON_ENE:
    casos[col] = casos[col].str.replace("�", "N")
casos = clean_df(casos)
casos["ubigeo"] = casos["ubigeo"].str.replace(r"\D", "", regex=True).str.zfill(6)
casos["ano"] = pd.to_numeric(casos["ano"], errors="coerce").astype("Int64")
casos["semana"] = pd.to_numeric(casos["semana"], errors="coerce").astype("Int64")
casos["edad"] = pd.to_numeric(casos["edad"], errors="coerce")
casos.to_parquet(INTERIM / "casos_typed.parquet", index=False)
print(f"  casos: {len(casos):,} filas x {casos.shape[1]} cols")

# ----------------------------------------------------------------- clima ----
print("Importando clima...")
clima = pd.read_csv(RAW / "clima_celda_semana.csv", dtype={"celda_grilla": str})
clima = clean_df(clima)
num_cols = [c for c in clima.columns if c not in ("celda_grilla", "ano", "semana")]
clima[num_cols] = clima[num_cols].apply(pd.to_numeric, errors="coerce")
clima["ano"] = clima["ano"].astype(int)
clima["semana"] = clima["semana"].astype(int)
clima.to_parquet(INTERIM / "clima_typed.parquet", index=False)
print(f"  clima: {len(clima):,} filas x {clima.shape[1]} cols, "
      f"{clima['celda_grilla'].nunique()} celdas")

# ------------------------------------------------------------------ icen ----
print("Importando ICEN...")
icen = pd.read_csv(RAW / "icen_mensual.csv")
icen = clean_df(icen)
icen.to_parquet(INTERIM / "icen_typed.parquet", index=False)
print(f"  icen: {len(icen):,} filas")

# ------------------------------------------------------------------ mapa ----
print("Importando mapa ubigeo->celda...")
mapa = pd.read_csv(RAW / "mapa_ubigeo_celda.csv", dtype={"ubigeo": str})
mapa = clean_df(mapa)
mapa["ubigeo"] = mapa["ubigeo"].str.replace(r"\D", "", regex=True).str.zfill(6)
mapa.to_parquet(INTERIM / "mapa_typed.parquet", index=False)
print(f"  mapa: {len(mapa):,} filas, ubigeos duplicados: "
      f"{mapa['ubigeo'].duplicated().sum()}")

# ------------------------------------------------------------- distritos ----
print("Importando distritos/centroides...")
distritos = pd.read_csv(RAW / "distritos_dengue_centroides.csv", dtype={"ubigeo": str})
distritos = clean_df(distritos)
distritos["ubigeo"] = distritos["ubigeo"].str.replace(r"\D", "", regex=True).str.zfill(6)
distritos.to_parquet(INTERIM / "distritos_typed.parquet", index=False)
print(f"  distritos: {len(distritos):,} filas")

print("Listo. Intermedios en data/interim/")
