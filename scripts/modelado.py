# -*- coding: utf-8 -*-
"""
modelado.py - Funciones compartidas por los notebooks de modelado (09 y 11).

  - preparar_panel: log1p(casos), estacionalidad y todos los rezagos de casos,
    clima e ICEN que pueda usar cualquier horizonte.
  - candidatas / seleccionar: seleccion de una variable por categoria climatica
    (|Spearman| con log1p(casos), solo con los datos que se le pasen).
  - modelo_xgb / objetivo_de / a_casos: XGBoost con objetivo "log", "poisson"
    o "delta" (cambio respecto a la semana T - h).
  - umbral_brote / marcar_brotes: canal endemico para definir las semanas de
    brote y evaluar alertas.

Todas las variables de la fila de la semana T usan informacion hasta T - h.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from comun import ENTRENAMIENTO, EXCLUIR_MODELO, HORIZONTES, MIN_CASOS_TRAIN

SEMILLA = 42
N_REZAGOS = 4                 # rezagos h .. h + 3 (casos y clima)
MAX_REZAGO_SENSIBILIDAD = 12  # rezagos h .. 12 en la prueba de sensibilidad
CATEGORIAS = {
    "temperatura": ["temp_media", "temp_max", "temp_min", "rango_term"],
    "lluvia": ["precip_total", "precip_max_d", "dias_lluvia"],
    "humedad": ["hum_rel"],
    "viento": ["viento"],
}
ICEN = ["icen_lag3", "icen_lag4", "icen_lag5", "icen_lag6"]
REJILLA = [
    {"max_depth": d, "min_child_weight": m, "objetivo": o}
    for d, m, o in itertools.product([2, 3, 4], [1, 5], ["log", "poisson", "delta"])
]
PARAMS_FIJOS = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                    random_state=SEMILLA, n_jobs=4)
MAX_ARBOLES, PARADA_TEMPRANA = 1000, 50

# Canal endemico (definicion de brote "B2" del proyecto anterior del equipo)
ANIOS_EPIDEMICOS = (2017, 2023, 2024)
ANIOS_CANAL = 7
MIN_CASOS_BROTE = 5


# ------------------------------------------------------------ datos -------
def preparar_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Agrega log1p(casos), estacionalidad y los rezagos por departamento."""
    panel = panel.sort_values(["departamento", "ano", "semana"]).reset_index(drop=True)
    panel["log_casos"] = np.log1p(panel["casos"])
    panel["sem_sin"] = np.sin(2 * np.pi * panel["semana"] / 52)
    panel["sem_cos"] = np.cos(2 * np.pi * panel["semana"] / 52)
    g = panel.groupby("departamento", sort=False)

    # Todos los rezagos que pueda usar cualquier horizonte (1 .. 12)
    rezagos_max = max(MAX_REZAGO_SENSIBILIDAD, max(HORIZONTES) + N_REZAGOS - 1)
    nuevas = {}
    for k in range(1, rezagos_max + 1):
        nuevas[f"casos_l{k}"] = g["log_casos"].shift(k)
        for v in itertools.chain.from_iterable(CATEGORIAS.values()):
            nuevas[f"{v}_l{k}"] = g[v].shift(k)
    for h in HORIZONTES:
        for c in ICEN:
            nuevas[f"{c}_s{h}"] = g[c].shift(h)  # valor conocido en la semana T - h
    return pd.concat([panel, pd.DataFrame(nuevas)], axis=1)


def departamentos_modelables(panel: pd.DataFrame) -> tuple[list[str], pd.Series]:
    """Departamentos con >= MIN_CASOS_TRAIN casos en entrenamiento, sin los excluidos."""
    casos_train = panel[panel["ano"].between(*ENTRENAMIENTO)].groupby("departamento")["casos"].sum()
    deptos = [d for d in casos_train[casos_train >= MIN_CASOS_TRAIN].index if d not in EXCLUIR_MODELO]
    return deptos, casos_train


# ---------------------------------------------- seleccion de variables ----
def candidatas(h: int, ultimo_rezago: int) -> dict[str, dict[str, list[str]]]:
    """Columnas candidatas por categoria y variable para el horizonte h."""
    rezagos = range(h, ultimo_rezago + 1)
    cand = {cat: {v: [f"{v}_l{k}" for k in rezagos] for v in vs} for cat, vs in CATEGORIAS.items()}
    cand["icen"] = {c: [f"{c}_s{h}"] for c in ICEN}
    return cand


def seleccionar(train: pd.DataFrame, cand: dict) -> tuple[list[str], list[dict]]:
    """Por categoria, la variable con mayor |Spearman| (en alguno de sus rezagos) con log1p(casos)."""
    columnas, registro = [], []
    for cat, variables in cand.items():
        todas = [c for cols in variables.values() for c in cols]
        # NaN (variable o casos constantes, p. ej. un departamento sin casos en los
        # primeros anos) cuenta como correlacion 0
        corr = train[todas].corrwith(train["log_casos"], method="spearman").abs().fillna(0)
        mejor_var = max(variables, key=lambda v: corr[variables[v]].max())
        columnas += variables[mejor_var]
        registro.append({"categoria": cat, "variable": mejor_var,
                         "rezago_mas_correlacionado": corr[variables[mejor_var]].idxmax(),
                         "abs_spearman": round(float(corr[variables[mejor_var]].max()), 3)})
    return columnas, registro


# ------------------------------------------------------------ modelo ------
def modelo_xgb(config: dict, n_arboles: int, parada: bool) -> XGBRegressor:
    objetivo = "count:poisson" if config["objetivo"] == "poisson" else "reg:squarederror"
    extra = {"early_stopping_rounds": PARADA_TEMPRANA} if parada else {}
    return XGBRegressor(n_estimators=n_arboles, max_depth=config["max_depth"],
                        min_child_weight=config["min_child_weight"], objective=objetivo,
                        **PARAMS_FIJOS, **extra)


def objetivo_de(config: dict, df: pd.DataFrame, h: int) -> pd.Series:
    if config["objetivo"] == "log":
        return df["log_casos"]
    if config["objetivo"] == "delta":
        return df["log_casos"] - df[f"casos_l{h}"]
    return df["casos"]


def a_casos(config: dict, pred: np.ndarray, df: pd.DataFrame, h: int) -> np.ndarray:
    if config["objetivo"] == "log":
        casos = np.expm1(pred)
    elif config["objetivo"] == "delta":
        casos = np.expm1(df[f"casos_l{h}"].to_numpy() + pred)
    else:
        casos = pred
    return np.clip(casos, 0, None)


def metricas(y, p) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


# ---------------------------------------------------- canal endemico ------
def umbral_brote(casos_depto: pd.DataFrame) -> pd.DataFrame:
    """Umbral de brote por departamento, ano y semana.

    Q3 (percentil 75) de los casos de la MISMA semana epidemiologica en los 7
    anos previos no epidemicos (se excluyen 2017, 2023 y 2024). Solo usa anos
    anteriores, asi que el umbral se conoce de antemano. La semana 53 usa la
    historia de la 52. casos_depto: departamento, ano, semana, casos (con
    historia desde 2000, sin filas para las semanas sin casos).
    """
    anios = range(int(casos_depto["ano"].min()), int(casos_depto["ano"].max()) + 1)
    deptos = casos_depto["departamento"].unique()
    completo = (casos_depto.assign(semana=casos_depto["semana"].clip(upper=52))
                .groupby(["departamento", "ano", "semana"])["casos"].sum()
                .reindex(pd.MultiIndex.from_product([deptos, anios, range(1, 53)],
                                                    names=["departamento", "ano", "semana"]), fill_value=0))
    tabla = completo.unstack("ano")  # filas: (departamento, semana); columnas: anos
    filas = []
    for anio in anios:
        previos = [a for a in anios if a < anio and a not in ANIOS_EPIDEMICOS][-ANIOS_CANAL:]
        if len(previos) < ANIOS_CANAL:
            continue
        q3 = tabla[previos].quantile(0.75, axis=1).rename("umbral").reset_index()
        filas.append(q3.assign(ano=anio))
    return pd.concat(filas, ignore_index=True)


def marcar_brotes(df: pd.DataFrame, umbrales: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'umbral' y 'brote' (casos > umbral y casos >= 5) a una tabla con departamento, ano, semana y casos."""
    clave = df.assign(sem52=df["semana"].clip(upper=52))
    salida = clave.merge(umbrales.rename(columns={"semana": "sem52"}), on=["departamento", "ano", "sem52"],
                         how="left").drop(columns="sem52")
    salida["brote"] = (salida["casos"] > salida["umbral"]) & (salida["casos"] >= MIN_CASOS_BROTE)
    return salida
