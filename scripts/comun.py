# -*- coding: utf-8 -*-
"""
comun.py - Funciones compartidas por los scripts del pipeline.

Semana epidemiologica (estandar MMWR, el mismo que usa el CDC-MINSA): va de
domingo a sabado y la semana 1 es la que contiene el 4 de enero. NO coincide
con "1 de enero + 7*(semana-1)" (error frecuente 7 de la guia).
"""
import datetime as dt

import pandas as pd
from epiweeks import Week


def fecha_central(ano: int, semana: int) -> pd.Timestamp:
    """Miercoles (dia central) de la semana epidemiologica.

    Se cuenta desde la semana 1 del ano para aceptar tambien una semana 53
    en anos que no la tienen.
    """
    inicio = Week(int(ano), 1, system="cdc").startdate()
    return pd.Timestamp(inicio + dt.timedelta(weeks=int(semana) - 1, days=3))


def agregar_fecha_y_mes(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'fecha' (dia central de la semana epidemiologica) y 'mes' a una tabla con ano y semana."""
    calendario = df[["ano", "semana"]].drop_duplicates()
    calendario = calendario.assign(
        fecha=[fecha_central(a, s) for a, s in zip(calendario["ano"], calendario["semana"])]
    )
    calendario["mes"] = calendario["fecha"].dt.month
    return df.merge(calendario, on=["ano", "semana"], how="left")
