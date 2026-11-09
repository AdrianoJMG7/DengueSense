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

# Division temporal de la fase de modelado (por departamento). Nunca al azar.
#   - entrenamiento: ajustar los modelos
#   - validacion: elegir hiperparametros y comparar modelos sin tocar la prueba
#   - prueba: evaluacion final; al final se reentrena con entrenamiento + validacion
ENTRENAMIENTO = (2010, 2021)
VALIDACION = (2022, 2023)
PRUEBA = (2024, 2024)

# Horizontes de pronostico en semanas (guia: 1 a 4)
HORIZONTES = [1, 2, 3, 4]
# Regiones principales (guia, fase 0): costa norte, selva baja y selva alta.
# Alternativas para la tercera: UCAYALI o MADRE DE DIOS.
REGIONES = ["PIURA", "LORETO", "SAN MARTIN"]
# Departamentos con datos suficientes: >= MIN_CASOS_TRAIN casos en entrenamiento
MIN_CASOS_TRAIN = 1000
# Fuera del modelo principal:
#   - LIMA: epidemia nueva (96 % de sus casos 2010-2024 son de 2023-2024; nunca
#     paso de 362 casos al ano antes de 2021), asi que el entrenamiento casi no
#     tiene ejemplos. Se analiza aparte como dengue emergente.
EXCLUIR_MODELO = ["LIMA"]


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
