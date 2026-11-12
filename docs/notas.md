# Notas del pipeline — decisiones y pendientes

## Decisiones tomadas
- **Lenguaje**: Python 3.14 + pandas (no hay R instalado en el equipo). Misma lógica de la guía.
- **Unidad de análisis**: panel principal a nivel **celda x semana** (`panel_celda_semana.parquet`). También se generó `casos_depto_semana.parquet` y `celda_depto.parquet` por si se agrega a departamento.
- **Horizonte h = 2 semanas** (como la guía). Regla aplicada: solo predictores con rezago >= 2.
- **Split temporal**: train 2010–2023, test 2024 (el clima cubre 2010–2024; los casos 2000–2024, pero solo 2010–2024 entran al panel).
- **Definición de caso**: todo el archivo ya es dengue CIE-10 A97.x (3 formas clínicas). No existe `tipo_dx` en este archivo (2000–2024, 14 columnas); no hubo filas que filtrar por diagnóstico. La versión 2000–2025 del CDC sí trae `tipo_dx`, solo con C (confirmado) y P (probable), es decir, sin descartados: la definición equivale a "confirmados + probables".
- **Duplicados**: 152,091 filas (14.77%) son copias sobrantes de otra fila idéntica; contando todas las filas de cada grupo repetido son 247,969 (24.09%). Se conservaron (no hay ID de caso); la columna `duplicado_exacto` (True desde la 2.a copia) queda en `data/interim/casos_limpios.parquet` para el análisis de sensibilidad. Conteos en `outputs/casos_controles.csv`.
- **Fuga de información**: excluidas `*_anomalia` (posiblemente calculadas con todos los años) y todas las variables ICEN del mes en curso. Del ICEN solo se usan `icen_lag3..6` (`MIN_REZAGO_ICEN` en `06_modelado_prep.py`): con h = 2, en el 38 % de las semanas el mes de `icen_lag1` aún no termina al predecir, e `icen_lag2` puede no estar publicado (el ICEN usa medias móviles de 3 meses; supuesto a confirmar con ENFEN).
- **Fechas**: `fecha` es el miércoles (día central) de la semana epidemiológica estándar, calculada con `epiweeks` en `scripts/comun.py`. Antes se usaba 1-ene + 7·(semana−1), que asignaba mal el mes del ICEN en 65 de 782 semanas (8 %).
- **Edades imposibles**: 14 casos con edad > 110 años quedan con `edad_anos` = NA; el caso se conserva.

## Hallazgos del control de calidad
- Clima: 181 celdas x 782 semanas, series completas, 0 violaciones de rangos físicos, rezagos verificados exactos (max diff = 0).
- Casos: solo 2 filas (0.00%) descartadas por ubigeo sin celda.
- **Letras perdidas en la fuente**: el CSV de casos trae la Ñ ya dañada como el carácter U+FFFD (`FERRE�AFE`, 41,651 líneas); la versión 2000–2025 del CDC la trae como `_`. En `provincia` y `distrito` siempre era una Ñ y se reemplaza por N; en `localidad` no siempre (por ejemplo `N� 11`), así que ahí se deja. No afecta las uniones, que van por `ubigeo`: no usar nombres como llave.
- Con pandas 3 el texto se lee como dtype `str`; la normalización de `01_importar.py` ahora usa `is_string_dtype` (antes comparaba con `object` y no se aplicaba).
- Panel: 141,542 filas; 937,119 casos (2010–2024). El resto de los 1,029,419 casos limpios corresponde a 2000–2009, fuera del rango del clima.
- 77.2% de las filas celda-semana tienen 0 casos (esperado: unidad fina).

## Pendientes / a verificar con fuentes oficiales
1. **`semana`**: asumir semana epidemiológica MINSA en casos y clima (consistente). La fecha del panel es el miércoles de la semana epidemiológica estándar.
2. **`mm4_lag2`**: confirmar definición exacta antes de usar con h > 2.
3. **Periodo de referencia de `*_anomalia`**: si se recalculan solo con train, podrían reintroducirse como predictores.
4. **Unidad de `viento`**: parece m/s, confirmar.
5. **Validación externa**: falta OpenDengue y sala situacional CDC-Minsa. Mientras tanto, `distritos_dengue_centroides.csv` tiene `casos_2011_2024` para sanity check.
6. **Incidencia**: requiere población (INEI no está en los CSV; `distritos_dengue_centroides.csv` tiene `pob_densidad_2020` pero no población absoluta).
7. **ICEN costo vs. selva**: el índice es costero; considerar interacciones por región si el modelo lo amerita.

## Reproducir
```powershell
python scripts/01_importar.py   # ~2 min (108 MB CSV)
python scripts/02_casos.py
python scripts/03_clima.py
python scripts/04_icen.py
python scripts/05_panel.py
python scripts/06_modelado_prep.py
python scripts/07_panel_region.py
python scripts/10_validacion_externa.py   # opcional: descarga OpenDengue (55 MB)
```

Los entrenamientos están en notebooks (ejecutar en este orden, con el entorno `.venv` activado):

```powershell
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 08_lineas_base.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 09_xgboost_shap.ipynb   # ~15 min
```
También se pueden abrir y ejecutar celda por celda en VS Code o Jupyter.

## Fase 2 de la guía metodológica: líneas base (`scripts/07_panel_region.py` y `notebooks/08_lineas_base.ipynb`)
- **Panel departamento × semana** (`07_panel_region.py` → `data/processed/panel_depto_semana.parquet`): 23 departamentos × 782 semanas = 17,986 filas, 937,119 casos (los mismos del panel de celdas). Los casos se suman por ubigeo → departamento; el clima es el promedio simple de las celdas que tienen algún distrito del departamento (24 celdas abarcan más de un departamento y cuentan en ambos).
- **Población e incidencia** (casos por 100,000 habitantes): INEI, población proyectada por distrito 2018–2026 (`data/external/poblacion_distrital_2018_2026.csv`), sumando **todos** los distritos de cada departamento. **Supuesto:** 2010–2017 no están en la fuente y se estiman hacia atrás con el crecimiento anual compuesto 2018–2024 de cada departamento (−0.39 % a 3.39 %); la columna `poblacion_estimada` las marca. Las proyecciones 2018+ incluyen la migración venezolana, así que 2010–2017 podría quedar algo subestimada.
- **División temporal** (`scripts/comun.py`): entrenamiento 2010–2021, validación 2022–2023 y prueba 2024. La validación sirve para elegir hiperparámetros y comparar modelos; para la prueba se reajusta con 2010–2023. (`06_modelado_prep.py`, a nivel celda, mantiene su corte 2010–2023 / 2024.)
- **Regiones principales**: Piura (costa norte), Loreto (selva baja) y San Martín (selva alta), constante `REGIONES` en `scripts/comun.py`. Alternativas para la tercera: Ucayali o Madre de Dios. Las líneas base se calculan para los 18 departamentos con ≥ 1,000 casos en 2010–2021.
- **Pronóstico a h = 1..4 semanas**: la fila de la semana T solo usa información hasta T − h.
- **Modelos**: persistencia (casos de T − h), promedio estacional (media de la misma semana en los años anteriores) y regresión lineal de log1p(casos) sobre los casos de T − h a T − h − 3, una por departamento y horizonte.
- **Resultados (R², h = 1 / 2 / 3 / 4 semanas)**:
  - Piura, regresión: validación 0.87 / 0.73 / 0.56 / 0.40; prueba 0.92 / 0.84 / 0.74 / 0.61.
  - Loreto, regresión: validación 0.81 / 0.65 / 0.46 / 0.27; prueba 0.82 / 0.63 / 0.41 / 0.22.
  - San Martín, persistencia (mejor que la regresión): validación 0.79 / 0.66 / 0.45 / 0.30; prueba 0.88 / 0.82 / 0.65 / 0.51.
  - El promedio estacional falla en 2022–2024 (R² cercano a 0 o negativo): son años epidémicos atípicos.
  - Detalle en `outputs/fase2_lineas_base_metricas.csv` y `outputs/fase2_lineas_base_2024.png`.
- **Lectura**: a 1 semana casi todo lo explica la inercia de los casos; desde 3–4 semanas el R² cae rápido. Ese es el espacio que el clima rezagado (fase 3, XGBoost) tiene que mejorar. La regresión subestima los picos al volver de la escala log (sesgo de retransformación).

## Revisión de datos antes de la fase 3
- **Validación externa** (`10_validacion_externa.py`, paso 11 de la guía de limpieza): el archivo del equipo (2000–2024) es idéntico al CSV del CDC 2000–2025 en esos años. Frente a OpenDengue V1.3.1 por provincia (construido con el mismo archivo abierto, no independiente) el 73 % de los departamento-año son idénticos y la mayor diferencia es de 222 casos. Frente a los totales nacionales independientes, el panel tiene menos casos: −6 % en 2023 (Minsa AE 006-2024: 273,684), −3 % en 2024, pero −10 a −32 % en años de pocos casos (2016–2020, 2022). **Limitación:** los datos abiertos subestiman el total oficial.
- **Clima ponderado por casos** (`07_panel_region.py`): el promedio simple de celdas incluía celdas de sierra sin dengue (Piura salía 2.2 °C más fría y 5 puntos más húmeda que donde ocurren los casos). Ahora cada celda pesa según los casos 2010–2021 de sus distritos (solo entrenamiento, sin fuga). Comparación en `outputs/clima_ponderado_vs_simple.csv`.
- **Lima fuera del modelo principal** (`EXCLUIR_MODELO` en `comun.py`): el 96 % de sus casos 2010–2024 son de 2023–2024 (antes de 2021 nunca pasó de 362 al año) y el 87 % están en Lima Metropolitana. Analizarla aparte como dengue emergente (hipótesis H3).

## Fase 3 de la guía metodológica: XGBoost y SHAP (`notebooks/09_xgboost_shap.ipynb`)
- **Modelos** (un XGBoost por departamento y horizonte h = 1..4, 17 departamentos): sin clima (casos rezagados T−h..T−h−3 + estacionalidad), con clima (+ clima rezagado T−h..T−h−3 + ICEN conocido en T−h) y con clima rezagado hasta 12 semanas (sensibilidad).
- **Selección de variables** (como el artículo): por categoría (temperatura, lluvia, humedad, viento, ICEN) la variable con mayor |Spearman| con log1p(casos), solo con entrenamiento. Detalle en `outputs/fase3_variables_seleccionadas.csv`.
- **Hiperparámetros**: profundidad, min_child_weight y objetivo, elegidos por MAE en validación 2022–2023 con parada temprana; para la prueba se reentrena con 2010–2023.
- **Decisión clave — objetivo "delta"**: los árboles no extrapolan fuera del rango de entrenamiento. Prediciendo el nivel de casos, San Martín 2024 quedaba clavado en 80–150 casos semanales con 190–380 reales (R² negativo). Predecir el cambio log1p(casos_T) − log1p(casos_{T−h}) lo resuelve; la validación eligió "delta" en el 92 % de los modelos.
- **Resultados (R² en prueba 2024, h = 1 / 2 / 3 / 4)**:
  - Piura: con clima 0.92 / **0.87** / **0.76** / 0.53; regresión 0.92 / 0.84 / 0.74 / 0.61; persistencia 0.94 / 0.85 / 0.71 / 0.52.
  - Loreto: sin clima **0.85** / 0.73 / **0.54** / **0.42**; con clima 0.78 / **0.74** / 0.44 / 0.37; persistencia 0.83 / 0.66 / 0.43 / 0.26.
  - San Martín: todos los XGBoost ≈ persistencia (0.88 / 0.82 / 0.65 / 0.51).
- **H2 (¿el clima aporta?)**: en **validación** sí, sobre todo a más semanas: con clima tiene menor MAE que sin clima en 10/17, 9/17, 12/17 y **16/17** departamentos para h = 1..4 (mediana +3.7 % de mejora a 4 semanas). En la **prueba 2024 no se sostiene**: gana en 8, 9, 6 y 6 de 17 (mediana −3 a −5 % a 3–4 semanas). Lectura honesta: el clima ayuda en 2022–2023, pero en el año epidémico atípico 2024 su aporte es inestable; con dos años de validación la elección de modelos es ruidosa.
- **H1 (factores por región, SHAP a 2 semanas, `outputs/fase3_shap_categorias.png`)**: en la costa norte pesa la temperatura (Piura 51 % de la importancia climática, Lambayeque 65 %, Ica 47 %, Áncash 88 %); en la selva el peso se reparte entre humedad, lluvia y viento (Loreto, Ucayali 40 % humedad, Madre de Dios 35 % humedad). Los casos pasados y la estacionalidad siguen explicando la mayor parte (34–96 % del total). El viento aparece alto en varios departamentos de sierra y selva alta: puede estar actuando como marcador de estación y no como causa. **SHAP muestra asociaciones, no causalidad.**
- **Salidas**: `outputs/fase3_metricas.csv` (incluye persistencia y regresión en las mismas semanas), `outputs/fase3_shap_importancia.csv`, `outputs/fase3_comparacion_regiones.png`, `outputs/fase3_shap_regiones_h2.png`, `data/processed/predicciones_xgboost.parquet`. Tarda ~15 min.
- **Pendientes**: validación cruzada temporal (varios cortes, no solo 2022–2023) para que la elección de modelos sea menos ruidosa; análisis de Lima; intervalos de incertidumbre; métricas de detección de brotes (precisión/recall con un umbral de alerta).

## Fase 5: validación con varios cortes temporales y alertas (`notebooks/11_alertas_y_validacion_temporal.ipynb`)
- **Validación con origen móvil**: para cada año Y de 2016 a 2024, selección de variables e hiperparámetros con años ≤ Y−2 y validación en Y−1, modelo final con ≤ Y−1, evaluación en Y. Solo el objetivo "delta" y 6 configuraciones por modelo. Las funciones de modelado se movieron a `scripts/modelado.py`, que comparten los notebooks 09 y 11 (el 09 da métricas idénticas tras el cambio).
- **H2 con 9 años**: el XGBoost con clima reduce el error total 2016–2024 en 8, 10, 8 y 11 de 17 departamentos (h = 1..4), con mejora mediana entre −1.3 % y +1.6 %; gana la mayoría solo en 2–4 de 9 años. 2022 y 2023 fueron años favorables al clima, por eso la validación del notebook 09 parecía confirmarlo. **Conclusión: el clima no agrega una mejora robusta.**
- **XGBoost sí supera a las líneas base** en error total (Piura a 2 semanas: 108 casos/semana frente a 146 de la persistencia), pero la ventaja no viene del clima.
- **Alertas**: brote = casos > Q3 de la misma semana en los 7 años previos no epidémicos y ≥ 5 casos (`umbral_brote` en `scripts/modelado.py`). Precisión y recall de 0.85–0.94 para todos los modelos (los brotes duran semanas; el 45 % de las semanas evaluadas son de brote). **Brotes nuevos** (sin brote en las 4 semanas previas, 134 en total): todos anticipan solo el 12–28 %; el XGBoost con clima anticipa algo más que el sin clima (20.1 % frente a 16.4 % a 2 semanas), pero no supera a la persistencia y la muestra es pequeña.
- **Decisión**: con "la semana anterior no era de brote" el 60 % de los inicios eran reentradas al umbral de episodios en curso (331 frente a 134), por eso se exige no tener brote en las 4 semanas previas.
- `REENTRENAR = False` en el notebook reutiliza `data/processed/predicciones_validacion_temporal.parquet`; con `True` reentrena todo (~30 min).
- Salidas: `outputs/fase5_validacion_temporal_metricas.csv`, `outputs/fase5_alertas.csv`, `outputs/fase5_h2_por_anio.png`, `outputs/fase5_alertas.png`, `outputs/fase5_alertas_piura.png`.

## Verificación de supuestos (`scripts/11_verificar_supuestos.py`, `outputs/supuestos_verificacion.csv`)
- **`*_mm4_lag2`** = media de las semanas t−2 a t−5 (diferencia 0 con los datos). Usa información de t−2, así que solo es válida para h ≤ 2: en los modelos a 3–4 semanas no se usa (solo se usan rezagos ≥ h).
- **`*_anomalia`** = valor − media de la misma celda y semana en **2010–2024** (diferencia 0). Incluye los años de prueba: **fuga de información confirmada**, por eso siguen excluidas.
- **Semanas del clima**: siguen el calendario epidemiológico (MMWR): semana 53 en 2014 y 2020, igual que el oficial (el ISO la pone en 2015 y 2020). Casos y clima usan el mismo calendario.
- **ICEN**: en la tabla, `icen_lag1..6` son el ICEN de 1–6 meses antes; `icen_mm3` es la media de los meses m, m−1 y m−2; `icen_tendencia` = icen − icen de 3 meses antes; `icen_cap` recorta los valores altos (máx. 2.23 frente a 2.92); `icen_trad` es otra versión (correlación 0.998). Según el ENFEN, el ICEN es la media móvil de 3 meses de la anomalía de TSM en Niño 1+2 (ERSSTv5) y el informe técnico de un mes sale a mediados del siguiente (p. ej. el de febrero 2022, el 14 de marzo); los meses recientes son un "ICEN temporal" que se corrige después. **Usar desde `icen_lag3` es prudente**; `icen_lag2` podría usarse solo si se acepta el valor temporal.
- **`semana` del CDC** = semana de **inicio de síntomas**; **`ubigeo`** = **lugar probable de infección** (no residencia). Fuente: descripción del dataset en datosabiertos.gob.pe. **Limitación:** los modelos suponen que los casos de la semana T − h se conocen completos en T − h; en la práctica los casos de las últimas semanas llegan con retraso de notificación, así que el desempeño en tiempo real sería algo menor.
- **Variables del artículo que faltan**: la tabla de clima no trae punto de rocío ni presión a nivel del mar (el artículo usa ERA5 con ambas). Se declara como limitación; agregarlas requiere descargar ERA5 (Copernicus CDS, con registro).
