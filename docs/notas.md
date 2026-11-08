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
```
