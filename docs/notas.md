# Notas del pipeline — decisiones y pendientes

## Decisiones tomadas
- **Lenguaje**: Python 3.14 + pandas (no hay R instalado en el equipo). Misma lógica de la guía.
- **Unidad de análisis**: panel principal a nivel **celda x semana** (`panel_celda_semana.parquet`). También se generó `casos_depto_semana.parquet` y `celda_depto.parquet` por si se agrega a departamento.
- **Horizonte h = 2 semanas** (como la guía). Regla aplicada: solo predictores con rezago >= 2.
- **Split temporal**: train 2010–2023, test 2024 (el clima cubre 2010–2024; los casos 2000–2024, pero solo 2010–2024 entran al panel).
- **Definición de caso**: todo el archivo ya es dengue CIE-10 A97.x (3 formas clínicas). No existe `tipo_dx` en la fuente; no hubo filas que filtrar por diagnóstico.
- **Duplicados**: 24.09% de las filas de casos son idénticas exactas. Se conservaron (no hay ID de caso); columna `duplicado_exacto` disponible en el parquet raw tipado.
- **Fuga de información**: excluidas `*_anomalia` (posiblemente calculadas con todos los años) y todas las variables ICEN del mes en curso (solo `icen_lag1..6`).

## Hallazgos del control de calidad
- Clima: 181 celdas x 782 semanas, series completas, 0 violaciones de rangos físicos, rezagos verificados exactos (max diff = 0).
- Casos: solo 2 filas (0.00%) descartadas por ubigeo sin celda.
- Panel: 141,542 filas; 937,119 casos (2010–2024). El resto de los 1,029,419 casos limpios corresponde a 2000–2009, fuera del rango del clima.
- 77.2% de las filas celda-semana tienen 0 casos (esperado: unidad fina).

## Pendientes / a verificar con fuentes oficiales
1. **`semana`**: asumir semana epidemiológica MINSA en casos y clima (consistente). La fecha del panel es aproximada (lunes de la semana epidemiológica).
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
