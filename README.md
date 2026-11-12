# DengueSense 🦟

**Pronóstico semanal de dengue y alertas de brote por departamento en el Perú, con 1 a 4 semanas de anticipación.**

En 2023 y 2024 el Perú vivió las peores epidemias de dengue de su historia, con más de 250 000 casos al año. La respuesta suele ser reactiva: se actúa cuando los casos ya superaron el canal endémico. DengueSense pronostica los casos semanales de 17 departamentos con aprendizaje automático (XGBoost y LSTM). Usa los casos notificados, el clima y el Índice Costero El Niño (ICEN), y explica qué factores climáticos pesan en cada región con SHAP.

**Para ver los resultados sin instalar nada:** abre [`DengueSense.html`](DengueSense.html) con doble clic. Tiene mapas de riesgo y gráficos interactivos, y funciona sin internet. El artículo completo está en [`docs/articulo/`](docs/articulo/), en Word y PDF y en formato IEEE.

## Resultados principales

| Pregunta | Resultado |
|---|---|
| ¿Se puede pronosticar? | Sí. XGBoost aprendiendo el **cambio** semanal de casos supera a la persistencia en 9 a 13 de 17 departamentos (2016–2024). En **2025**, un año que ningún modelo vio, la supera en 11 a 13 de 17 |
| **H1:** ¿los factores climáticos cambian por región? | **Sí.** En la costa norte domina la temperatura; en la selva, la humedad, la lluvia y el viento (SHAP) |
| **H2:** ¿el clima mejora el pronóstico? | **No de forma consistente.** Con 9 años de prueba, el clima reduce el error en solo 8 a 11 de 17 departamentos, con una mejora mediana de −1,3 % a +1,6 % |
| **H3:** ¿el desempeño cae en periodos atípicos? | **Sí.** En Lima (dengue emergente en 2023), un modelo con solo la historia local falla; uno entrenado con la costa reduce el error un 30 % |
| Alertas de brote | Avisar que un brote **sigue** es fácil (≈ 90 % de recall y precisión). Anticipar un brote **nuevo** es difícil: el 12–21 % a 2–4 semanas |

Más detalle en el artículo y en [`docs/notas.md`](docs/notas.md), que registra cada decisión y sus cifras.

## Estructura

```
DengueSense.html          interfaz completa en un solo archivo (abrir con doble clic)
app/tablero.py            la misma interfaz en Streamlit
scripts/                  limpieza, panel y utilidades
  01_importar.py ... 06_modelado_prep.py   limpieza y panel celda × semana
  07_panel_region.py      panel departamento × semana (clima ponderado, población, incidencia)
  10_validacion_externa.py  comparación con OpenDengue y el Minsa
  11_verificar_supuestos.py comprobación de mm4_lag2, anomalías, ICEN y calendario
  12_exportar_interfaz_html.py  genera DengueSense.html
  comun.py, modelado.py   división temporal, regiones y funciones de modelado compartidas
  articulo/               genera las figuras, tablas y el Word del artículo
notebooks/
  08_lineas_base           persistencia, promedio estacional y regresión
  09_xgboost_shap          XGBoost con y sin clima, y SHAP por región
  11_alertas_y_validacion_temporal   validación de origen móvil 2016–2024 y alertas de brote
  12_lima_dengue_emergente           Lima y El Niño costero de 2023
  13_modelos_finales_y_pronostico    modelos finales, prueba 2025 y pronóstico 2026
  14_lstm                  extensión: red LSTM
outputs/                  métricas (CSV) y figuras
docs/
  notas.md, diccionario.csv   decisiones, hallazgos y diccionario de datos
  articulo/                   artículo IEEE (Word y PDF)
data/                     datos (no se suben a git; ver abajo)
```

## Cómo reproducirlo

**1. Entorno** (Python 3.11 o superior):
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# para el LSTM en CPU, si pip no encuentra torch:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**2. Datos.** Los archivos pesados no están en git. Colócalos en `data/raw/`:
- `datos_abiertos_vigilancia_dengue_2000_2024.csv` y `datos_abiertos_vigilancia_dengue_2000_2025.csv`, del CDC-MINSA ([datos abiertos](https://www.datosabiertos.gob.pe/dataset/vigilancia-epidemiol%C3%B3gica-de-dengue)).
- `clima_celda_semana.csv`, `icen_mensual.csv`, `mapa_ubigeo_celda.csv` y `distritos_dengue_centroides.csv`.

La población del INEI ya está en `data/external/`.

**3. Limpieza y panel:**
```powershell
python scripts/01_importar.py
python scripts/02_casos.py
python scripts/03_clima.py
python scripts/04_icen.py
python scripts/05_panel.py
python scripts/06_modelado_prep.py
python scripts/07_panel_region.py
python scripts/10_validacion_externa.py      # opcional, descarga OpenDengue
python scripts/11_verificar_supuestos.py     # opcional
```

**4. Modelos.** Ejecuta los notebooks en orden: 08, 09, 11, 12, 13 y 14. El 09 tarda unos 15 minutos. El 11 reutiliza sus predicciones guardadas; con `REENTRENAR = True` vuelve a entrenar en unos 30 minutos.
```powershell
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 08_lineas_base.ipynb
```

**5. Interfaz y artículo:**
```powershell
streamlit run app/tablero.py                  # interfaz en el navegador
python scripts/12_exportar_interfaz_html.py   # regenera DengueSense.html
python scripts/articulo/generar_material.py   # figuras y tablas del artículo
```

## Limitaciones
- Los datos abiertos tienen algo menos de casos que el total oficial: −3 % a −6 % en años epidémicos.
- Los modelos suponen que los casos recientes ya están completos; en la práctica llegan con retraso de notificación.
- El clima es el promedio de celdas de unos 55 km y le faltan el punto de rocío y la presión.
- La población de 2010–2017 es estimada.
- Lima se analiza aparte porque su epidemia es reciente.

## Equipo
Universidad Nacional Mayor de San Marcos, Facultad de Ingeniería de Sistemas e Informática.
- Adriano Jesús Medrano Garay · adriano.medrano@unmsm.edu.pe
- Julio Daniel Chavez Huamani · julio.chavezh@unmsm.edu.pe
- Oder Pedro Quincho Cordova · oder.quincho@unmsm.edu.pe

## Fuentes y licencias
CDC-MINSA, datos abiertos de vigilancia de dengue (ODC-By) · ENFEN, Índice Costero El Niño · INEI, población proyectada · OpenDengue (validación) · mapa de departamentos: *peru-geojson* (MPL-2.0).
