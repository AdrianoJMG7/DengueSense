# DengueSense: interfaz

```powershell
.venv\Scripts\activate
streamlit run app/tablero.py
```

Se abre en http://localhost:8501. No entrena nada: usa resultados ya calculados.
- `data/processed/tablero.parquet` (notebook 13) y `data/processed/panel_depto_semana.parquet` (script 07).
- `outputs/*.csv` (notebooks 09, 11, 12, 13 y 14, y script 10).

## Pestañas
1. **Mapa de riesgo:** mapa por departamento (pronóstico / umbral del canal endémico) y ranking de la semana. Tiene tres periodos (validación 2016–2024, prueba 2025 y pronóstico 2026), de 1 a 4 semanas de anticipación, y permite elegir el modelo.
2. **Departamento:** serie interactiva con casos, pronóstico, umbral y semanas de brote, más indicadores del departamento.
3. **Desempeño de los modelos:** error en 9 años de validación, ¿ayuda el clima? (H2), alertas de brote, prueba real en 2025 y comparación con el LSTM.
4. **Factores climáticos:** mapa de calor SHAP por departamento y variables más importantes (H1).
5. **Lima:** dengue emergente, El Niño costero de 2023 y pronóstico 2023–2024 (H3).
6. **Acerca del proyecto:** método, limitaciones, validación externa y fuentes.

## Archivos
- `tablero.py`: la aplicación. El tema está en `../.streamlit/config.toml`.
- `peru_departamental.geojson`: límites de los departamentos, solo para dibujar el mapa. Viene de [*peru-geojson*](https://github.com/juaneladio/peru-geojson) de Juan Eladio, basado en el INEI, con licencia **MPL-2.0**.

## Versión en un solo archivo HTML
`DengueSense.html`, en la raíz del proyecto, tiene la misma interfaz en un solo archivo. **Cualquier persona puede abrirlo con doble clic**, sin Python, sin instalar nada y sin internet, porque los datos y Plotly.js van incrustados (unos 9 MB). Para regenerarlo cuando cambien los resultados:

```powershell
python scripts/12_exportar_interfaz_html.py
```

La plantilla está en `scripts/plantilla_interfaz.html`.
