# Tablero de riesgo de dengue

```powershell
.venv\Scripts\activate
streamlit run app/tablero.py
```

Necesita `data/processed/tablero.parquet`, que genera `notebooks/13_modelos_finales_y_pronostico.ipynb`.

## Archivos
- `tablero.py`: la aplicación de Streamlit.
- `peru_departamental.geojson`: límites de los departamentos, solo para dibujar el mapa. Viene del repositorio [*peru-geojson*](https://github.com/juaneladio/peru-geojson) de Juan Eladio, basado en el INEI, y tiene licencia **MPL-2.0**.
