# AGENTS.md

Guia operativa para agentes que trabajen en este repositorio.

## Objetivo del proyecto

`medidor` es una herramienta de escritorio + CLI (y UI web con Streamlit) para:
- medir velocidad de internet (`download`, `upload`, `ping`)
- guardar datos en CSV y/o PostgreSQL
- limpiar/analizar datos
- entrenar un modelo de prediccion
- predecir el siguiente valor

## Stack principal

- Python 3.11+
- `customtkinter` (GUI de escritorio)
- `streamlit` (UI web)
- `scikit-learn` + `joblib` (forecasting)
- PostgreSQL opcional

## Archivos clave

- `gui.py`: aplicacion de escritorio
- `web_ui.py`: interfaz web Streamlit
- `main.py`: recoleccion de muestras
- `analyze.py`: limpieza/analisis de CSV
- `train_forecast.py`: entrenamiento
- `predict_next.py`: inferencia
- `postgres/import_csv.py`: CSV -> PostgreSQL
- `postgres/export_to_csv.py`: PostgreSQL -> CSV
- `start_gui.bat`: launcher GUI en Windows
- `start_web.bat`: launcher Web UI en Windows

## Setup recomendado

En Windows:

```bat
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Comandos frecuentes

Ejecutar GUI:

```bat
start_gui.bat
```

Ejecutar Web UI:

```bat
start_web.bat
```

Recolectar muestras por CLI:

```powershell
python main.py --samples 5 --interval-seconds 60 --timeout 20 --storage csv
```

Limpiar/analizar CSV:

```powershell
python analyze.py --input internet_speed_data.csv --rolling-window 5 --output-clean internet_speed_data_clean.csv
```

Entrenar modelo:

```powershell
python train_forecast.py --input internet_speed_data.csv --target download_mbps --horizon 1 --lags 5 --test-size 0.2
```

Predecir siguiente valor:

```powershell
python predict_next.py --input internet_speed_data.csv --model models/medidor_forecast.joblib
```

## Reglas para cambios de codigo

- Mantener cambios pequenos y enfocados.
- No romper la compatibilidad de los scripts CLI existentes.
- Si se cambia la interfaz de argumentos (`argparse`), actualizar `README.md`.
- Evitar introducir dependencias nuevas sin necesidad.
- Si se modifica flujo de datos, validar que:
  - el CSV se sigue escribiendo correctamente
  - no se rompe el guardado/lectura en PostgreSQL
  - entrenamiento y prediccion siguen funcionando con rutas por defecto

## Verificacion minima al terminar cambios

Ejecutar (cuando aplique):

1. `python -m py_compile main.py gui.py web_ui.py analyze.py train_forecast.py predict_next.py`
2. comando CLI tocado en modo corto (pocas muestras o con archivos existentes)
3. si hay cambios de UI, abrir GUI/Streamlit y validar arranque

## Notas operativas

- Revisar `app.log` ante errores silenciosos.
- No commitear artefactos generados (`.log`, CSV de salida, modelos entrenados) salvo que el usuario lo pida.
- Mantener consistencia con estilo existente del archivo tocado.
