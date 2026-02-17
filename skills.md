# skills.md

Registro de skills para este repositorio.

## Estado actual

Este proyecto ya tiene 1 skill local propia dentro del repo.

Skills disponibles en la sesion (globales del sistema Codex):
- `skill-creator`: crear/actualizar skills.
- `skill-installer`: listar/instalar skills desde catalogo o GitHub.

## Skills locales del proyecto

### model-train-check
- Descripcion: ejecutar y validar el flujo de analisis, entrenamiento e inferencia del modelo de `medidor`.
- Ruta: `skills/model-train-check/SKILL.md`
- Trigger: usar cuando se pida entrenar, validar metricas, depurar fallos del pipeline de forecasting o verificar artefactos `models/*.joblib` y `models/*metrics.json`.

## Objetivo de este archivo

Centralizar las skills que el proyecto use de forma recurrente para que cualquier agente sepa:
- cuales existen
- donde estan
- cuando aplicarlas

## Plantilla para agregar una skill local

Usar este formato cuando se cree una skill del proyecto:

```md
### <nombre-skill>
- Descripcion: <que resuelve>
- Ruta: <ruta al SKILL.md>
- Trigger: <cuando debe usarse>
```

## Siguiente paso recomendado

Agregar una segunda skill local para flujo de PostgreSQL (`import_csv.py`/`export_to_csv.py`) con validaciones de conexion y schema.
