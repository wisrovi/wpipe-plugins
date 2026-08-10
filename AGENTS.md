# AGENTS.md — Reglas del Repositorio (wpipe-plugins)

Reglas obligatorias para cualquier agente o contribuidor que trabaje en este repo.
Complementa al `AGENTS.md` de la raíz del workspace (`../AGENTS.md`): para el ecosistema
manda `wpipe_os/AGENTS.md`; este archivo es la guía específica de `wpipe-plugins`.

## 1. Leer primero (obligatorio)

- `../AGENTS.md` (reglas del ecosistema `wpipe_os`) y este archivo.
- `../PROJECT_MAP.md` (arquitectura del ecosistema: módulos, repos hermanos, versionado).

## 2. Estructura del repo

```text
wpipe-plugins/
├── pyproject.toml           # paquete instalable: metadatos + deps globales
├── .ruff.toml               # config ruff ACTIVA (tiene prioridad sobre pyproject)
├── src/wpipe_plugins/       # código instalable (editable install)
│   └── <dominio>/           # categoría (vision, connectivity, ...)
│       └── <plugin>/        # paquete del plugin (snake_case)
├── tests/                   # espejo de src/ + tests globales
├── examples/                # ejemplos runnables de usuario (p.ej. eigecam.py)
├── demo_real/               # modelos/imágenes reales para verificación GPU
└── steps_catalog.json       # catálogo público de plugins
```

Instalación editable confirmada: `wpipe_plugins.__file__` apunta a
`src/wpipe_plugins/__init__.py`, así que los cambios en `src/` se reflejan sin reinstalar.

## 3. Estructura estándar de un plugin/estado (obligatoria)

Cada nuevo estado/plugin es un paquete bajo `src/wpipe_plugins/<dominio>/<plugin>/`:

```text
ecam_yolo/                      # referencia (modelo a seguir)
├── __init__.py                 # exports públicos: from .states... import X; __all__
├── config/                     # constantes (DEVICE, umbrales, ...)
├── exceptions/                 # excepciones custom (módulos snake_case)
├── requirements.txt            # deps específicas del plugin (la referencia steps_catalog.json)
├── schemas/                    # DTOs pydantic para validar el input del run (@to_obj)
├── states/                     # lógica del step (clase decorada @step)
├── utils/                      # helpers de parsing/render
├── wrappers/                   # adaptadores (p.ej. YOLO → CAM)
└── examples/                   # example_*.py, un script por caso de uso
```

Reglas de estructura:

- Nombres de módulos y carpetas en `snake_case` (`states/ecam_yolo.py`,
  `exceptions/yolo_error.py`). Nunca CamelCase en nombres de archivo (ruff N999).
- Cada subpaquete lleva `__init__.py` con docstring de módulo (D104).
- El `__init__.py` del plugin exporta solo la API pública (paso + config), con `__all__`.

## 4. Contrato de un step

- El step es una clase decorada con `@step(...)` de wpipe.
- El `__call__` recibe los datos del run convertidos a un DTO con `@to_obj(MyInferenceObj)`.
- **Siempre retorna un dict `{"results": [...]}`** (contrato de merge de wpipe); una
  entrada por imagen procesada, cada una con `model_results` (lista de dicts).
- **Campos inmutables fijados internamente por el step** (no deben llegarse en el run):
  p. ej. `save`, `verbose`, `image_name` (derivado del source). No exponerlos en el
  data contract del run.
- **Degradación elegante**: si falta un requisito (modelo, imágenes), NO lanzar
  excepción; retornar un resultado con `status` informativo (`no_model`, `no_images`)
  y `message`, para que el pipeline no se rompa. Un path de modelo inválido sí lanza.
- **Modelo seleccionable**: `model_path` puede venir en el config (constructor) o en los
  datos del run (el del run tiene prioridad/override). En los **ejemplos** se usa uno u
  otro, nunca ambos (ver §5).

## 5. Convenciones de ejemplos

- Ubicación principal: `src/wpipe_plugins/<dominio>/<plugin>/examples/`.
- **Importan SIEMPRE desde la librería instalada**, nunca con imports relativos:
  `from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO`.
  Un usuario final copia el ejemplo fuera del repo; los imports relativos lo rompen.
- **CLI-parametrizables**: sin rutas hardcodeadas de modelos/imágenes inexistentes.
  Exponer las entradas como argumentos CLI (`--model`, `--image`, `--images`,
  `--images-dir`, `--output-dir`, `--device`, `--conf`) y una función
  `run_*_example(...)` que reciba `model_path` y la entrada como parámetros obligatorios.
- **Modelo en init o en run, no ambos**: si `model_path` va en `ECAMConfig`, no pasarlo
  en `inference_data` (y viceversa). `example_model_in_run.py` demuestra el patrón
  solo-en-run (config sin modelo).
- Cada ejemplo debe tener tests de import + end-to-end (ver §6).

## 6. Tests

- Viven en `tests/` espejando `src/`: `tests/<dominio>/<plugin>/`.
- **No requieren GPU ni pesos reales**: cada plugin provee un `conftest.py` con fakes
  (mockear YOLO y EigenCAM) y `monkeypatch.chdir(tmp_path)`.
- Por ejemplo, para cada ejemplo:
  - `test_examples_import.py`: el módulo importa y expone su runner.
  - `test_examples_run.py`: el pipeline real corre de extremo a extremo con fakes y se
    verifican artefactos (visualización no vacía, `results` mergeable, `status == "ok"`).
- Además, tests a nivel step para el contrato: modelo override en run, modelo solo-en-run,
  sin modelo (graceful), lista de imágenes, directorio, directorio vacío.
- Tests globales en `tests/` raíz (p. ej. `test_imports.py`).
- Ejecutar desde la raíz de `wpipe-plugins/` con el venv del workspace:

  ```bash
  ../../.venv/bin/python -m pytest -q
  ../../.venv/bin/python -m pytest tests/vision/ecam_yolo/ -q
  ```

- `test_examples.py` de `wpipe/` (workspace) no es gate de regresión: es lento y tiene
  fallos pre-existentes no relacionados. No usarlo como verificación.

## 7. Ruff (obligatorio)

- Config activa: `.ruff.toml` (tiene prioridad sobre `[tool.ruff]` de `pyproject.toml`).
  `line-length = 120`, docstrings google, reglas D/E/F/W/I/N/UP/B/A/C4/SIM/RUF/PL/PT.
- Correr SIEMPRE antes de dar por terminado un cambio:

  ```bash
  ruff check src/wpipe_plugins/
  ruff check tests/
  ruff check examples/
  ```

- Debe quedar **100% limpio**, sin `# noqa` de conveniencia.
- Normas concretas que se aplican de facto:
  - Módulos `snake_case` (N999); docstrings google-style en todo lo público (D100–D415)
    y docstring de módulo en cada paquete (D104).
  - Tipos modernos: `X | None` (UP045), `str | int` (UP007), `collections.abc.Callable`
    (UP035), `from torch import nn` (PLR0402).
  - `zip(..., strict=True)` (B905).
  - Sin `global`: usar una clase con estado, p. ej. `_ColorPalette` (PLW0603).
  - Identificadores `snake_case` (N806); controlar complejidad extrayendo helpers para
    no exceder PLR0912/PLR0915.

## 8. Verificación de regresión

1. Plugin: `../../.venv/bin/python -m pytest -q` → todo en verde.
2. Regresión del ecosistema (desde `wpipe/`):
   `pytest test/ --ignore=test/test_examples.py -q` → 143 tests.
3. Si el cambio toca estilo/tipos: `ruff check` (y opcional `mypy`; en este entorno mypy
   puede fallar por stubs de torch vs Python 3.13 — limitación pre-existente).

## 9. Catálogo y mapa

- Al añadir un plugin, registrar su entrada en `steps_catalog.json` (nombre, `func_name`,
  namespace, `requirements` apuntando al `requirements.txt` del plugin, `how_to_use`).
- Cualquier cambio de estructura/API/estrategia de pruebas debe reflejarse en
  `../PROJECT_MAP.md` en el mismo paso de trabajo.

## 10. Prohibido

- No editar `wpipe/old_wpipe` ni `old_wpipe/` (versiones históricas).
- No crear documentación nueva por iniciativa propia; mantener la existente.
- No commitear salvo que se pida explícitamente.
