"""Ejemplo simple: detección YOLO + Eigen-CAM con el plugin ecam_yolo.

Sin argumentos de línea de comandos: usa los modelos e imágenes de ejemplo que
viven en ``wpipe-plugins/demo_real/`` (yolov8n.pt, bus.jpg). La visualización
se guarda en ``output_eigecam/`` junto a este script.

Ejecución:
    python eigecam.py
"""

import os
from pathlib import Path

from wpipe import Pipeline

from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO

# Recursos de ejemplo del repo (wpipe-plugins/demo_real/)
DEMO_DIR = Path(__file__).resolve().parents[3] / "demo_real"
MODEL = str(DEMO_DIR / "yolov8n.pt")
IMAGE = str(DEMO_DIR / "bus.jpg")
DEVICE = 0
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_eigecam")


def main() -> None:
    """Run the Eigen-CAM detection demo with the demo_real assets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipe = Pipeline(pipeline_name="eigecam_demo", verbose=False)
    pipe.set_steps(
        [
            ImageECamYOLO(
                ECAMConfig(
                    confidence_threshold=0.25,
                )
            ),
        ]
    )

    # 3. Run: imagen + modelo (override) + dónde guardar la visualización.
    context = pipe.run(
        {
            "image_data": IMAGE,
            "model_path": MODEL,
            "output_dir": OUTPUT_DIR,
        }
    )

    # 4. Resultados
    for result in context["results"]:
        detections = [obj["name"] for obj in result["model_results"]]
        print(f"Detectado en {IMAGE}: {detections}")
        print(f"Visualización guardada en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
