"""Ejemplo simple: detección YOLO + Eigen-CAM con el plugin ecam_yolo.

Uso (desde wpipe-plugins/demo_real/):
    python demo_simple.py --model yolov8n.pt --image bus.jpg --device 0
"""

import argparse
import os

from wpipe import Pipeline
from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejemplo simple de ecam_yolo")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--image", default="bus.jpg")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", default="output/simple")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Config + step (el modelo también puede pasarse solo en el run)
    config = ECAMConfig(model_path=args.model, confidence_threshold=0.25, device=args.device)
    step = ImageECamYOLO(config)

    # 2. Pipeline con el step
    pipe = Pipeline(pipeline_name="simple_demo", verbose=False)
    pipe.set_steps([step])

    # 3. Run: imagen + modelo (override) + dónde guardar la visualización
    context = pipe.run(
        {
            "image_data": args.image,
            "image_name": os.path.basename(args.image),
            "model_path": args.model,
            "save": True,
            "output_dir": args.output_dir,
            "verbose": False,
        }
    )

    # 4. Resultados
    for result in context["results"]:
        detections = [obj["name"] for obj in result["model_results"]]
        print(f"Detectado en {args.image}: {detections}")
        print(f"Visualización guardada en {args.output_dir}")


if __name__ == "__main__":
    main()
