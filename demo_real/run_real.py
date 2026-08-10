"""Run the ECAM-YOLO step with REAL models and REAL images through a wpipe Pipeline."""
import sys
import os

from wpipe import Pipeline
from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO

CWD = os.path.dirname(os.path.abspath(__file__))

TASKS = {
    "detect": {
        "model": os.path.join(CWD, "yolov8n.pt"),
        "image": os.path.join(CWD, "bus.jpg"),
        "output_dir": os.path.join(CWD, "output", "detect"),
    },
    "segment": {
        "model": os.path.join(CWD, "yolov8n-seg.pt"),
        "image": os.path.join(CWD, "bus.jpg"),
        "output_dir": os.path.join(CWD, "output", "segment"),
    },
    "classify": {
        "model": os.path.join(CWD, "yolov8n-cls.pt"),
        "image": os.path.join(CWD, "zidane.jpg"),
        "output_dir": os.path.join(CWD, "output", "classify"),
    },
}

for task, cfg in TASKS.items():
    print(f"\n{'='*60}\nTASK: {task.upper()}\n{'='*60}")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    config = ECAMConfig(model_path=cfg["model"], confidence_threshold=0.25, device=0)
    step = ImageECamYOLO(config)

    pipe = Pipeline(pipeline_name=f"real_{task}", verbose=False)
    pipe.set_steps([step])

    context = pipe.run({
        "image_data": cfg["image"],
        "image_name": os.path.basename(cfg["image"]),
        "save": True,
        "output_dir": cfg["output_dir"],
        "verbose": False,
    })

    for r in context.get("results", []):
        for obj in r.get("model_results", []):
            status = obj.get("status")
            if status != "ok":
                print(f"  {status}")
                continue
            if task == "classify":
                print(f"  top1 -> {obj['name']} ({obj['confidence']*100:.1f}%)")
            elif task == "segment":
                print(f"  {obj['name']} bbox={obj['bbox']} points={obj['points_count']}")
            else:
                print(f"  {obj['name']} bbox={obj['bbox']}")

    print(f"  visualization -> {cfg['output_dir']}")
    for f in sorted(os.listdir(cfg["output_dir"])):
        print(f"    {f}  ({os.path.getsize(os.path.join(cfg['output_dir'], f))} bytes)")
