"""Build a private Kaggle script kernel from src/runner_body.py and an experiment config.

usage: python tools/make_kernel.py <config.json> [--cpu]
The config needs "slug" (kernel suffix) and "experiments". Output: kernels/<slug>/ ready for `kaggle kernels push -p`.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg_path = Path(sys.argv[1])
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
gpu = "--cpu" not in sys.argv
slug = f"s6e10-{cfg['slug']}"
out = ROOT / "kernels" / slug
out.mkdir(parents=True, exist_ok=True)
if cfg.get("kind") == "darwin":
    body = (ROOT / "src" / "darwin_core.py").read_text(encoding="utf-8") + "\n\n" + \
        (ROOT / "src" / "darwin_body.py").read_text(encoding="utf-8")
elif cfg.get("kind") == "stack":
    body = (ROOT / "src" / "stack_body.py").read_text(encoding="utf-8")
else:
    body = (ROOT / "src" / "runner_body.py").read_text(encoding="utf-8")
(out / f"{slug}.py").write_text(f"CONFIG = {cfg!r}\n\n{body}", encoding="utf-8")
meta = {"id": f"franciscoangulo/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": False, "dataset_sources": ["arseniyshutko/binary-aviation-satisfaction-129k"],
        "competition_sources": ["playground-series-s6e10"], "kernel_sources": cfg.get("kernel_sources", []),
        "model_sources": []}
meta["dataset_sources"] += cfg.get("extra_datasets", [])
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(out)
