from __future__ import annotations

import json

from fraudbench.config import PROJECT_ROOT

if __name__ == "__main__":
    metrics_path = PROJECT_ROOT / "outputs" / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit("Run python training_model/train.py first")
    print(json.dumps(json.loads(metrics_path.read_text()), indent=2))
