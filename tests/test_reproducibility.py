from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interpreterule.pipeline import run_full_pipeline


def test_core_metrics_reproduce_assignment_results(tmp_path):
    artifacts = run_full_pipeline(
        dataset_path=ROOT / "data" / "raw" / "machlearnia.csv",
        output_dir=tmp_path,
        include_extensions=True,
    )

    metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))

    assert metrics["clan_classification"]["selected_features"] == ["avoidance", "peril"]
    assert metrics["clan_classification"]["best_k"] == 9
    assert round(metrics["clan_classification"]["train_accuracy"], 3) == 0.948

    assert metrics["rank_rule_learning"]["train_accuracy"] == 1.0
    learned_features = {row["clan"]: row["feature"] for row in metrics["rank_rule_learning"]["rules"]}
    assert learned_features == {
        "haldane": "forgiveness",
        "irvine": "spike_rate",
        "lewis": "angst",
        "wheatley": "charring",
    }

    assert metrics["prowess_formula"]["features"] == ["charring", "sliminess"]
    assert metrics["prowess_formula"]["integer_weights"] == [-27796, 6, 887]
    assert round(metrics["prowess_formula"]["train_r2"], 4) == 0.9993
