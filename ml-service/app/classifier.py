"""Read the existing TF-IDF artifact stored outside app/ (do not retrain)."""
import os
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("AVITO_MODEL_PATH", str(ROOT / "models/baseline/tfidf_baseline.joblib")))
TARGETS = {102, 103, 104, 105, 106}


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}; keep models/ outside app/")
    # joblib/pickle can execute code. Load only the locally trained trusted artifact.
    model = joblib.load(MODEL_PATH)
    if set(map(int, model["ids"])) != TARGETS:
        raise ValueError("Unexpected classifier target IDs")
    return model


def classify(model: dict, description: str, source_id: int) -> list[int]:
    """One label per target category. Source category is always excluded."""
    probabilities = model["pipeline"].predict_proba([description])[0]
    if len(probabilities) != len(model["ids"]):
        raise ValueError("Classifier probability shape mismatch")
    selected = [int(category_id) for category_id, p, threshold in zip(
        model["ids"], probabilities, model["thresholds"]
    ) if int(category_id) in TARGETS and int(category_id) != source_id and p >= threshold]
    return sorted(set(selected))
