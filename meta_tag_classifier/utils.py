from pathlib import Path
import joblib

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_artifact(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_artifact(path):
    return joblib.load(Path(path))
