import json
from pathlib import Path
from collections import Counter

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
AUDIO_EXT = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
TEXT_EXT = {'.txt', '.csv', '.json', '.tsv'}

def _read_meta(meta_path: Path):
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def detect_data_type(data_dir: str):
    p = Path(data_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    meta_path = p / "meta.json"
    meta = _read_meta(meta_path) if meta_path.exists() else None
    if meta and isinstance(meta, dict):
        t = meta.get("type")
        if isinstance(t, str) and t.lower() in {"text","image","audio","tabular"}:
            return t.lower()

    exts = Counter()
    for f in p.rglob('*'):
        if f.is_file():
            ext = f.suffix.lower()
            exts[ext] += 1

    if not exts:
        return "tabular"

    type_counts = {
        'image': sum(count for ext, count in exts.items() if ext in IMAGE_EXT),
        'audio': sum(count for ext, count in exts.items() if ext in AUDIO_EXT),
        'text': sum(count for ext, count in exts.items() if ext in TEXT_EXT),
    }

    dominant = max(type_counts, key=lambda k: type_counts[k])
    if type_counts[dominant] == 0:
        return "tabular"

    if dominant == "text" and exts.get('.csv', 0) > 1:
        return "tabular"

    return dominant
