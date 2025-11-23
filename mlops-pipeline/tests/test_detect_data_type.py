import json
from src.utils.data_router import detect_data_type

def test_meta_detection(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"type": "image"}))
    assert detect_data_type(str(tmp_path)) == "image"

def test_image_ext(tmp_path):
    (tmp_path / "img1.jpg").write_text("x")
    (tmp_path / "img2.png").write_text("x")
    assert detect_data_type(str(tmp_path)) == "image"

def test_audio_ext(tmp_path):
    (tmp_path / "a.wav").write_text("x")
    (tmp_path / "b.mp3").write_text("x")
    assert detect_data_type(str(tmp_path)) == "audio"

def test_tabular_from_csv(tmp_path):
    (tmp_path / "d1.csv").write_text("a")
    (tmp_path / "d2.csv").write_text("b")
    assert detect_data_type(str(tmp_path)) == "tabular"
