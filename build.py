"""Rebuild index.html from template.html + the data/*.json files.

Run this after regenerating species_data.json, acoustic_features.json or
frame_features.json (e.g. after re-running the extract_*.py / frame_features.py
pipeline) to produce the final, self-contained page.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent

def main():
    html = (ROOT / "template.html").read_text(encoding="utf-8")
    species = json.loads((ROOT / "species_data.json").read_text(encoding="utf-8"))
    audio = json.loads((ROOT / "acoustic_features.json").read_text(encoding="utf-8"))
    frames = json.loads((ROOT / "frame_features.json").read_text(encoding="utf-8"))

    assert html.count("__SPECIES_JSON__") == 1, "template must contain exactly one __SPECIES_JSON__ placeholder"
    assert html.count("__AUDIO_JSON__") == 1, "template must contain exactly one __AUDIO_JSON__ placeholder"
    assert html.count("__FRAMES_JSON__") == 1, "template must contain exactly one __FRAMES_JSON__ placeholder"

    html = html.replace("__SPECIES_JSON__", json.dumps(species))
    html = html.replace("__AUDIO_JSON__", json.dumps(audio))
    html = html.replace("__FRAMES_JSON__", json.dumps(frames))

    out = ROOT / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html.encode('utf-8')):,} bytes)")

if __name__ == "__main__":
    main()
