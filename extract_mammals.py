import json
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc
import sys
sys.path.insert(0, ".")
import extract_acoustics as ea

ROOT = Path(__file__).parent

# (species_common, class, avoid_source)
TARGETS = [
    ("Humpback Whale",           "Mammal", "Watkins"),
    ("Sperm Whale",              "Mammal", "Watkins"),
    ("California Sea Lion",      "Mammal", None),
    ("Eastern Gray Squirrel",    "Mammal", None),
    ("Red Fox",                  "Mammal", None),
    ("Mexican Free-tailed Bat",  "Mammal", None),
    ("Western Roe Deer",         "Mammal", None),
    ("Arctic Wolf",              "Mammal", None),
]

def main():
    names = [t[0] for t in TARGETS]
    table = pq.read_table(
        ROOT / "train.parquet",
        columns=["species_common", "species_scientific", "url", "source"],
        filters=[("species_common", "in", names)],
    )
    df = table.to_pandas(types_mapper=None)
    df = df.dropna(subset=["species_common", "url"])

    results = []
    for sp, klass, avoid in TARGETS:
        rows = df[df["species_common"] == sp]
        if avoid:
            rows = rows[rows["source"] != avoid]
        if rows.empty:
            print(f"[{sp}] no usable rows, skipping")
            continue
        m = rows["species_scientific"].dropna().mode()
        scientific = m.iloc[0] if len(m) else ""

        cand = rows.sample(n=min(6, len(rows)), random_state=11)
        ok = False
        for _, row in cand.iterrows():
            url = row["url"]
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            raw_path = ea.RAW_DIR / (ea.slug(sp) + "_raw")
            print(f"[{sp}] trying {url[:90]}")
            if not ea.download(url, raw_path):
                print("  failed / too large")
                continue
            wav_tmp = ea.RAW_DIR / (ea.slug(sp) + "_tmp.wav")
            if not ea.ffmpeg_to_wav(raw_path, wav_tmp):
                print("  ffmpeg decode failed")
                continue
            feats = ea.acoustic_features(wav_tmp)
            if feats is None:
                print("  feature extraction failed (too short/silent)")
                continue
            mp3_path = ea.CLIP_DIR / (ea.slug(sp) + ".mp3")
            ea.ffmpeg_to_mp3(raw_path, mp3_path)
            feats["species"] = sp
            feats["scientific"] = scientific or ""
            feats["clip"] = mp3_path.name if mp3_path.exists() else None
            feats["source_used"] = row["source"]
            feats["class"] = klass
            results.append(feats)
            print("  OK:", feats)
            ok = True
            break
        if not ok:
            print(f"[{sp}] ALL CANDIDATES FAILED")

    with open(ROOT / "acoustic_features_mammals.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {len(results)}/{len(TARGETS)} species succeeded.")

if __name__ == "__main__":
    main()
