import json
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, ".")
import extract_acoustics as ea

ROOT = Path(__file__).parent

TARGETS = [
    ("Killer Whale",            "Mammal",    "Orcinus orca"),
    ("Humpback Whale",          "Mammal",    None),
    ("Common Dolphin",          "Mammal",    "Delphinus delphis"),
    ("Coyote",                  "Mammal",    None),
    ("Boreal Chorus Frog",      "Amphibian", None),
    ("Gray Treefrog",           "Amphibian", None),
    ("American Bullfrog",       "Amphibian", None),
    ("Common True Katydid",     "Insect",    None),
    ("Spotted Wattle Cicada",   "Insect",    None),
    ("European Field Cricket",  "Insect",    None),
    ("Tokay Gecko",             "Reptile",   None),
]

def main():
    df = pd.read_parquet(ROOT / "train.parquet")
    df = df.dropna(subset=["species_common", "url"])

    results = []
    for sp, klass, sci_override in TARGETS:
        rows = df[df["species_common"] == sp]
        if rows.empty:
            print(f"[{sp}] no rows, skipping")
            continue
        scientific = sci_override
        if not scientific:
            m = rows["species_scientific"].dropna().mode()
            scientific = m.iloc[0] if len(m) else ""

        cand = rows.sample(n=min(5, len(rows)), random_state=7)
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

    with open(ROOT / "acoustic_features_extra.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {len(results)}/{len(TARGETS)} species succeeded.")

if __name__ == "__main__":
    main()
