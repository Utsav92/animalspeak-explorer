import json
from pathlib import Path
import sys
sys.path.insert(0, ".")
import extract_acoustics as ea

ROOT = Path(__file__).parent

# species name kept consistent with AnimalSpeak's own naming where a corpus
# row already exists; scientific names filled manually since Watkins rows
# (the only AnimalSpeak source for these) carry no species_scientific value.
TARGETS = [
    ("Killer Whale", "Orcinus orca",
     "https://www.fisheries.noaa.gov/s3/2023-04/Oror-Multisound-AWI-Van-Opzeeland-01-killer-clip.mp3"),
    ("Common Dolphin", "Delphinus delphis",
     "https://www.fisheries.noaa.gov/s3/2023-04/Dede-whistles-NOAA-PAGroup-01-short-beaked-common-dolphin-clip.mp3"),
    ("Long-finned Pilot Whale", "Globicephala melas",
     "https://www.fisheries.noaa.gov/s3/2023-04/Glsp-Multisound-NOAA-PAGroup-01-pilot-whale-clip.mp3"),
    ("Risso's Dolphin", "Grampus griseus",
     "https://www.fisheries.noaa.gov/s3/2023-04/Grgr-clicks-NOAA-PAGroup-08-rissos-dolphin-clip.mp3"),
    ("Striped Dolphin", "Stenella coeruleoalba",
     "https://www.fisheries.noaa.gov/s3/2023-04/Stco-Multisound-NOAA-PAGroup-01-striped-dolphin-clip.mp3"),
    ("Bottlenose Dolphin", "Tursiops truncatus",
     "https://www.fisheries.noaa.gov/s3/2023-04/Tutr-multisound-NOAA-PAGroup-03-bottlenose-dolphin-clip.mp3"),
]

SOURCE_LABEL = "NOAA Fisheries (public domain — AnimalSpeak's own Watkins link was unreachable)"

def main():
    results = []
    for sp, scientific, url in TARGETS:
        raw_path = ea.RAW_DIR / (ea.slug(sp) + "_raw")
        print(f"[{sp}] downloading {url}")
        if not ea.download(url, raw_path):
            print("  failed / too large")
            continue
        wav_tmp = ea.RAW_DIR / (ea.slug(sp) + "_tmp.wav")
        if not ea.ffmpeg_to_wav(raw_path, wav_tmp):
            print("  ffmpeg decode failed")
            continue
        feats = ea.acoustic_features(wav_tmp)
        if feats is None:
            print("  feature extraction failed")
            continue
        mp3_path = ea.CLIP_DIR / (ea.slug(sp) + ".mp3")
        ea.ffmpeg_to_mp3(raw_path, mp3_path)
        feats["species"] = sp
        feats["scientific"] = scientific
        feats["clip"] = mp3_path.name if mp3_path.exists() else None
        feats["source_used"] = SOURCE_LABEL
        feats["class"] = "Mammal"
        results.append(feats)
        print("  OK:", feats)

    with open(ROOT / "acoustic_features_noaa.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {len(results)}/{len(TARGETS)} species succeeded.")

if __name__ == "__main__":
    main()
