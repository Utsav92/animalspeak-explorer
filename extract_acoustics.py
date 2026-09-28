import json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw_audio"
CLIP_DIR = ROOT / "audio"
RAW_DIR.mkdir(exist_ok=True)
CLIP_DIR.mkdir(exist_ok=True)

TOP_N = 18            # species to attempt
CANDIDATES_PER_SP = 4  # url attempts per species
MAX_BYTES = 18_000_000
DL_TIMEOUT = 25
CLIP_SECONDS = 6

def slug(s):
    return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")

def download(url, dest):
    try:
        r = requests.get(url, timeout=DL_TIMEOUT, stream=True,
                          headers={"User-Agent": "Mozilla/5.0 (research; AnimalSpeak explorer)"})
        if r.status_code != 200:
            return False
        total = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                total += len(chunk)
                if total > MAX_BYTES:
                    return False
                f.write(chunk)
        return total > 2000
    except Exception as e:
        print("  download error:", e)
        return False

def ffmpeg_to_wav(src, dst_wav, seconds=CLIP_SECONDS, sr=22050):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", str(seconds),
           "-ac", "1", "-ar", str(sr), "-f", "wav", str(dst_wav)]
    p = subprocess.run(cmd, capture_output=True)
    return p.returncode == 0 and dst_wav.exists() and dst_wav.stat().st_size > 1000

def ffmpeg_to_mp3(src, dst_mp3, seconds=CLIP_SECONDS):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", str(seconds),
           "-ac", "1", "-ar", "22050", "-b:a", "48k", str(dst_mp3)]
    p = subprocess.run(cmd, capture_output=True)
    return p.returncode == 0 and dst_mp3.exists() and dst_mp3.stat().st_size > 500

def acoustic_features(wav_path):
    sr, x = wavfile.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if np.issubdtype(x.dtype, np.floating) is False:
        pass
    maxabs = np.max(np.abs(x)) or 1.0
    x = x / maxabs

    duration_s = len(x) / sr

    # trim leading/trailing near-silence for cleaner spectral stats
    thresh = 0.02
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) > sr * 0.2:
        x_active = x[idx[0]:idx[-1]]
    else:
        x_active = x

    # highpass at 150Hz to remove wind/handling rumble common in phone
    # recordings (iNaturalist m4a) before any level or spectral stats
    nyq = sr / 2.0
    b, a = butter(6, 300.0 / nyq, btype="highpass")
    x_active = filtfilt(b, a, x_active)

    rms = float(np.sqrt(np.mean(x_active ** 2)))

    signs = np.sign(x_active)
    signs[signs == 0] = 1
    zcr = float(np.mean(np.abs(np.diff(signs)) > 0))

    n = len(x_active)
    if n < 256:
        return None
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(x_active * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # ignore residual sub-350Hz energy for peak/centroid
    mask = freqs > 350
    f = freqs[mask]
    m = spec[mask]
    if m.sum() <= 0:
        return None

    peak_freq = float(f[np.argmax(m)])
    centroid = float(np.sum(f * m) / np.sum(m))
    bandwidth = float(np.sqrt(np.sum(m * (f - centroid) ** 2) / np.sum(m)))

    cumulative = np.cumsum(m)
    rolloff_idx = np.searchsorted(cumulative, 0.85 * cumulative[-1])
    rolloff = float(f[min(rolloff_idx, len(f) - 1)])

    return dict(
        duration_s=round(duration_s, 2),
        peak_freq_hz=round(peak_freq, 1),
        spectral_centroid_hz=round(centroid, 1),
        spectral_bandwidth_hz=round(bandwidth, 1),
        spectral_rolloff_hz=round(rolloff, 1),
        rms_loudness=round(rms, 4),
        zero_crossing_rate=round(zcr, 4),
    )

def main():
    df = pd.read_parquet(ROOT / "train.parquet")
    excl = {"Soundscape", "Identity unknown", "Human", "Unknown", "Mystery"}
    df = df[~df["species_common"].isin(excl)].dropna(subset=["species_common", "url"])

    top_species = (df["species_common"].value_counts().head(TOP_N).index.tolist())

    results = []
    for sp in top_species:
        rows = df[df["species_common"] == sp]
        # prefer rows with a scientific name and shorter captions (often shorter clips)
        cand = rows.sample(n=min(CANDIDATES_PER_SP, len(rows)), random_state=42)
        scientific = rows["species_scientific"].dropna().mode()
        scientific = scientific.iloc[0] if len(scientific) else ""

        ok = False
        for _, row in cand.iterrows():
            url = row["url"]
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            ext = ".mp3" if url.lower().split("?")[0].endswith(".mp3") else ".wav"
            raw_path = RAW_DIR / (slug(sp) + ext)
            print(f"[{sp}] trying {url[:80]}")
            if not download(url, raw_path):
                print("  failed / too large")
                continue
            wav_tmp = RAW_DIR / (slug(sp) + "_tmp.wav")
            if not ffmpeg_to_wav(raw_path, wav_tmp):
                print("  ffmpeg decode failed")
                continue
            feats = acoustic_features(wav_tmp)
            if feats is None:
                print("  feature extraction failed (too short/silent)")
                continue
            mp3_path = CLIP_DIR / (slug(sp) + ".mp3")
            ffmpeg_to_mp3(raw_path, mp3_path)
            feats["species"] = sp
            feats["scientific"] = scientific
            feats["clip"] = mp3_path.name if mp3_path.exists() else None
            feats["source_used"] = row["source"]
            results.append(feats)
            print("  OK:", feats)
            ok = True
            break
        if not ok:
            print(f"[{sp}] ALL CANDIDATES FAILED")

    with open(ROOT / "acoustic_features.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {len(results)}/{len(top_species)} species succeeded.")

if __name__ == "__main__":
    main()
