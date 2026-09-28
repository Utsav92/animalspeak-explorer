import json, subprocess
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / "audio"
TMP = ROOT / "raw_audio" / "_frame_tmp.wav"

WINDOW = 2048
HOP = 1024
SR = 22050
SILENCE_RMS = 0.012

def to_wav(mp3_path, wav_path, sr=SR):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(mp3_path), "-ac", "1", "-ar", str(sr), "-f", "wav", str(wav_path)]
    p = subprocess.run(cmd, capture_output=True)
    return p.returncode == 0

def frame_series(wav_path):
    sr, x = wavfile.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    maxabs = np.max(np.abs(x)) or 1.0
    x = x / maxabs

    nyq = sr / 2.0
    b, a = butter(6, 300.0 / nyq, btype="highpass")
    x = filtfilt(b, a, x)

    n = len(x)
    frames = []
    window = np.hanning(WINDOW)
    last_freq = 0.0
    last_centroid = 0.0

    start = 0
    while start + WINDOW <= n:
        seg = x[start:start + WINDOW] * window
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(WINDOW, d=1.0 / sr)
        rms = float(np.sqrt(np.mean(seg ** 2)))

        mask = freqs > 300
        f = freqs[mask]
        m = spec[mask]

        if rms > SILENCE_RMS and m.sum() > 0:
            peak = float(f[np.argmax(m)])
            centroid = float(np.sum(f * m) / np.sum(m))
            last_freq, last_centroid = peak, centroid
        else:
            peak, centroid = last_freq, last_centroid

        frames.append({
            "t": round(start / sr, 3),
            "freq": round(peak, 1),
            "centroid": round(centroid, 1),
            "rms": round(rms, 4),
        })
        start += HOP

    return frames

def main():
    out = {}
    clips = sorted(AUDIO_DIR.glob("*.mp3"))
    for clip in clips:
        print("processing", clip.name)
        if not to_wav(clip, TMP):
            print("  ffmpeg failed"); continue
        frames = frame_series(TMP)
        if len(frames) < 4:
            print("  too few frames"); continue
        out[clip.stem] = frames
        print(f"  {len(frames)} frames, duration {frames[-1]['t']:.2f}s")

    with open(ROOT / "frame_features.json", "w") as f:
        json.dump(out, f)
    print(f"\nDone. {len(out)} clips, {sum(len(v) for v in out.values())} total frames.")

if __name__ == "__main__":
    main()
