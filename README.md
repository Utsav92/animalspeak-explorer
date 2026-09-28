# AnimalSpeak Explorer

Interactive 3D visualization of [`davidrrobinson/AnimalSpeak`](https://huggingface.co/datasets/davidrrobinson/AnimalSpeak), a Hugging Face dataset of ~894K wildlife field recordings paired with natural-language captions.

**Live page:** https://claude.ai/artifact/RCXhPk8RGpGNUWnSg6wx1b

## What's here

The AnimalSpeak dataset itself ships only text metadata (captions, source, recordist, species names) — no waveform-derived acoustics. This project goes two directions:

1. **Corpus plot** — a 3D scatter over 60 species built directly from the dataset's own caption metadata: recording counts, recordist/source diversity, caption length.
2. **Acoustic fingerprints** — for 39 species spanning birds, mammals (including six whales and dolphins), amphibians, insects and a reptile, real audio was downloaded, highpass-filtered, and run through an FFT to measure genuine numeric parameters (peak frequency, spectral centroid/bandwidth/rolloff, loudness, zero-crossing rate). Each species also gets a **frame-by-frame pitch/loudness trajectory** (short-time FFT every ~46ms) rendered as an animated 3D path that a playhead follows in sync with audio playback.

Audio sources:
- Most clips are AnimalSpeak's own linked recordings (Xeno-canto, iNaturalist, Tierstimmenarchiv).
- Six cetacean clips (Killer Whale, Common Dolphin, Long-finned Pilot Whale, Risso's Dolphin, Striped Dolphin, Bottlenose Dolphin) come from [NOAA Fisheries](https://www.fisheries.noaa.gov/national/science-data/sounds-ocean)' public-domain ocean sound library instead — AnimalSpeak's own source for these species is the Watkins Marine Mammal Sound Database, whose host (`cis.whoi.edu`) has been down for maintenance throughout this project.
- The dataset has **no fish recordings** at all — Xeno-canto and iNaturalist skew almost entirely to birds, and Watkins covers marine mammals, not fish.

## Repo layout

```
template.html            page source (Plotly 3D charts, vanilla JS) with __SPECIES_JSON__ / __AUDIO_JSON__ / __FRAMES_JSON__ placeholders
build.py                 substitutes the three data/*.json files into template.html -> index.html
index.html               the built, self-contained page (references audio/*.mp3 by relative path)
audio/                   39 short (<=6s) mp3 clips, one per acoustically-sampled species
species_data.json        per-species corpus stats (80 species)
acoustic_features.json   per-species clip-level acoustic stats (39 species)
frame_features.json      per-species frame-by-frame pitch/loudness series (39 species, ~46ms hops)
extract_acoustics.py     core download + FFT feature extraction helpers, run for the first 18 bird species
extract_extra.py         extraction for the first batch of non-bird species (amphibians/insects/reptile/coyote)
extract_mammals.py       extraction for additional mammals (AnimalSpeak's own non-Watkins links)
extract_noaa.py          extraction for the six NOAA-sourced whale/dolphin clips
frame_features.py        re-decodes audio/*.mp3 and extracts the per-frame pitch/loudness series
```

## Rebuilding from scratch

```bash
pip install pandas pyarrow requests scipy numpy
# ffmpeg must also be on PATH

# 1. Download the dataset's metadata parquet (~66MB) into this directory
curl -L -o train.parquet "https://huggingface.co/datasets/davidrrobinson/AnimalSpeak/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

# 2. Run the extraction pipeline (downloads real audio clips + computes features)
python extract_acoustics.py
python extract_extra.py
python extract_mammals.py
python extract_noaa.py
python frame_features.py

# 3. Assemble the final page
python build.py
```

`train.parquet` and the intermediate `raw_audio/` directory are gitignored (66MB+ of source audio/metadata not needed once `audio/*.mp3` and the `*.json` feature files exist).

## Notes on the acoustic numbers

These are derived by this project, not published ground truth: one representative recording per species, highpass-filtered at 300Hz to suppress handling/wind noise, analyzed with a Hanning-windowed FFT. Treat them as illustrative, not authoritative per-species averages.
