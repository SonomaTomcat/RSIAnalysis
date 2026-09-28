# 🦘 Continuous CMJ RSI Video Analysis

> **Low-cost, zero-hardware alternative to Optojump / force plates (¥20k-100k)**  
> Upload continuous CMJ video → auto count all jumps → per-jump `RSI = flight / contact` → curves + mean/best + height | Proof-of-Concept

A monocular-vision system that replaces contact mats and force plates for schools, training centers and gyms. Built on off-the-shelf small models (YOLOv8n + YOLOv8n-Pose, <7M params), no training required.

---

## Features

- **Fully automatic**: Upload 5–10 continuous CMJs → jump count, per-jump RSI, contact/flight, height, fatigue decay, annotated video & CSV
- **Robust small-target**: Two-stage `detect → crop → pose` with 15% padding + EMA smoothing; validated on 30% scale (100×192 person in 1280×720)
- **Physiologically grounded contact**: Ankle collapse + shoe deformation proxy (`ankle y near ground & velocity≈0`), excludes toe-touch; takeoff at ankle no-longer-rising + shoe recovery (before toe-off)
- **Max FPS**: No downsampling by default; optional downsampling range **60fps – native** (native ≤60 keeps full rate) for contact precision (~150ms needs ≥60fps)
- **Height prediction**: Manual height + trochanter height input → physics `h = g·flight²/8` + pixel scaling, **prediction only, not accurate** (flagged in UI)
- **Clean architecture**: Inheritance, decoupling, SRP, Dependency Inversion

---

## Architecture

```
core/interfaces.py          # Abstractions: IDetector, IPoseEstimator, IFilter, IPeakDetector, IContactDetector, IRSICalculator + Value Objects BBox/Keypoints/Jump
core/config.py              # Dataclass configs (DetectorConfig/PoseConfig/RSIConfig)
core/factory.py             # Factory + DI (decouples creation)
core/detection/yolo_detector.py  # YOLODetector : IDetector (+ BBoxSmoother SRP)
core/pose/yolo_pose.py           # YOLOPoseEstimator : IPoseEstimator (limbs only, no face)
core/signal/filter.py            # ButterworthFilter : IFilter
core/signal/peak.py              # PeakDetector : IPeakDetector
core/rsi/contact.py              # AnkleCollapseContactDetector : IContactDetector (ground vs collapse masks)
core/rsi/calculator.py           # RSICalculator : IRSICalculator (orchestrates filter+peak+contact)
core/rsi/helpers.py              # hip_center_y / ankle_center_y / estimate_trochanter_height
core/video/processor.py          # VideoReader / VideoWriter (SRP: IO only)
app.py                      # Thin orchestration (UI only, <150 lines)
```

## Quick Start

```bash
python3 -m pip install -r requirements.txt  # streamlit, ultralytics, opencv, scipy, torch
python3 -m streamlit run app.py --server.port 8501
# Browser: http://localhost:8501
```

**Shooting (side view recommended, not mandatory)**: Tripod 1.5–3m side view recommended (higher accuracy, front view also works), single person centered 15%+, 720p 30fps+, 5–10 continuous CMJs, flat ground, good light.

---

## Usage

1. Sidebar: adjust `Detection Confidence / NMS / Bandpass / Prominence / Jump Interval` and input `Height cm` + `Trochanter Height cm (manual)` (default 180cm → 95.4cm via 0.53 ratio)
2. Upload video (mp4/avi/mov) or view synthetic 5-jump 60fps demo
3. Click **Start Analysis** → progress + preview → results:
   - Metrics: jump count, mean/best RSI, mean contact/flight, mean height, fatigue
   - Table: per-jump `contact / flight / RSI / height`
   - Tabs: RSI curve + height curve, waveform with peaks/troughs, annotated video
   - Downloads: annotated video, frame CSV, jump CSV

---

## Verification

```bash
python3 -m py_compile app.py core/rsi/calculator.py && echo ok
python3 -c "from core.factory import create_rsi_calculator; from core.config import RSIConfig; import numpy as np; c=create_rsi_calculator(RSIConfig()); t=np.linspace(0,5,300); hip=[0.6-0.12*np.sin(2*np.pi*1.25*x) for x in t]; ankle=[0.85+0.08*np.sin(2*np.pi*1.25*x+np.pi) for x in t]; print(c.calculate(hip,60,ankle_signal=ankle, user_height_cm=180, trochanter_cm=95.4)['jump_count'])"
# expect 5

# Small-target robustness already validated: 30% scale still detected BBox(505,298,604,490)
```

- Synthetic 60fps 5 jumps → 5 jumps, mean RSI ~2.27, contact ~0.22s flight ~0.50s
- Static small-target → 0 jumps, no false positive
- Real translation video (40px, 1.25Hz, 30fps) → 3 jumps, mean RSI 1.87

---

## VC Pitch Summary

**Gap**: Force plates / Optojump ¥20k–100k, inaccessible for schools/gyms; existing apps require manual frame picking.
**Tech**: Off-the-shelf YOLOv8n (3.2M) + YOLOv8n-Pose (6.5M) + 1D signal processing (no training, <7M, edge-deployable).
**Modification**: Two-stage small-target boost, ankle-collapse + shoe-deform contact (excludes toe, takeoff before toe-off when ankle stops rising + shoe recovers), full native FPS maximized (60–native selectable), manual trochanter height for height prediction (flagged prediction only).
**PoC**: Streamlit upload → per-jump RSI curve + annotated video + CSV, validated on synthetic + real small-target; height with disclaimer.

---

## Project Structure

```
src/
├── app.py               # Thin UI orchestration
├── core/                # Elegant core
│   ├── interfaces.py    # ABCs + Value Objects
│   ├── config.py
│   ├── factory.py
│   ├── detection/yolo_detector.py
│   ├── pose/yolo_pose.py
│   ├── signal/{filter,peak}.py
│   ├── rsi/{calculator,contact,helpers}.py
│   └── video/processor.py
├── assets/              # demo videos
└── requirements.txt
```
