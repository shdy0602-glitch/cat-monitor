#!/usr/bin/env python3
"""Diagnose breathing rate discrepancy: record frame timestamps + raw optical flow signal.

Deploy: scp to cloud server, then:
  docker cp diagnose_breath.py cat_monitor:/app/
  docker exec cat_monitor python3 /app/diagnose_breath.py

Outputs /app/data/diagnose_YYYYMMDD_HHMMSS.json with raw data for offline analysis.
"""

import json
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from ultralytics import YOLO

CAMERA_SHOT_URL = "http://127.0.0.1:18080/shot.jpg"
CAPTURE_DURATION = 30
FRAME_SKIP = 2
CAT_CLASS = 15
CAT_ALT_CLASSES = {77, 16}

# ---------- capture with timestamps ----------

print("📷 Capturing with timestamp recording...")
frames = []
timestamps = []
start = time.time()
frame_count = 0
consecutive_fails = 0

while time.time() - start < CAPTURE_DURATION:
    t_before = time.time()
    try:
        resp = requests.get(CAMERA_SHOT_URL, timeout=1)
        t_after = time.time()
        if resp.status_code == 200 and len(resp.content) > 100:
            consecutive_fails = 0
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is not None:
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue
                frames.append(frame)
                timestamps.append(t_after)
        else:
            consecutive_fails += 1
    except Exception:
        consecutive_fails += 1
    if consecutive_fails > 5:
        break
    time.sleep(0.5)

wall_elapsed = time.time() - start
print(f"Captured {len(frames)} frames ({frame_count} raw) in {wall_elapsed:.2f}s")

if len(frames) < 5:
    print("❌ Not enough frames")
    exit(1)

# Frame spacing stats
deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)]
if deltas:
    print(f"Frame spacing: mean={np.mean(deltas):.3f}s median={np.median(deltas):.3f}s "
          f"min={np.min(deltas):.3f}s max={np.max(deltas):.3f}s std={np.std(deltas):.3f}s")

# Actual time span of signal data
actual_span = timestamps[-1] - timestamps[0]
print(f"Actual capture span (first→last frame): {actual_span:.2f}s")
print(f"Calculated fps_effective (old method): {len(frames) / CAPTURE_DURATION:.3f} Hz")
print(f"Actual fps (frames / actual_span): {len(frames) / actual_span:.3f} Hz")

# ---------- detect cat ----------

print("\n🐱 Detecting cat...")
model = YOLO("yolov8n.pt")
cat_bboxes = []
sample_indices = np.linspace(0, len(frames) - 1, min(len(frames), 8), dtype=int)
for idx in sample_indices:
    results = model(frames[idx], conf=0.08, verbose=False)
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) in (CAT_CLASS, *CAT_ALT_CLASSES) and box.conf[0] > 0.05:
                cat_bboxes.append([int(v) for v in box.xyxy[0].tolist()])

if len(cat_bboxes) < 3:
    print(f"❌ Cat not detected ({len(cat_bboxes)} bboxes)")
    exit(1)

avg_bbox = [0, 0, 0, 0]
for b in cat_bboxes:
    for i in range(4):
        avg_bbox[i] += b[i]
avg_bbox = [v // len(cat_bboxes) for v in avg_bbox]
x1, y1, x2, y2 = avg_bbox
print(f"Cat bbox: {avg_bbox}")

# ---------- optical flow with precise timestamps ----------

print("\n🌊 Computing optical flow...")
torso_y1 = y1 + int((y2 - y1) * 0.3)
torso_y2 = y1 + int((y2 - y1) * 0.7)
torso_x1 = x1 + int((x2 - x1) * 0.2)
torso_x2 = x2 - int((x2 - x1) * 0.2)

prev_gray = None
motion_signals = []
flow_timestamps = []

for i, frame in enumerate(frames):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    torso_roi = gray[torso_y1:torso_y2, torso_x1:torso_x2]
    if torso_roi.size == 0:
        continue
    if prev_gray is not None:
        prev_roi = prev_gray[torso_y1:torso_y2, torso_x1:torso_x2]
        if prev_roi.shape != torso_roi.shape:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, torso_roi, None, 0.7, 5, 21, 3, 7, 1.5, 0
        )
        mag = float(np.median(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
        motion_signals.append(mag)
        flow_timestamps.append(timestamps[i])  # timestamp of the "current" frame
    prev_gray = gray

print(f"Motion signals: {len(motion_signals)}")

# ---------- Analysis with BOTH fps methods ----------

signal_arr = np.array(motion_signals)
signal_arr = signal_arr - np.mean(signal_arr)

# Old method
fps_old = len(motion_signals) / CAPTURE_DURATION
# New method: actual span between first and last flow pair
flow_span = flow_timestamps[-1] - flow_timestamps[0] if len(flow_timestamps) > 1 else CAPTURE_DURATION
fps_actual = len(motion_signals) / flow_span if flow_span > 0 else 1.0

print(f"\n📊 Sampling rate comparison:")
print(f"  fps_old (signal_count / 30s): {fps_old:.3f} Hz")
print(f"  fps_actual (signal_count / actual_flow_span): {fps_actual:.3f} Hz")
print(f"  Ratio (actual / old): {fps_actual/fps_old:.3f}")

# Peak counting with BOTH
smoothed = uniform_filter1d(signal_arr, size=max(3, len(signal_arr)//15))

for label, fps in [("OLD", fps_old), ("ACTUAL", fps_actual)]:
    print(f"\n--- Peak counting [{label}] fps={fps:.3f} ---")
    distance = max(1, int(fps * 0.3))
    peaks, props = find_peaks(smoothed, distance=distance, prominence=np.std(smoothed) * 0.15)

    if len(peaks) < 2:
        print(f"  Only {len(peaks)} peaks")
        continue

    intervals = np.diff(peaks) / fps
    q1, q3 = np.percentile(intervals, [25, 75])
    iqr = q3 - q1
    valid = intervals[(intervals >= q1 - 1.5*iqr) & (intervals <= q3 + 1.5*iqr)]
    if len(valid) >= 2:
        peak_rate = 60.0 / np.mean(valid)
        print(f"  Peaks: {len(peaks)}, valid intervals: {len(valid)}")
        print(f"  Peak rate: {peak_rate:.1f} bpm")
        print(f"  Raw intervals (s): {[f'{v:.1f}' for v in intervals[:10]]}...")

    # FFT
    n = len(signal_arr)
    fft_a = np.abs(np.fft.rfft(signal_arr))
    fft_f = np.fft.rfftfreq(n, d=1.0/fps)
    band = (fft_f >= 0.17) & (fft_f <= 1.2)
    if np.any(band):
        bf = fft_f[band]
        ba = fft_a[band]
        top_i = np.argsort(ba)[-5:][::-1]
        print(f"  FFT peaks: {', '.join(f'{bf[i]*60:.0f}bpm(p={ba[i]:.4f})' for i in top_i[:5])}")

# ---------- Save raw data ----------

output = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "CAPTURE_DURATION": CAPTURE_DURATION,
        "FRAME_SKIP": FRAME_SKIP,
    },
    "capture": {
        "wall_elapsed_s": round(wall_elapsed, 3),
        "raw_frame_count": frame_count,
        "frames_after_skip": len(frames),
        "frame_spacing_stats": {
            "mean": float(np.mean(deltas)) if deltas else None,
            "median": float(np.median(deltas)) if deltas else None,
            "min": float(np.min(deltas)) if deltas else None,
            "max": float(np.max(deltas)) if deltas else None,
            "std": float(np.std(deltas)) if deltas else None,
        },
        "actual_frame_span_s": round(actual_span, 3),
        "fps_calculated_old": round(fps_old, 3),
        "fps_actual": round(fps_actual, 3),
        "fps_ratio": round(fps_actual / fps_old, 3) if fps_old > 0 else None,
    },
    "bbox": avg_bbox,
    "torso_roi": [torso_x1, torso_y1, torso_x2, torso_y2],
    "signal": {
        "count": len(motion_signals),
        "flow_span_s": round(flow_span, 3),
        "timestamps": [round(t - timestamps[0], 3) for t in flow_timestamps],
        "values": [round(v, 8) for v in motion_signals],
    },
}

out_path = f"/app/data/diagnose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved raw data to {out_path}")
print("Copy to local: docker cp cat_monitor:/app/data/diagnose_*.json .")
