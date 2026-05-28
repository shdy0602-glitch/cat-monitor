#!/usr/bin/env python3
"""Cat breath rate monitor using IP Webcam + YOLOv8 + optical flow."""

import csv
import json
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import requests
from scipy.signal import find_peaks
from ultralytics import YOLO

# ============ CONFIG ============
# Load .env file if present
def _load_dotenv():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip("\"'"))

_load_dotenv()

CAMERA_URL = os.environ.get("CAMERA_URL", "")
CAMERA_SHOT_URL = os.environ.get("CAMERA_SHOT_URL", "")
if not CAMERA_SHOT_URL:
    # Derive snapshot URL from video URL
    for suffix in ("/video", "/videofeed"):
        if CAMERA_URL.endswith(suffix):
            CAMERA_SHOT_URL = CAMERA_URL[: -len(suffix)] + "/shot.jpg"
            break
    if not CAMERA_SHOT_URL:
        CAMERA_SHOT_URL = CAMERA_URL.rstrip("/") + "/shot.jpg"
CAPTURE_DURATION = 30          # seconds of video per detection cycle
CYCLE_INTERVAL = 300           # seconds between detection cycles (5 min)
FRAME_SKIP = 2                 # process every Nth frame to reduce load
BREATH_RATE_MAX = 35           # breaths/min threshold for alert
ALERT_COOLDOWN = 1800          # 30 minutes between same-type alerts
FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
FEISHU_CHAT_ID = os.environ.get("FEISHU_CHAT_ID", "")
# cached tenant access token (valid 2 hours)
_token_cache = {"token": None, "expires_at": 0}

LOG_DIR = Path(__file__).parent / "data"
LOG_FILE = LOG_DIR / "log.csv"
STATE_FILE = LOG_DIR / "state.json"

CAT_CLASS = 15  # COCO class ID for cat
CAT_ALT_CLASSES = {77, 16}  # teddy bear + dog — curled-up cats often misdetected


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _get_feishu_token():
    """Get or refresh Feishu tenant access token."""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    try:
        r = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
            timeout=10,
        )
        if r.status_code == 200 and r.json().get("code") == 0:
            _token_cache["token"] = r.json()["tenant_access_token"]
            _token_cache["expires_at"] = now + r.json().get("expire", 7200)
            return _token_cache["token"]
    except Exception as e:
        print(f"  ❌ Feishu auth error: {e}")
    return None


def send_alert(title, content, state, alert_key):
    """Send Feishu bot alert with cooldown."""
    now = time.time()
    last = state.get(alert_key, 0)
    if now - last < ALERT_COOLDOWN:
        return False
    token = _get_feishu_token()
    if not token:
        print(f"  ❌ Cannot get Feishu token, alert not sent: {title}")
        return False
    try:
        r = requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "receive_id": FEISHU_CHAT_ID,
                "msg_type": "text",
                "content": json.dumps({"text": f"{title}\n\n{content}"}),
            },
            timeout=10,
        )
        if r.status_code == 200 and r.json().get("code") == 0:
            state[alert_key] = now
            save_state(state)
            print(f"  ✅ Feishu alert sent: {title}")
            return True
        else:
            print(f"  ❌ Feishu failed: {r.json()}")
    except Exception as e:
        print(f"  ❌ Feishu error: {e}")
    return False


def detect_breathing_rate(frames, bbox, frame_timestamps=None):
    """Estimate breathing rate from optical flow using FFT-based frequency analysis."""
    x1, y1, x2, y2 = bbox
    # Focus on torso: middle 30-70% vertically, narrow horizontally
    torso_y1 = y1 + int((y2 - y1) * 0.3)
    torso_y2 = y1 + int((y2 - y1) * 0.7)
    torso_x1 = x1 + int((x2 - x1) * 0.2)
    torso_x2 = x2 - int((x2 - x1) * 0.2)

    prev_gray = None
    motion_signals = []

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        torso_roi = gray[torso_y1:torso_y2, torso_x1:torso_x2]

        if torso_roi.size == 0:
            continue

        if prev_gray is not None:
            prev_roi = prev_gray[torso_y1:torso_y2, torso_x1:torso_x2]
            if prev_roi.shape != torso_roi.shape:
                continue
            # Finer optical flow for subtle breathing detection
            flow = cv2.calcOpticalFlowFarneback(
                prev_roi, torso_roi, None, 0.7, 5, 21, 3, 7, 1.5, 0
            )
            # Use median for outlier resilience
            mag = np.median(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            motion_signals.append(mag)

        prev_gray = gray

    if len(motion_signals) < 20:
        print(f"  ❌ Breath: too few motion signals ({len(motion_signals)})")
        return None

    if frame_timestamps is not None and len(frame_timestamps) > 1:
        actual_span = frame_timestamps[-1] - frame_timestamps[0]
        fps_effective = len(motion_signals) / actual_span if actual_span > 0 else 1.0
    else:
        fps_effective = len(motion_signals) / CAPTURE_DURATION
    print(f"  ⏱️  fps_effective={fps_effective:.3f} Hz ({len(motion_signals)} signals)")
    signal_arr = np.array(motion_signals)
    signal_arr = signal_arr - np.mean(signal_arr)

    # Time-domain peak counting for breathing rate
    # Apply light smoothing for noise reduction
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(signal_arr, size=max(3, len(signal_arr)//15))

    # Find peaks with breathing-appropriate spacing
    distance = max(1, int(fps_effective * 0.3))  # at least 0.3s between breaths
    peaks, props = find_peaks(smoothed, distance=distance, prominence=np.std(smoothed) * 0.08)

    if len(peaks) < 2:
        print(f"  ❌ Breath: only {len(peaks)} peaks found (need ≥2)")
        return None

    # Calculate intervals between peaks
    intervals = np.diff(peaks) / fps_effective
    # Filter outliers: keep intervals within 1.5x IQR
    q1, q3 = np.percentile(intervals, [25, 75])
    iqr = q3 - q1
    valid = intervals[(intervals >= q1 - 1.5*iqr) & (intervals <= q3 + 1.5*iqr)]

    if len(valid) < 2:
        print(f"  ❌ Breath: {len(peaks)} peaks but only {len(valid)} valid intervals")
        return None

    mean_interval = np.mean(valid)
    peak_rate = 60.0 / mean_interval
    breath_rate = peak_rate  # default to peak-counting

    # FFT cross-check: use strongest peak among top candidates
    fft_n = len(signal_arr)
    fft_a = np.abs(np.fft.rfft(signal_arr))
    fft_f = np.fft.rfftfreq(fft_n, d=1.0/fps_effective)
    band = (fft_f >= 0.17) & (fft_f <= 1.2)
    if np.any(band):
        bf = fft_f[band]
        ba = fft_a[band]
        top_i = np.argsort(ba)[-5:][::-1]  # top 5 peaks by power
        median_pwr = np.median(ba)
        strong = [(bf[i]*60, ba[i]) for i in top_i if ba[i] > median_pwr * 1.3]
        if strong:
            strong_sorted = sorted(strong, key=lambda x: x[1], reverse=True)
            # When peak-counting has good data (>=4 valid intervals), use it as anchor
            if len(valid) >= 4:
                # Find FFT peak closest to peak_rate for corroboration
                closest_fft = min(strong_sorted, key=lambda x: abs(x[0] - peak_rate))
                if abs(closest_fft[0] - peak_rate) < 8:
                    # FFT confirms — weighted average toward peak-counting
                    breath_rate = peak_rate * 0.7 + closest_fft[0] * 0.3
                else:
                    # FFT disagrees but peak data is solid — trust peaks
                    breath_rate = peak_rate
            else:
                # Sparse peaks: rely on FFT with harmonic check
                fft_rate = strong_sorted[0][0]
                for rate, power in strong_sorted[1:]:
                    if abs(fft_rate - 2 * rate) < 6 and power > strong_sorted[0][1] * 0.35:
                        fft_rate = rate
                        break
                if abs(fft_rate - peak_rate) < 10:
                    breath_rate = (fft_rate + peak_rate) / 2
                else:
                    breath_rate = fft_rate
        else:
            breath_rate = peak_rate

        peak_info = ", ".join(f"{bf[i]*60:.0f}bpm(p={ba[i]:.4f})" for i in top_i[:3])
        print(f"  📊 Peaks: {len(peaks)} found, {len(valid)} valid, FFT: {peak_info}  →  {breath_rate:.1f} bpm")

    # Sanity check
    if breath_rate < 10 or breath_rate > 80:
        return None

    return round(breath_rate, 1)


def is_cat_stationary(bboxes_history):
    """Check if cat has been relatively stationary (YOLO jitter resistant)."""
    if len(bboxes_history) < 3:
        return False
    # Apply median smoothing to filter single-frame YOLO jitter
    smoothed = []
    for i in range(len(bboxes_history)):
        start_i = max(0, i - 1)
        end_i = min(len(bboxes_history), i + 2)
        window = bboxes_history[start_i:end_i]
        smoothed.append([int(np.median([b[j] for b in window])) for j in range(4)])
    centers_x = [(b[0] + b[2]) / 2 for b in smoothed]
    centers_y = [(b[1] + b[3]) / 2 for b in smoothed]
    x_range = np.percentile(centers_x, 90) - np.percentile(centers_x, 10)
    y_range = np.percentile(centers_y, 90) - np.percentile(centers_y, 10)
    avg_width = np.mean([b[2] - b[0] for b in smoothed])
    rel_movement = max(x_range, y_range) / avg_width if avg_width > 0 else 999
    return rel_movement < 0.35


def run_detection(model):
    """Run one detection cycle: capture 30s → detect cat → estimate breath rate."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*50}")
    print(f"[{timestamp}] Starting detection cycle...")

    # Pre-check: verify camera is reachable before capture
    try:
        test = requests.get(CAMERA_SHOT_URL, timeout=3)
        if test.status_code != 200 or len(test.content) < 100:
            print("  ❌ Camera unreachable")
            return {"timestamp": timestamp, "cat_detected": False, "breath_rate": None, "reason": "stream_failed"}
    except Exception:
        print("  ❌ Camera unreachable")
        return {"timestamp": timestamp, "cat_detected": False, "breath_rate": None, "reason": "stream_failed"}

    # Capture frames via HTTP snapshot polling
    frames = []
    frame_timestamps = []
    start = time.time()
    frame_count = 0
    consecutive_fails = 0

    print(f"  📷 Capturing snapshots...")

    while time.time() - start < CAPTURE_DURATION:
        try:
            resp = requests.get(CAMERA_SHOT_URL, timeout=1)
            if resp.status_code == 200 and len(resp.content) > 100:
                consecutive_fails = 0
                img_array = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_count += 1
                    if frame_count % FRAME_SKIP != 0:
                        continue
                    frames.append(frame)
                    frame_timestamps.append(time.time())
            else:
                consecutive_fails += 1
        except Exception:
            consecutive_fails += 1

        # Abort early if stream is down
        if consecutive_fails > 5:
            break

        time.sleep(0.5)

    if not frames:
        print("  ❌ No frames captured")
        return {"timestamp": timestamp, "cat_detected": False, "breath_rate": None, "reason": "no_frames"}

    print(f"  📷 Captured {len(frames)} frames in {time.time() - start:.1f}s")

    # Run full YOLO detection on key frames
    cat_bboxes = []
    sample_indices = np.linspace(0, len(frames) - 1, min(len(frames), 8), dtype=int)
    for idx in sample_indices:
        results = model(frames[idx], conf=0.08, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in (CAT_CLASS, *CAT_ALT_CLASSES) and box.conf[0] > 0.05:
                    cat_bboxes.append([int(v) for v in box.xyxy[0].tolist()])

    if len(cat_bboxes) < 3:
        print("  🐱 Cat not detected (or detected too briefly)")
        return {"timestamp": timestamp, "cat_detected": False, "breath_rate": None, "reason": "cat_not_found"}

    print(f"  🐱 Cat detected in {len(cat_bboxes)} frames")

    # Average the bounding box
    avg_bbox = [0, 0, 0, 0]
    for b in cat_bboxes:
        for i in range(4):
            avg_bbox[i] += b[i]
    avg_bbox = [v // len(cat_bboxes) for v in avg_bbox]

    if not is_cat_stationary(cat_bboxes):
        # Diagnostic: show actual drift for debugging
        cx = [(b[0]+b[2])/2 for b in cat_bboxes]
        cy = [(b[1]+b[3])/2 for b in cat_bboxes]
        xr = np.percentile(cx, 90) - np.percentile(cx, 10)
        yr = np.percentile(cy, 90) - np.percentile(cy, 10)
        aw = np.mean([b[2]-b[0] for b in cat_bboxes])
        rel = max(xr, yr)/aw if aw > 0 else 999
        print(f"  🏃 Cat moving (drift {rel:.1%}, {xr:.0f}x{yr:.0f}px, {len(cat_bboxes)} bboxes)")
        return {"timestamp": timestamp, "cat_detected": True, "breath_rate": None, "reason": "cat_moving"}

    print("  😴 Cat is stationary — analyzing breath rate...")
    breath_rate = detect_breathing_rate(frames, avg_bbox, frame_timestamps)

    if breath_rate is None:
        print("  ❓ Could not determine breath rate")
        return {"timestamp": timestamp, "cat_detected": True, "breath_rate": None, "reason": "signal_unclear"}

    print(f"  🌬️  Breath rate: {breath_rate} breaths/min")
    return {"timestamp": timestamp, "cat_detected": True, "breath_rate": breath_rate, "reason": None}


def write_log(result):
    """Append result to CSV log."""
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "cat_detected", "breath_rate", "reason"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def check_hourly(state):
    """Send hourly summary: detection count and breath rates, or alert if none."""
    if not LOG_FILE.exists():
        return
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    detections = []  # list of (timestamp, breath_rate)
    with open(LOG_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                if t > one_hour_ago and row.get("cat_detected") == "True":
                    rate = row.get("breath_rate", "")
                    detections.append((row["timestamp"], rate))
            except (ValueError, KeyError):
                continue

    if not detections:
        print("  ⚠️  Hourly: 0 detections in the past hour")
        send_alert("⚠️ 猫咪监护 — 过去1小时无检测",
                   f"过去60分钟内没有成功检测到猫咪。请检查摄像头是否正常。\n时间：{now.strftime('%Y-%m-%d %H:%M:%S')}",
                   state, "last_hourly_alert")
    else:
        lines = [f"• {ts} — {r} 次/分钟" if r else f"• {ts} — 检测到但未获得呼吸率"
                 for ts, r in detections]
        summary = "\n".join(lines)
        print(f"  📊 Hourly summary: {len(detections)} detections")
        send_alert("📊 猫咪监护 — 过去1小时汇总",
                   f"成功检测 {len(detections)} 次\n\n{summary}",
                   state, "last_hourly_alert")


def main():
    print("🐱 Cat Breath Rate Monitor v1.0")
    print(f"   Camera: {CAMERA_URL}")
    print(f"   Cycle interval: {CYCLE_INTERVAL}s ({CYCLE_INTERVAL // 60} min)")
    print(f"   Alert threshold: {BREATH_RATE_MAX} breaths/min")
    print(f"   Feishu alerts: {'✅ configured (bot → chat)' if FEISHU_APP_ID else '⚠️  not set'}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 model
    print("\n📦 Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("   ✅ Model loaded")

    state = load_state()
    last_hourly_check = state.get("last_hourly_check", 0)
    signal.signal(signal.SIGINT, lambda sig, frame: graceful_exit())
    signal.signal(signal.SIGTERM, lambda sig, frame: graceful_exit())

    print("\n🔄 Monitoring started. Press Ctrl+C to stop.\n")

    while True:
        try:
            result = run_detection(model)
            write_log(result)

            # Alert if breathing rate exceeds threshold
            if result.get("breath_rate") and result["breath_rate"] > BREATH_RATE_MAX:
                print(f"  🚨 HIGH BREATH RATE: {result['breath_rate']}/min!")
                send_alert("猫咪呼吸急促警报",
                           f"检测到猫咪呼吸频率异常！\n"
                           f"呼吸频率：{result['breath_rate']} 次/分钟\n"
                           f"阈值：{BREATH_RATE_MAX} 次/分钟\n"
                           f"时间：{result['timestamp']}",
                           state, "last_breath_alert")

            # Hourly check
            now = time.time()
            if now - last_hourly_check > 3600:
                check_hourly(state)
                last_hourly_check = now
                state["last_hourly_check"] = last_hourly_check
                save_state(state)

        except Exception as e:
            print(f"  ❌ Detection error: {e}")
            write_log({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "cat_detected": False, "breath_rate": None, "reason": f"error: {e}"})

        # Wait until next cycle
        next_cycle = datetime.now() + timedelta(seconds=CYCLE_INTERVAL)
        print(f"\n  ⏰ Next detection: {next_cycle.strftime('%H:%M:%S')}")
        time.sleep(CYCLE_INTERVAL)


def graceful_exit():
    print("\n🛑 Monitor stopped.")
    sys.exit(0)


if __name__ == "__main__":
    main()
