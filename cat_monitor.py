#!/usr/bin/env python3
"""Cat breath rate monitor — grid-based optical flow detection, no YOLO."""

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
from collections import Counter
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ============ CONFIG ============
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
    for suffix in ("/video", "/videofeed"):
        if CAMERA_URL.endswith(suffix):
            CAMERA_SHOT_URL = CAMERA_URL[: -len(suffix)] + "/shot.jpg"
            break
    if not CAMERA_SHOT_URL:
        CAMERA_SHOT_URL = CAMERA_URL.rstrip("/") + "/shot.jpg"
CAPTURE_DURATION = 30
CYCLE_INTERVAL = 300
FRAME_SKIP = 2
BREATH_RATE_MAX = 35
ALERT_COOLDOWN = 1800
FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
FEISHU_CHAT_ID = os.environ.get("FEISHU_CHAT_ID", "")
_token_cache = {"token": None, "expires_at": 0}

# Grid detection config
GRID_ROWS = 4
GRID_COLS = 4
GRID_CANDIDATE_CELLS = 5   # check the N lowest-motion cells
GRID_BREATH_BAND = (0.17, 0.8)  # Hz → 10–48 bpm
GRID_SNR_THRESHOLD = 1.25       # FFT SNR to consider a cell as "breathing"
GRID_SCALE_WIDTH = 640          # resize frames to this width for optical flow speed

LOG_DIR = Path(__file__).parent / "data"
LOG_FILE = LOG_DIR / "log.csv"
DEBUG_LOG_FILE = LOG_DIR / "debug_log.csv"
STATE_FILE = LOG_DIR / "state.json"
_last_diag = {}

# Grid locking state
_grid_lock = {"locked_cell": None, "cell_history": [], "snr_history": [],
              "warmup_cycles": 3, "lock_threshold": 2, "unlock_snr_ratio": 0.6,
              "unlock_consecutive": 2}

# Historical breath rates for FFT peak weighting
_rate_history = []  # last N successful breath_rate values
_RATE_HISTORY_SIZE = 4
_RATE_HISTORY_BONUS = 1.5  # weight multiplier for FFT peaks near historical mean


# ============ STATE / FEISHU ============
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def _get_feishu_token():
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


# ============ SIGNAL ANALYSIS ============
def _analyze_signal(signal_arr, fps_effective):
    """Run peak-counting + autocorrelation + FFT on a 1D signal.
    Returns (breath_rate, debug_dict).  debug_dict has all intermediate values.
    """
    global _rate_history
    diag = {}
    n = len(signal_arr)

    # ---- autocorrelation ----
    ac_rate = None
    ac_strength = 0.0
    if n > 10:
        ac = np.correlate(signal_arr, signal_arr, mode='full')
        ac = ac[len(ac) // 2:]
        ac = ac / max(ac[0], 1e-10)
        lag_min = max(1, int(fps_effective * 60 / 40))
        lag_max = min(len(ac) - 1, int(fps_effective * 60 / 10))
        if lag_max > lag_min:
            ac_seg = ac[lag_min:lag_max + 1]
            ac_peaks_raw, _ = find_peaks(ac_seg, height=0.05,
                                          distance=max(1, int(fps_effective * 0.5)))
            if len(ac_peaks_raw) > 0:
                best_i = max(ac_peaks_raw, key=lambda i: ac_seg[i])
                ac_lag = best_i + lag_min
                ac_strength = float(ac[ac_lag])
                ac_rate = 60.0 * fps_effective / ac_lag

    diag["ac_strength"] = round(ac_strength, 3)
    diag["ac_rate"] = round(ac_rate, 1) if ac_rate else None

    # Note: grid scan already filters by FFT SNR, so we don't reject on low ac_strength.
    # Instead, we just note it and continue.

    # ---- peak counting ----
    smoothed = uniform_filter1d(signal_arr, size=max(3, n // 12))
    distance = max(1, int(fps_effective * 0.3))
    peaks, props = find_peaks(smoothed, distance=distance,
                               prominence=np.std(smoothed) * 0.04)

    peak_rate = None
    valid_count = 0
    if len(peaks) >= 2:
        intervals = np.diff(peaks) / fps_effective
        q1, q3 = np.percentile(intervals, [25, 75])
        iqr = q3 - q1
        valid = intervals[(intervals >= q1 - 1.5 * iqr) & (intervals <= q3 + 1.5 * iqr)]
        valid_count = len(valid)
        if valid_count >= 2:
            peak_rate = 60.0 / np.mean(valid)

    diag["peak_rate"] = round(peak_rate, 1) if peak_rate else None
    diag["peaks_found"] = len(peaks)
    diag["valid_intervals"] = valid_count

    # ---- FFT ----
    fft_a = np.abs(np.fft.rfft(signal_arr))
    fft_f = np.fft.rfftfreq(n, d=1.0 / fps_effective)
    band = (fft_f >= 0.25) & (fft_f <= 1.2)
    fft_rate = None
    peak_info = ""
    if np.any(band):
        bf = fft_f[band]
        ba = fft_a[band]
        top_i = np.argsort(ba)[-6:][::-1]
        median_pwr = np.median(ba)
        strong = [(bf[i] * 60, ba[i]) for i in top_i if ba[i] > median_pwr * 1.2]
        peak_info = ", ".join(f"{bf[i] * 60:.0f}bpm(p={ba[i]:.4f})" for i in top_i[:3])
        fft_rate = strong[0][0] if strong else None
        diag["fft_rate"] = round(fft_rate, 1) if fft_rate else None

        # ---- consensus ----
        candidates = []
        if peak_rate and valid_count >= 3:
            candidates.append(("peaks", peak_rate, valid_count / 10))
        if fft_rate:
            snr_weight = min(1.0, strong[0][1] / (np.median(ba) + 1e-10) / 5)
            # Harmonic suppression: if fft > 25 and another method ≈ fft/2, force low value
            if fft_rate > 25:
                for _name, other_rate in [("peaks", peak_rate), ("ac", ac_rate)]:
                    if other_rate and abs(fft_rate / 2 - other_rate) < 6:
                        snr_weight = 0
                        print(f"  🔇 Harmonic suppressed: fft={fft_rate:.0f} → zero weight")
                        break
            candidates.append(("fft", fft_rate, snr_weight))

        # Historical weighting: boost FFT peaks near recent mean
        if len(_rate_history) >= 2:
            hist_mean = np.mean(_rate_history)
            for fft_bpm, fft_pwr in strong[:4]:
                if abs(fft_bpm - hist_mean) < 5:
                    hist_weight = min(1.0, fft_pwr / (np.median(ba) + 1e-10) / 5)
                    candidates.append(("fft_hist", fft_bpm, hist_weight * _RATE_HISTORY_BONUS))
                    break  # only boost the closest peak
        if ac_rate and ac_strength > 0.15:
            candidates.append(("ac", ac_rate, ac_strength * 2))

        if not candidates:
            breath_rate = peak_rate
        elif len(candidates) == 1:
            breath_rate = candidates[0][1]
        else:
            # Harmonic filter + physiological weighting
            rates = sorted(candidates, key=lambda x: x[1])
            filtered = []
            for method, rate, weight in rates:
                is_harmonic = False
                for method2, rate2, _ in rates:
                    if rate2 >= rate:
                        continue
                    for mult in [2, 3]:
                        if abs(rate - mult * rate2) < 8:
                            is_harmonic = True
                            break
                    for mult in [2, 3]:
                        if abs(rate * mult - rate2) < 8:
                            is_harmonic = True
                            break
                if is_harmonic:
                    continue
                phys_weight = weight
                if 12 <= rate <= 35:
                    phys_weight *= 1.3
                elif rate < 10 or rate > 50:
                    phys_weight *= 0.4
                filtered.append((method, rate, phys_weight))

            if not filtered:
                breath_rate = min(candidates, key=lambda x: x[1])[1]
            else:
                total_w = sum(w for _, _, w in filtered)
                breath_rate = sum(r * w for _, r, w in filtered) / max(total_w, 1e-10)

            if ac_rate and ac_strength > 0.25:
                close = [c for c in candidates if abs(c[1] - ac_rate) < 8]
                if len(close) >= 2:
                    others = [c[1] for c in close if c[0] != 'ac']
                    if others:
                        breath_rate = ac_rate * 0.6 + np.mean(others) * 0.4

            # Sub-harmonic correction
            if breath_rate and 10 <= breath_rate <= 17:
                best_sh = None
                for fft_bpm, _ in strong[:4]:
                    if abs(fft_bpm - breath_rate * 2) < 8 and 18 <= fft_bpm <= 35:
                        # Prefer peaks closer to 21 bpm (expected cat resting rate)
                        score = 1.0 / (abs(fft_bpm - 21) + 1)
                        if best_sh is None or score > best_sh[1]:
                            best_sh = (fft_bpm, score)
                if best_sh:
                    print(f"  ⚠️  Sub-harmonic: {breath_rate:.1f} → {best_sh[0]:.1f} bpm")
                    breath_rate = best_sh[0]
                    diag["sub_harmonic_corrected"] = True
    else:
        breath_rate = peak_rate

    if breath_rate is None:
        diag["breath_rate"] = None
        return None, diag, "no_consensus"
    diag["sub_harmonic_corrected"] = False

    if breath_rate < 10 or breath_rate > 80:
        diag["breath_rate"] = None
        return None, diag, "out_of_range"

    final_rate = round(float(breath_rate), 1)
    diag["breath_rate"] = final_rate

    # Update rate history for FFT historical weighting
    _rate_history.append(final_rate)
    if len(_rate_history) > _RATE_HISTORY_SIZE:
        _rate_history = _rate_history[-_RATE_HISTORY_SIZE:]

    print(f"  📊 Peaks: {diag['peaks_found']} found, {diag['valid_intervals']} valid, "
          f"FFT: {peak_info}  →  {final_rate:.1f} bpm")
    return final_rate, diag, None


# ============ GRID-BASED BREATHING DETECTION ============
def grid_detect_breathing(frames, frame_timestamps):
    """Divide frame into grid, find low-motion cells, detect periodic breathing signal."""
    global _last_diag
    _last_diag = {"fps_effective": 0, "signal_count": 0, "ac_strength": 0, "ac_rate": None,
                  "peak_rate": None, "peaks_found": 0, "valid_intervals": 0,
                  "fft_rate": None, "breath_rate": None,
                  "grid_cell": None, "grid_motion": None, "grid_snr": None}

    if len(frames) < 5:
        print("  ❌ Grid: too few frames")
        return None

    # Resize frames for optical flow speed
    scale = GRID_SCALE_WIDTH / frames[0].shape[1]
    scaled_h = int(frames[0].shape[0] * scale)
    scaled_w = GRID_SCALE_WIDTH

    h, w = scaled_h, scaled_w
    cell_h, cell_w = h // GRID_ROWS, w // GRID_COLS

    # Accumulate median flow magnitude per cell over time
    cell_signals = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    prev_gray = None
    flow_count = 0

    for frame in frames:
        small = cv2.resize(frame, (scaled_w, scaled_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.7, 5, 21, 3, 7, 1.5, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    y1, y2 = r * cell_h, (r + 1) * cell_h
                    x1, x2 = c * cell_w, (c + 1) * cell_w
                    cell_mag = float(np.median(mag[y1:y2, x1:x2]))
                    cell_signals[r][c].append(cell_mag)
            flow_count += 1
        prev_gray = gray

    if flow_count < 10:
        print("  ❌ Grid: too few flow frames")
        return None

    # fps from timestamps
    if frame_timestamps is not None and len(frame_timestamps) > 1:
        actual_span = frame_timestamps[-1] - frame_timestamps[0]
        fps_effective = flow_count / actual_span if actual_span > 0 else 1.0
    else:
        fps_effective = flow_count / CAPTURE_DURATION

    _last_diag["fps_effective"] = round(fps_effective, 3)
    _last_diag["signal_count"] = flow_count
    print(f"  📐 Grid: {GRID_ROWS}x{GRID_COLS}, {flow_count} flow frames, "
          f"fps={fps_effective:.3f}")

    # ----- grid locking -----
    lock = _grid_lock
    selected_cell = None
    selected_sig = None
    selected_snr = 0.0
    selected_mean_motion = 0.0
    locked = False

    # Check if we should use locked cell
    if lock["locked_cell"] is not None:
        lr, lc = lock["locked_cell"]
        sig = np.array(cell_signals[lr][lc], dtype=np.float64)
        sig = sig - np.mean(sig)
        # Compute SNR for locked cell
        n = len(sig)
        fft_a = np.abs(np.fft.rfft(sig))
        fft_f = np.fft.rfftfreq(n, d=1.0 / fps_effective)
        band = (fft_f >= GRID_BREATH_BAND[0]) & (fft_f <= GRID_BREATH_BAND[1])
        cur_snr = 0.0
        if np.any(band):
            bf = fft_f[band]
            ba = fft_a[band]
            cur_snr = float(ba.max()) / (float(np.median(ba)) + 1e-10)
        lock["snr_history"].append(cur_snr)
        # Check unlock condition: 2 consecutive SNR < 60% of mean
        mean_snr = np.mean(lock["snr_history"][:-1]) if len(lock["snr_history"]) > 1 else cur_snr
        recent_low = sum(1 for s in lock["snr_history"][-2:] if s < mean_snr * lock["unlock_snr_ratio"])
        if len(lock["snr_history"]) >= 2 and recent_low >= 2:
            print(f"  🔓 Unlocking cell ({lr},{lc}) — SNR degraded (recent: "
                  f"{lock['snr_history'][-2]:.1f}x, {lock['snr_history'][-1]:.1f}x vs mean {mean_snr:.1f}x)")
            lock["locked_cell"] = None
            lock["cell_history"] = []
            lock["snr_history"] = []
        elif cur_snr > GRID_SNR_THRESHOLD:
            locked = True
            selected_cell = (lr, lc)
            selected_sig = sig
            selected_snr = cur_snr
            selected_mean_motion = float(np.mean(np.abs(sig)))
            print(f"  🔒 Locked cell ({lr},{lc}) snr={cur_snr:.1f}x")

    # Normal grid search (if not locked)
    if not locked:
        cell_rank = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                sig = np.array(cell_signals[r][c], dtype=np.float64)
                sig_centered = sig - np.mean(sig)
                mean_motion = float(np.mean(np.abs(sig_centered)))
                cell_rank.append((mean_motion, r, c, sig_centered))

        cell_rank.sort(key=lambda x: x[0])
        candidates = cell_rank[:GRID_CANDIDATE_CELLS]

        print(f"  🔍 Checking {len(candidates)} lowest-motion cells for breathing...")

        best = None
        for mean_motion, r, c, sig in candidates:
            n = len(sig)
            fft_a = np.abs(np.fft.rfft(sig))
            fft_f = np.fft.rfftfreq(n, d=1.0 / fps_effective)
            band = (fft_f >= GRID_BREATH_BAND[0]) & (fft_f <= GRID_BREATH_BAND[1])
            if not np.any(band):
                continue
            bf = fft_f[band]
            ba = fft_a[band]
            peak_i = int(np.argmax(ba))
            peak_power = float(ba[peak_i])
            median_pwr = float(np.median(ba)) if len(ba) > 1 else 1.0
            snr = peak_power / median_pwr if median_pwr > 0 else 0
            if snr > GRID_SNR_THRESHOLD:
                score = snr / (mean_motion + 1e-8)
                print(f"    cell({r},{c}) mean_motion={mean_motion:.4f} "
                      f"peak={bf[peak_i]*60:.1f}bpm snr={snr:.1f}x")
                if best is None or score > best[0]:
                    best = (score, r, c, sig, snr, bf[peak_i])

        if best is None:
            print("  ❌ Grid: no cell with periodic breathing signal")
            lock["cell_history"].append(None)
            return None

        _, best_r, best_c, best_sig, best_snr, _ = best
        selected_cell = (best_r, best_c)
        selected_sig = best_sig
        selected_snr = best_snr
        selected_mean_motion = float(cell_rank[0][0])

        # Update cell history for locking
        lock["cell_history"].append(selected_cell)
        if len(lock["cell_history"]) > lock["warmup_cycles"]:
            lock["cell_history"] = lock["cell_history"][-lock["warmup_cycles"]:]

        # Check if we should lock
        hist = [c for c in lock["cell_history"] if c is not None]
        if len(lock["cell_history"]) >= lock["warmup_cycles"] and lock["locked_cell"] is None:
            cell_counts = Counter(hist)
            most_common = cell_counts.most_common(1)[0]
            if most_common[1] >= lock["lock_threshold"]:
                lock["locked_cell"] = most_common[0]
                lock["snr_history"] = [best_snr]
                print(f"  🔒 Locked cell ({most_common[0][0]},{most_common[0][1]}) "
                      f"— selected {most_common[1]}/{lock['warmup_cycles']} times")

    print(f"  ✅ Best cell: ({selected_cell[0]},{selected_cell[1]}) "
          f"snr={selected_snr:.1f}x {'[locked]' if locked else ''}— running full analysis")

    _last_diag["grid_cell"] = f"({selected_cell[0]},{selected_cell[1]})"
    _last_diag["grid_motion"] = round(selected_mean_motion, 5)
    _last_diag["grid_snr"] = round(selected_snr, 1)

    breath_rate, diag, reason = _analyze_signal(selected_sig, fps_effective)
    _last_diag.update(diag)
    return breath_rate


# ============ MAIN CYCLE ============
def run_detection():
    """Run one detection cycle: capture → grid detect → analyze."""
    global _last_diag
    _last_diag = {"fps_effective": 0, "signal_count": 0, "ac_strength": 0, "ac_rate": None,
                  "peak_rate": None, "peaks_found": 0, "valid_intervals": 0,
                  "fft_rate": None, "breath_rate": None,
                  "grid_cell": None, "grid_motion": None, "grid_snr": None}

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*50}")
    print(f"[{timestamp}] Starting detection cycle...")

    # Pre-check camera
    try:
        test = requests.get(CAMERA_SHOT_URL, timeout=3)
        if test.status_code != 200 or len(test.content) < 100:
            print("  ❌ Camera unreachable")
            return {"timestamp": timestamp, "breath_rate": None, "reason": "stream_failed"}
    except Exception:
        print("  ❌ Camera unreachable")
        return {"timestamp": timestamp, "breath_rate": None, "reason": "stream_failed"}

    # Capture frames
    frames = []
    frame_timestamps = []
    start = time.time()
    frame_count = 0
    consecutive_fails = 0

    print("  📷 Capturing snapshots...")
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
        if consecutive_fails > 5:
            break
        time.sleep(0.5)

    elapsed = time.time() - start
    print(f"  📷 Captured {len(frames)} frames ({frame_count} raw) in {elapsed:.1f}s")

    if len(frames) < 5:
        print("  ❌ Too few frames")
        write_debug_log(timestamp, len(frames), frame_count, None)
        return {"timestamp": timestamp, "breath_rate": None, "reason": "no_frames"}

    # Grid detection
    breath_rate = grid_detect_breathing(frames, frame_timestamps)

    if breath_rate is None:
        print("  ❓ No breathing signal detected")
        write_debug_log(timestamp, len(frames), frame_count, None,
                        reason="no_breathing_detected")
        return {"timestamp": timestamp, "breath_rate": None, "reason": "no_breathing_detected"}

    print(f"  🌬️  Breath rate: {breath_rate} breaths/min")
    write_debug_log(timestamp, len(frames), frame_count, breath_rate, reason=None)
    return {"timestamp": timestamp, "breath_rate": breath_rate, "reason": None}


# ============ LOGGING ============
def write_log(result):
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "breath_rate", "reason"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def write_debug_log(timestamp, frames_captured, raw_frame_count, breath_rate, reason=None):
    d = _last_diag
    file_exists = DEBUG_LOG_FILE.exists()
    fields = ["timestamp", "frames_captured", "raw_frame_count", "fps_effective",
              "grid_cell", "grid_motion", "grid_snr", "ac_strength",
              "peak_rate", "fft_rate", "ac_rate", "breath_rate", "reason"]
    with open(DEBUG_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": timestamp,
            "frames_captured": frames_captured,
            "raw_frame_count": raw_frame_count,
            "fps_effective": d.get("fps_effective", ""),
            "grid_cell": d.get("grid_cell", ""),
            "grid_motion": d.get("grid_motion", ""),
            "grid_snr": d.get("grid_snr", ""),
            "ac_strength": d.get("ac_strength", ""),
            "peak_rate": d.get("peak_rate", ""),
            "fft_rate": d.get("fft_rate", ""),
            "ac_rate": d.get("ac_rate", ""),
            "breath_rate": breath_rate if breath_rate else "",
            "reason": reason or "",
        })


def check_hourly(state):
    if not LOG_FILE.exists():
        return
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)

    ac_map = {}
    if DEBUG_LOG_FILE.exists():
        with open(DEBUG_LOG_FILE) as f:
            for row in csv.DictReader(f):
                ts = row.get("timestamp", "")
                ac_s = row.get("ac_strength", "")
                if ts and ac_s:
                    ac_map[ts] = ac_s

    detections = []
    with open(LOG_FILE) as f:
        for row in csv.DictReader(f):
            try:
                t = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                if t > one_hour_ago and row.get("breath_rate"):
                    detections.append((row["timestamp"], row["breath_rate"]))
            except (ValueError, KeyError):
                continue

    if not detections:
        print("  ⚠️  Hourly: 0 detections in the past hour")
        send_alert("⚠️ 猫咪监护 — 过去1小时无检测",
                   f"过去60分钟内没有检测到猫咪呼吸信号。请检查摄像头是否正常。\n"
                   f"时间：{now.strftime('%Y-%m-%d %H:%M:%S')}",
                   state, "last_hourly_alert")
    else:
        lines = []
        for ts, r in detections:
            ac_s = ac_map.get(ts, "")
            ac_info = f" 信号质量: {ac_s}" if ac_s else ""
            lines.append(f"• {ts} — {r} 次/分钟{ac_info}")
        summary = "\n".join(lines)
        print(f"  📊 Hourly summary: {len(detections)} detections")
        send_alert("📊 猫咪监护 — 过去1小时汇总",
                   f"成功检测 {len(detections)} 次\n\n{summary}",
                   state, "last_hourly_alert")


# ============ MAIN ============
def main():
    print("🐱 Cat Breath Rate Monitor v2.0 (grid-based, no YOLO)")
    print(f"   Camera: {CAMERA_URL}")
    print(f"   Cycle interval: {CYCLE_INTERVAL}s ({CYCLE_INTERVAL // 60} min)")
    print(f"   Alert threshold: {BREATH_RATE_MAX} breaths/min")
    print(f"   Grid: {GRID_ROWS}x{GRID_COLS}, breathing band "
          f"{GRID_BREATH_BAND[0]}-{GRID_BREATH_BAND[1]} Hz")
    print(f"   Feishu alerts: {'✅ configured' if FEISHU_APP_ID else '⚠️  not set'}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    last_hourly_check = state.get("last_hourly_check", 0)
    signal.signal(signal.SIGINT, lambda sig, frame: graceful_exit())
    signal.signal(signal.SIGTERM, lambda sig, frame: graceful_exit())

    print("\n🔄 Monitoring started. Press Ctrl+C to stop.\n")

    while True:
        try:
            result = run_detection()
            write_log(result)

            if result.get("breath_rate") and result["breath_rate"] > BREATH_RATE_MAX:
                ac_s = _last_diag.get("ac_strength", "N/A")
                print(f"  🚨 HIGH BREATH RATE: {result['breath_rate']}/min!")
                send_alert("猫咪呼吸急促警报",
                           f"检测到猫咪呼吸频率异常！\n"
                           f"呼吸频率：{result['breath_rate']} 次/分钟\n"
                           f"阈值：{BREATH_RATE_MAX} 次/分钟\n"
                           f"信号质量 (ac_strength)：{ac_s}\n"
                           f"时间：{result['timestamp']}",
                           state, "last_breath_alert")

            now = time.time()
            if now - last_hourly_check > 3600:
                check_hourly(state)
                last_hourly_check = now
                state["last_hourly_check"] = last_hourly_check
                save_state(state)

        except Exception as e:
            print(f"  ❌ Detection error: {e}")
            write_log({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "breath_rate": None, "reason": f"error: {e}"})

        next_cycle = datetime.now() + timedelta(seconds=CYCLE_INTERVAL)
        print(f"\n  ⏰ Next detection: {next_cycle.strftime('%H:%M:%S')}")
        time.sleep(CYCLE_INTERVAL)


def graceful_exit():
    print("\n🛑 Monitor stopped.")
    sys.exit(0)


if __name__ == "__main__":
    main()
