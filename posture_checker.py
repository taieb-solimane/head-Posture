"""
Posture Checker - live webcam demo (OpenCV + MediaPipe)      [reviewed version]

Install:  pip install opencv-python mediapipe numpy
Run:      python posture_checker.py

Setup: put the camera at the SIDE of your body (camera axis ~perpendicular to
your shoulders), at roughly shoulder/head height. Ear, shoulder and nose must be
visible. A 3/4 angle works but UNDER-estimates the lean (see README notes).
Keys:  C = recalibrate   R = reset session stats   Q = quit

What the score means: how far your head has moved FORWARD, in degrees, compared
with YOUR OWN calibrated posture. It is a heuristic, not a clinical measurement.
"""
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

try:
    import winsound
    def _beep(): winsound.Beep(900, 250)
except ImportError:
    def _beep(): print("\a", end="", flush=True)


def beep():
    """Play the sound in a thread so the video does not freeze while it plays."""
    threading.Thread(target=_beep, daemon=True).start()


# ---------------- settings ----------------
CALIB_SECONDS = 3        # hold still, visible, this long at start
CALIB_MIN_SAMPLES = 10   # need at least this many valid frames in calibration
CALIB_MAX_STD = 3.0      # deg; more movement than this -> calibration is rejected
MAX_BASELINE_DEG = 15.0  # deg; warn if the calibration posture already leans this far forward
ALERT_AFTER = 3          # seconds of continuous red before the alarm
ALERT_COOLDOWN = 8       # seconds between alarms
VIS_MIN = 0.4            # minimum landmark visibility
SMOOTH_TAU = 0.15        # s, smoothing time constant (== old 0.8/0.2 filter at 30 fps)

DEG_TO_ZERO = 30         # degrees of forward deviation that gives a score of 0
GOOD_SCORE = 75          # score >= 75  <=>  deviation <= 7.5 deg  (green)
BAD_SCORE = 50           # score <  50  <=>  deviation >  15  deg  (red)
RECOVER_SCORE = 60       # alert timer only resets once score is back above this (hysteresis)
GREEN, ORANGE, RED = (80, 200, 80), (0, 165, 255), (60, 60, 230)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def head_angle(lm, w, h, side=None):
    """
    SIGNED forward-head angle in degrees, between the image vertical and the
    shoulder->ear line.  Positive = ear is in FRONT of the shoulder (the direction
    the nose points), negative = ear is behind the shoulder.

    side = "left" / "right" to force a body side, or None to pick the clearer one.
    Returns (angle, (shoulder_px, ear_px), side_used) or (None, None, side_used).
    """
    L = mp_pose.PoseLandmark
    pairs = {"left": (lm[L.LEFT_EAR], lm[L.LEFT_SHOULDER]),
             "right": (lm[L.RIGHT_EAR], lm[L.RIGHT_SHOULDER])}
    if side is None:
        side = max(pairs, key=lambda s: pairs[s][0].visibility + pairs[s][1].visibility)
    ear, sh = pairs[side]
    nose = lm[L.NOSE]
    if min(ear.visibility, sh.visibility, nose.visibility) < VIS_MIN:
        return None, None, side

    # normalized -> pixels, so x and y have the same scale (needed for a true angle)
    dx, dy = (ear.x - sh.x) * w, (ear.y - sh.y) * h
    if dy >= 0:                          # ear must be ABOVE the shoulder (y grows downward)
        return None, None, side
    if abs(nose.x - ear.x) < 0.01:       # nose ~ ear in x: facing the camera, direction unknown
        return None, None, side

    facing = 1.0 if nose.x > ear.x else -1.0     # which way the person looks in the image
    angle = np.degrees(np.arctan2(facing * dx, -dy))
    return angle, ((int(sh.x * w), int(sh.y * h)), (int(ear.x * w), int(ear.y * h))), side


def draw_text(img, text, pos, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the camera.")
        return
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    baseline, calib_start, calib_samples = None, None, []
    side = None                                   # body side, locked after first valid frame
    calib_msg, calib_msg_until = "", 0
    smooth, history = None, deque(maxlen=200)
    bad_since, last_alert = None, 0
    good_time = total_time = 0.0                  # seconds (time-weighted, not frame counts)
    session_start = last_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time()
        dt = min(now - last_t, 0.2)
        last_t = now
        status, color, score, angle = "No person detected", (200, 200, 200), None, None

        if res.pose_landmarks:
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            angle, pts, used = head_angle(res.pose_landmarks.landmark, w, h, side)
            if angle is None:
                status = "Show ear, shoulder and nose (side view)"
            else:
                if side is None:
                    side = used
                # exponential smoothing; alpha is time-based so it does not depend on FPS
                alpha = 1 - np.exp(-dt / SMOOTH_TAU)
                smooth = angle if smooth is None else smooth + alpha * (angle - smooth)

                if baseline is None:  # ---- calibration phase ----
                    if calib_start is None:
                        calib_start = now
                    calib_samples.append(smooth)
                    left = max(0.0, CALIB_SECONDS - (now - calib_start))
                    status, color = f"Sit straight... {left:.1f}s", ORANGE
                    if left == 0:
                        samples = np.array(calib_samples)
                        if len(samples) < CALIB_MIN_SAMPLES or samples.std() > CALIB_MAX_STD:
                            calib_msg, calib_msg_until = "Calibration failed: hold still", now + 3
                            calib_start, calib_samples = None, []
                        else:
                            baseline = float(np.median(samples))
                            bad_since = None
                else:  # ---- live scoring ----
                    dev = max(0.0, smooth - baseline)      # only FORWARD movement is penalised
                    score = int(np.clip(100 - dev * 100 / DEG_TO_ZERO, 0, 100))
                    history.append(score)
                    total_time += dt
                    if score >= GOOD_SCORE:
                        status, color = "Aligned with your baseline", GREEN
                        good_time += dt
                        bad_since = None
                    elif score >= BAD_SCORE:
                        status, color = "Head drifting forward", ORANGE
                        if score >= RECOVER_SCORE:
                            bad_since = None
                    else:
                        status, color = "HEAD TOO FAR FORWARD!", RED
                        if bad_since is None:
                            bad_since = now
                        if now - bad_since > ALERT_AFTER and now - last_alert > ALERT_COOLDOWN:
                            beep()
                            last_alert = now
                    cv2.line(frame, pts[0], pts[1], color, 4)

        if angle is None:  # no person / unreliable landmarks: nothing carries over
            smooth = None
            bad_since = None
            if baseline is None:             # calibration needs CONTINUOUS valid frames
                calib_start, calib_samples = None, []

        # ---- UI ----
        cv2.rectangle(frame, (0, 0), (w, 70), (25, 25, 25), -1)
        draw_text(frame, status, (15, 45), 0.9, color, 2)
        if baseline is None and calib_start is None:
            draw_text(frame, "Sit straight, side view, to start", (15, h - 20), 0.7)
        if now < calib_msg_until:
            draw_text(frame, calib_msg, (15, h - 55), 0.7, RED)
        if baseline is not None and baseline > MAX_BASELINE_DEG:
            draw_text(frame, f"Warning: baseline already leans {baseline:.0f} deg (press C)",
                      (15, 100), 0.6, ORANGE)
        if score is not None:
            draw_text(frame, f"Score: {score}", (w - 200, 45), 1.0, color, 2)
            # live chart
            cw, ch, x0, y0 = min(w - 30, 400), 90, 15, h - 110
            cv2.rectangle(frame, (x0, y0), (x0 + cw, y0 + ch), (25, 25, 25), -1)
            pts_c = [(x0 + int(i * cw / history.maxlen), y0 + ch - int(s * ch / 100))
                     for i, s in enumerate(history)]
            if len(pts_c) > 1:
                cv2.polylines(frame, [np.array(pts_c)], False, color, 2)
            pct = 100 * good_time / max(total_time, 1e-6)
            draw_text(frame, f"Aligned: {pct:.0f}%  |  {int((now - session_start) // 60)} min",
                      (x0, y0 - 10), 0.6)
        if score is not None and score < BAD_SCORE:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 8)

        cv2.imshow("Posture Checker  (C = recalibrate, Q = quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            baseline, calib_start, calib_samples = None, None, []
            side, smooth, bad_since = None, None, None
            history.clear()
        if key == ord("r"):
            history.clear()
            good_time = total_time = 0.0
            session_start = time.time()

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    if total_time > 0:
        print(f"Session summary: {100 * good_time / total_time:.0f}% of time aligned with baseline")


if __name__ == "__main__":
    main()