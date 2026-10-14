# Posture Checker

A live webcam-based posture monitor that estimates how far your head is leaning forward relative to your own calibrated baseline. It uses OpenCV and MediaPipe to detect body landmarks, computes a forward-head angle in real time, and gives a simple score from 0 to 100 with visual and audio alerts.

This project is designed as a lightweight desktop demo for posture awareness, not as a medical diagnostic tool.

## Features

- Real-time webcam posture tracking
- Side-view setup to estimate forward head posture
- User calibration against a neutral baseline
- Continuous posture score from 0 to 100
- Status colors and on-screen guidance
- Audio beep alert when posture remains too far forward
- Session summary showing percentage of time aligned with the baseline
- Recalibration and reset controls via keyboard

## How it works

The script uses MediaPipe Pose to detect the nose, ears, and shoulders. It computes a signed forward-head angle based on the ear-to-shoulder line relative to the image vertical. The system then compares the current angle against a baseline recorded during the initial calibration phase.

The score is derived as:

- 100 = aligned with your baseline
- 75+ = good
- 50- = warning range
- below 50 = excessive forward head posture

The implementation smooths the angle over time and only penalizes forward movement, not backward movement, so it reflects the specific posture issue the app is designed to monitor.

## Requirements

- Python 3.8+
- Webcam
- OpenCV
- MediaPipe
- NumPy

## Installation

```bash
pip install opencv-python mediapipe numpy
```

## Usage

From the project directory:

```bash
python posture_checker.py
```

### Controls

- C: recalibrate
- R: reset session stats
- Q: quit

## Camera setup

For best results:

- Position the camera at the side of your body, roughly perpendicular to your shoulders
- Keep the camera near shoulder or head height
- Ensure the ear, shoulder, and nose are visible
- Maintain a 3/4 angle for reasonable results; the app notes that this tends to under-estimate the lean

## Calibration

When the app starts, it waits for a short calibration period while the user sits still and upright. During this time:

- It collects valid pose samples
- Rejects calibration if the user moves too much
- Requires a minimum number of valid frames
- Uses the median of the samples as the baseline posture

If the baseline itself is already too far forward, the app displays a warning.

## Configuration

The script contains a few tunable constants near the top of the file:

- CALIB_SECONDS: calibration duration
- CALIB_MIN_SAMPLES: minimum valid samples required
- CALIB_MAX_STD: maximum allowed motion during calibration
- MAX_BASELINE_DEG: warns if the baseline is already too leaned forward
- ALERT_AFTER: seconds of continuous red before alerting
- ALERT_COOLDOWN: minimum time between audible warnings
- VIS_MIN: minimum visibility threshold for landmarks
- SMOOTH_TAU: smoothing constant used to stabilize readings

These can be modified depending on your environment, camera placement, or tolerance preferences.

## Notes and limitations

- This project is a heuristic and not a clinical or medical measurement.
- It depends on the camera angle and proper landmark visibility.
- Poor lighting or obstructed landmarks may reduce accuracy.
- It is most reliable when the user maintains a consistent side profile and remains still during calibration.

## Project structure

```text
.
├── posture_checker.py
├── README.md
└── posture_env/
```

## Example use cases

- Desk posture awareness during work sessions
- Ergonomic habit feedback
- Quick personal posture monitoring without external hardware

## License

No explicit license is included in the project files. Use this code according to your local or organizational requirements.
