# Face Access Control Security Experiments

This project focuses on experimenting with security challenges in a face access control system, especially presentation-based spoofing attacks.

## What This Project Shows

- `src/ui_baseline.py`: pre-mitigated version that can be attacked with spoofing (for example presentation attacks like printed or screen-displayed faces, as shown in `outputs/attack_screenshots/`).
- `src/ui_mitigated.py`: mitigated version that adds liveness detection to reduce these spoofing attacks.

## Project Structure

- `data/authorized/` -> enrolled user images
- `data/test/` -> test images
- `models/embeddings/` -> saved face embeddings
- `src/` -> source code (`enroll.py`, `recognize.py`, UIs)

## Setup

### 1. Go to project directory

```bash
cd face_access_control
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

## Usage

### 1. Enroll authorized users

This builds and saves whitelist embeddings:

```bash
python src/enroll.py
```

### 2. Run baseline system (pre-mitigated)

```bash
python src/ui_baseline.py
```

### 3. Run mitigated system (with liveness detection)

```bash
python src/ui_mitigated.py
```

Expected outputs include:

- Grant Access
- Deny Access
- Retry (for invalid frame conditions)

## Notes

- Use one face in frame and good lighting for better results.
- Confidence is a heuristic score, not a true probability.
- Attack examples are shown in `outputs/attack_screenshots/`.
