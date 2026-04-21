# Face Access Control Security Experiments

This project focuses on experimenting with security challenges in a face access control system, especially presentation-based spoofing attacks.

## What This Project Shows

- `src/ui_baseline.py`: pre-mitigated version that can be attacked with spoofing (for example presentation attacks like printed or screen-displayed faces, as shown in `outputs/attack_screenshots/`).
- `src/ui_mitigated.py`: mitigated version that adds liveness detection to reduce these spoofing attacks.

## Project Structure

- `data/authorized/` -> enrolled user images
- `data/test/` -> test images
- `models/embeddings/` -> saved face embeddings
- `notebooks/` -> development and experiment workflow:
  - `create_public_datasets.ipynb`: build subset datasets from public sources for controlled testing.
  - `system_development.ipynb`: prototype and tune baseline recognition pipeline logic before moving stable code into `src/`.
  - `mitigation_development.ipynb`: prototype and tune liveness detection mitigation pipeline logic before moving stable code into `src/`.
- `src/` -> source code:
  - `enroll.py`: saves face embeddings for authorized users.
  - `recognize.py`: compares a face with enrolled users and returns the access control decision and results.
  - `liveness.py`: checks if the face is live (not a spoof from a photo/screen).
  - `ui_baseline.py` and `ui_mitigated.py`: UIs that show the access control interface.

## Setup

### 1. Go to repository root

If you cloned the full `SEC-301` repository, start from the `SEC-301` folder (parent of `face_access_control`).

```bash
cd SEC-301
```

### 2. Use Python 3.12

This project is tested on Python 3.12. Using Python 3.14 may cause dependency compatibility issues.

```bash
python3.12 --version
```

### 3. Create virtual environment (recommended)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 4. Go to project folder

```bash
cd face_access_control
```

### 5. Upgrade installer tools and pin setuptools

`face_recognition_models` still depends on `pkg_resources`, so keep `setuptools` below 81.

If `setuptools==80.9.0` is already pinned in `requirements.txt`, this step is optional but recommended on fresh environments.

```bash
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==80.9.0"
```

### 6. Install requirements

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
