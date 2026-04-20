# Face Access Control System

This project is a facial recognition based access control system. It allows only authorized users (whitelisted faces) to gain access by comparing their face with stored embeddings.

## Project Structure

- data/authorized/ -> training images for authorized users
- data/test/ -> test images
- models/embeddings/ -> saved face embeddings
- src/ -> source code (enroll, recognize, UI)

## Setup

### 1. Create virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### Important

Make sure you are inside the project directory:

```bash
cd face_access_control
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

## Usage

### 1. Enroll authorized users

This builds and saves face embeddings:

```bash
python src/enroll.py
```

### 2. Run the system (UI)

```bash
python src/UI.py
```

- Capture your face using the webcam
- System will output:
  - Grant Access
  - Deny Access
  - Retry (if no or multiple faces detected)

## Notes

- Only one face should be in the image
- Good lighting improves detection
- Confidence is a heuristic score, not a true probability

## Authorized Users

- Thanawat Kositjaroenkul (Taming)
- Hardik Joshi (Josh)
