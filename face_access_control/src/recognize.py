import pickle
from pathlib import Path

import face_recognition
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_FILE = PROJECT_ROOT / "models" / "embeddings" / "whitelist_embeddings.pkl"


def load_whitelist(embeddings_file):
    with embeddings_file.open("rb") as f:
        whitelist = pickle.load(f)

    print(f"[INFO] Loaded embeddings from: {embeddings_file}")
    return whitelist

def compute_confidence(distance, threshold=0.40, k=15):
    return 1 / (1 + np.exp(k * (distance - threshold)))

def recognize_face(image_path, whitelist, threshold=0.40):
    image = face_recognition.load_image_file(image_path)

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if len(face_encodings) == 0:
        return "Retry", "No face detected", None, None

    if len(face_encodings) > 1:
        return "Retry", "Multiple faces detected", None, None

    input_encoding = face_encodings[0]

    best_match = None
    best_distance = float("inf")

    for person_name, embeddings in whitelist.items():
        distances = face_recognition.face_distance(embeddings, input_encoding)

        if len(distances) == 0:
            continue

        min_distance = np.min(distances)

        if min_distance < best_distance:
            best_distance = min_distance
            best_match = person_name

    # Calculate confidence based on distance
    confidence = compute_confidence(best_distance, threshold)

    if best_distance < threshold:
        return "Grant Access", best_match, best_distance, confidence
    else:
        return "Deny Access", "Unknown", best_distance, confidence

def main():
    whitelist = load_whitelist(EMBEDDINGS_FILE)

    test_image = PROJECT_ROOT / "data" / "test" / "authorized" / "taming" / "taming_test_1.jpg"

    decision, label, distance, confidence = recognize_face(test_image, whitelist)

    print(f"Decision: {decision}")
    print(f"Label: {label}")
    print(f"Distance: {distance}")
    print(f"Confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()