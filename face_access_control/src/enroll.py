import pickle # For pkl files to save the embeddings
import os
from pathlib import Path

import face_recognition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_DIR = PROJECT_ROOT / "data" / "authorized"
EMBEDDINGS_DIR = PROJECT_ROOT / "models" / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "whitelist_embeddings.pkl"


def build_whitelist(authorized_dir):
    whitelist = {}

    for person_name in os.listdir(authorized_dir):
        person_path = authorized_dir / person_name

        if not person_path.is_dir():
            continue

        embeddings = []

        for file_name in os.listdir(person_path):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = person_path / file_name
            image = face_recognition.load_image_file(image_path)

            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if len(face_encodings) == 1:
                embeddings.append(face_encodings[0])
            else:
                print(f"[WARNING] Skipping {image_path}, detected {len(face_encodings)} faces")

        whitelist[person_name] = embeddings
        print(f"{person_name}: {len(embeddings)} valid images")

    return whitelist


# save whitelist embeddings to disk as a pickle file.
def save_whitelist(whitelist, output_file: Path):
    
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("wb") as f:
        pickle.dump(whitelist, f)

    print(f"[INFO] Saved whitelist embeddings to: {output_file}")

def main():
    whitelist = build_whitelist(AUTHORIZED_DIR)

    total_embeddings = sum(len(v) for v in whitelist.values())
    if total_embeddings == 0:
        raise ValueError("No valid embeddings were created. Check your authorized images.")
    save_whitelist(whitelist, EMBEDDINGS_FILE)

    print("[INFO] Enrollment completed successfully.")
    print(f"[INFO] Total identities: {len(whitelist)}")
    print(f"[INFO] Total embeddings: {total_embeddings}")

if __name__ == "__main__":
    main()