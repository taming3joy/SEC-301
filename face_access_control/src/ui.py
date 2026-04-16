import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from pathlib import Path
from recognize import load_whitelist, recognize_face  # import functions from recognize.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_FILE = PROJECT_ROOT / "models" / "embeddings" / "whitelist_embeddings.pkl"

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Access Control")
        self.root.geometry("500x600")

        # Load whitelist once
        self.whitelist = load_whitelist(EMBEDDINGS_FILE)

        # Image display
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

        # Result label
        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Upload button
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        # Display image
        img = Image.open(file_path)
        max_size = 300

        # Get original size
        w, h = img.size

        # Compute scaling factor
        scale = max_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size)

        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image)

        # Run recognition
        decision, label, distance, confidence = recognize_face(file_path, self.whitelist)

        # Display result
        if decision == "Retry":
            self.result_label.config(
                text=f"{label}\nPlease upload another image.",
                fg="orange"
            )
        else:
            self.result_label.config(
                text=f"{decision}\nUser: {label}\nAuthorized User Confidence: {confidence:.2f}\nDistance: {distance:.3f}",
                fg="green" if decision == "Grant Access" else "red"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()