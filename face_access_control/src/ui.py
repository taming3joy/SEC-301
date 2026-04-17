import tkinter as tk
from pathlib import Path
import tempfile
import cv2
from PIL import Image, ImageTk

from recognize import load_whitelist, recognize_face  # import functions from recognize.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_FILE = PROJECT_ROOT / "models" / "embeddings" / "whitelist_embeddings.pkl"

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Access Control")
        self.root.geometry("700x550")

        # Load whitelist once
        self.whitelist = load_whitelist(EMBEDDINGS_FILE)

        # Open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        self.current_frame_bgr = None
        self.frozen_frame_bgr = None
        self.is_frozen = False
        self.threshold = 0.45

        # Webcam and image display
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

        # Result label
        self.result_label = tk.Label(root, text="Press Capture to take a photo", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Capture button
        self.capture_btn = tk.Button(root, text="Capture", command=self.handle_capture)
        self.capture_btn.pack(pady=10)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

        # Start live preview
        self.update_frame()

    def resize_for_display(self, frame_bgr, max_size=500):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        w, h = img.size
        scale = max_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size)

        return ImageTk.PhotoImage(img)

    def update_frame(self):
        if not self.is_frozen:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_bgr = frame.copy()
                self.tk_image = self.resize_for_display(frame)
                self.image_label.config(image=self.tk_image)

        self.root.after(20, self.update_frame)

    def handle_capture(self):
        if not self.is_frozen:
            self.capture_image()
        else:
            self.reset_to_live()

    def capture_image(self):
        if self.current_frame_bgr is None:
            self.result_label.config(
                text="No frame captured from webcam.",
                fg="orange"
            )
            return

        self.is_frozen = True
        self.frozen_frame_bgr = self.current_frame_bgr.copy()

        # Show frozen frame
        self.tk_image = self.resize_for_display(self.frozen_frame_bgr)
        self.image_label.config(image=self.tk_image)

        # Save frozen frame to temporary file for recognition
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_path = tmp_file.name
            cv2.imwrite(temp_path, self.frozen_frame_bgr)

        # Run recognition
        decision, label, distance, confidence = recognize_face(temp_path, self.whitelist)

        # Show result
        if decision == "Retry":
            self.result_label.config(
                text=f"{label}\nPlease capture again.",
                fg="orange"
            )
        else:
            self.result_label.config(
                text=(
                    f"{decision}\n"
                    f"User: {label}\n"
                    f"Authorized User Confidence: {confidence:.2f}\n"
                    f"Distance: {distance:.3f} (Threshold: {self.threshold})"
                ),
                fg="green" if decision == "Grant Access" else "red"
            )

        self.capture_btn.config(text="Capture Again")

    def reset_to_live(self):
        self.is_frozen = False
        self.frozen_frame_bgr = None
        self.result_label.config(
            text="Press Capture to take a photo",
            fg="black"
        )
        self.capture_btn.config(text="Capture")

    def close_app(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()