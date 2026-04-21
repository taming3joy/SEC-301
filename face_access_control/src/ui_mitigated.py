import tkinter as tk
from pathlib import Path
import tempfile
import cv2
from PIL import Image, ImageTk

from liveness import LivenessDetector # import class from liveness.py
from recognize import load_whitelist, recognize_face  # import functions from recognize.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_FILE = PROJECT_ROOT / "models" / "embeddings" / "whitelist_embeddings.pkl"

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Access Control with Liveness Detection")
        self.root.geometry("700x600")

        # Load whitelist once
        self.whitelist = load_whitelist(EMBEDDINGS_FILE)

        # Open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Liveness detector
        self.liveness = LivenessDetector()

        # App state
        # modes: preview, liveness, frozen
        self.mode = "preview"

        self.current_frame_bgr = None
        self.frozen_frame_bgr = None
        self.threshold = 0.40

        # Webcam / image display
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

        # Result / instruction label
        self.result_label = tk.Label(
            root,
            text="Press Capture to start liveness check",
            font=("Arial", 14),
            wraplength=650,
            justify="center"
        )
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
        if self.mode != "frozen":
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_bgr = frame.copy()

                if self.mode == "preview":
                    self.tk_image = self.resize_for_display(frame)
                    self.image_label.config(image=self.tk_image)

                elif self.mode == "liveness":
                    result = self.liveness.process_frame(frame)
                    display_frame = result.annotated_frame_bgr

                    self.tk_image = self.resize_for_display(display_frame)
                    self.image_label.config(image=self.tk_image)

                    if result.passed:
                        # Use the frame captured at the end of the forward-hold step.
                        self.frozen_frame_bgr = (
                            result.captured_frame_bgr.copy()
                            if result.captured_frame_bgr is not None
                            else frame.copy()
                        )
                        self.mode = "frozen"
                        self.run_recognition_on_frozen_frame()
                    else:
                        status_text = result.debug_text.get("status", "Liveness check in progress...")
                        lower_status = status_text.lower()

                        if result.restarted:
                            self.result_label.config(
                                text=status_text,
                                fg="orange"
                            )
                        elif result.current_instruction:
                            self.result_label.config(
                                text=f"Liveness check in progress...\n{result.current_instruction}",
                                fg="blue"
                            )
                        elif "face" in lower_status or "restart" in lower_status:
                            self.result_label.config(
                                text=status_text,
                                fg="orange"
                            )
                        else:
                            self.result_label.config(
                                text=status_text,
                                fg="blue"
                            )

        self.root.after(20, self.update_frame)

    def handle_capture(self):
        if self.mode == "preview":
            if self.current_frame_bgr is None:
                self.result_label.config(
                    text="No frame captured from webcam.",
                    fg="orange"
                )
                return

            self.mode = "liveness"
            self.liveness.start()
            self.result_label.config(
                text="Liveness check started...",
                fg="blue"
            )
            self.capture_btn.config(state="disabled")

        elif self.mode == "frozen":
            self.reset_to_live()

    def run_recognition_on_frozen_frame(self):
        if self.frozen_frame_bgr is None:
            self.result_label.config(
                text="No frozen frame available for recognition.",
                fg="orange"
            )
            self.capture_btn.config(text="Capture Again", state="normal")
            return

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
                text=f"Liveness Passed\n{label}\nPlease capture again.",
                fg="orange"
            )
        else:
            self.result_label.config(
                text=(
                    f"Liveness Passed\n"
                    f"{decision}\n"
                    f"User: {label}\n"
                    f"Authorized User Confidence: {confidence:.2f}\n"
                    f"Distance: {distance:.3f} (Threshold: {self.threshold})"
                ),
                fg="green" if decision == "Grant Access" else "red"
            )

        self.capture_btn.config(text="Capture Again", state="normal")

    def reset_to_live(self):
        self.mode = "preview"
        self.frozen_frame_bgr = None
        self.liveness.reset()

        self.result_label.config(
            text="Press Capture to start liveness check",
            fg="black"
        )
        self.capture_btn.config(text="Capture", state="normal")

    def close_app(self):
        if self.cap.isOpened():
            self.cap.release()
        self.liveness.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()