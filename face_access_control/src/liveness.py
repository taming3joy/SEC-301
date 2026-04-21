import random
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import cv2
import mediapipe as mp
import numpy as np


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_point(landmarks, idx, w, h):
    return np.array([
        int(landmarks[idx].x * w),
        int(landmarks[idx].y * h)
    ])


def make_random_challenge() -> List[Dict[str, Any]]:
    first_turn = random.choice(["turn_left", "turn_right"])
    second_turn = random.choice(["turn_left", "turn_right"])
    blink_target = random.randint(2, 3)

    return [
        {"type": first_turn},
        {"type": "blink", "count": blink_target},
        {"type": second_turn},
        {"type": "forward_hold", "duration": 2.0},
    ]


def step_to_instruction(step: Dict[str, Any]) -> str:
    step_type = step["type"]

    if step_type == "turn_left":
        return "Turn head left"
    if step_type == "turn_right":
        return "Turn head right"
    if step_type == "forward":
        return "Face forward"
    if step_type == "blink":
        count = step["count"]
        if count == 1:
            return "Face forward and blink once"
        return f"Face forward and blink {count} times"
    if step_type == "forward_hold":
        return f"Face forward and hold still for {step['duration']:.0f} seconds"

    return ""


@dataclass
class LivenessResult:
    face_detected: bool
    passed: bool
    restarted: bool
    current_instruction: Optional[str]
    debug_text: Dict[str, str]
    annotated_frame_bgr: np.ndarray
    captured_frame_bgr: Optional[np.ndarray] = None


class LivenessDetector:
    def __init__(
        self,
        face_retry_delay_seconds: float = 2.0,
        blink_threshold: float = 0.21,
        turn_threshold: float = 0.075,
        forward_tolerance: float = 0.03,
        min_closed_frames: int = 2,
        step_delay_seconds: float = 1.5,
    ):
        self.face_retry_delay_seconds = face_retry_delay_seconds
        self.blink_threshold = blink_threshold
        self.turn_threshold = turn_threshold
        self.forward_tolerance = forward_tolerance
        self.min_closed_frames = min_closed_frames
        self.step_delay_seconds = step_delay_seconds

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.active = False
        self.reset()

    def reset(self):
        self.retry_available_at = None
        self.active = False
        self.state = {
            "challenge": [],
            "step": 0,
            "passed": False,
            "waiting_until": 0.0,
            "blink_count": 0,
            "closed_frames": 0,
            "hold_started_at": None,
            "awaiting_final_blink": False,
            "post_capture_blink_count": 0,
            "post_capture_closed_frames": 0,
            "captured_face_frame": None,
        }

    def start(self):
        self.active = True
        self.state = {
            "challenge": make_random_challenge(),
            "step": 0,
            "passed": False,
            "waiting_until": time.time() + self.step_delay_seconds,
            "blink_count": 0,
            "closed_frames": 0,
            "hold_started_at": None,
            "awaiting_final_blink": False,
            "post_capture_blink_count": 0,
            "post_capture_closed_frames": 0,
            "captured_face_frame": None,
        }

    def close(self):
        self.face_mesh.close()

    def _get_direction_and_offset(self, face_landmarks, w, h):
        nose = get_point(face_landmarks, 1, w, h)
        left_face = get_point(face_landmarks, 234, w, h)
        right_face = get_point(face_landmarks, 454, w, h)

        face_center_x = (left_face[0] + right_face[0]) / 2
        nose_offset = (nose[0] - face_center_x) / w

        if nose_offset < -self.turn_threshold:
            direction = "Turned Left"
        elif nose_offset > self.turn_threshold:
            direction = "Turned Right"
        elif abs(nose_offset) <= self.forward_tolerance:
            direction = "Forward"
        else:
            direction = "Slight Turn"

        return direction, nose_offset, nose, left_face, right_face

    def process_frame(self, frame_bgr: np.ndarray) -> LivenessResult:
        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape
        now = time.time()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        avg_ear = 0.0
        direction = "No face"
        nose_offset = 0.0
        current_instruction = None
        restarted = False

        if not self.active:
            return LivenessResult(
                face_detected=False,
                passed=False,
                restarted=False,
                current_instruction=None,
                debug_text={
                    "ear": "0.000",
                    "direction": "Inactive",
                    "offset": "0.000",
                    "status": "Liveness inactive",
                },
                annotated_frame_bgr=annotated,
                captured_frame_bgr=None,
            )

        face_count = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0

        if self.retry_available_at is not None and now >= self.retry_available_at:
            self.retry_available_at = None

        if face_count != 1:
            restarted = False
            if self.retry_available_at is None:
                self.start()
                self.retry_available_at = now + self.face_retry_delay_seconds
                restarted = True

            remaining = max(0.0, self.retry_available_at - now) if self.retry_available_at is not None else self.face_retry_delay_seconds
            if face_count == 0:
                status_text = f"Keep your face within the frame. Retrying in {remaining:.1f}s"
            else:
                status_text = f"Keep only one face within the frame. Retrying in {remaining:.1f}s"

            return LivenessResult(
                face_detected=False,
                passed=False,
                restarted=restarted,
                current_instruction=None,
                debug_text={
                    "ear": "0.000",
                    "direction": "No face" if face_count == 0 else "Multiple faces",
                    "offset": "0.000",
                    "status": status_text,
                },
                annotated_frame_bgr=annotated,
                captured_frame_bgr=None,
            )

        if self.retry_available_at is not None and now < self.retry_available_at:
            remaining = self.retry_available_at - now
            return LivenessResult(
                face_detected=True,
                passed=False,
                restarted=False,
                current_instruction=None,
                debug_text={
                    "ear": "0.000",
                    "direction": "Waiting",
                    "offset": "0.000",
                    "status": f"Retrying in {remaining:.1f}s",
                },
                annotated_frame_bgr=annotated,
                captured_frame_bgr=None,
            )

        face_landmarks = results.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        direction, nose_offset, nose, left_face, right_face = self._get_direction_and_offset(
            face_landmarks, w, h
        )

        if not self.state["passed"] and not self.state["awaiting_final_blink"]:
            if now < self.state["waiting_until"]:
                current_instruction = None
            else:
                if self.state["step"] >= len(self.state["challenge"]):
                    self.state["captured_face_frame"] = frame_bgr.copy()
                    self.state["awaiting_final_blink"] = True
                    self.state["post_capture_blink_count"] = 0
                    self.state["post_capture_closed_frames"] = 0
                    self.state["hold_started_at"] = None
                    current_instruction = None
                else:
                    current_step = self.state["challenge"][self.state["step"]]
                    current_instruction = step_to_instruction(current_step)

                    if current_step["type"] == "turn_left":
                        if direction == "Turned Left":
                            self.state["step"] += 1
                            self.state["waiting_until"] = now + self.step_delay_seconds

                    elif current_step["type"] == "turn_right":
                        if direction == "Turned Right":
                            self.state["step"] += 1
                            self.state["waiting_until"] = now + self.step_delay_seconds

                    elif current_step["type"] == "forward":
                        if direction == "Forward":
                            self.state["step"] += 1
                            self.state["waiting_until"] = now + self.step_delay_seconds

                    elif current_step["type"] == "blink":
                        if direction == "Forward":
                            if avg_ear < self.blink_threshold:
                                self.state["closed_frames"] += 1
                            else:
                                if self.state["closed_frames"] >= self.min_closed_frames:
                                    self.state["blink_count"] += 1
                                self.state["closed_frames"] = 0
                        else:
                            self.state["closed_frames"] = 0

                        if self.state["blink_count"] >= current_step["count"]:
                            self.state["step"] += 1
                            self.state["waiting_until"] = now + self.step_delay_seconds
                            self.state["blink_count"] = 0
                            self.state["closed_frames"] = 0

                    elif current_step["type"] == "forward_hold":
                        if direction == "Forward":
                            if self.state["hold_started_at"] is None:
                                self.state["hold_started_at"] = now

                            held_for = now - self.state["hold_started_at"]
                            if held_for >= current_step["duration"]:
                                self.state["step"] += 1
                                self.state["waiting_until"] = now + self.step_delay_seconds
                                self.state["hold_started_at"] = None
                        else:
                            self.state["hold_started_at"] = None

                if self.state["step"] >= len(self.state["challenge"]) and not self.state["awaiting_final_blink"]:
                    self.state["captured_face_frame"] = frame_bgr.copy()
                    self.state["awaiting_final_blink"] = True
                    self.state["post_capture_blink_count"] = 0
                    self.state["post_capture_closed_frames"] = 0
                    self.state["hold_started_at"] = None
                    current_instruction = None
                elif self.state["step"] < len(self.state["challenge"]):
                    if now < self.state["waiting_until"]:
                        current_instruction = None
                    else:
                        current_step = self.state["challenge"][self.state["step"]]
                        current_instruction = step_to_instruction(current_step)

                        if current_step["type"] == "blink":
                            current_instruction += (
                                f" ({self.state['blink_count']}/{current_step['count']})"
                            )
                        elif current_step["type"] == "forward_hold" and self.state["hold_started_at"] is not None:
                            held_for = now - self.state["hold_started_at"]
                            current_instruction += (
                                f" ({min(held_for, current_step['duration']):.1f}/{current_step['duration']:.1f}s)"
                            )

        elif self.state["awaiting_final_blink"] and not self.state["passed"]:
            if direction == "Forward":
                if avg_ear < self.blink_threshold:
                    self.state["post_capture_closed_frames"] += 1
                else:
                    if self.state["post_capture_closed_frames"] >= self.min_closed_frames:
                        self.state["post_capture_blink_count"] += 1
                    self.state["post_capture_closed_frames"] = 0
            else:
                self.state["post_capture_closed_frames"] = 0

            current_instruction = (
                "Face captured. Blink once to confirm liveness "
                f"({self.state['post_capture_blink_count']}/1)"
            )

            if self.state["post_capture_blink_count"] >= 1:
                self.state["passed"] = True
                self.state["awaiting_final_blink"] = False
                current_instruction = None

        if self.state["passed"]:
            status_text = "Passed"
        elif current_instruction is not None:
            status_text = current_instruction
        elif self.state["awaiting_final_blink"]:
            status_text = "Face captured. Blink once to confirm liveness"
        else:
            status_text = "Running"

        return LivenessResult(
            face_detected=True,
            passed=self.state["passed"],
            restarted=restarted,
            current_instruction=current_instruction,
            debug_text={
                "ear": f"{avg_ear:.3f}",
                "direction": direction,
                "offset": f"{nose_offset:.3f}",
                "status": status_text,
            },
            annotated_frame_bgr=annotated,
            captured_frame_bgr=self.state["captured_face_frame"] if self.state["passed"] else None,
        )