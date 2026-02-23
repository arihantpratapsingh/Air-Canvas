import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Air Canvas v2", layout="wide")
st.title("🖌️ Air Canvas v2 (Draw • Erase • Save)")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands


class AirCanvasProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.prev_x = None
        self.prev_y = None
        self.canvas = None

        self.color = (255, 0, 0)  # Default Blue
        self.thickness = 5
        self.eraser_thickness = 40
        self.eraser_mode = False

    # Detect which fingers are up
    def fingers_up(self, hand_landmarks):
        fingers = []

        # Index finger
        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
            fingers.append(1)
        else:
            fingers.append(0)

        # Middle finger
        if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
            fingers.append(1)
        else:
            fingers.append(0)

        return fingers

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Create persistent canvas
        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        # Draw UI buttons
        cv2.rectangle(img, (50, 10), (150, 60), (255, 0, 0), -1)    # Blue
        cv2.rectangle(img, (200, 10), (300, 60), (0, 255, 0), -1)  # Green
        cv2.rectangle(img, (350, 10), (450, 60), (0, 0, 255), -1)  # Red
        cv2.rectangle(img, (500, 10), (600, 60), (0, 0, 0), -1)    # Eraser

        cv2.putText(img, "ERASER", (505, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                # Get index finger tip coordinates
                x = int(lm[8].x * w)
                y = int(lm[8].y * h)

                fingers = self.fingers_up(hand_landmarks)

                # ✌ Selection Mode (2 fingers up)
                if fingers[0] == 1 and fingers[1] == 1:
                    self.prev_x, self.prev_y = None, None

                    if 50 < x < 150 and 10 < y < 60:
                        self.color = (255, 0, 0)
                        self.eraser_mode = False

                    elif 200 < x < 300 and 10 < y < 60:
                        self.color = (0, 255, 0)
                        self.eraser_mode = False

                    elif 350 < x < 450 and 10 < y < 60:
                        self.color = (0, 0, 255)
                        self.eraser_mode = False

                    elif 500 < x < 600 and 10 < y < 60:
                        self.eraser_mode = True

                # ☝ Drawing Mode (Only index finger up)
                elif fingers[0] == 1 and fingers[1] == 0:

                    if self.prev_x is None:
                        self.prev_x, self.prev_y = x, y

                    if self.eraser_mode:
                        cv2.line(
                            self.canvas,
                            (self.prev_x, self.prev_y),
                            (x, y),
                            (0, 0, 0),
                            self.eraser_thickness,
                        )
                    else:
                        cv2.line(
                            self.canvas,
                            (self.prev_x, self.prev_y),
                            (x, y),
                            self.color,
                            self.thickness,
                        )

                    self.prev_x, self.prev_y = x, y

                else:
                    self.prev_x, self.prev_y = None, None

        # Merge canvas with live frame
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        final = cv2.add(img_bg, canvas_fg)

        return av.VideoFrame.from_ndarray(final, format="bgr24")


# Start WebRTC stream
ctx = webrtc_streamer(
    key="air-canvas",
    video_processor_factory=AirCanvasProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    },
)

# Save Drawing Button
if ctx.video_processor:
    canvas = ctx.video_processor.canvas
    if canvas is not None:
        success, png = cv2.imencode(".png", canvas)
        if success:
            st.download_button(
                label="💾 Save Drawing as PNG",
                data=png.tobytes(),
                file_name="air_canvas_drawing.png",
                mime="image/png",
            ) 