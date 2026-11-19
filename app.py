import streamlit as st
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

st.set_page_config(page_title="AI Air Drawing", page_icon="🎨")

st.title("🎨 AI Air Drawing")
st.write("Draw in the air using your hand gestures!")

run = st.checkbox("Start Air Drawing")

FRAME_WINDOW = st.image([])
canvas = None
detector = HandDetector(maxHands=1, detectionCon=0.7)
previous_point = None

if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera could not be opened.")
    else:
        while run:
            success, frame = cap.read()

            if not success:
                st.error("Unable to read camera.")
                break

            frame = cv2.flip(frame, 1)

            if canvas is None:
                canvas = np.zeros_like(frame)

            hands, frame = detector.findHands(frame)

            if hands:
                lmList = hands[0]["lmList"]

                index_tip = lmList[8]
                middle_tip = lmList[12]

                x, y = index_tip[0], index_tip[1]

                if lmList[8][1] < lmList[6][1] and lmList[12][1] > lmList[10][1]:
                    if previous_point is not None:
                        cv2.line(canvas, previous_point, (x, y),
                                 (0, 255, 0), 5)
                    previous_point = (x, y)
                else:
                    previous_point = None

            combined = cv2.addWeighted(frame, 1, canvas, 1, 0)
            FRAME_WINDOW.image(
                cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            )

        cap.release()