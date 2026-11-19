import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ==========================================
# AI AIR DRAWING WITH FINGER GESTURE CONTROL
# ==========================================

# 1. WEBCAM
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Webcam not detected!")
    exit()

# 2. MEDIAPIPE
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. CANVAS
canvas = None
previous_point = None
smooth_point = None

# 4. COLOURS
colors = {
    "BLUE": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "PURPLE": (255, 0, 255),
    "WHITE": (255, 255, 255)
}

color_name = "BLUE"
draw_color = colors[color_name]
brush_size = 8
eraser = False


# ==========================================
# 5. GESTURE DETECTION
# ==========================================

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_states(hand):

    lm = hand.landmark
    wrist = lm[0]

    index = (
        distance(lm[8], wrist)
        > distance(lm[5], wrist) * 1.15
    )

    middle = (
        distance(lm[12], wrist)
        > distance(lm[9], wrist) * 1.15
    )

    ring = (
        distance(lm[16], wrist)
        > distance(lm[13], wrist) * 1.15
    )

    pinky = (
        distance(lm[20], wrist)
        > distance(lm[17], wrist) * 1.15
    )

    return index, middle, ring, pinky


# ==========================================
# 6. MAIN PROGRAM
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera frame error!")
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    if canvas is None:
        canvas = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    status = "NO HAND"

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        # 7. FINGERTIP POSITION

        x = int(lm[8].x * width)
        y = int(lm[8].y * height)

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        # Smooth cursor
        if smooth_point is None:

            smooth_point = np.array(
                [x, y],
                dtype=float
            )

        else:

            smooth_point = (
                0.7 * smooth_point
                + 0.3 * np.array([x, y], dtype=float)
            )

        current_point = (
            int(smooth_point[0]),
            int(smooth_point[1])
        )

        # 8. RECOGNIZE GESTURE

        index, middle, ring, pinky = finger_states(hand)

        # Show fingertip cursor
        cv2.circle(
            frame,
            current_point,
            12,
            (0, 255, 255),
            -1
        )

        # 9. DRAWING GESTURE

        if index and not middle and not ring and not pinky:

            status = "DRAWING"

            if previous_point is not None:

                cv2.line(
                    canvas,
                    previous_point,
                    current_point,
                    draw_color,
                    brush_size,
                    cv2.LINE_AA
                )

            previous_point = current_point

        elif index and middle and not ring and not pinky:

            status = "STOP DRAWING"
            previous_point = None

        elif not index and not middle and not ring and not pinky:

            status = "CLOSED FIST"
            previous_point = None

        else:

            status = "TRACKING"
            previous_point = None

        # 10. DISPLAY HAND LANDMARKS

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Debug finger values
        debug = (
            f"I:{int(index)} "
            f"M:{int(middle)} "
            f"R:{int(ring)} "
            f"P:{int(pinky)}"
        )

        cv2.putText(
            frame,
            debug,
            (15, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    else:

        previous_point = None
        smooth_point = None

    # ==========================================
    # 11. COMBINE CAMERA AND CANVAS
    # ==========================================

    output = cv2.addWeighted(
        frame,
        0.5,
        canvas,
        0.5,
        0
    )

    # Toolbar
    cv2.rectangle(
        output,
        (0, 0),
        (width, 75),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        output,
        "AI AIR DRAWING",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "1-6: Colours | E: Eraser | C: Clear | S: Save | Q: Quit",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Status
    cv2.putText(
        output,
        "GESTURE: " + status,
        (15, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        "COLOUR: " + (
            "ERASER" if eraser else color_name
        ),
        (15, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AI Air Drawing Project",
        output
    )

    # ==========================================
    # 12. KEYBOARD CONTROLS
    # ==========================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('1'):
        color_name = "BLUE"
        eraser = False
        draw_color = colors[color_name]

    elif key == ord('2'):
        color_name = "GREEN"
        eraser = False
        draw_color = colors[color_name]

    elif key == ord('3'):
        color_name = "RED"
        eraser = False
        draw_color = colors[color_name]

    elif key == ord('4'):
        color_name = "YELLOW"
        eraser = False
        draw_color = colors[color_name]

    elif key == ord('5'):
        color_name = "PURPLE"
        eraser = False
        draw_color = colors[color_name]

    elif key == ord('6'):
        color_name = "WHITE"
        eraser = False
        draw_color = colors[color_name]

    elif key == ord('e'):
        eraser = True
        draw_color = (0, 0, 0)
        brush_size = max(brush_size, 20)

    elif key == ord('c'):
        canvas[:] = 0
        previous_point = None

    elif key == ord('s'):

        filename = (
            "air_drawing_"
            + str(int(time.time()))
            + ".png"
        )

        cv2.imwrite(filename, canvas)
        print("Drawing saved:", filename)

    elif key in (ord('+'), ord('=')):
        brush_size = min(brush_size + 2, 40)

    elif key == ord('-'):
        brush_size = max(brush_size - 2, 2)


# ==========================================
# 13. CLOSE APPLICATION
# ==========================================

cap.release()
hands.close()
cv2.destroyAllWindows()

print("AI Air Drawing closed successfully!")