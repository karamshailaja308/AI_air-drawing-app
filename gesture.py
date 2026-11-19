import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def finger_up(hand, tip, pip, mcp):
    # Compare finger tip and base joint positions
    return (
        hand.landmark[tip].y < hand.landmark[pip].y
        and hand.landmark[pip].y < hand.landmark[mcp].y
    )

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "NO HAND"

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        index = finger_up(hand, 8, 6, 5)
        middle = finger_up(hand, 12, 10, 9)
        ring = finger_up(hand, 16, 14, 13)
        pinky = finger_up(hand, 20, 18, 17)

        if index and not middle and not ring and not pinky:
            gesture = "DRAW"

        elif index and middle and not ring and not pinky:
            gesture = "STOP DRAWING"

        elif not index and not middle and not ring and not pinky:
            gesture = "FIST"

        else:
            gesture = "OTHER GESTURE"

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        debug = (
            f"I:{int(index)} M:{int(middle)} "
            f"R:{int(ring)} P:{int(pinky)}"
        )

        cv2.putText(
            frame, debug, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )

    cv2.putText(
        frame, gesture, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 3
    )

    cv2.imshow("Gesture Testing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()