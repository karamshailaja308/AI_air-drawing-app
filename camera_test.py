import cv2

# Open laptop webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("Camera started successfully!")

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to read camera")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Display camera
    cv2.imshow("AI Air Drawing - Camera Test", frame)

    # Press Q to close camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()