import numpy as np
import cv2
import mediapipe as mp

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe Hands module for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam video stream
cap = cv2.VideoCapture(0)

# The output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize for faster detection
    frame = cv2.resize(frame, (640, 480))

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people in the image (HOG + SVM for human detection)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # Detect faces in the image (Haar Cascade)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Convert the BGR frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw rectangles around detected people (HOG-based)
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)  # Green rectangle for human body

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle for face

    # Draw landmarks around detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks for each detected hand
            for landmark in hand_landmarks.landmark:
                # Convert the landmark coordinates to pixel values
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                # Draw a small circle for each landmark
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red circle for landmarks

    # Write the output video
    out.write(frame.astype('uint8'))

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and output video
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
cv2.waitKey(1)
