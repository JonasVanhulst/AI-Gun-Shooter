import threading
import time
import cv2 as cv
import mediapipe as mp
import numpy as np
import serial as srl

SERIAL_PORT: str = 'COM12'
SERIAL_BAUD: int = 115200
BUFFER_LIMIT = 50  # Maximum commands in the message queue

tilt_correction = 0
rotation_correction = 0

# Shared variable for serial messages
message_queue = []

# Add this variable outside the loop
fired = False
center_threshold = 20  # Define a threshold for being "centered"


# Setup serial connection
try:
    ser = srl.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
except srl.SerialException as e:
    print(f"Error opening serial port: {e}")
    ser = None

# Serial worker function with buffer limit
def serial_worker():
    while True:
        if message_queue:
            try:
                message = message_queue.pop(0)
                ser.write(message.encode())
            except (srl.SerialException, AttributeError) as e:
                print(f"Serial write error: {e}")
        time.sleep(0.01)

def add_to_message_queue(command):
    if len(message_queue) < BUFFER_LIMIT:
        message_queue.append(command)
    else:
        message_queue.clear()
        print("Queue was full -> queue is now cleared")

# Start serial thread
serial_thread = threading.Thread(target=serial_worker, daemon=True)
serial_thread.start()

mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose

# Initialize webcam
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
screen_center_x = 640 // 2
screen_center_y = 480 // 2

count: int = 0
tilt: int = 0  # Used to keep track of the tilt, between 0 and 100
rotate: int = 0  # Used to keep track of the rotation, also between 0 and 100
frame_counter = 0

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector, \
        mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detector:
    start_time = time.time()
    
    add_to_message_queue("tilt 0\n")
    add_to_message_queue("rotate 0\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read a frame from the webcam.")
            #break

        frame_counter += 1
        

        # Convert BGR to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert RGB to Grayscale
        grayscale_image = cv.cvtColor(rgb_frame, cv.COLOR_RGB2GRAY)

        # Equalize the grayscale image (for better contrast)
        equalized_image = cv.equalizeHist(grayscale_image)

        # Apply Gaussian blur to the equalized image
        blurred_frame = cv.GaussianBlur(equalized_image, (5, 5), 0)

        # Convert the processed grayscale image back to RGB
        processed_rgb_frame = cv.cvtColor(blurred_frame, cv.COLOR_GRAY2RGB)

        # Use the processed RGB frame for face detection
        face_results = face_detector.process(processed_rgb_frame)


        #pose_results = pose_detector.process(grayscale_image)

        frame_height, frame_width, c = frame.shape

        # Flags to track detections
        face_detected = False

        # Draw face detection results
        if face_results.detections:
            face_detected = True
            for face in face_results.detections:
                face_react = np.multiply(
                    [
                        face.location_data.relative_bounding_box.xmin,
                        face.location_data.relative_bounding_box.ymin,
                        face.location_data.relative_bounding_box.width,
                        face.location_data.relative_bounding_box.height,
                    ],
                    [frame_width, frame_height, frame_width, frame_height]).astype(int)

                cv.rectangle(frame, face_react, color=(255, 255, 255), thickness=2)
                key_points = np.array([(p.x, p.y) for p in face.location_data.relative_keypoints])
                key_points_coords = np.multiply(key_points, [frame_width, frame_height]).astype(int)
                for p in key_points_coords:
                    cv.circle(frame, p, 4, (255, 255, 255), 2)
                    cv.circle(frame, p, 2, (0, 0, 0), -1)
                break

        # If no detection, adjust rotation
        if not face_detected:
            if frame_counter % 10 == 0:  # Send message every 10 frames
                rotate = (rotate + 10) % 100
                add_to_message_queue(f"rotate {rotate}\n")

        # Face correction logic
        if frame_counter % 30 == 0 and face_detected:
            face_center_x = face_react[0] + face_react[2] // 2
            face_center_y = face_react[1] + face_react[3] // 2

            # Calculate offsets
            offset_x = face_center_x - screen_center_x
            offset_y = face_center_y - screen_center_y

            dynamic_step_x = min(5, abs(offset_x) // 40)  # Adjust scaling factor (e.g., 40)
            dynamic_step_y = min(5, abs(offset_y) // 40)

            rotation_correction = 0
            tilt_correction = 0

            if offset_x >= 20 and rotate >= dynamic_step_x:
                rotation_correction = -dynamic_step_x
            elif offset_x < -20 and rotate <= 100 - dynamic_step_x:
                rotation_correction = dynamic_step_x

            if offset_y >= 20 and tilt >= dynamic_step_y:
                tilt_correction = -dynamic_step_y
            elif offset_y < -20 and tilt <= 100 - dynamic_step_y:
                tilt_correction = dynamic_step_y

            # Apply corrections
            # Apply corrections
            rotate = max(0, min(100, rotate + rotation_correction))
            tilt = max(0, min(100, tilt + tilt_correction))

            # Debugging output
            print(f"Offset Y: {offset_y}, Tilt Correction: {tilt_correction}, New Tilt: {tilt}")

            add_to_message_queue(f"tilt {tilt}\n")
            add_to_message_queue(f"rotate {rotate}\n")
            
            
            # Check if the face is centered
            if abs(offset_x) < center_threshold :   #and abs(offset_y) < center_threshold
                if not fired:
                    print("Face is centered. Firing!")
                    add_to_message_queue("fire 1\n")
                    fired = True
            else:
                # Reset the fired flag if the face moves out of the center
                fired = False
            
        # Display FPS
        fps = frame_counter / (time.time() - start_time)
        cv.putText(frame, f"FPS: {fps:.2f}", (30, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)

        # Show the frame
        cv.imshow("rgb_frame", rgb_frame)
        cv.imshow("grayscale_image", grayscale_image)
        #cv.imshow("equalized_image", equalized_image)
        cv.imshow("blurred_frame", blurred_frame)
        #cv.imshow("flipped_face", flipped_face)
        cv.imshow("frame", frame)
        
        
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    #cap.release()
    cv.destroyAllWindows()

