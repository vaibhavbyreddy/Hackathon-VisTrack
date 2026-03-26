import cv2
import socket
import time

# --- CONFIGURATION ---
CAMERA_ID = "CAM-01"
TARGET_IP = "127.0.0.1"  # Loopback for local testing
TARGET_PORT = 5005
MOTION_THRESHOLD = 25    # Sensitivity to pixel changes
MIN_AREA = 100           # Ignore tiny blobs (sensor noise)

# --- NETWORK SETUP ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- CAMERA SETUP ---
cap = cv2.VideoCapture(0) # 0 is usually your laptop's built-in webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(5)
ret, prev_frame = cap.read()
if not ret:
    print("Failed to grab initial frame. Is your webcam occupied?")
    exit()

# Convert to grayscale and blur to reduce noise
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

print(f"Starting motion tracking for {CAMERA_ID}...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 1. Compute absolute difference between frames
    frame_diff = cv2.absdiff(prev_gray, gray)

    # 2. Threshold to get a clean black/white mask
    _, thresh = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 3. Find contours (the blobs of motion)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignore tiny movements
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        # 4. Calculate the centroid (center of mass) of the blob
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 5. Construct and send the UDP packet
            # Format: CAMERA_ID, X, Y, AREA
            message = f"{CAMERA_ID},{cx},{cy},{int(cv2.contourArea(contour))}"
            sock.sendto(message.encode('utf-8'), (TARGET_IP, TARGET_PORT))

            # Draw a red dot on the video feed for local visual debugging
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Show the video (You will remove this line later when running headless on the Pi)
    cv2.imshow("Edge Node View", frame)

    # Update previous frame for the next loop
    prev_gray = gray

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
