import socket
import cv2
import time
import numpy as np
from picamera2 import Picamera2

# --- CONFIGURATION ---
NODE_ID = socket.gethostname()
TARGET_IP = "Vaibhavs-MacBook-Pro-2.local"
TARGET_PORT = 5005

WIDTH, HEIGHT = 320, 240
MOTION_THRESHOLD = 25
MIN_AREA = 50

# --- NETWORK ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"[{NODE_ID}] Booting targeting system...")

# --- CAMERA ---
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format": "YUV420", "size": (WIDTH, HEIGHT)})
picam2.configure(config)
picam2.start()

print(f"[{NODE_ID}] Sensor active. Extracting motion...")

prev_frame = None

try:
    while True:
        # 1. Capture the luminance (Y channel) for fast grayscale processing
        yuv_array = picam2.capture_array()
        gray = yuv_array[:HEIGHT, :] # Slice the Y-plane from the 2D array
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # 2. Extract Motion
        frame_diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 3. Find Coordinates
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest moving object
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > MIN_AREA:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # 4. Transmit Telemetry
                    packet = f"{NODE_ID},{cx},{cy}"
                    sock.sendto(packet.encode('utf-8'), (TARGET_IP, TARGET_PORT))

        prev_frame = gray
        time.sleep(0.05)  # ~20 FPS loop

except KeyboardInterrupt:
    print("Shutting down...")
finally:
    picam2.stop()
    sock.close()