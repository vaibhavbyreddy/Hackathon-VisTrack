import socket
import cv2
import time
import threading
import numpy as np
from picamera2 import Picamera2

# --- CONFIGURATION ---
NODE_ID = socket.gethostname()
TARGET_IP = "Vaibhavs-MacBook-Pro-2.local"
TELEMETRY_PORT = 5005
UPLINK_PORT = 5006

WIDTH, HEIGHT = 320, 240
MIN_AREA = 50

# --- SHARED STATE (Thread-Safe Variables) ---
current_mode = "TRACK"
motion_threshold = 25
target_cx = -1
target_cy = -1
node_status = "SEARCHING"


# --- THREAD 1: UPLINK LISTENER (Command & Control) ---
def uplink_listener():
    global current_mode, motion_threshold
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', UPLINK_PORT))

    print(f"[{NODE_ID}] C2 Uplink active. Listening for commands...")
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            cmd = data.decode('utf-8').strip().split(':')

            # Parse Commands: CMD:MODE:TRACK or CMD:THRESH:40
            if len(cmd) == 3 and cmd[0] == "CMD":
                if cmd[1] == "MODE":
                    current_mode = cmd[2]
                    print(f"[{NODE_ID}] Switched to {current_mode} mode")
                elif cmd[1] == "THRESH":
                    motion_threshold = int(cmd[2])
                    print(f"[{NODE_ID}] Motion Threshold updated to {motion_threshold}")
        except Exception as e:
            pass


# --- THREAD 2: TELEMETRY DOWNLINK (10 Hz Heartbeat) ---
def telemetry_downlink():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    print(f"[{NODE_ID}] Telemetry Transponder active.")
    while True:
        # Format: NODE_ID,STATUS,CX,CY
        packet = f"{NODE_ID},{node_status},{target_cx},{target_cy}"
        sock.sendto(packet.encode('utf-8'), (TARGET_IP, TELEMETRY_PORT))
        time.sleep(0.1)  # 10 Hz cycle


# --- START BACKGROUND THREADS ---
threading.Thread(target=uplink_listener, daemon=True).start()
threading.Thread(target=telemetry_downlink, daemon=True).start()

# --- ARUCO SETUP ---
# Standardize on 4x4 ArUco dict.
try:
    # Older OpenCV 4.x syntax
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
except AttributeError:
    # Newer OpenCV 4.7+ syntax (Bookworm usually ships with this)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# --- CAMERA SETUP ---
print(f"[{NODE_ID}] Initializing sensor matrix...")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format": "YUV420", "size": (WIDTH, HEIGHT)})
picam2.configure(config)
picam2.start()

prev_frame = None

print(f"[{NODE_ID}] Vision Pipeline running. Ready for flight.")

try:
    while True:
        # 1. Pull the raw luminance array for ultra-fast grayscale processing
        yuv_array = picam2.capture_array()
        gray = yuv_array[:HEIGHT, :]

        # Default to nothing found
        cx, cy = -1, -1
        current_status = "SEARCHING"

        # --- BRANCH A: MOTION TRACKING ---
        if current_mode == "TRACK":
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            if prev_frame is None:
                prev_frame = blurred
                continue

            frame_diff = cv2.absdiff(prev_frame, blurred)
            _, thresh = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > MIN_AREA:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        current_status = "LOCKED"

            prev_frame = blurred

        # --- BRANCH B: ARUCO CALIBRATION ---
        elif current_mode == "CALIBRATE":
            # Bookworm OS API compatibility check
            if 'aruco_detector' in locals():
                corners, ids, rejected = aruco_detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

            if ids is not None and 0 in ids:
                # Find the index of Marker ID 0
                idx = np.where(ids == 0)[0][0]
                marker_corners = corners[idx][0]

                # Calculate the exact center of the 4 corners
                cx = int(np.mean(marker_corners[:, 0]))
                cy = int(np.mean(marker_corners[:, 1]))
                current_status = "LOCKED"

            # We don't need prev_frame for ArUco, but we keep it updated so motion tracking doesn't glitch when we switch back
            prev_frame = cv2.GaussianBlur(gray, (21, 21), 0)

            # 3. Update the global state for the Telemetry thread to transmit
        target_cx, target_cy = cx, cy
        node_status = current_status

        # Keep the main thread from burning 100% CPU
        time.sleep(0.02)

except KeyboardInterrupt:
    print(f"[{NODE_ID}] Shutting down...")
finally:
    picam2.stop()