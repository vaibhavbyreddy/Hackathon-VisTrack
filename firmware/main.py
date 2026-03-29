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

# --- SHARED STATE ---
current_mode = "TRACK"
motion_threshold = 25
target_cx, target_cy = -1, -1
node_status = "SEARCHING"
current_fps = 0.0


# --- THREAD 1: UPLINK LISTENER ---
def uplink_listener():
    global current_mode, motion_threshold
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', UPLINK_PORT))
    print(f"[{NODE_ID}] Command Listener Active on Port {UPLINK_PORT}")

    while True:
        try:
            data, _ = sock.recvfrom(1024)
            cmd = data.decode('utf-8').strip().split(':')
            if len(cmd) == 3 and cmd[0] == "CMD":
                if cmd[1] == "MODE":
                    current_mode = cmd[2]
                    print(f"\n>>> [{NODE_ID}] Switched to {current_mode} mode <<<\n")
                elif cmd[1] == "THRESH":
                    motion_threshold = int(cmd[2])
                    print(f"\n>>> [{NODE_ID}] Threshold updated to {motion_threshold} <<<\n")
        except Exception:
            pass


# --- THREAD 2: TELEMETRY DOWNLINK ---
def telemetry_downlink():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        packet = f"{NODE_ID},{node_status},{target_cx},{target_cy},{current_fps:.1f}"
        try:
            sock.sendto(packet.encode('utf-8'), (TARGET_IP, TELEMETRY_PORT))
        except Exception:
            pass
        time.sleep(0.1)


threading.Thread(target=uplink_listener, daemon=True).start()
threading.Thread(target=telemetry_downlink, daemon=True).start()

# --- ARUCO SETUP ---
try:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
except AttributeError:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# --- CAMERA SETUP ---
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format": "YUV420", "size": (WIDTH, HEIGHT)})
picam2.configure(config)
picam2.start()

prev_frame = None
last_time = time.time()
frame_count = 0

try:
    while True:
        curr_time = time.time()
        frame_count += 1
        if curr_time - last_time >= 1.0:
            current_fps = frame_count / (curr_time - last_time)
            frame_count = 0
            last_time = curr_time

        yuv_array = picam2.capture_array()
        gray = yuv_array[:HEIGHT, :]

        cx, cy = -1, -1
        current_status = "SEARCHING"

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
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        current_status = "LOCKED"
            prev_frame = blurred

        elif current_mode == "CALIBRATE":
            if 'aruco_detector' in locals():
                corners, ids, _ = aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

            if ids is not None and 0 in ids:
                idx = np.where(ids == 0)[0][0]
                marker_corners = corners[idx][0]
                cx, cy = int(np.mean(marker_corners[:, 0])), int(np.mean(marker_corners[:, 1]))
                current_status = "LOCKED"
            prev_frame = cv2.GaussianBlur(gray, (21, 21), 0)

        target_cx, target_cy = cx, cy
        node_status = current_status
        time.sleep(0.01)

except KeyboardInterrupt:
    picam2.stop()