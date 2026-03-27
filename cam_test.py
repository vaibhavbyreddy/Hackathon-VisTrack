import socket
import cv2
import numpy as np

LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))

print(f"Listening for Node video streams on port {LISTEN_PORT}...")
print("Press 'q' to quit.")

while True:
    try:
        # 65536 is the maximum possible size for a UDP packet
        data, addr = sock.recvfrom(65536)

        # 1. Convert the received bytes back into a numpy array
        np_data = np.frombuffer(data, dtype=np.uint8)

        # 2. Decode the JPEG back into a standard OpenCV image
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        # 3. Display the video!
        if frame is not None:
            # Scale it up x2 so it's easier to see on your Mac screen
            frame_large = cv2.resize(frame, (640, 480))
            cv2.imshow("Node Camera Smoke Test", frame_large)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        # If a packet drops or is corrupted, just ignore it and catch the next one
        pass

cv2.destroyAllWindows()
sock.close()