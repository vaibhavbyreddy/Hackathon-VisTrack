import socket
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
PORT = 5005
FOV_H = np.radians(70)  # Assume 70 degree horizontal FOV for Arducam
W, H = 320, 240
RAY_LIFESPAN = 10.0  # seconds

# Calculate focal length in pixels
focal_length = (W / 2) / np.tan(FOV_H / 2)

# --- NETWORK ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', PORT))
sock.setblocking(False)  # Non-blocking so the plot doesn't freeze

# --- PLOT SETUP ---
plt.ion()  # Interactive mode on
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Data structure: list of tuples -> (timestamp, x, y, z)
active_rays = []

print("Listening for trajectory telemetry...")

try:
    while True:
        # 1. Drain the network buffer
        try:
            while True:
                data, addr = sock.recvfrom(1024)
                payload = data.decode('utf-8').split(',')
                node_id = payload[0]
                cx, cy = int(payload[1]), int(payload[2])

                # 2. Convert to 3D Vector
                vx = cx - (W / 2)
                vy = (H / 2) - cy  # Flip Y
                vz = focal_length

                # Normalize the vector to a standard length for visualization
                length = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
                ray_end = (vx / length * 10, vy / length * 10, vz / length * 10)  # 10 units long

                active_rays.append((time.time(), *ray_end))
        except BlockingIOError:
            pass  # No new packets right now, continue to plotting

        # 3. Purge old rays
        current_time = time.time()
        active_rays = [ray for ray in active_rays if current_time - ray[0] < RAY_LIFESPAN]

        # 4. Render the Scene
        ax.cla()

        # Draw Camera Base
        ax.scatter(0, 0, 0, color='black', marker='s', s=100, label="Camera Origin")

        # Draw Rays
        for ray in active_rays:
            ax.plot([0, ray[1]], [0, ray[2]], [0, ray[3]], color='cyan', alpha=0.6)
            ax.scatter(ray[1], ray[2], ray[3], color='red', s=10)  # Hit point marker

        # Lock the axes so the camera doesn't spasm
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([0, 15])
        ax.set_xlabel('X (Horizontal)')
        ax.set_ylabel('Y (Vertical)')
        ax.set_zlabel('Z (Boresight)')
        ax.set_title("Real-Time Telemetry Plotter")

        plt.pause(0.05)  # Allow matplotlib to refresh the GUI

except KeyboardInterrupt:
    print("Shutting down ground control...")
    plt.close()