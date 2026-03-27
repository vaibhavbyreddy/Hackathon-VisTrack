import customtkinter as ctk
import socket
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- CONFIGURATION ---
TELEMETRY_PORT = 5005
UPLINK_PORT = 5006
TARGET_BCAST = "192.168.50.255"

# Camera Math
FOV_H = np.radians(70)
W, H = 320, 240
FOCAL_LENGTH = (W / 2) / np.tan(FOV_H / 2)

# Swarm Management
ASSIGNED_COLORS = ['#00ffff', '#ff00ff', '#ffff00', '#00ff00']  # Cyan, Magenta, Yellow, Lime


class GroundControlApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisTrack Swarm C2")
        self.geometry("1400x800")
        ctk.set_appearance_mode("dark")

        # State Data
        self.mode = "TRACK"  # "TRACK" or "CALIBRATE"
        self.swarm_data = {}  # {node_id: {'last_seen': time, 'status': str, 'cx': int, 'cy': int, 'color': str}}
        self.color_index = 0

        # Networking
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(('0.0.0.0', TELEMETRY_PORT))
        self.sock_recv.setblocking(False)

        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self.setup_ui()

        # Start Threads
        threading.Thread(target=self.telemetry_listener, daemon=True).start()
        self.update_gui_loop()

    def setup_ui(self):
        # --- LEFT PANEL: UPLINK & STATUS ---
        self.frame_c2 = ctk.CTkFrame(self, width=300)
        self.frame_c2.pack(side="left", fill="y", padx=20, pady=20)

        ctk.CTkLabel(self.frame_c2, text="VisTrack Uplink", font=("Roboto", 24, "bold")).pack(pady=20)

        # Mode Controls
        self.btn_mode = ctk.CTkButton(self.frame_c2, text="MODE: TRACKING", command=self.toggle_mode,
                                      fg_color="#1f538d")
        self.btn_mode.pack(pady=10, fill="x", padx=20)

        # Live Sliders (Uplink Commands)
        ctk.CTkLabel(self.frame_c2, text="Motion Threshold", font=("Roboto", 12)).pack(pady=(20, 0))
        self.slider_thresh = ctk.CTkSlider(self.frame_c2, from_=5, to=100, command=self.send_threshold)
        self.slider_thresh.set(25)
        self.slider_thresh.pack(padx=20, pady=5, fill="x")

        # Swarm Health (Traffic Lights)
        ctk.CTkLabel(self.frame_c2, text="SWARM HEALTH", font=("Roboto", 16, "bold")).pack(pady=(40, 10))
        self.health_frame = ctk.CTkFrame(self.frame_c2, fg_color="transparent")
        self.health_frame.pack(fill="both", expand=True, padx=20)
        self.node_ui_elements = {}

        # --- RIGHT PANEL: PLOTS ---
        self.frame_plot = ctk.CTkFrame(self)
        self.frame_plot.pack(side="right", fill="both", expand=True, padx=(0, 20), pady=20)

        # Create Matplotlib Figure with 2 subplots (1 row, 2 columns)
        self.fig = plt.figure(figsize=(12, 6))
        self.fig.patch.set_facecolor('#2b2b2b')

        # 2D Sensor Plot
        self.ax2d = self.fig.add_subplot(121)
        self.ax2d.set_facecolor('#1e1e1e')
        self.ax2d.set_title("2D Sensor Planes", color='white')

        # 3D Airspace Plot
        self.ax3d = self.fig.add_subplot(122, projection='3d')
        self.ax3d.set_facecolor('#2b2b2b')
        self.ax3d.set_title("3D Airspace", color='white')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- COMMAND ACTIONS ---
    def toggle_mode(self):
        if self.mode == "TRACK":
            self.mode = "CALIBRATE"
            self.btn_mode.configure(text="MODE: CALIBRATING (ARUCO)", fg_color="#cc7000")
        else:
            self.mode = "TRACK"
            self.btn_mode.configure(text="MODE: TRACKING (MOTION)", fg_color="#1f538d")

        # Blast mode change to swarm
        self.sock_send.sendto(f"CMD:MODE:{self.mode}".encode('utf-8'), (TARGET_BCAST, UPLINK_PORT))

    def send_threshold(self, value):
        # Blast new threshold to swarm dynamically
        self.sock_send.sendto(f"CMD:THRESH:{int(value)}".encode('utf-8'), (TARGET_BCAST, UPLINK_PORT))

    # --- NETWORK LISTENER ---
    def telemetry_listener(self):
        """Expected packet: NODE_ID,STATUS,CX,CY"""
        while True:
            try:
                data, addr = self.sock_recv.recvfrom(1024)
                payload = data.decode('utf-8').split(',')

                if len(payload) == 4:
                    node_id, status, cx, cy = payload

                    # Assign a permanent color to new nodes
                    if node_id not in self.swarm_data:
                        color = ASSIGNED_COLORS[self.color_index % len(ASSIGNED_COLORS)]
                        self.color_index += 1
                        self.swarm_data[node_id] = {'color': color}

                    # Update node state
                    self.swarm_data[node_id].update({
                        'last_seen': time.time(),
                        'status': status,
                        'cx': int(cx),
                        'cy': int(cy)
                    })

            except BlockingIOError:
                pass
            time.sleep(0.005)

    # --- MAIN RENDER LOOP ---
    def update_gui_loop(self):
        current_time = time.time()

        # 1. Update Traffic Lights
        for node_id, data in list(self.swarm_data.items()):
            # Create UI element if it doesn't exist
            if node_id not in self.node_ui_elements:
                lbl = ctk.CTkLabel(self.health_frame, text=node_id, text_color=data['color'],
                                   font=("Roboto", 14, "bold"))
                lbl.pack(anchor="w", pady=2)
                self.node_ui_elements[node_id] = lbl

            # Determine Health (Timeout after 1 second of silence)
            if current_time - data['last_seen'] > 1.0:
                self.node_ui_elements[node_id].configure(text=f"🔴 {node_id} (OFFLINE)", text_color="gray")
            elif data['status'] == "LOCKED":
                self.node_ui_elements[node_id].configure(text=f"🟢 {node_id} (LOCKED)", text_color=data['color'])
            else:
                self.node_ui_elements[node_id].configure(text=f"🟡 {node_id} (SEARCHING)", text_color="yellow")

        # 2. Render Plots
        self.ax2d.cla()
        self.ax3d.cla()

        # Format 2D Camera Frame limits (320x240)
        self.ax2d.set_xlim([0, W])
        self.ax2d.set_ylim([H, 0])  # Invert Y so top of image is top of plot
        self.ax2d.set_facecolor('#1e1e1e')
        self.ax2d.tick_params(colors='white')

        # Mock origins for 3D plot to space out the cameras visually
        origins = {
            "vistrack-node-01": np.array([-2, 0, 0]),
            "vistrack-node-02": np.array([0, 0, 0]),
            "vistrack-node-03": np.array([2, 0, 0])
        }

        for node_id, data in self.swarm_data.items():
            if current_time - data['last_seen'] > 1.0:
                continue  # Skip dead nodes

            cx, cy = data['cx'], data['cy']
            color = data['color']

            # If camera actually sees a target (not -1, -1)
            if cx != -1 and cy != -1:
                # Draw on 2D Plot
                self.ax2d.scatter(cx, cy, color=color, s=100, edgecolors='white')
                self.ax2d.text(cx + 10, cy, node_id, color='white', fontsize=8)

                # Draw on 3D Plot
                origin = origins.get(node_id, np.array([0, 0, 0]))
                vx = cx - (W / 2)
                vy = (H / 2) - cy
                vz = FOCAL_LENGTH

                # Normalize length for display
                length = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
                ray = (vx / length * 10, vy / length * 10, vz / length * 10)

                self.ax3d.scatter(*origin, color=color, marker='s', s=50)  # Camera lens
                self.ax3d.plot([origin[0], origin[0] + ray[0]],
                               [origin[1], origin[1] + ray[1]],
                               [origin[2], origin[2] + ray[2]], color=color, linewidth=2, alpha=0.7)

        # Format 3D Airspace limits
        self.ax3d.set_xlim([-10, 10])
        self.ax3d.set_ylim([-10, 10])
        self.ax3d.set_zlim([0, 15])
        self.ax3d.set_facecolor('#2b2b2b')
        self.ax3d.tick_params(colors='white')
        self.ax3d.xaxis.pane.fill = False
        self.ax3d.yaxis.pane.fill = False
        self.ax3d.zaxis.pane.fill = False

        self.canvas.draw_idle()

        # Loop at ~15 FPS
        self.after(60, self.update_gui_loop)


if __name__ == "__main__":
    app = GroundControlApp()
    app.mainloop()