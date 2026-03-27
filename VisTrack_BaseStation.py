import customtkinter as ctk
import socket
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
HEARTBEAT_PORT = 5006
TELEMETRY_PORT = 5005
TARGET_BCAST = "192.168.50.255"  # Where Mac blasts the sync pulse

FOV_H = np.radians(70)
W, H = 320, 240
FOCAL_LENGTH = (W / 2) / np.tan(FOV_H / 2)
POINTS_NEEDED = 50  # How many paired frames we need for calibration


class GroundControlApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisTrack Swarm Ground Control")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State Variables
        self.mode = "IDLE"  # IDLE -> PREPPING -> COLLECTING -> TRACKING
        self.active_nodes = {}  # {node_id: last_seen_timestamp}
        self.current_rays = {}  # {node_id: (vx, vy, vz)}
        self.paired_points = []  # Holds the data for the heavy math later
        self.temp_sync_buffer = {}  # Groups incoming points by timestamp

        # Networking Setup
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(('0.0.0.0', TELEMETRY_PORT))
        self.sock_recv.setblocking(False)

        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self.setup_ui()

        # Start Background Threads
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        threading.Thread(target=self.telemetry_listener, daemon=True).start()

        # Start GUI Update Loop
        self.update_gui_loop()

    def setup_ui(self):
        # Left Panel: Command & Control
        self.frame_c2 = ctk.CTkFrame(self, width=300)
        self.frame_c2.pack(side="left", fill="y", padx=20, pady=20)

        self.lbl_title = ctk.CTkLabel(self.frame_c2, text="VisTrack C2", font=("Roboto", 24, "bold"))
        self.lbl_title.pack(pady=20)

        self.lbl_status = ctk.CTkLabel(self.frame_c2, text="STATUS: IDLE", text_color="gray", font=("Roboto", 16))
        self.lbl_status.pack(pady=10)

        self.btn_action = ctk.CTkButton(self.frame_c2, text="INITIATE CALIBRATION", command=self.btn_click, height=50)
        self.btn_action.pack(pady=20, fill="x", padx=20)

        self.lbl_progress = ctk.CTkLabel(self.frame_c2, text="Paired Frames: 0 / 50")
        self.lbl_progress.pack(pady=5)
        self.progress_bar = ctk.CTkProgressBar(self.frame_c2)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=20, pady=5)

        # Telemetry Indicators
        self.lbl_telemetry = ctk.CTkLabel(self.frame_c2, text="NODE VISUAL LOCKS", font=("Roboto", 14, "bold"))
        self.lbl_telemetry.pack(pady=(30, 10))
        self.node_labels = {}

        # Right Panel: 3D Matplotlib Render
        self.frame_plot = ctk.CTkFrame(self)
        self.frame_plot.pack(side="right", fill="both", expand=True, padx=(0, 20), pady=20)

        self.fig = plt.figure(figsize=(8, 8))
        self.fig.patch.set_facecolor('#2b2b2b')  # Match dark mode
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#2b2b2b')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def btn_click(self):
        if self.mode == "IDLE":
            self.mode = "PREPPING"
            self.lbl_status.configure(text="STATUS: WAITING FOR MARKER", text_color="orange")
            self.btn_action.configure(text="START WAVING", fg_color="orange", hover_color="#cc7000")

        elif self.mode == "PREPPING":
            self.mode = "COLLECTING"
            self.paired_points.clear()
            self.lbl_status.configure(text="STATUS: COLLECTING DATA", text_color="cyan")
            self.btn_action.configure(state="disabled", text="CALIBRATING...")

    def heartbeat_loop(self):
        """Blasts a sync pulse to the swarm 10 times a second."""
        while True:
            if self.mode in ["COLLECTING", "TRACKING"]:
                sync_timestamp = str(time.time())
                packet = f"SYNC,{sync_timestamp}"
                self.sock_send.sendto(packet.encode('utf-8'), (TARGET_BCAST, HEARTBEAT_PORT))
            time.sleep(0.1)

    def telemetry_listener(self):
        """Listens for returning targeting data from the nodes."""
        while True:
            try:
                data, addr = self.sock_recv.recvfrom(1024)
                payload = data.decode('utf-8').split(',')
                # Expected format: NODE_ID, TIMESTAMP, CX, CY
                if len(payload) == 4:
                    node_id, timestamp, cx, cy = payload
                    cx, cy = int(cx), int(cy)

                    self.active_nodes[node_id] = time.time()

                    # 3D Ray Math (Assuming camera at origin looking UP)
                    vx = cx - (W / 2)
                    vy = (H / 2) - cy
                    vz = FOCAL_LENGTH

                    # Normalize for plotting
                    length = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
                    self.current_rays[node_id] = (vx / length * 10, vy / length * 10, vz / length * 10)

                    # Group by exact timestamp for stereo calibration
                    if self.mode == "COLLECTING":
                        if timestamp not in self.temp_sync_buffer:
                            self.temp_sync_buffer[timestamp] = {}
                        self.temp_sync_buffer[timestamp][node_id] = (cx, cy)

                        # If we have 2 nodes responding to the SAME sync pulse
                        if len(self.temp_sync_buffer[timestamp]) >= 2:
                            self.paired_points.append(self.temp_sync_buffer[timestamp])
                            del self.temp_sync_buffer[timestamp]  # Clear it out

            except BlockingIOError:
                pass
            time.sleep(0.005)

    def update_gui_loop(self):
        """Updates the Tkinter UI and Matplotlib canvas smoothly."""
        current_time = time.time()

        # Update Node Visual Lock Status
        for node_id, last_seen in list(self.active_nodes.items()):
            if node_id not in self.node_labels:
                lbl = ctk.CTkLabel(self.frame_c2, text=f"{node_id}: OFFLINE", text_color="red")
                lbl.pack(pady=2)
                self.node_labels[node_id] = lbl

            # If heard from in the last 0.5 seconds, it has a lock
            if current_time - last_seen < 0.5:
                self.node_labels[node_id].configure(text=f"{node_id}: VISUAL LOCK", text_color="#00ff00")
            else:
                self.node_labels[node_id].configure(text=f"{node_id}: SEARCHING...", text_color="red")
                if node_id in self.current_rays:
                    del self.current_rays[node_id]  # Erase ray if lost

        # Update Calibration Progress
        if self.mode == "COLLECTING":
            progress = len(self.paired_points) / POINTS_NEEDED
            self.progress_bar.set(progress)
            self.lbl_progress.configure(text=f"Paired Frames: {len(self.paired_points)} / {POINTS_NEEDED}")

            if len(self.paired_points) >= POINTS_NEEDED:
                self.mode = "TRACKING"
                self.lbl_status.configure(text="STATUS: CALIBRATED (MOCKED)", text_color="#00ff00")
                self.btn_action.configure(state="normal", text="SYSTEM ARMED", fg_color="green")
                print("DATA COLLECTION COMPLETE. READY FOR ESSENTIAL MATRIX MATH.")

        # Render 3D Rays
        self.ax.cla()

        # Hardcoded origins for now (Assuming they are sitting near each other)
        mock_origins = {
            "vistrack-node-01": np.array([0, 0, 0]),
            "vistrack-node-02": np.array([2, 0, 0])  # Assume node 2 is 2 units to the right
        }

        for node_id, ray in self.current_rays.items():
            origin = mock_origins.get(node_id, np.array([0, 0, 0]))
            self.ax.scatter(*origin, color='white', marker='s', s=50)  # Draw camera
            self.ax.plot([origin[0], origin[0] + ray[0]],
                         [origin[1], origin[1] + ray[1]],
                         [origin[2], origin[2] + ray[2]], color='cyan', linewidth=2)

        # Plot formatting
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([0, 15])
        self.ax.set_axis_off()  # Hide grid for clean hacker look

        self.canvas.draw_idle()

        # Loop at ~20 FPS
        self.after(50, self.update_gui_loop)


if __name__ == "__main__":
    app = GroundControlApp()
    app.mainloop()