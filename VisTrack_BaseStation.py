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

FOV_H = np.radians(100)
W, H = 320, 240
FOCAL_LENGTH = (W / 2) / np.tan(FOV_H / 2)
ASSIGNED_COLORS = ['#00ffff', '#ff00ff', '#ffff00', '#00ff00']

# Cameras spaced 20cm apart for table testing
ORIGINS = {
    "vistrack-node-01": np.array([-0.2, 0.0, 0.0]),
    "vistrack-node-02": np.array([0.0, 0.0, 0.0]),
    "vistrack-node-03": np.array([0.2, 0.0, 0.0])
}


class GroundControlApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisTrack Swarm C2")
        self.geometry("1400x800")
        ctk.set_appearance_mode("dark")

        self.mode = "TRACK"
        self.swarm_data = {}  # Now stores 'ip' as well
        self.color_index = 0
        self.calib_end_time = 0
        self.intersection_threshold = 0.05  # 5cm threshold

        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(('0.0.0.0', TELEMETRY_PORT))
        self.sock_recv.setblocking(False)

        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.setup_ui()
        threading.Thread(target=self.telemetry_listener, daemon=True).start()
        self.update_gui_loop()

    def setup_ui(self):
        self.frame_c2 = ctk.CTkFrame(self, width=320)
        self.frame_c2.pack(side="left", fill="y", padx=20, pady=20)

        ctk.CTkLabel(self.frame_c2, text="VisTrack Uplink", font=("Roboto", 24, "bold")).pack(pady=10)

        self.btn_mode = ctk.CTkButton(self.frame_c2, text="MODE: TRACKING", command=self.toggle_mode,
                                      fg_color="#1f538d")
        self.btn_mode.pack(pady=10, fill="x", padx=20)

        self.lbl_timer = ctk.CTkLabel(self.frame_c2, text="", font=("Roboto", 18, "bold"), text_color="orange")
        self.lbl_timer.pack()

        ctk.CTkLabel(self.frame_c2, text="Motion Threshold", font=("Roboto", 12)).pack(pady=(10, 0))
        self.slider_thresh = ctk.CTkSlider(self.frame_c2, from_=5, to=100, command=self.send_threshold)
        self.slider_thresh.set(25)
        self.slider_thresh.pack(padx=20, pady=5, fill="x")

        ctk.CTkLabel(self.frame_c2, text="Intersection Threshold (m)", font=("Roboto", 12)).pack(pady=(10, 0))
        self.slider_intersect = ctk.CTkSlider(self.frame_c2, from_=0.01, to=0.20, command=self.set_intersect_thresh)
        self.slider_intersect.set(0.05)
        self.slider_intersect.pack(padx=20, pady=5, fill="x")

        ctk.CTkLabel(self.frame_c2, text="SWARM HEALTH", font=("Roboto", 16, "bold")).pack(pady=(20, 10))
        self.health_frame = ctk.CTkFrame(self.frame_c2, fg_color="transparent")
        self.health_frame.pack(fill="both", expand=True, padx=20)
        self.node_ui_elements = {}

        self.frame_plot = ctk.CTkFrame(self)
        self.frame_plot.pack(side="right", fill="both", expand=True, padx=(0, 20), pady=20)

        self.fig = plt.figure(figsize=(12, 6))
        self.fig.patch.set_facecolor('#2b2b2b')

        self.ax2d = self.fig.add_subplot(121)
        self.ax3d = self.fig.add_subplot(122, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def blast_command(self, cmd_str):
        """Bypasses the firewall by using the mDNS reflector for each node"""
        for node_id in self.swarm_data.keys():
            target_hostname = f"{node_id}.local"
            try:
                self.sock_send.sendto(cmd_str.encode('utf-8'), (target_hostname, UPLINK_PORT))
                print(f"Uplink sent to {target_hostname}: {cmd_str}")
            except Exception as e:
                print(f"Failed to route to {target_hostname}")

    def toggle_mode(self):
        if self.mode == "TRACK":
            self.mode = "CALIBRATE"
            self.calib_end_time = time.time() + 30.0
            self.btn_mode.configure(text="MODE: CALIBRATING (ARUCO)", fg_color="#cc7000")
        else:
            self.mode = "TRACK"
            self.lbl_timer.configure(text="")
            self.btn_mode.configure(text="MODE: TRACKING (MOTION)", fg_color="#1f538d")

        self.blast_command(f"CMD:MODE:{self.mode}")

    def send_threshold(self, value):
        self.blast_command(f"CMD:THRESH:{int(value)}")

    def set_intersect_thresh(self, value):
        self.intersection_threshold = float(value)

    def calculate_intersection(self, p1, d1, p2, d2):
        cross_d = np.cross(d1, d2)
        denom = np.linalg.norm(cross_d) ** 2
        if denom < 1e-6:
            return None, float('inf')

        t1 = np.dot(np.cross(p2 - p1, d2), cross_d) / denom
        t2 = np.dot(np.cross(p2 - p1, d1), cross_d) / denom

        pt1 = p1 + t1 * d1
        pt2 = p2 + t2 * d2
        return (pt1 + pt2) / 2, np.linalg.norm(pt1 - pt2)

    def telemetry_listener(self):
        while True:
            try:
                # Harvest IP address from the addr tuple
                data, addr = self.sock_recv.recvfrom(1024)
                node_ip = addr[0]
                payload = data.decode('utf-8').split(',')

                if len(payload) == 5:
                    node_id, status, cx, cy, fps = payload
                    if node_id not in self.swarm_data:
                        color = ASSIGNED_COLORS[self.color_index % len(ASSIGNED_COLORS)]
                        self.color_index += 1
                        self.swarm_data[node_id] = {'color': color}

                    self.swarm_data[node_id].update({
                        'ip': node_ip,
                        'last_seen': time.time(),
                        'status': status,
                        'cx': int(cx),
                        'cy': int(cy),
                        'fps': float(fps)
                    })
            except BlockingIOError:
                pass
            time.sleep(0.005)

    def update_gui_loop(self):
        current_time = time.time()

        if self.mode == "CALIBRATE":
            remaining = int(self.calib_end_time - current_time)
            if remaining > 0:
                self.lbl_timer.configure(text=f"WAVE MARKER: {remaining}s")
            else:
                self.toggle_mode()

        for node_id, data in list(self.swarm_data.items()):
            if node_id not in self.node_ui_elements:
                lbl = ctk.CTkLabel(self.health_frame, text="", font=("Roboto", 14, "bold"))
                lbl.pack(anchor="w", pady=2)
                self.node_ui_elements[node_id] = lbl

            if current_time - data['last_seen'] > 1.0:
                self.node_ui_elements[node_id].configure(text=f"🔴 {node_id} (OFF)", text_color="gray")
            elif data['status'] == "LOCKED":
                self.node_ui_elements[node_id].configure(text=f"🟢 {node_id} [{data['fps']} FPS]",
                                                         text_color=data['color'])
            else:
                self.node_ui_elements[node_id].configure(text=f"🟡 {node_id} [{data['fps']} FPS]", text_color="yellow")

        self.ax2d.cla()
        self.ax3d.cla()

        self.ax2d.set_xlim([0, W])
        self.ax2d.set_ylim([H, 0])
        self.ax2d.set_facecolor('#1e1e1e')
        self.ax2d.set_title("2D Sensor Planes", color='white')

        active_rays = {}

        for node_id, data in self.swarm_data.items():
            if current_time - data['last_seen'] > 1.0:
                continue

            cx, cy, color = data['cx'], data['cy'], data['color']
            origin = ORIGINS.get(node_id, np.array([0.0, 0.0, 0.0]))
            self.ax3d.scatter(*origin, color=color, marker='s', s=50)

            if cx != -1 and cy != -1:
                self.ax2d.scatter(cx, cy, color=color, s=100, edgecolors='white')

                vx, vy, vz = cx - (W / 2), (H / 2) - cy, FOCAL_LENGTH
                direction = np.array([vx, vy, vz])
                direction = direction / np.linalg.norm(direction)
                active_rays[node_id] = (origin, direction, color)

                ray_end = origin + direction * 1.5  # 1.5 meters long
                self.ax3d.plot([origin[0], ray_end[0]], [origin[1], ray_end[1]], [origin[2], ray_end[2]], color=color,
                               alpha=0.5)

        ray_keys = list(active_rays.keys())
        for i in range(len(ray_keys)):
            for j in range(i + 1, len(ray_keys)):
                o1, d1, _ = active_rays[ray_keys[i]]
                o2, d2, _ = active_rays[ray_keys[j]]

                midpoint, distance = self.calculate_intersection(o1, d1, o2, d2)

                # Check intersection distance AND verify it's not sitting right on the camera lenses (< 10cm)
                if midpoint is not None and distance < self.intersection_threshold:
                    if np.linalg.norm(midpoint) > 0.1:
                        self.ax3d.scatter(*midpoint, color='red', marker='*', s=300)
                        self.ax3d.text(midpoint[0], midpoint[1], midpoint[2] + 0.1, f"TARGET\nErr:{distance:.2f}m",
                                       color='red')

        # 1 Meter Cube
        self.ax3d.set_xlim([-1.0, 1.0])
        self.ax3d.set_ylim([-1.0, 1.0])
        self.ax3d.set_zlim([0, 1.0])
        self.ax3d.set_facecolor('#2b2b2b')
        self.ax3d.set_title("3D Airspace (1m)", color='white')
        self.ax3d.tick_params(colors='white')

        self.canvas.draw_idle()
        self.after(50, self.update_gui_loop)


if __name__ == "__main__":
    app = GroundControlApp()
    app.mainloop()