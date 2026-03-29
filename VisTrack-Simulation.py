import customtkinter as ctk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os

# --- CONFIGURATION ---
VIDEO_DIR = os.path.expanduser("~/Desktop/vistrack_sim/")
W, H = 400, 250
FOV_H = np.radians(100)
FOCAL_LENGTH = (W / 2) / np.tan(FOV_H / 2)

# Set back to positive 45. The corrected math handles the "upward" tilt properly now.
PITCH_ANGLE = np.radians(45)

ORIGINS = [
    np.array([-2.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([2.0, 0.0, 0.0])
]

COLORS = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
HEX_COLORS = ['#ff0000', '#00ff00', '#00ffff', '#ff00ff', '#ffff00']


class VisTrackDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisTrack Central Processor")
        self.geometry("1600x900")
        ctk.set_appearance_mode("dark")

        self.frame_count = 0
        self.history = []  # Stores historical 3D positions for the trajectory trail

        # --- VIDEO SETUP ---
        self.caps = []
        self.prev_grays = [None] * 5
        for i in range(5):
            path = os.path.join(VIDEO_DIR, f"VisTrack-Node-0{i + 1}.mp4")
            self.caps.append(cv2.VideoCapture(path))

        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        self.frame_videos = ctk.CTkFrame(self, width=450)
        self.frame_videos.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.frame_videos, text="LIVE SENSOR FEEDS", font=("Roboto", 18, "bold")).pack(pady=10)

        self.video_labels = []
        for i in range(5):
            lbl = ctk.CTkLabel(self.frame_videos, text=f"Loading Node {i + 1}...")
            lbl.pack(pady=5)
            self.video_labels.append(lbl)

        self.frame_plot = ctk.CTkFrame(self)
        self.frame_plot.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig = plt.figure(figsize=(10, 8))
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax3d = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def calc_intersection(self, p1, d1, p2, d2):
        cross_d = np.cross(d1, d2)
        denom = np.linalg.norm(cross_d) ** 2
        if denom < 1e-6:
            return None, float('inf')

        t1 = np.dot(np.cross(p2 - p1, d2), cross_d) / denom
        t2 = np.dot(np.cross(p2 - p1, d1), cross_d) / denom

        pt1 = p1 + t1 * d1
        pt2 = p2 + t2 * d2
        return (pt1 + pt2) / 2, np.linalg.norm(pt1 - pt2)

    def update_loop(self):
        self.frame_count += 1
        active_rays = []

        for i, cap in enumerate(self.caps):
            ret, frame = cap.read()

            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.video_labels[i].configure(text="FEED OFFLINE")
                    continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            if self.prev_grays[i] is None:
                self.prev_grays[i] = blurred
                continue

            frame_diff = cv2.absdiff(self.prev_grays[i], blurred)
            _, thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 5:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                        cv2.putText(frame, "TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # CORRECTED RAY MATH
                        v_right = cx - (W / 2)
                        v_up = (H / 2) - cy  # Flipped here so positive is physically "up" in the sky
                        v_fwd = FOCAL_LENGTH

                        # Apply 3D Rotation (pitching up)
                        world_x = v_right
                        world_y = v_fwd * np.cos(PITCH_ANGLE) - v_up * np.sin(PITCH_ANGLE)
                        world_z = v_fwd * np.sin(PITCH_ANGLE) + v_up * np.cos(PITCH_ANGLE)

                        direction = np.array([world_x, world_y, world_z])
                        direction = direction / np.linalg.norm(direction)
                        active_rays.append((ORIGINS[i], direction, HEX_COLORS[i]))

            self.prev_grays[i] = blurred

            frame_resized = cv2.resize(frame, (280, 175))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self.video_labels[i].configure(image=img, text="")
            self.video_labels[i].image = img

        if self.frame_count % 3 == 0:
            self.ax3d.cla()

            for i, origin in enumerate(ORIGINS):
                self.ax3d.scatter(*origin, color=HEX_COLORS[i], marker='s', s=100)

            ray_pts = []
            for origin, direction, color in active_rays:
                ray_end = origin + direction * 400
                self.ax3d.plot([origin[0], ray_end[0]], [origin[1], ray_end[1]], [origin[2], ray_end[2]], color=color,
                               alpha=0.3)
                ray_pts.append((origin, direction))

            valid_intersections = []
            for i in range(len(ray_pts)):
                for j in range(i + 1, len(ray_pts)):
                    p1, d1 = ray_pts[i]
                    p2, d2 = ray_pts[j]
                    midpoint, distance = self.calc_intersection(p1, d1, p2, d2)

                    if midpoint is not None and distance < 10.0:
                        valid_intersections.append(midpoint)

            if valid_intersections:
                avg_target = np.mean(valid_intersections, axis=0)
                self.ax3d.scatter(*avg_target, color='white', marker='*', s=400, edgecolors='red')
                self.ax3d.text(avg_target[0], avg_target[1], avg_target[2] + 15, f"ALT: {avg_target[2]:.1f}m",
                               color='white', fontsize=12)

                # Update history
                self.history.append(avg_target)
                if len(self.history) > 150:  # Cap trail length to prevent lag
                    self.history.pop(0)

            # DRAW TRAJECTORY PATH
            if len(self.history) > 1:
                hx = [p[0] for p in self.history]
                hy = [p[1] for p in self.history]
                hz = [p[2] for p in self.history]
                self.ax3d.plot(hx, hy, hz, color='cyan', linewidth=2)

            # CALCULATE & DRAW PREDICTED PATH
            if len(self.history) >= 5:
                p_curr = self.history[-1]
                p_old = self.history[-5]
                # Calculate velocity over the last 4 frames
                velocity = (p_curr - p_old) / 4.0

                # Predict the next 30 frames
                pred_x = [p_curr[0] + velocity[0] * i for i in range(1, 30)]
                pred_y = [p_curr[1] + velocity[1] * i for i in range(1, 30)]
                pred_z = [p_curr[2] + velocity[2] * i for i in range(1, 30)]

                # Connect the current position to the prediction line
                self.ax3d.plot([p_curr[0]] + pred_x,
                               [p_curr[1]] + pred_y,
                               [p_curr[2]] + pred_z,
                               color='yellow', linestyle=':', linewidth=2)

            # Setup 3D View bounds
            self.ax3d.set_xlim([-100, 100])
            self.ax3d.set_ylim([-100, 400])
            self.ax3d.set_zlim([0, 100])
            self.ax3d.set_xlabel("X (Width)")
            self.ax3d.set_ylabel("Y (Depth)")
            self.ax3d.set_zlabel("Z (Altitude)")
            self.ax3d.set_facecolor('#2b2b2b')
            self.ax3d.tick_params(colors='white')
            self.ax3d.xaxis.label.set_color('white')
            self.ax3d.yaxis.label.set_color('white')
            self.ax3d.zaxis.label.set_color('white')

            self.canvas.draw_idle()

        self.after(30, self.update_loop)


if __name__ == "__main__":
    app = VisTrackDashboard()
    app.mainloop()