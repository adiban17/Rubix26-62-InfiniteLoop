import cv2
import mediapipe as mp
import numpy as np
import socketio
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk 
import subprocess
import time
import threading
import sys
import os

# --- GLOBAL CONFIG ---
ALLOWED_APPS = [
    "LeetCode", "ProctorHQ", "Exam Portal", "127.0.0.1", "localhost",
    "python", "python3", "Terminal", "iTerm2", "Code", "Finder" 
]

USER_DETAILS = {
    "name": "",
    "roll": "",
    "sap": ""
}

# THE LEGAL TEXT
TERMS_TEXT = """PROCTORHQ CANDIDATE AGREEMENT

1. MONITORING CONSENT
By proceeding, you consent to being monitored via webcam, microphone, and screen activity for the duration of this exam. Data is transmitted securely to the proctoring server.

2. PROHIBITED ACTIONS
- No switching browser tabs or applications.
- No other persons allowed in the room.
- No usage of mobile phones or external devices.
- No looking away from the screen for extended periods.

3. DATA PRIVACY
Your session data is used solely for the purpose of academic integrity verification.

4. AUTOMATED FLAGGING
The system uses AI to detect suspicious behavior. Multiple violations may result in automatic disqualification.

By clicking "I Agree", you certify that you are the person registered for this exam and you will adhere to these rules."""

class ProctorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ProctorHQ | Watchdog Edition")
        self.root.geometry("1100x850")
        
        # KILL SWITCHES
        self.is_exam_running = False
        self.stop_threads = False 
        
        # Colors & Network
        self.colors = {"primary": "#6366f1", "bg": "#ffffff", "text": "#1e293b", "danger": "#dc2626"}
        self.sio = socketio.Client()
        self.setup_vision()
        
        # VISION STATE
        self.pitch_offset = 0; self.yaw_offset = 0; self.is_calibrated = False
        self.smooth_pitch = 0; self.smooth_yaw = 0
        self.last_sent_status = ""
        self.cap = None
        
        # WATCHDOG STATE
        self.current_window_status = "Status: Initializing..."
        self.violation_counter = 0 

        self.setup_styles()
        self.build_login_ui()

    def setup_vision(self):
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils 

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"])
        style.configure("Modern.TButton", background=self.colors["primary"], foreground="white", font=("Segoe UI", 11, "bold"), borderwidth=0)
        style.map("Modern.TButton", background=[("active", "#4f46e5")])
        style.configure("Danger.TButton", background=self.colors["danger"], foreground="white", font=("Segoe UI", 11, "bold"), borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#b91c1c")])

    # --- APPLE SCRIPT (Safe Version) ---
    def check_active_window(self):
        script = '''
        try
            with timeout of 1 second
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                end tell

                if frontApp is "Google Chrome" then
                    tell application "Google Chrome"
                        return frontApp & "|||" & title of active tab of front window & "|||" & URL of active tab of front window
                    end tell
                else if frontApp is "Safari" then
                    tell application "Safari"
                        return frontApp & "|||" & name of front document & "|||" & URL of front document
                    end tell
                else
                    return frontApp & "|||None|||None"
                end if
            end timeout
        on error
            return "Unknown|||Unknown|||Unknown"
        end try
        '''
        try:
            result = subprocess.check_output(['osascript', '-e', script], stderr=subprocess.STDOUT)
            parts = result.decode('utf-8').strip().split("|||")
            if len(parts) >= 3: return parts[0], parts[1], parts[2]
            return parts[0], "Unknown", ""
        except:
            return "Unknown", "Unknown", ""

    # --- WATCHDOG THREAD ---
    def start_watchdog(self):
        def watchdog_loop():
            while not self.stop_threads:
                if self.is_exam_running:
                    self.update_window_status()
                time.sleep(1.0) 
        
        t = threading.Thread(target=watchdog_loop)
        t.daemon = True 
        t.start()

    def update_window_status(self):
        app, title, url = self.check_active_window()
        is_safe = False
        full_context = f"{app} {title} {url}".lower()
        
        for safe_word in ALLOWED_APPS:
            if safe_word.lower() in full_context:
                is_safe = True; break
        
        if is_safe:
            self.violation_counter = 0 
            if "VIOLATION" in self.current_window_status:
                self.current_window_status = "Status: Normal"
        else:
            self.violation_counter += 1
            if self.violation_counter >= 2:
                clean_url = url.replace("https://", "")[:30] if url else ""
                self.current_window_status = f"VIOLATION: {app} ({clean_url})"

    # --- UI: LOGIN & TERMS ---
    def build_login_ui(self):
        for w in self.root.winfo_children(): w.destroy()
        
        header = tk.Frame(self.root, bg=self.colors["primary"], height=100)
        header.pack(fill="x")
        tk.Label(header, text="ProctorHQ Login", bg=self.colors["primary"], fg="white", font=("Segoe UI", 24, "bold")).place(relx=0.5, rely=0.5, anchor="center")

        frame = ttk.Frame(self.root, padding=40)
        frame.pack(expand=True)

        self.create_input(frame, "Full Name", "name_entry")
        self.create_input(frame, "Roll Number", "roll_entry")
        self.create_input(frame, "SAP ID", "sap_entry")

        # --- TERMS AND CONDITIONS SECTION ---
        terms_frame = tk.Frame(frame, bg=self.colors["bg"])
        terms_frame.pack(pady=20)

        self.agree_var = tk.IntVar()
        
        # Checkbox
        chk = tk.Checkbutton(terms_frame, text="I accept the", variable=self.agree_var, 
                             bg=self.colors["bg"], activebackground=self.colors["bg"], font=("Segoe UI", 9))
        chk.pack(side="left")

        # Clickable Link
        link = tk.Label(terms_frame, text="Terms & Conditions", font=("Segoe UI", 9, "bold", "underline"),
                        bg=self.colors["bg"], fg=self.colors["primary"], cursor="hand2")
        link.pack(side="left", padx=5)
        link.bind("<Button-1>", lambda e: self.show_terms_popup())

        ttk.Button(frame, text="Start Exam", style="Modern.TButton", command=self.attempt_login).pack(fill="x", pady=20)

    def show_terms_popup(self):
        # Create a popup window
        popup = tk.Toplevel(self.root)
        popup.title("Terms & Conditions")
        popup.geometry("600x500")
        popup.configure(bg="white")
        
        # Header
        tk.Label(popup, text="Exam Integrity Policy", font=("Segoe UI", 16, "bold"), bg="white", fg=self.colors["text"]).pack(pady=20)
        
        # Scrollable Text Area
        text_area = scrolledtext.ScrolledText(popup, width=60, height=15, font=("Segoe UI", 10), padx=10, pady=10)
        text_area.pack(pady=10, padx=20)
        text_area.insert(tk.END, TERMS_TEXT)
        text_area.configure(state='disabled') # Read-only
        
        # Accept Button
        def accept():
            self.agree_var.set(1) # Check the box
            popup.destroy()
            
        ttk.Button(popup, text="I Agree to these Terms", style="Modern.TButton", command=accept).pack(pady=20)


    def create_input(self, parent, label, var_name):
        ttk.Label(parent, text=label, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 5))
        entry = ttk.Entry(parent, width=40)
        entry.pack(fill="x")
        setattr(self, var_name, entry)

    def attempt_login(self):
        if not self.name_entry.get() or not self.roll_entry.get():
            messagebox.showwarning("Error", "Please fill in all details.")
            return
            
        if self.agree_var.get() == 0:
            messagebox.showwarning("Compliance", "You must accept the Terms & Conditions to proceed.")
            # Optional: Open the popup automatically if they forgot
            self.show_terms_popup()
            return

        USER_DETAILS.update({"name": self.name_entry.get(), "roll": self.roll_entry.get(), "sap": self.sap_entry.get()})
        try:
            self.sio.connect('http://localhost:3000')
            self.sio.emit('student-connect', USER_DETAILS)
            self.start_exam_mode()
        except Exception as e:
            messagebox.showerror("Error", f"Server connection failed.\nIs Node.js running?\n\nError: {e}")

    def start_exam_mode(self):
        for w in self.root.winfo_children(): w.destroy()
        self.root.configure(bg="black")
        self.cap = cv2.VideoCapture(0)
        self.is_exam_running = True
        self.start_watchdog()

        controls = tk.Frame(self.root, bg="white", height=100)
        controls.pack(side=tk.BOTTOM, fill=tk.X)
        controls.pack_propagate(False)

        info_frame = tk.Frame(controls, bg="white")
        info_frame.pack(side="left", padx=20, pady=10)
        
        self.status_lbl = tk.Label(info_frame, text="Face: Calibrating...", font=("Segoe UI", 14, "bold"), fg="orange", bg="white", anchor="w")
        self.status_lbl.pack(fill="x")
        self.window_lbl = tk.Label(info_frame, text="App Monitor: Active", font=("Segoe UI", 10), fg="#64748b", bg="white", anchor="w")
        self.window_lbl.pack(fill="x")

        ttk.Button(controls, text="END EXAM", style="Danger.TButton", command=self.end_exam).pack(side="right", padx=20)
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.process_video_loop()

    def process_video_loop(self):
        if not self.is_exam_running: return
        success, frame = self.cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, face_status, face_color = self.analyze_frame(frame)
            final_status = face_status; final_color = face_color
            
            if "VIOLATION" in self.current_window_status:
                final_status = self.current_window_status; final_color = "#dc2626"

            img = Image.fromarray(processed_frame)
            win_h = self.root.winfo_height() - 100; win_w = self.root.winfo_width()
            if win_h > 1:
                ratio = img.width / img.height
                new_w = int(win_h * ratio)
                if new_w > win_w: new_w = win_w
                img = img.resize((new_w, win_h))

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.status_lbl.config(text=final_status, fg=final_color)
            self.window_lbl.config(text=self.current_window_status)
            
            if self.is_calibrated and final_status != self.last_sent_status:
                self.sio.emit('student-status-update', final_status)
                self.last_sent_status = final_status
        self.root.after(20, self.process_video_loop)

    def analyze_frame(self, image):
        img_h, img_w, _ = image.shape
        status_text = "Status: Normal"; tk_color = "green" 
        fd_results = self.mp_face_detection.process(image)
        fm_results = self.mp_face_mesh.process(image)
        
        face_count = 0
        if fd_results.detections:
            face_count = len(fd_results.detections)
            for detection in fd_results.detections: self.mp_drawing.draw_detection(image, detection)

        if not self.is_calibrated:
            cv2.putText(image, "Look at screen & Press 'C'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            return image, "Action Required: Press 'C' to Calibrate", "#f59e0b"
        
        if face_count == 0: return image, "VIOLATION: NO FACE", "#dc2626"
        elif face_count > 1: return image, "VIOLATION: MULTIPLE FACES", "#dc2626"
        elif face_count == 1 and fm_results.multi_face_landmarks:
            for face_landmarks in fm_results.multi_face_landmarks:
                pitch, yaw, _ = self.get_head_pose(image, face_landmarks, img_w, img_h)
                alpha = 0.2
                self.smooth_pitch = (pitch * alpha) + (self.smooth_pitch * (1.0 - alpha))
                self.smooth_yaw = (yaw * alpha) + (self.smooth_yaw * (1.0 - alpha))
                if self.is_calibrated:
                    final_pitch = self.smooth_pitch - self.pitch_offset
                    final_yaw = self.smooth_yaw - self.yaw_offset
                    if abs(final_pitch) > 25: return image, "VIOLATION: LOOKING AWAY", "#dc2626"
                    elif abs(final_yaw) > 40: return image, "VIOLATION: SIDEWAYS LOOK", "#dc2626"
        return image, status_text, tk_color

    def get_head_pose(self, image, face_landmarks, img_w, img_h):
        face_3d = np.array([(0.0, 0.0, 0.0), (0.0, 330.0, -65.0), (-225.0, -170.0, -135.0), (225.0, -170.0, -135.0), (-150.0, 150.0, -125.0), (150.0, 150.0, -125.0)], dtype=np.float64)
        face_2d = np.array([[int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h)] for idx in [1, 152, 33, 263, 61, 291]], dtype=np.float64)
        for p in face_2d: cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h/2], [0, focal_length, img_w/2], [0, 0, 1]])
        dist_matrix = np.zeros((4,1), dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0], angles[1], angles[2]

    def end_exam(self):
        if messagebox.askyesno("Confirm", "End Exam?"):
            self.stop_threads = True 
            self.is_exam_running = False
            if self.cap: self.cap.release()
            try: self.sio.disconnect()
            except: pass
            self.root.destroy()
            os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = ProctorApp(root)
    root.bind('<c>', lambda e: setattr(app, 'is_calibrated', True)) 
    root.bind('<C>', lambda e: setattr(app, 'is_calibrated', True)) 
    root.mainloop()