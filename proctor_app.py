import cv2
import mediapipe as mp
import numpy as np
import socketio
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk 
import subprocess
import time
import threading
import sys
import os
import platform

# --- GLOBAL CONFIG ---
ALLOWED_APPS = [
    "LeetCode", "ProctorHQ", "Exam Portal", "127.0.0.1", "localhost",
    "python", "python3", "Terminal", "iTerm2", "Code", "Finder", "Explorer"
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
        
        # --- THEME CONFIGURATION (DARK MODE) ---
        self.colors = {
            "bg": "#0f172a",         # Very dark slate (Main Background)
            "card": "#1e293b",       # Lighter slate (Panels)
            "input_bg": "#334155",   # Inputs
            "text": "#f8fafc",       # Almost white
            "subtext": "#94a3b8",    # Gray text
            "primary": "#6366f1",    # Indigo (Brand)
            "primary_hover": "#4f46e5",
            "danger": "#ef4444",     # Red
            "success": "#22c55e",    # Neon Green
            "warning": "#f59e0b"     # Amber
        }
        
        self.root.configure(bg=self.colors["bg"])
        self.sio = socketio.Client()
        
        # --- 1. NEURAL NETWORK LAYER ---
        print("System: Loading Neural Network for Phone Detection...")
        try:
            self.object_detector = YOLO("yolov8n.pt") 
            self.phone_class_id = 67 
        except Exception as e:
            print(f"Warning: YOLO failed to load. Phone detection disabled. {e}")
            self.object_detector = None
        
        self.setup_vision()
        
        # VISION STATE
        self.pitch_offset = 0; self.yaw_offset = 0; self.is_calibrated = False
        self.smooth_pitch = 0; self.smooth_yaw = 0
        self.last_sent_status = ""
        self.cap = None
        
        # WATCHDOG STATE
        self.current_window_status = "Status: Initializing..."
        self.violation_counter = 0 
        self.system_os = platform.system() 

        self.setup_styles()
        self.build_login_ui()

    def setup_vision(self):
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils 

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure frames and labels to match dark theme
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Card.TFrame", background=self.colors["card"], relief="flat")
        
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=self.colors["card"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.colors["card"], foreground=self.colors["text"], font=("Segoe UI", 20, "bold"))
        
        # Modern Button Styling
        style.configure("Modern.TButton", 
                        background=self.colors["primary"], 
                        foreground="white", 
                        font=("Segoe UI", 11, "bold"), 
                        borderwidth=0, 
                        focuscolor=self.colors["primary"])
        style.map("Modern.TButton", background=[("active", self.colors["primary_hover"])])
        
        style.configure("Danger.TButton", 
                        background=self.colors["danger"], 
                        foreground="white", 
                        font=("Segoe UI", 10, "bold"), 
                        borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#dc2626")])

        # Scrollbar (darkish)
        style.configure("Vertical.TScrollbar", troughcolor=self.colors["card"], background=self.colors["input_bg"], borderwidth=0, arrowsize=0)

    # --- APPLE SCRIPT (Safe Version) ---
    def check_active_window(self):
        if self.system_os != "Darwin":
            return "OS-Not-Mac", "Unknown", ""

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
        
        if self.system_os != "Darwin":
            is_safe = True

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

    # --- UI: LOGIN & TERMS (DARK MODE REDESIGN) ---
    def build_login_ui(self):
        for w in self.root.winfo_children(): w.destroy()
        
        # 1. Main Container (Centers the content)
        main_container = tk.Frame(self.root, bg=self.colors["bg"])
        main_container.pack(fill="both", expand=True)
        
        # 2. The "Card" (Login Box)
        # We use standard tk.Frame to control background color precisely
        login_card = tk.Frame(main_container, bg=self.colors["card"], padx=40, pady=40)
        login_card.place(relx=0.5, rely=0.5, anchor="center")
        
        # Brand Header
        tk.Label(login_card, text="PROCTOR HQ", bg=self.colors["card"], fg=self.colors["primary"], font=("Segoe UI", 24, "bold")).pack(pady=(0, 5))
        tk.Label(login_card, text="Secure Exam Environment", bg=self.colors["card"], fg=self.colors["subtext"], font=("Segoe UI", 10)).pack(pady=(0, 30))

        # Inputs
        self.create_dark_input(login_card, "Full Name", "name_entry")
        self.create_dark_input(login_card, "Roll Number", "roll_entry")
        self.create_dark_input(login_card, "SAP ID", "sap_entry")

        # Terms
        terms_frame = tk.Frame(login_card, bg=self.colors["card"])
        terms_frame.pack(pady=20, fill="x")

        self.agree_var = tk.IntVar()
        
        # Custom Dark Checkbox hack: Text only, styled
        chk = tk.Checkbutton(terms_frame, text="I accept the", variable=self.agree_var, 
                             bg=self.colors["card"], fg=self.colors["text"], 
                             selectcolor=self.colors["bg"], # Checkbox inner color
                             activebackground=self.colors["card"], activeforeground=self.colors["text"],
                             font=("Segoe UI", 9))
        chk.pack(side="left")

        link = tk.Label(terms_frame, text="Terms & Conditions", font=("Segoe UI", 9, "bold", "underline"),
                        bg=self.colors["card"], fg=self.colors["primary"], cursor="hand2")
        link.pack(side="left", padx=5)
        link.bind("<Button-1>", lambda e: self.show_terms_popup())

        # Big Button
        ttk.Button(login_card, text="AUTHENTICATE & START", style="Modern.TButton", command=self.attempt_login).pack(fill="x", pady=10)

    def create_dark_input(self, parent, label_text, var_name):
        """Creates a modern dark themed input field"""
        # Label
        tk.Label(parent, text=label_text.upper(), bg=self.colors["card"], fg=self.colors["subtext"], 
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(10, 2))
        
        # Entry (Using tk.Entry for custom background color)
        entry = tk.Entry(parent, width=35, bg=self.colors["input_bg"], fg="white", 
                         insertbackground="white", # Cursor color
                         font=("Segoe UI", 11), relief="flat", bd=5)
        entry.pack(fill="x", ipady=3) # Internal padding for height
        setattr(self, var_name, entry)

    def show_terms_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Terms & Conditions")
        popup.geometry("600x500")
        popup.configure(bg=self.colors["card"])
        
        tk.Label(popup, text="Exam Integrity Policy", font=("Segoe UI", 16, "bold"), 
                 bg=self.colors["card"], fg=self.colors["text"]).pack(pady=20)
        
        text_area = scrolledtext.ScrolledText(popup, width=60, height=15, font=("Segoe UI", 10), 
                                            padx=10, pady=10, bg=self.colors["input_bg"], fg=self.colors["text"], relief="flat")
        text_area.pack(pady=10, padx=20)
        text_area.insert(tk.END, TERMS_TEXT)
        text_area.configure(state='disabled')
        
        def accept():
            self.agree_var.set(1)
            popup.destroy()
            
        ttk.Button(popup, text="I Agree", style="Modern.TButton", command=accept).pack(pady=20)

    def attempt_login(self):
        if not self.name_entry.get() or not self.roll_entry.get():
            messagebox.showwarning("Incomplete", "Please fill in all identity fields.")
            return
            
        if self.agree_var.get() == 0:
            messagebox.showwarning("Compliance", "You must accept the Terms & Conditions.")
            return

        USER_DETAILS.update({"name": self.name_entry.get(), "roll": self.roll_entry.get(), "sap": self.sap_entry.get()})
        try:
            self.sio.connect('http://localhost:3000')
            self.sio.emit('student-connect', USER_DETAILS)
            self.start_exam_mode()
        except Exception as e:
            messagebox.showerror("Connection Error", f"Server unreachable.\nIs Node.js running?\n\nError: {e}")

    # --- 2. EXAM MODE UI UPDATES (HUD) ---
    def start_exam_mode(self):
        for w in self.root.winfo_children(): w.destroy()
        self.root.configure(bg="black") # Video background is black
        
        # 1. Initialize Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Hardware Error", "Camera access denied.")
            return
        
        # 2. HUD (Heads Up Display) - Bottom Bar
        hud_frame = tk.Frame(self.root, bg=self.colors["card"], height=100)
        hud_frame.pack(side=tk.BOTTOM, fill=tk.X)
        hud_frame.pack_propagate(False)

        # Status Indicators
        info_frame = tk.Frame(hud_frame, bg=self.colors["card"])
        info_frame.pack(side="left", padx=30, pady=20)
        
        self.status_lbl = tk.Label(info_frame, text="SYSTEM: CALIBRATING...", 
                                   font=("Segoe UI", 14, "bold"), fg=self.colors["warning"], bg=self.colors["card"], anchor="w")
        self.status_lbl.pack(fill="x")
        
        self.window_lbl = tk.Label(info_frame, text="APP MONITOR: ACTIVE", 
                                   font=("Segoe UI", 9, "bold"), fg=self.colors["subtext"], bg=self.colors["card"], anchor="w")
        self.window_lbl.pack(fill="x")

        # End Button
        ttk.Button(hud_frame, text="SUBMIT & EXIT", style="Danger.TButton", command=self.end_exam).pack(side="right", padx=30)
        
        # Video Feed
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.show_start_button()

        self.is_exam_running = True 
        self.process_video_loop()
        self.start_watchdog()

    # --- 3. THE NEW START BUTTON ---
    def show_start_button(self):
        # We create a button that looks like a floating overlay
        self.start_btn = tk.Button(
            self.root, 
            text="INITIALIZE EXAM\n(Look at camera and click)", 
            font=("Segoe UI", 16, "bold"), 
            bg=self.colors["primary"], 
            fg="white",
            activebackground=self.colors["primary_hover"],
            activeforeground="white",
            padx=40,
            pady=20,
            relief="flat",
            cursor="hand2",
            command=self.start_exam_sequence
        )
        self.start_btn.place(relx=0.5, rely=0.5, anchor="center")

    def start_exam_sequence(self):
        if self.cap is None: return
        ret, frame = self.cap.read()
        if ret:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                pitch, yaw, _ = self.get_head_pose(frame, results.multi_face_landmarks[0], frame.shape[1], frame.shape[0])
                self.pitch_offset = pitch
                self.yaw_offset = yaw
                self.is_calibrated = True
                
                self.start_btn.destroy()
                print("System: Calibration Complete. Monitoring Active.")
                self.status_lbl.config(text="SYSTEM: ONLINE (MONITORING)", fg=self.colors["success"])
            else:
                messagebox.showwarning("Calibration Failed", "Face not detected. Please ensure good lighting.")

    def process_video_loop(self):
        if not self.is_exam_running: return
        success, frame = self.cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, face_status, face_color = self.analyze_frame(frame)
            final_status = face_status; final_color = face_color
            
            if "VIOLATION" in self.current_window_status:
                final_status = self.current_window_status
                final_color = self.colors["danger"]

            img = Image.fromarray(processed_frame)
            
            # Smart Resize to fill available space
            win_h = self.root.winfo_height() - 100
            win_w = self.root.winfo_width()
            if win_h > 1:
                ratio = img.width / img.height
                new_w = int(win_h * ratio)
                if new_w > win_w: new_w = win_w
                img = img.resize((new_w, win_h))

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            if self.is_calibrated:
                self.status_lbl.config(text=final_status.upper(), fg=final_color)
                if final_status != self.last_sent_status:
                    self.sio.emit('student-status-update', final_status)
                    self.last_sent_status = final_status
                    
            self.window_lbl.config(text=self.current_window_status.upper())
            
        self.root.after(20, self.process_video_loop)

    def analyze_frame(self, image):
        img_h, img_w, _ = image.shape
        status_text = "System: Normal"
        tk_color = self.colors["success"]
        
        # 1. PHONE DETECTION (YOLOv8)
        phone_detected = False
        if self.object_detector:
            obj_results = self.object_detector(image, verbose=False, conf=0.5)
            for result in obj_results:
                for box in result.boxes:
                    if int(box.cls) == self.phone_class_id:
                        phone_detected = True
                        status_text = "VIOLATION: PHONE DETECTED"
                        tk_color = self.colors["danger"]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Draw Cyan Box for "Tech" feel
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(image, "UNAUTHORIZED DEVICE", (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 2. FACE MESH
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fd_results = self.mp_face_detection.process(image_rgb)
        fm_results = self.mp_face_mesh.process(image_rgb)
        
        face_count = 0
        if fd_results.detections:
            face_count = len(fd_results.detections)
            for detection in fd_results.detections: 
                self.mp_drawing.draw_detection(image, detection)

        # 3. CALIBRATION CHECK
        if not self.is_calibrated:
            if phone_detected:
                return image, "VIOLATION: PHONE DETECTED", self.colors["danger"]
            return image, "Action Required: Click Initialize", self.colors["warning"]
        
        # 4. VIOLATION LOGIC
        if face_count == 0: 
            if not phone_detected:
                return image, "VIOLATION: NO FACE DETECTED", self.colors["danger"]
        elif face_count > 1: 
            if not phone_detected:
                return image, "VIOLATION: MULTIPLE PERSONS", self.colors["danger"]
        elif face_count == 1 and fm_results.multi_face_landmarks:
            for face_landmarks in fm_results.multi_face_landmarks:
                pitch, yaw, _ = self.get_head_pose(image, face_landmarks, img_w, img_h)
                
                self.smooth_pitch = (pitch * 0.2) + (self.smooth_pitch * 0.8)
                self.smooth_yaw = (yaw * 0.2) + (self.smooth_yaw * 0.8)
                
                if self.is_calibrated:
                    final_pitch = self.smooth_pitch - self.pitch_offset
                    final_yaw = self.smooth_yaw - self.yaw_offset
                    
                    if abs(final_pitch) > 25: 
                        if not phone_detected:
                            return image, "VIOLATION: LOOKING AWAY ", self.colors["danger"]
                    elif abs(final_yaw) > 40: 
                        if not phone_detected:
                            return image, "VIOLATION: LOOKING AWAY ", self.colors["danger"]

        return image, status_text, tk_color

    def get_head_pose(self, image, face_landmarks, img_w, img_h):
        face_3d = np.array([(0.0, 0.0, 0.0), (0.0, 330.0, -65.0), (-225.0, -170.0, -135.0), (225.0, -170.0, -135.0), (-150.0, 150.0, -125.0), (150.0, 150.0, -125.0)], dtype=np.float64)
        face_2d = np.array([[int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h)] for idx in [1, 152, 33, 263, 61, 291]], dtype=np.float64)
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h/2], [0, focal_length, img_w/2], [0, 0, 1]])
        dist_matrix = np.zeros((4,1), dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0], angles[1], angles[2]

    def end_exam(self):
        if messagebox.askyesno("Submit Exam", "Are you sure you want to finish the session?"):
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
    root.mainloop()