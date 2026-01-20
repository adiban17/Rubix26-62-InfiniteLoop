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
import os
import platform
import math
import base64

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

# --- SECURITY THRESHOLDS ---
THRESHOLDS = {
    "PHONE": 0.0,           # Instant violation
    "LOOKING_AWAY": 1.0,    
    "GAZE_ISSUE": 0.5,      
    "NO_FACE": 2.5,         
    "MULTIPLE_FACES": 2.0,  
    "NORMAL": 0.0
}

# --- SENSITIVITY CALIBRATION ---
# 1. HEAD ANGLES (Degrees)
MAX_PITCH_DOWN = 12   
MAX_PITCH_UP = 25     
MAX_YAW = 20          

# 2. GAZE SENSITIVITY (Dynamic)
GAZE_DROP_THRESH_VERT = 0.12  
GAZE_DEV_THRESH_HORIZ = 0.10  

# 3. HYBRID TRAPS
COMBO_PITCH_DOWN = 5      
COMBO_GAZE_DOWN = 0.08      
COMBO_YAW = 10              
COMBO_GAZE_HORIZ = 0.06     

# --- LOW LIGHT CONFIG ---
MIN_BRIGHTNESS = 90  

TERMS_TEXT = """PROCTORHQ CANDIDATE AGREEMENT

1. MONITORING CONSENT
By proceeding, you consent to being monitored via webcam, microphone, and screen activity for the duration of this exam. Data is transmitted securely to the proctoring server.

2. PROHIBITED ACTIONS
- No switching browser tabs or applications.
- No other persons allowed in the room.
- No usage of mobile phones or external devices.
- No looking away from the screen.
- Gaze must remain within the screen boundaries.

3. DATA PRIVACY
Your session data is used solely for the purpose of academic integrity verification.

4. AUTOMATED FLAGGING
The system uses AI to detect suspicious behavior. Multiple violations may result in automatic disqualification.

By clicking "I Agree", you certify that you are the person registered for this exam and you will adhere to these rules."""

class ProctorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel | Night Vision & Lateral Gaze")
        self.root.geometry("1100x850")
        
        # KILL SWITCHES
        self.is_exam_running = False
        self.stop_threads = False 
        
        # --- THEME CONFIGURATION (DARK MODE) ---
        self.colors = {
            "bg": "#0f172a",         
            "card": "#1e293b",       
            "input_bg": "#334155",   
            "text": "#f8fafc",       
            "subtext": "#94a3b8",    
            "primary": "#6366f1",    
            "primary_hover": "#4f46e5",
            "danger": "#ef4444",     
            "success": "#22c55e",    
            "warning": "#f59e0b"     
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
        self.is_calibrated = False
        self.pitch_offset = 0
        self.yaw_offset = 0
        self.baseline_gaze_vert = 0.5 
        self.baseline_gaze_horiz = 0.5 
        
        self.smooth_pitch = 0
        self.smooth_yaw = 0
        self.last_sent_status = ""
        self.cap = None
        
        # --- STATE MACHINE FOR VIOLATION DEBOUNCING ---
        self.potential_violation_type = None  
        self.violation_start_time = None      
        self.confirmed_status = "SYSTEM: ACTIVE" 
        self.confirmed_color = self.colors["success"]

        # WATCHDOG STATE
        self.current_window_status = "Status: Initializing..."
        self.violation_counter = 0 
        self.system_os = platform.system() 

        self.setup_styles()
        self.build_login_ui()

    def setup_vision(self):
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7, 
            refine_landmarks=True 
        )
        self.mp_drawing = mp.solutions.drawing_utils 

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Card.TFrame", background=self.colors["card"], relief="flat")
        
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=self.colors["card"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.colors["card"], foreground=self.colors["text"], font=("Segoe UI", 20, "bold"))
        
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

        style.configure("Vertical.TScrollbar", troughcolor=self.colors["card"], background=self.colors["input_bg"], borderwidth=0, arrowsize=0)

    # --- APPLE SCRIPT ---
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
        
        if app == "Unknown":
            self.violation_counter = 0
            if "VIOLATION" in self.current_window_status:
                self.current_window_status = "Status: Normal"
            return 

        is_safe = False
        full_context = f"{app} {title} {url}".lower()
        
        if self.system_os != "Darwin": is_safe = True
        
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
        
        main_container = tk.Frame(self.root, bg=self.colors["bg"])
        main_container.pack(fill="both", expand=True)
        
        login_card = tk.Frame(main_container, bg=self.colors["card"], padx=40, pady=40)
        login_card.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(login_card, text="SENTINEL", bg=self.colors["card"], fg=self.colors["primary"], font=("Segoe UI", 24, "bold")).pack(pady=(0, 5))
        tk.Label(login_card, text="AI Secure Environment", bg=self.colors["card"], fg=self.colors["subtext"], font=("Segoe UI", 10)).pack(pady=(0, 30))

        self.create_dark_input(login_card, "Full Name", "name_entry")
        self.create_dark_input(login_card, "Roll Number", "roll_entry")
        self.create_dark_input(login_card, "SAP ID", "sap_entry")

        terms_frame = tk.Frame(login_card, bg=self.colors["card"])
        terms_frame.pack(pady=20, fill="x")

        self.agree_var = tk.IntVar()
        
        chk = tk.Checkbutton(terms_frame, text="I accept the", variable=self.agree_var, 
                             bg=self.colors["card"], fg=self.colors["text"], 
                             selectcolor=self.colors["bg"],
                             activebackground=self.colors["card"], activeforeground=self.colors["text"],
                             font=("Segoe UI", 9))
        chk.pack(side="left")

        link = tk.Label(terms_frame, text="Terms & Conditions", font=("Segoe UI", 9, "bold", "underline"),
                        bg=self.colors["card"], fg=self.colors["primary"], cursor="hand2")
        link.pack(side="left", padx=5)
        link.bind("<Button-1>", lambda e: self.show_terms_popup())

        ttk.Button(login_card, text="AUTHENTICATE & START", style="Modern.TButton", command=self.attempt_login).pack(fill="x", pady=10)

    def create_dark_input(self, parent, label_text, var_name):
        tk.Label(parent, text=label_text.upper(), bg=self.colors["card"], fg=self.colors["subtext"], 
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(10, 2))
        entry = tk.Entry(parent, width=35, bg=self.colors["input_bg"], fg="white", 
                         insertbackground="white", font=("Segoe UI", 11), relief="flat", bd=5)
        entry.pack(fill="x", ipady=3)
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
            self.agree_var.set(1); popup.destroy()
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
            messagebox.showerror("Connection Error", f"Server unreachable.\n\nError: {e}")

    # --- 2. EXAM MODE UI ---
    def start_exam_mode(self):
        for w in self.root.winfo_children(): w.destroy()
        self.root.configure(bg="black")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Hardware Error", "Camera access denied.")
            return
        
        hud_frame = tk.Frame(self.root, bg=self.colors["card"], height=100)
        hud_frame.pack(side=tk.BOTTOM, fill=tk.X)
        hud_frame.pack_propagate(False)

        info_frame = tk.Frame(hud_frame, bg=self.colors["card"])
        info_frame.pack(side="left", padx=30, pady=20)
        
        self.status_lbl = tk.Label(info_frame, text="SYSTEM: READY TO START", 
                                   font=("Segoe UI", 14, "bold"), fg=self.colors["primary"], bg=self.colors["card"], anchor="w")
        self.status_lbl.pack(fill="x")
        self.window_lbl = tk.Label(info_frame, text="APP MONITOR: ACTIVE", 
                                   font=("Segoe UI", 9, "bold"), fg=self.colors["subtext"], bg=self.colors["card"], anchor="w")
        self.window_lbl.pack(fill="x")

        # Dynamic Action Button
        self.action_btn = ttk.Button(hud_frame, text="START EXAM", style="Modern.TButton", command=self.start_calibration)
        self.action_btn.pack(side="right", padx=30)

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.is_exam_running = True 
        self.process_video_loop()
        self.start_watchdog()

    def start_calibration(self):
        if self.cap is None: return
        ret, frame = self.cap.read()
        if ret:
            frame = self.enhance_low_light(frame)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # 1. Calibrate Head Pose
                pitch, yaw, _ = self.get_head_pose(frame, landmarks, frame.shape[1], frame.shape[0])
                self.pitch_offset = pitch
                self.yaw_offset = yaw
                
                # 2. Calibrate Gaze
                _, _, base_vert, base_horiz = self.check_gaze(landmarks, frame.shape[1], frame.shape[0])
                self.baseline_gaze_vert = base_vert
                self.baseline_gaze_horiz = base_horiz
                
                self.is_calibrated = True
                
                # Change button to "SUBMIT & EXIT"
                self.action_btn.configure(text="SUBMIT & EXIT", style="Danger.TButton", command=self.end_exam)
                
                print(f"System: Calibrated. Pitch:{pitch:.1f}, VertGaze:{base_vert:.2f}, HorizGaze:{base_horiz:.2f}")
                self.confirmed_status = "SYSTEM: ACTIVE"
                self.status_lbl.config(text=self.confirmed_status, fg=self.colors["success"])
            else:
                messagebox.showwarning("Calibration Failed", "Face not detected. Adjust lighting.")

    def enhance_low_light(self, frame):
        # Adaptive Night Vision
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        avg_brightness = np.mean(v)
        
        if avg_brightness < MIN_BRIGHTNESS:
            # Gamma Correction
            gamma = 1.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            frame = cv2.LUT(frame, table)
            
            # CLAHE on V channel
            hsv_gamma = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_g, s_g, v_g = cv2.split(hsv_gamma)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            v_enhanced = clahe.apply(v_g)
            
            hsv_final = cv2.merge([h_g, s_g, v_enhanced])
            frame = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
            
            cv2.putText(frame, "NIGHT VISION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        return frame

    def process_video_loop(self):
        if not self.is_exam_running: return
        success, frame = self.cap.read()
        if success:
            frame = self.enhance_low_light(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, raw_violation_code = self.analyze_frame(frame, frame_rgb)
            
            # --- STATUS DECISION LOGIC ---
            if "VIOLATION" in self.current_window_status:
                self.confirmed_status = self.current_window_status
                self.confirmed_color = self.colors["danger"]
            else:
                current_time = time.time()
                if raw_violation_code == "NORMAL":
                    self.potential_violation_type = None
                    self.violation_start_time = None
                    self.confirmed_status = "SYSTEM: ACTIVE"
                    self.confirmed_color = self.colors["success"]
                else:
                    if self.potential_violation_type == raw_violation_code:
                        elapsed_time = current_time - self.violation_start_time
                        required_threshold = THRESHOLDS.get(raw_violation_code, 2.0)
                        if elapsed_time > required_threshold:
                            readable_status = raw_violation_code.replace("_", " ")
                            self.confirmed_status = f"VIOLATION: {readable_status}"
                            self.confirmed_color = self.colors["danger"]
                    else:
                        self.potential_violation_type = raw_violation_code
                        self.violation_start_time = current_time
            
            # --- RENDER VIDEO ---
            img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            
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
            
            # --- EMIT STATUS & EVIDENCE ---
            if self.is_calibrated:
                self.status_lbl.config(text=self.confirmed_status, fg=self.confirmed_color)
                self.window_lbl.config(text=self.current_window_status.upper())

                if self.confirmed_status != self.last_sent_status:
                    payload = {"status": self.confirmed_status, "image": None}
                    
                    # Capture evidence ONLY if it's a vision violation (not a window violation)
                    # Window violations are text-only to save bandwidth/confusion
                    is_window_violation = (self.confirmed_status == self.current_window_status)
                    
                    if "VIOLATION" in self.confirmed_status and not is_window_violation:
                        try:
                            # Encode the processed frame (which includes the bounding boxes/text)
                            _, buffer = cv2.imencode('.jpg', processed_frame)
                            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                            payload["image"] = jpg_as_text
                            print(f"System: Captured evidence for {self.confirmed_status}")
                        except Exception as e:
                            print(f"Error capturing evidence: {e}")

                    self.sio.emit('student-status-update', payload)
                    self.last_sent_status = self.confirmed_status

        self.root.after(20, self.process_video_loop)

    # --- GAZE ENGINE ---
    def get_iris_center(self, landmarks, idx_list, img_w, img_h):
        x_total = 0; y_total = 0
        for idx in idx_list:
            x_total += landmarks.landmark[idx].x * img_w
            y_total += landmarks.landmark[idx].y * img_h
        return (int(x_total / len(idx_list)), int(y_total / len(idx_list)))

    def check_gaze(self, landmarks, img_w, img_h):
        left_iris = self.get_iris_center(landmarks, [468], img_w, img_h)
        right_iris = self.get_iris_center(landmarks, [473], img_w, img_h)
        
        l_top = landmarks.landmark[159].y * img_h
        l_bot = landmarks.landmark[145].y * img_h
        l_eye_h = l_bot - l_top
        
        r_top = landmarks.landmark[386].y * img_h
        r_bot = landmarks.landmark[374].y * img_h
        r_eye_h = r_bot - r_top
        
        l_dist_to_bot = l_bot - left_iris[1]
        r_dist_to_bot = r_bot - right_iris[1]
        
        l_v_ratio = l_dist_to_bot / l_eye_h if l_eye_h > 0 else 0
        r_v_ratio = r_dist_to_bot / r_eye_h if r_eye_h > 0 else 0
        avg_vert_ratio = (l_v_ratio + r_v_ratio) / 2
        
        l_outer_x = landmarks.landmark[33].x * img_w
        l_inner_x = landmarks.landmark[133].x * img_w
        l_iris_x = landmarks.landmark[468].x * img_w
        l_width = l_inner_x - l_outer_x
        l_h_ratio = (l_iris_x - l_outer_x) / l_width if l_width > 0 else 0.5
        
        r_inner_x = landmarks.landmark[362].x * img_w
        r_outer_x = landmarks.landmark[263].x * img_w
        r_iris_x = landmarks.landmark[473].x * img_w
        r_width = r_outer_x - r_inner_x
        r_h_ratio = (r_iris_x - r_inner_x) / r_width if r_width > 0 else 0.5
        
        avg_horiz_ratio = (l_h_ratio + r_h_ratio) / 2
        
        return left_iris, right_iris, avg_vert_ratio, avg_horiz_ratio

    def analyze_frame(self, display_image, processing_image):
        img_h, img_w, _ = display_image.shape
        detection_code = "NORMAL"
        
        # 1. PHONE DETECTION
        phone_detected = False
        if self.object_detector:
            obj_results = self.object_detector(display_image, verbose=False, conf=0.3, classes=[self.phone_class_id])
            for result in obj_results:
                if len(result.boxes) > 0:
                    phone_detected = True; detection_code = "PHONE"
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(display_image, "DEVICE DETECTED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 2. FACE MESH
        fd_results = self.mp_face_detection.process(processing_image)
        fm_results = self.mp_face_mesh.process(processing_image)
        
        face_count = 0
        if fd_results.detections:
            face_count = len(fd_results.detections)
            for detection in fd_results.detections: self.mp_drawing.draw_detection(display_image, detection)

        # 3. LOGIC
        if not self.is_calibrated: return display_image, "NORMAL"
        if phone_detected: return display_image, "PHONE"
        if face_count == 0: return display_image, "NO_FACE"
        elif face_count > 1: return display_image, "MULTIPLE_FACES"
            
        elif face_count == 1 and fm_results.multi_face_landmarks:
            landmarks = fm_results.multi_face_landmarks[0]
            
            l_iris, r_iris, vert_ratio, horiz_ratio = self.check_gaze(landmarks, img_w, img_h)
            pitch, yaw, _ = self.get_head_pose(display_image, landmarks, img_w, img_h)
            
            self.smooth_pitch = (pitch * 0.2) + (self.smooth_pitch * 0.8)
            self.smooth_yaw = (yaw * 0.2) + (self.smooth_yaw * 0.8)
            final_pitch = self.smooth_pitch - self.pitch_offset
            final_yaw = self.smooth_yaw - self.yaw_offset
            
            cv2.circle(display_image, l_iris, 2, (0, 255, 0), -1)
            cv2.circle(display_image, r_iris, 2, (0, 255, 0), -1)
            
            if abs(final_yaw) > MAX_YAW: return display_image, "LOOKING_AWAY"
            
            if final_pitch > MAX_PITCH_DOWN: return display_image, "LOOKING_UP"
            
            gaze_drop = self.baseline_gaze_vert - vert_ratio
            if gaze_drop > GAZE_DROP_THRESH_VERT and final_pitch > -10: 
                 cv2.putText(display_image, f"V-GAZE DROP: {gaze_drop:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                 return display_image, "GAZE_ISSUE"

            horiz_dev = abs(self.baseline_gaze_horiz - horiz_ratio)
            if horiz_dev > GAZE_DEV_THRESH_HORIZ:
                cv2.putText(display_image, f"H-GAZE DEV: {horiz_dev:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                return display_image, "LOOKING_AWAY_EYES"

            if abs(final_yaw) > COMBO_YAW and horiz_dev > COMBO_GAZE_HORIZ:
                cv2.putText(display_image, "HYBRID SIDE TRAP", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                return display_image, "LOOKING_SIDEWAYS_SUSPICIOUS"

            if final_pitch > COMBO_PITCH_DOWN and gaze_drop > COMBO_GAZE_DOWN:
                return display_image, "LOOKING_DOWN_SUSPICIOUS"

        return display_image, detection_code

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