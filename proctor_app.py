import cv2
import mediapipe as mp
import numpy as np
import socketio  # The new communication library

# --- NETWORK SETUP ---
sio = socketio.Client()
print("Connecting to server...")

try:
    sio.connect('http://localhost:3000')
    print("Connected to Proctor Server!")
except Exception as e:
    print(f"Connection Failed: {e}")
    print("Ensure server.js is running first!")
    exit()

# --- STUDENT LOGIN ---
print("\n--- EXAM PORTAL ---")
student_name = input("Enter Name: ")
student_roll = input("Enter Roll No: ")
student_sap = input("Enter SAP ID: ")

# Register student with server
sio.emit('student-connect', {'name': student_name, 'roll': student_roll, 'sap': student_sap})

# --- INITIALIZATION ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)

mp_drawing = mp.solutions.drawing_utils 

cap = cv2.VideoCapture(0)

# --- CALIBRATION VARIABLES ---
pitch_offset = 0
yaw_offset = 0
roll_offset = 0
is_calibrated = False

# --- SMOOTHING VARIABLES ---
alpha = 0.2
smooth_pitch = 0
smooth_yaw = 0
smooth_roll = 0

# Track previous status to avoid spamming the server
last_sent_status = ""

def get_head_pose(image, face_landmarks):
    img_h, img_w, img_c = image.shape
    face_3d = np.array([
        (0.0, 0.0, 0.0), (0.0, 330.0, -65.0), (-225.0, -170.0, -135.0),
        (225.0, -170.0, -135.0), (-150.0, 150.0, -125.0), (150.0, 150.0, -125.0)
    ], dtype=np.float64)

    face_2d = []
    target_indices = [1, 152, 33, 263, 61, 291]
    
    for idx in target_indices:
        lm = face_landmarks.landmark[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    face_2d = np.array(face_2d, dtype=np.float64)
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h/2], [0, focal_length, img_w/2], [0, 0, 1]])
    dist_matrix = np.zeros((4,1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2] # Pitch, Yaw, Roll

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, img_c = image.shape

    fd_results = face_detection.process(image)
    fm_results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_count = 0
    status_text = "Status: Normal"
    color = (0, 255, 0)

    if fd_results.detections:
        face_count = len(fd_results.detections)
        for detection in fd_results.detections:
            mp_drawing.draw_detection(image, detection)

    cv2.putText(image, f"Faces: {face_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    if not is_calibrated:
        cv2.putText(image, "Look at screen & Press 'C'", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- VIOLATION LOGIC ---
    if face_count == 0:
        status_text = "VIOLATION: NO FACE"
        color = (0, 0, 255)
    elif face_count > 1:
        status_text = "VIOLATION: MULTIPLE FACES"
        color = (0, 0, 255)
    elif face_count == 1:
        if fm_results.multi_face_landmarks:
            for face_landmarks in fm_results.multi_face_landmarks:
                raw_pitch, raw_yaw, raw_roll = get_head_pose(image, face_landmarks)
                smooth_pitch = (raw_pitch * alpha) + (smooth_pitch * (1.0 - alpha))
                smooth_yaw = (raw_yaw * alpha) + (smooth_yaw * (1.0 - alpha))
                smooth_roll = (raw_roll * alpha) + (smooth_roll * (1.0 - alpha))

                final_pitch = smooth_pitch - pitch_offset
                final_yaw = smooth_yaw - yaw_offset

                if abs(final_pitch) > 25:
                    status_text = "VIOLATION: LOOKING AWAY"
                    color = (0, 0, 255)
                elif abs(final_yaw) > 40:
                    status_text = "VIOLATION: SIDEWAYS LOOK"
                    color = (0, 0, 255)
        else:
            status_text = "VIOLATION: FACE NOT CLEAR"
            color = (0, 0, 255)

    if is_calibrated:
        cv2.putText(image, status_text, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # --- SEND TO SERVER ---
        # Only emit if status changed to prevent lag
        if status_text != last_sent_status:
            sio.emit('student-status-update', status_text)
            last_sent_status = status_text

    cv2.imshow('Proctoring Vision Core', image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('c'):
        pitch_offset = smooth_pitch
        yaw_offset = smooth_yaw
        roll_offset = smooth_roll
        is_calibrated = True

cap.release()
cv2.destroyAllWindows()
sio.disconnect()