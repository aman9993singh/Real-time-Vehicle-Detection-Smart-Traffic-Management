from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from werkzeug.utils import secure_filename
from Vehicle_detection import TrafficDetector
from Traffic_logic import TrafficLogic

app = Flask(__name__)

# --- Configuration ---
ADMIN_USER = "admin"
ADMIN_PASS = "mtech2026"
UPLOAD_FOLDER = 'static/uploads'
FIXED_SIZE = (640, 480)  # The "Secret Sauce" for visual consistency

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize System Objects
detector = TrafficDetector(r"models\yolo26n.pt", r"models\best.pt")
logic = TrafficLogic()

video_paths = {1: None, 2: None, 3: None, 4: None}

@app.route('/')
def index():
    return render_template('admin_login.html')

@app.route('/login', methods=['POST'])
def login():
    user = request.form.get('username')
    pw = request.form.get('password')
    if user == ADMIN_USER and pw == ADMIN_PASS:
        return redirect(url_for('upload'))
    return "<h3>Invalid Credentials. Please go back and try again.</h3>"

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        for i in range(1, 5):
            file = request.files.get(f'lane{i}')
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"lane_{i}_{filename}")
                file.save(save_path)
                video_paths[i] = save_path
        return redirect(url_for('dashboard'))
    return render_template('upload_video.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    global video_paths
    video_paths = {1: None, 2: None, 3: None, 4: None}
    return redirect(url_for('index'))

def draw_visuals(frame, lane_id, count, is_ambulance):
    # Since we resize in gen_frames, h will always be 480 and w will be 640
    h, w = frame.shape[:2]
    
    status = logic.get_lane_status(lane_id).upper()
    timer_val = logic.get_timer_text(lane_id)

    # 1. FULL-HEIGHT TRAFFIC LIGHT BOX (Left side)
    cv2.rectangle(frame, (0, 0), (75, h), (35, 35, 35), -1)
    
    center_x = 37
    OFF, RED, ORANGE, GREEN = (20, 20, 20), (0, 0, 255), (0, 165, 255), (0, 255, 0)
    
    # Precise positioning for 480px height
    cv2.circle(frame, (center_x, 80), 28, RED if status == 'RED' else OFF, -1)
    cv2.circle(frame, (center_x, 240), 28, ORANGE if status == 'ORANGE' else OFF, -1)
    cv2.circle(frame, (center_x, 400), 28, GREEN if status == 'GREEN' else OFF, -1)

    # 2. DATA DESCRIPTION OVERLAY (Bottom bar)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Text rendering
    text = f"Density: {count} | Amb: {'YES' if is_ambulance else 'No'} | {timer_val}"
    cv2.putText(frame, text, (90, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Lane Label (Top Right)
    cv2.putText(frame, f"LANE {lane_id}", (w-120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def gen_frames(lane_id):
    path = video_paths.get(lane_id)
    if not path: return
    cap = cv2.VideoCapture(path)
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # --- CRITICAL FIX: Resize BEFORE processing and drawing ---
        frame = cv2.resize(frame, FIXED_SIZE)
        
        # Detection
        processed, count, amb = detector.process_frame(frame)
        
        # Logic Update
        logic.update_state(lane_id, count, amb)
        
        # Draw Visuals (Now perfectly consistent across all lanes)
        final_frame = draw_visuals(processed, lane_id, count, amb)
        
        ret, buffer = cv2.imencode('.jpg', final_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed/<int:lane_id>')
def video_feed(lane_id):
    return Response(gen_frames(lane_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)