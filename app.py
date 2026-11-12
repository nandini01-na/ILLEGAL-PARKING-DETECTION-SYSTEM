import os
import time
from datetime import datetime
import cv2
import numpy as np
import serial
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# ==========================================
# 1. HARDWARE CONFIGURATION
# ==========================================
SERIAL_PORT = 'COM3'  
BAUD_RATE = 9600
ser = None

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    ser.write(b'C')
    print(f"[HW] Hardware connected on {SERIAL_PORT}")
except Exception as e:
    print(f"[HW] Hardware warning: {e}. Running simulation.")

def send_hardware_signal(cmd):
    global ser
    if ser is not None and ser.is_open:
        try:
            ser.write(cmd.encode())
        except Exception as e:
            print(f"[HW] Serial Error: {e}")

# ==========================================
# 2. STREAM & DYNAMIC ZONE CONFIGURATION
# ==========================================
PHONE_IP = "192.168.137.44"
DROIDCAM_URL = f"http://{PHONE_IP}:4747/video"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), 'static', 'violations')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

current_mode = "file"  
video_file_path = None
cap = None

ZONE_PRESETS = {
    "center": np.array([[180, 180], [460, 180], [520, 450], [120, 450]], np.int32),
    "left": np.array([[30, 180], [280, 180], [330, 450], [10, 450]], np.int32),
    "right": np.array([[360, 180], [610, 180], [630, 450], [310, 450]], np.int32),
    "wide": np.array([[60, 160], [580, 160], [620, 460], [20, 460]], np.int32)
}
current_zone_name = "wide"
ZONE_POLYGON = ZONE_PRESETS["wide"]

# ==========================================
# 3. AI MODEL & TRACKING STATE
# ==========================================
model = YOLO('yolov8n.pt')
ALLOWED_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

vehicle_start_times = {}
vehicle_last_seen = {}
vehicle_positions = {}
logged_violations = set()
incident_logs = []
current_system_state = "CLEAR"
last_sent_signal = 'C'
current_max_timer = 0.0

FINE_PER_VIOLATION = 500

# ==========================================
# 4. COMPUTER VISION ENGINE
# ==========================================
frame_counter = 0
last_boxes = []
last_ids = []
last_classes = []

def process_frame(frame):
    global current_system_state, last_sent_signal, frame_counter
    global last_boxes, last_ids, last_classes, ZONE_POLYGON, current_max_timer
    global vehicle_start_times, vehicle_last_seen, vehicle_positions

    if frame is None:
        return frame

    cv2.polylines(frame, [ZONE_POLYGON], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.putText(frame, f"ZONE: {current_zone_name.upper()}", (ZONE_POLYGON[0][0], max(ZONE_POLYGON[0][1] - 8, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    frame_counter += 1
    current_time = time.time()
    max_stop_duration = 0.0

    if frame_counter % 2 == 0:
        results = model.track(frame, persist=True, verbose=False, imgsz=480, conf=0.15)
        if results and results[0].boxes and results[0].boxes.id is not None:
            last_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            last_ids = results[0].boxes.id.cpu().numpy().astype(int)
            last_classes = results[0].boxes.cls.cpu().numpy().astype(int)
        else:
            last_boxes, last_ids, last_classes = [], [], []

    for box, track_id, cls_id in zip(last_boxes, last_ids, last_classes):
        if cls_id not in ALLOWED_CLASSES:
            continue

        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        is_inside = (cv2.pointPolygonTest(ZONE_POLYGON, (float(cx), float(cy)), False) >= 0) or \
                    (cv2.pointPolygonTest(ZONE_POLYGON, (float(cx), float(y2)), False) >= 0)

        if is_inside:
            vehicle_last_seen[track_id] = current_time

            if track_id not in vehicle_start_times:
                vehicle_start_times[track_id] = current_time
                vehicle_positions[track_id] = (cx, cy)

            prev_x, prev_y = vehicle_positions[track_id]
            dist_moved = np.hypot(cx - prev_x, cy - prev_y)

            # High movement threshold (180px) taaki camera angle ya bus ke halka hilne se timer zero na ho
            if dist_moved > 180.0:
                vehicle_start_times[track_id] = current_time
                vehicle_positions[track_id] = (cx, cy)
                duration = 0.0
            else:
                duration = current_time - vehicle_start_times[track_id]

            if duration > max_stop_duration:
                max_stop_duration = duration

            # Strict Timeline: 0-10s Clear, 10-30s Warning, 30s+ Violation
            if duration >= 30.0:
                color = (0, 0, 255)
                label = f"VIOLATION: ID {track_id} ({duration:.0f}s)"

                if track_id not in logged_violations:
                    logged_violations.add(track_id)
                    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    fname = f"viol_{track_id}_{int(current_time)}.jpg"
                    fpath = os.path.join(SNAPSHOT_DIR, fname)
                    cv2.imwrite(fpath, frame)

                    incident_logs.insert(0, {
                        "vehicle_id": f"VEH-{track_id}",
                        "class": model.names[cls_id].title(),
                        "timestamp": ts_str,
                        "fine": f"₹{FINE_PER_VIOLATION}",
                        "snapshot": f"/static/violations/{fname}"
                    })

            elif duration >= 10.0:
                color = (0, 165, 255)
                label = f"WARNING: ID {track_id} ({duration:.0f}s)"
            else:
                color = (0, 255, 0)
                label = f"CLEAR: ID {track_id} ({duration:.1f}s)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # 15s buffer: agar tracker 1-2 second ke liye ID bhool jaye toh bhi timer safe rahe
    dead_ids = [vid for vid, l_time in vehicle_last_seen.items() if (current_time - l_time) > 15.0]
    for vid in dead_ids:
        vehicle_start_times.pop(vid, None)
        vehicle_last_seen.pop(vid, None)
        vehicle_positions.pop(vid, None)

    current_max_timer = max_stop_duration

    # State Machine: 10s par Warning ('W'), 30s par Violation ('V')
    target_state = "CLEAR"
    target_signal = 'C'
    if max_stop_duration >= 30.0:
        target_state = "VIOLATION"
        target_signal = 'V'
    elif max_stop_duration >= 10.0:
        target_state = "WARNING"
        target_signal = 'W'

    current_system_state = target_state
    if target_signal != last_sent_signal:
        send_hardware_signal(target_signal)
        last_sent_signal = target_signal

    return frame

# ==========================================
# 5. STREAM PIPELINE
# ==========================================
def get_capture():
    global current_mode, video_file_path, DROIDCAM_URL
    if current_mode == "phone":
        c = cv2.VideoCapture(DROIDCAM_URL)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return c
    elif current_mode == "file":
        if video_file_path and os.path.exists(video_file_path):
            return cv2.VideoCapture(video_file_path)
        else:
            files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.mp4', '.avi'))]
            if files:
                video_file_path = files[-1]
                return cv2.VideoCapture(video_file_path)
            current_mode = "webcam"
            return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    elif current_mode == "webcam":
        c = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return c
    return None

def generate_frames():
    global cap, current_mode

    while True:
        if current_mode == "stopped":
            time.sleep(0.3)
            continue

        if cap is None or not cap.isOpened():
            cap = get_capture()
            if cap is None or not cap.isOpened():
                time.sleep(0.5)
                continue

        if current_mode in ["phone", "webcam"]:
            cap.grab()

        success, frame = cap.read()
        if not success:
            if current_mode == "file":
                # Video loop hone par time aur ID safe rahega
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                time.sleep(0.1)
                continue

        frame = cv2.resize(frame, (640, 480))
        processed = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# 6. API ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_zone/<zone_type>')
def set_zone(zone_type):
    global ZONE_POLYGON, current_zone_name
    if zone_type in ZONE_PRESETS:
        ZONE_POLYGON = ZONE_PRESETS[zone_type]
        current_zone_name = zone_type
        return jsonify({"status": "success", "zone": zone_type}), 200
    return jsonify({"status": "error", "message": "Unknown zone"}), 400

@app.route('/start_phone')
def start_phone():
    global current_mode, cap
    current_mode = "phone"
    if cap is not None:
        cap.release()
        cap = None
    return jsonify({"status": "phone"}), 200

@app.route('/start_webcam')
def start_webcam():
    global current_mode, cap
    current_mode = "webcam"
    if cap is not None:
        cap.release()
        cap = None
    return jsonify({"status": "webcam"}), 200

@app.route('/stop_feed')
def stop_feed():
    global current_mode, cap
    current_mode = "stopped"
    if cap is not None:
        cap.release()
        cap = None
    send_hardware_signal('C')
    return jsonify({"status": "stopped"}), 200

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_mode, video_file_path, cap
    try:
        if 'video' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400

        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)
        video_file_path = save_path
        current_mode = "file"
        if cap is not None:
            cap.release()
            cap = None
        return jsonify({"status": "success", "file": file.filename}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stats')
def stats():
    total_violations = len(logged_violations)
    return jsonify({
        "status": current_system_state,
        "tracked": len(vehicle_start_times),
        "violations": total_violations,
        "total_fine": total_violations * FINE_PER_VIOLATION,
        "duration": round(current_max_timer, 1),
        "zone": current_zone_name,
        "mode": current_mode,
        "incident_logs": incident_logs[:10]
    }), 200

@app.route('/clear_logs')
def clear_logs():
    global incident_logs, logged_violations
    incident_logs.clear()
    logged_violations.clear()
    return jsonify({"status": "cleared"}), 200

@app.route('/reset')
def reset():
    global logged_violations, vehicle_start_times, vehicle_last_seen, vehicle_positions
    global incident_logs, current_system_state, last_sent_signal, current_max_timer
    logged_violations.clear()
    vehicle_start_times.clear()
    vehicle_last_seen.clear()
    vehicle_positions.clear()
    incident_logs.clear()
    current_max_timer = 0.0
    current_system_state = "CLEAR"
    last_sent_signal = 'C'
    send_hardware_signal('C')
    return jsonify({"status": "reset"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)