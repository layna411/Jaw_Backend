import time
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.spatial.transform import Rotation as R
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
import threading
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================
L = 100  # Jaw length (mm)
RADIUS = 100 # Protrusive arc radius (mm)
ANGLE_THRESHOLD = 0.5 # Degrees
DISP_THRESHOLD = 0.1 # mm
CAL_GAIN = 1.0 # Calibration multiplier
MAX_POINTS = 10000

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'jaw_rehab',
    'autocommit': True
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is up and running"}), 200

# Initialize database
def init_db():
    try:
        # Connect to MySQL server first to create DB
        conn = pymysql.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'])
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        conn.close()

        # Connect to specific DB
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS doctors (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    full_name VARCHAR(100) NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    phone VARCHAR(20),
                    hospital_name VARCHAR(100),
                    specialization VARCHAR(100),
                    password VARCHAR(255) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    unique_id VARCHAR(50) UNIQUE,
                    doctor_id INT,
                    patient_name VARCHAR(100) NOT NULL,
                    age INT,
                    gender VARCHAR(20),
                    phone VARCHAR(20),
                    medical_condition TEXT,
                    assigned_exercise TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doctor_id) REFERENCES doctors(id)
                )
            """)
            # Ensure unique_id column exists for existing tables
            try:
                cursor.execute("ALTER TABLE patients ADD COLUMN unique_id VARCHAR(50) UNIQUE AFTER id")
                print("Added unique_id column to patients table.")
            except Exception as e:
                if "Duplicate column name" not in str(e):
                    print(f"Alter table error: {e}")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_id VARCHAR(50),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    protrusive_angle FLOAT,
                    protrusive_disp FLOAT
                )
            """)
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Database Initialization Error: {e}")

# =========================================================
# KALMAN FILTER
# =========================================================
class KalmanFilter:
    def __init__(self):
        self.Q = 0.001
        self.R = 0.03
        self.x = 0.0
        self.P = 1.0

    def update(self, measurement, rate, dt):
        self.x += rate * dt
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)
        return self.x

# =========================================================
# STATE VARIABLES
# =========================================================
class State:
    def __init__(self):
        self.kf_roll_upper = KalmanFilter()
        self.kf_pitch_upper = KalmanFilter()
        self.kf_roll_lower = KalmanFilter()
        self.kf_pitch_lower = KalmanFilter()

        self.upper_roll = 0
        self.upper_pitch = 0
        self.upper_yaw = 0
        self.lower_roll = 0
        self.lower_pitch = 0
        self.lower_yaw = 0
        
        self.prev_time_upper = None
        self.prev_time_lower = None

        self.samples = []
        self.calibrated = False
        self.base = 0
        
        self.measure = False
        self.max_angle = 0
        self.max_disp = 0
        self.last_move = 0
        
        self.protrusive_angle = 0
        self.protrusive_disp = 0
        self.last_db_save_time = 0
        self.last_metrics_emit_time = 0
        self.last_articulator_print_time = 0
        self.last_incoming_print_time = 0

state = State()

def fuse_imu(ax, ay, az, gx, gy, gz, prev_roll, prev_pitch, prev_yaw, prev_time, kf_roll, kf_pitch, current_time):
    if prev_time is None:
        return prev_roll, prev_pitch, prev_yaw, current_time

    dt = current_time - prev_time
    if dt <= 0: dt = 0.001

    roll_acc = np.degrees(np.arctan2(ay, az))
    pitch_acc = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))

    acc_mag = np.sqrt(ax*ax + ay*ay + az*az)
    if abs(acc_mag - 1.0) > 0.15:
        kf_roll.R = 0.2
        kf_pitch.R = 0.2
    else:
        kf_roll.R = 0.03
        kf_pitch.R = 0.03

    roll = kf_roll.update(roll_acc, gx, dt)
    pitch = kf_pitch.update(pitch_acc, gy, dt)
    yaw = prev_yaw + gz * dt

    return roll, pitch, yaw, current_time

def get_quaternion(roll, pitch, yaw):
    from scipy.spatial.transform import Rotation as R
    return R.from_euler('xyz', [roll, pitch, yaw], degrees=True)

def process_realtime(patient_id):
    global state
    
    # 1. Check if we have enough orientation data
    # (Simplified: we use the latest fused values)
    
    u_quat = get_quaternion(state.upper_roll, state.upper_pitch, state.upper_yaw)
    l_quat = get_quaternion(state.lower_roll, state.lower_pitch, state.lower_yaw)
    
    # RELATIVE ORIENTATION
    rel = u_quat.inv() * l_quat
    rotvec = rel.as_rotvec()
    protrusive_raw = np.degrees(rotvec[1]) # Y-axis rotation (Pitch)

    # 2. CALIBRATION (100 samples median)
    if not state.calibrated:
        state.samples.append(protrusive_raw)
        progress = len(state.samples)
        socketio.emit('session_status', {'message': f'Calibrating {progress}/100', 'type': 'steady'})
        
        if len(state.samples) < 100:
            return None
            
        state.base = np.median(state.samples)
        state.calibrated = True
        state.measure = False
        print("✅ CALIBRATION DONE")
        socketio.emit('session_status', {'message': 'CALIBRATION DONE', 'type': 'success'})
        socketio.emit('session_status', {'message': 'READY TO MOVE JAW', 'type': 'ready'})
        return None

    # 3. MEASUREMENT TRIGGER (Always on after calibration for live view)
    angle = protrusive_raw - state.base
    if abs(angle) < ANGLE_THRESHOLD:
        angle = 0
        
    theta = np.radians(angle)
    raw_disp = 2 * RADIUS * np.sin(theta / 2)
    protrusive_disp = CAL_GAIN * raw_disp
    
    if abs(protrusive_disp) < DISP_THRESHOLD:
        protrusive_disp = 0

    # TRACK MAX
    if abs(angle) > abs(state.max_angle):
        state.max_angle = angle
        state.max_disp = protrusive_disp
        state.last_move = time.time()

    current_time = time.time()
    
    # Update state for reporting
    state.protrusive_angle = angle
    state.protrusive_disp = protrusive_disp

    # Database saving at 10Hz
    if (current_time - state.last_db_save_time) >= 0.1:
        state.last_db_save_time = current_time
        save_to_db(patient_id, state.protrusive_angle, state.protrusive_disp)

    # Emit results
    articulator_payload = f"<{state.protrusive_angle:.2f},{state.protrusive_disp:.2f}>"
    
    if (current_time - state.last_articulator_print_time) >= 2.0:
        state.last_articulator_print_time = current_time
        socketio.emit('articulator_cmd', articulator_payload)
        print(f"🔌 Broadcasted to Articulator (throttled): {articulator_payload}")

    return {
        "protrusive_angle": state.protrusive_angle,
        "protrusive_disp": state.protrusive_disp,
        "max_angle": state.max_angle,
        "max_disp": state.max_disp
    }

def save_to_db(patient_id, protrusive, protrusive_disp):
    def insert():
        try:
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO session_data 
                    (patient_id, protrusive_angle, protrusive_disp) 
                    VALUES (%s, %s, %s)
                """, (patient_id, protrusive, protrusive_disp))
            conn.close()
        except Exception as e:
            print(f"DB Insert Error: {e}")
    threading.Thread(target=insert).start()

# =========================================================
# SOCKETIO HANDLERS
# =========================================================
@socketio.on('connect')
def handle_connect():
    print('Client connected to websocket.')
    emit('status', {'message': 'Connected to Flask Backend'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from websocket.')

@socketio.on('register_articulator')
def handle_register_articulator():
    print("🤖 ESP32 Articulator connected and registered via WebSockets!")
    emit('status', {'message': 'Articulator registered successfully'})

@socketio.on('sensor_data')
def handle_sensor_data(data):
    """
    Expected JSON payload:
    {
      "patient_id": "123",
      "upper": [ax, ay, az, gx, gy, gz],
      "lower": [ax, ay, az, gx, gy, gz]
    }
    """
    try:
        current_time = time.time()
        patient_id = data.get('patient_id', 'Unknown')
        
        if (current_time - state.last_incoming_print_time) >= 2.0:
            state.last_incoming_print_time = current_time
            print(f"📦 Incoming Socket Data from Patient: {patient_id} (throttled)")
        
        # Process Upper
        upper = data.get('upper')
        if upper and len(upper) >= 6:
            ax, ay, az, gx, gy, gz = upper[:6]
            state.upper_roll, state.upper_pitch, state.upper_yaw, state.prev_time_upper = fuse_imu(
                ax, ay, az, gx, gy, gz,
                state.upper_roll, state.upper_pitch, state.upper_yaw, state.prev_time_upper,
                state.kf_roll_upper, state.kf_pitch_upper, current_time
            )

        # Process Lower
        lower = data.get('lower')
        if lower and len(lower) >= 6:
            ax, ay, az, gx, gy, gz = lower[:6]
            state.lower_roll, state.lower_pitch, state.lower_yaw, state.prev_time_lower = fuse_imu(
                ax, ay, az, gx, gy, gz,
                state.lower_roll, state.lower_pitch, state.lower_yaw, state.prev_time_lower,
                state.kf_roll_lower, state.kf_pitch_lower, current_time
            )

        # Compute Metrics
        metrics = process_realtime(patient_id)
        
        if metrics:
            if (current_time - state.last_metrics_emit_time) >= 2.0:
                state.last_metrics_emit_time = current_time
                print(f"📊 Emitting metrics to Patient {patient_id} (throttled): {metrics}")
                emit('metrics', metrics)

    except Exception as e:
        print(f"Error processing sensor data: {e}")

@app.route('/reset', methods=['POST'])
def reset_state():
    global state
    state = State()
    return jsonify({"message": "State reset successful."})

@app.route('/auth/register', methods=['POST'])
@app.route('/auth/register/', methods=['POST'])
def register():
    data = request.json
    full_name = data.get('full_name')
    email = data.get('email')
    phone = data.get('phone')
    hospital_name = data.get('hospital_name')
    specialization = data.get('specialization')
    password = data.get('password')

    if not all([full_name, email, password]):
        return jsonify({"message": "Missing required fields"}), 400

    hashed_password = generate_password_hash(password)

    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM doctors WHERE email = %s", (email,))
            if cursor.fetchone():
                conn.close()
                return jsonify({"message": "Email already registered"}), 409

            cursor.execute("""
                INSERT INTO doctors 
                (full_name, email, phone, hospital_name, specialization, password) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (full_name, email, phone, hospital_name, specialization, hashed_password))
        conn.close()
        return jsonify({"message": "Doctor registered successfully"}), 201
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({"message": "Internal server error"}), 500

@app.route('/auth/login', methods=['POST'])
@app.route('/auth/login/', methods=['POST'])
def login():
    # ... (existing login code)
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({"message": "Missing email or password"}), 400

    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("SELECT * FROM doctors WHERE email = %s", (email,))
            doctor = cursor.fetchone()
        conn.close()

        if doctor and check_password_hash(doctor['password'], password):
            doctor_data = {
                "id": doctor['id'],
                "full_name": doctor['full_name'],
                "email": doctor['email'],
                "phone": doctor['phone'],
                "hospital_name": doctor['hospital_name'],
                "specialization": doctor['specialization']
            }
            return jsonify({"message": "Login successful", "doctor": doctor_data}), 200
        else:
            return jsonify({"message": "Invalid email or password"}), 401
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"message": "Internal server error"}), 500

@app.route('/auth/update-doctor', methods=['POST'])
@app.route('/auth/update-doctor/', methods=['POST'])
def update_doctor():
    print("Received update_doctor request")
    data = request.json
    doctor_id = data.get('id')
    full_name = data.get('full_name')
    email = data.get('email')
    phone = data.get('phone')
    hospital_name = data.get('hospital_name')
    specialization = data.get('specialization')

    if not all([doctor_id, full_name, email]):
        return jsonify({"message": "Missing required fields"}), 400

    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            # Check if email is taken by another doctor
            cursor.execute("SELECT id FROM doctors WHERE email = %s AND id != %s", (email, doctor_id))
            if cursor.fetchone():
                conn.close()
                return jsonify({"message": "Email already in use"}), 409

            cursor.execute("""
                UPDATE doctors 
                SET full_name = %s, email = %s, phone = %s, hospital_name = %s, specialization = %s
                WHERE id = %s
            """, (full_name, email, phone, hospital_name, specialization, doctor_id))
        conn.close()
        return jsonify({"message": "Profile updated successfully"}), 200
    except Exception as e:
        print(f"Update error: {e}")
        return jsonify({"message": "Internal server error"}), 500

@app.route('/patients', methods=['POST'])
@app.route('/patients/', methods=['POST'])
def add_patient():
    data = request.json
    print(f"Received add_patient request: {data}")
    doctor_id = data.get('doctor_id')
    patient_name = data.get('patient_name')
    age = data.get('age')
    phone = data.get('phone')
    
    # Optional fields for backward compatibility if needed, but the user wants only name, age, phone
    gender = data.get('gender', 'Not Specified')
    medical_condition = data.get('medical_condition', '')
    assigned_exercise = data.get('assigned_exercise', '')

    if not doctor_id or not patient_name:
        print("Error: Missing doctor_id or patient_name")
        return jsonify({"message": "Doctor ID and Patient Name are required"}), 400

    # Generate Unique Patient ID
    unique_id = f"PAT-{int(time.time() * 1000) % 1000000:06d}"

    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            # Check if unique_id exists (highly unlikely but good practice)
            cursor.execute("SELECT id FROM patients WHERE unique_id = %s", (unique_id,))
            if cursor.fetchone():
                unique_id = f"PAT-{int(time.time() * 1000) % 1000000 + 1:06d}"

            cursor.execute("""
                INSERT INTO patients (unique_id, doctor_id, patient_name, age, gender, phone, medical_condition, assigned_exercise)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (unique_id, doctor_id, patient_name, age, gender, phone, medical_condition, assigned_exercise))
        conn.close()
        print(f"Patient {patient_name} (ID: {unique_id}) added successfully for doctor {doctor_id}")
        return jsonify({
            "message": "Patient added successfully",
            "unique_id": unique_id
        }), 201
    except Exception as e:
        print(f"Error adding patient: {e}")
        return jsonify({"message": str(e)}), 500


@app.route('/patients', methods=['GET'])
@app.route('/patients/', methods=['GET'])
def get_patients():
    doctor_id = request.args.get('doctor_id')
    if not doctor_id:
        return jsonify({"message": "Doctor ID is required"}), 400

    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT p.*, 
                (SELECT protrusive_angle FROM session_data WHERE patient_id = p.unique_id ORDER BY timestamp DESC LIMIT 1) as latest_angle
                FROM patients p WHERE p.doctor_id = %s
            """, (doctor_id,))
            patients = cursor.fetchall()

        conn.close()
        return jsonify(patients), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    doctor_id = request.args.get('doctor_id')
    if not doctor_id:
        return jsonify({"message": "Doctor ID is required"}), 400

    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # Total Patients
            cursor.execute("SELECT COUNT(*) as count FROM patients WHERE doctor_id = %s", (doctor_id,))
            total_patients = cursor.fetchone()['count']

            # Active Sessions (unique patients who had sessions in the last 7 days)
            cursor.execute("""
                SELECT COUNT(DISTINCT patient_id) as count 
                FROM session_data 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                AND patient_id IN (SELECT unique_id FROM patients WHERE doctor_id = %s)
            """, (doctor_id,))
            active_sessions = cursor.fetchone()['count']

            # Avg Recovery (based on max angle, benchmark 45°)
            cursor.execute("""
                SELECT AVG(protrusive_angle) as avg_angle 
                FROM session_data 
                WHERE patient_id IN (SELECT unique_id FROM patients WHERE doctor_id = %s)
            """, (doctor_id,))
            avg_angle = cursor.fetchone()['avg_angle'] or 0
            avg_recovery = (avg_angle / 45.0) * 100

            # Reports Today
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM session_data 
                WHERE DATE(timestamp) = CURDATE()
                AND patient_id IN (SELECT unique_id FROM patients WHERE doctor_id = %s)
            """, (doctor_id,))
            reports_today = cursor.fetchone()['count']

        conn.close()
        return jsonify({
            "total_patients": total_patients,
            "active_sessions": active_sessions,
            "avg_recovery": round(min(avg_recovery, 100.0), 1),
            "reports_today": reports_today
        }), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route('/sessions', methods=['GET'])

@app.route('/sessions/', methods=['GET'])
def get_sessions():
    patient_id = request.args.get('patient_id')
    if not patient_id:
        return jsonify({"message": "Patient ID is required"}), 400

    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # Get aggregated session data per day or raw data
            cursor.execute("""
                SELECT 
                    DATE(timestamp) as session_date,
                    MAX(protrusive_angle) as max_angle,
                    MAX(protrusive_disp) as max_disp
                FROM session_data 
                WHERE patient_id = %s
                GROUP BY DATE(timestamp)
                ORDER BY session_date DESC
            """, (patient_id,))
            sessions = cursor.fetchall()
            
            # Convert dates to strings for JSON serialization
            for session in sessions:
                if 'session_date' in session and session['session_date']:
                    session['session_date'] = session['session_date'].strftime('%Y-%m-%d')
                    
        conn.close()
        return jsonify(sessions), 200
    except Exception as e:
        print(f"Error fetching sessions: {e}")
        return jsonify({"message": str(e)}), 500

if __name__ == '__main__':
    init_db()
    print("Starting Flask SocketIO Server on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True, debug=True)