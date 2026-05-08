import time
import numpy as np
from scipy.signal import savgol_filter, find_peaks
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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_id VARCHAR(50),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    jaw_disp FLOAT,
                    velocity FLOAT,
                    acceleration FLOAT,
                    rom FLOAT,
                    symmetry FLOAT,
                    chewing_frequency FLOAT,
                    lateral_excursion FLOAT,
                    protrusive_angle FLOAT
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
        self.lower_roll = 0
        self.lower_pitch = 0
        self.prev_time_upper = None
        self.prev_time_lower = None

        self.upper_angles = []
        self.lower_angles = []

        self.full_x = []
        self.full_y = []
        self.full_z = []
        self.timestamps = []

        self.calibrated = False
        self.upper_base = None
        self.lower_base = None
        self.align_offset = None
        
        self.theta_buffer = []
        self.theta_ref = None

        self.prev_disp = 0
        self.prev_time = 0
        self.prev_velocity = 0

        self.max_disp = 0
        self.min_disp = 0

        self.jaw_disp = 0
        self.jaw_velocity = 0
        self.jaw_acceleration = 0
        self.rom = 0
        self.angular_velocity = 0
        self.symmetry = 0
        self.chewing_frequency = 0
        self.protrusive_angle = 0
        self.lateral_excursion = 0

        self.last_db_save_time = 0

state = State()

def clamp(v, limit=50):
    return max(min(v, limit), -limit)

def fuse_imu(ax, ay, az, gx, gy, prev_roll, prev_pitch, prev_time, kf_roll, kf_pitch, current_time):
    if prev_time is None:
        return prev_roll, prev_pitch, current_time

    dt = current_time - prev_time
    if dt <= 0:
        dt = 0.001

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

    if abs(gx) < 0.5 and abs(gy) < 0.5:
        roll *= 0.995
        pitch *= 0.995

    return roll, pitch, current_time

def process_realtime(patient_id):
    if len(state.upper_angles) < 50 or len(state.lower_angles) < 50:
        return None

    # CALIBRATION
    if not state.calibrated:
        state.upper_base = np.mean(state.upper_angles[:50], axis=0)
        state.lower_base = np.mean(state.lower_angles[:50], axis=0)
        state.calibrated = True
        print("✅ Calibration Done")
        return None

    upper = np.array(state.upper_angles[-1])
    lower = np.array(state.lower_angles[-1])

    upper_corr = upper - state.upper_base
    lower_corr = lower - state.lower_base

    # RELATIVE MOTION
    rel = upper_corr - lower_corr
    roll_rel, pitch_rel = rel
    protrusive_angle = pitch_rel

    # ALIGNMENT
    if state.align_offset is None:
        state.align_offset = np.array([roll_rel, pitch_rel])
        print("✅ Sensors aligned")
        return None

    roll_rel -= state.align_offset[0]
    pitch_rel -= state.align_offset[1]

    theta_pitch = np.radians(pitch_rel)
    theta_roll = np.radians(roll_rel)
    theta = np.sqrt(theta_pitch**2 + theta_roll**2)

    # ZERO LOCK
    if state.theta_ref is None:
        state.theta_buffer.append(theta)
        if len(state.theta_buffer) < 50:
            state.jaw_disp = 0
            state.protrusive_angle = 0
            state.lateral_excursion = 0
            return None
        state.theta_ref = np.mean(state.theta_buffer)
        print("✅ Zero locked")
        return None

    theta_rel = theta - state.theta_ref
    if abs(theta_rel) < np.radians(0.5):
        theta_rel = 0

    jaw_disp = 2 * L * np.sin(theta_rel / 2)
    if abs(jaw_disp) < 0.5:
        jaw_disp = 0

    lateral_excursion = 2 * L * np.sin(theta_roll / 2)
    if abs(lateral_excursion) < 0.5:
        lateral_excursion = 0

    current_time = time.time()
    dt = current_time - state.prev_time
    if dt <= 0:
        dt = 0.001

    jaw_velocity = (jaw_disp - state.prev_disp) / dt
    state.prev_disp = jaw_disp
    state.prev_time = current_time

    jaw_acceleration = (jaw_velocity - state.prev_velocity) / dt
    state.prev_velocity = jaw_velocity

    state.max_disp = max(state.max_disp, jaw_disp)
    state.min_disp = min(state.min_disp, jaw_disp)
    rom = state.max_disp - state.min_disp

    angular_velocity = np.sqrt(roll_rel**2 + pitch_rel**2)
    symmetry = abs(roll_rel)

    scale = 2
    lx = clamp(scale * pitch_rel)
    ly = clamp(scale * roll_rel)
    lz = clamp(jaw_disp)

    state.full_x.append(lx)
    state.full_y.append(ly)
    state.full_z.append(lz)
    state.timestamps.append(current_time)

    if len(state.full_x) >= 15:
        state.full_x[-1] = savgol_filter(state.full_x[-15:], 7, 2)[-1]
        state.full_y[-1] = savgol_filter(state.full_y[-15:], 7, 2)[-1]
        state.full_z[-1] = savgol_filter(state.full_z[-15:], 7, 2)[-1]

    if len(state.full_x) > MAX_POINTS:
        state.full_x.pop(0)
        state.full_y.pop(0)
        state.full_z.pop(0)
        state.timestamps.pop(0)

    chewing_frequency = 0
    if len(state.full_z) > 100:
        peaks, _ = find_peaks(state.full_z, distance=20)
        total_time = state.timestamps[-1] - state.timestamps[0]
        if total_time > 0:
            chewing_frequency = len(peaks) / total_time

    # Update state
    state.jaw_disp = jaw_disp
    state.jaw_velocity = jaw_velocity
    state.jaw_acceleration = jaw_acceleration
    state.rom = rom
    state.angular_velocity = angular_velocity
    state.symmetry = symmetry
    state.chewing_frequency = chewing_frequency
    state.protrusive_angle = protrusive_angle
    state.lateral_excursion = lateral_excursion

    # Database saving at 15Hz (approx 0.066s interval)
    if (current_time - state.last_db_save_time) >= (1.0 / 15.0):
        state.last_db_save_time = current_time
        save_to_db(patient_id, jaw_disp, jaw_velocity, jaw_acceleration, rom, symmetry, chewing_frequency, lateral_excursion, protrusive_angle)

    return {
        "disp": jaw_disp,
        "vel": jaw_velocity,
        "acc": jaw_acceleration,
        "rom": rom,
        "freq": chewing_frequency,
        "sym": symmetry,
        "pitch": lx,
        "roll": ly,
        "jaw_opening": lz
    }

def save_to_db(patient_id, disp, vel, acc, rom, sym, freq, lat, protrusive):
    def insert():
        try:
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO session_data 
                    (patient_id, jaw_disp, velocity, acceleration, rom, symmetry, chewing_frequency, lateral_excursion, protrusive_angle) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (patient_id, disp, vel, acc, rom, sym, freq, lat, protrusive))
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
        
        # Process Upper
        upper = data.get('upper')
        if upper and len(upper) == 6:
            ax, ay, az, gx, gy, gz = upper
            state.upper_roll, state.upper_pitch, state.prev_time_upper = fuse_imu(
                ax, ay, az, gx, gy,
                state.upper_roll, state.upper_pitch, state.prev_time_upper,
                state.kf_roll_upper, state.kf_pitch_upper, current_time
            )
            state.upper_angles.append([state.upper_roll, state.upper_pitch])

        # Process Lower
        lower = data.get('lower')
        if lower and len(lower) == 6:
            ax, ay, az, gx, gy, gz = lower
            state.lower_roll, state.lower_pitch, state.prev_time_lower = fuse_imu(
                ax, ay, az, gx, gy,
                state.lower_roll, state.lower_pitch, state.prev_time_lower,
                state.kf_roll_lower, state.kf_pitch_lower, current_time
            )
            state.lower_angles.append([state.lower_roll, state.lower_pitch])

        # Compute Metrics
        metrics = process_realtime(patient_id)
        
        if metrics:
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

@app.route('/patients', methods=['POST'])
@app.route('/patients/', methods=['POST'])
def add_patient():
    data = request.json
    print(f"Received add_patient request: {data}")
    doctor_id = data.get('doctor_id')
    patient_name = data.get('patient_name')
    age = data.get('age')
    gender = data.get('gender')
    phone = data.get('phone')
    medical_condition = data.get('medical_condition')
    assigned_exercise = data.get('assigned_exercise')

    if not doctor_id or not patient_name:
        print("Error: Missing doctor_id or patient_name")
        return jsonify({"message": "Doctor ID and Patient Name are required"}), 400

    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO patients (doctor_id, patient_name, age, gender, phone, medical_condition, assigned_exercise)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (doctor_id, patient_name, age, gender, phone, medical_condition, assigned_exercise))
        conn.close()
        print(f"Patient {patient_name} added successfully for doctor {doctor_id}")
        return jsonify({"message": "Patient added successfully"}), 201
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
            cursor.execute("SELECT * FROM patients WHERE doctor_id = %s", (doctor_id,))
            patients = cursor.fetchall()
        conn.close()
        return jsonify(patients), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500

if __name__ == '__main__':
    init_db()
    print("Starting Flask SocketIO Server on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)