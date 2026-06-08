/*
 * ============================================================
 *   ESP32 Jaw Articulator Controller
 *   Hardware: 2x 28BYJ-48 Stepper + 2x ULN2003 Driver Board
 * ============================================================
 *
 *  WHAT THIS CODE DOES:
 *  ---------------------
 *  1. Connects ESP32 to your WiFi network via a captive-portal
 *     (no re-flashing needed – saves credentials to flash).
 *  2. Connects to your Flask-SocketIO backend server.
 *  3. Listens for the 'articulator_cmd' event which carries
 *     a payload like  <angle, displacement>
 *     e.g.  <12.50, 3.20>
 *  4. Converts the floating-point angle & displacement values
 *     into stepper-motor STEPS and moves each motor accordingly.
 *
 *  MOTOR ASSIGNMENT:
 *  -----------------
 *  Motor-1 (Angle Motor)       → controls the opening angle of the jaw
 *  Motor-2 (Displacement Motor)→ controls the protrusive displacement
 *
 *  REQUIRED ARDUINO LIBRARIES (install via Library Manager):
 *  ----------------------------------------------------------
 *  1. WiFiManager by tzapu
 *  2. WebSockets  by Links2004
 *  3. SocketIOclient (included inside WebSockets library above)
 *
 * ============================================================
 */

// ── Standard / Platform ────────────────────────────────────
#include <WiFi.h>
#include <WebServer.h>
#include <WiFiManager.h>        // Captive-portal provisioning
#include <Preferences.h>        // NVS flash storage
#include <WebSocketsClient.h>
#include <SocketIOclient.h>

// ============================================================
//  STEPPER MOTOR PIN DEFINITIONS
// ============================================================
/*
 *  28BYJ-48 has 4 coil wires (IN1, IN2, IN3, IN4).
 *  Connect them to the matching pins on the ULN2003 board.
 *  Then connect the ULN2003's IN1-IN4 to ESP32 GPIOs below.
 *
 *  ┌──────────────┐   ULN2003 Board-1      ESP32
 *  │ Motor 1      │   IN1 ─────────────── GPIO 14
 *  │ (Angle)      │   IN2 ─────────────── GPIO 27
 *  │              │   IN3 ─────────────── GPIO 26
 *  │              │   IN4 ─────────────── GPIO 25
 *  └──────────────┘
 *
 *  ┌──────────────┐   ULN2003 Board-2      ESP32
 *  │ Motor 2      │   IN1 ─────────────── GPIO 33
 *  │ (Displace)   │   IN2 ─────────────── GPIO 32
 *  │              │   IN3 ─────────────── GPIO 19
 *  │              │   IN4 ─────────────── GPIO 18
 *  └──────────────┘
 *
 *  Power: Connect the ULN2003 board's "+" to 5V (external/USB)
 *         and "-" to GND (shared with ESP32 GND).
 *         ⚠️  Never power the stepper from the ESP32 3.3V pin!
 */

// Motor 1 – Angle
const int M1_IN1 = 14;
const int M1_IN2 = 27;
const int M1_IN3 = 26;
const int M1_IN4 = 25;

// Motor 2 – Displacement
const int M2_IN1 = 33;
const int M2_IN2 = 32;
const int M2_IN3 = 19;
const int M2_IN4 = 18;

// ============================================================
//  STEPPER MOTOR CONSTANTS
// ============================================================
/*
 *  28BYJ-48 specification:
 *   - Internal gear ratio : 1/64
 *   - Steps per revolution (full drive): 512 steps for 360°
 *     (more precisely 4096 half-steps; we use full-step here)
 *
 *  ANGLE MOTOR mapping:
 *   - Patient jaw opens 0° – 45°.
 *   - We map 0°  →   0 steps
 *             45° → 256 steps  (half rotation = 180° of motor output)
 *
 *  DISPLACEMENT MOTOR mapping:
 *   - Jaw protrusion 0 mm – 10 mm.
 *   - We map 0 mm  →   0 steps
 *             10 mm → 256 steps
 */
const int STEPS_PER_REVOLUTION = 512;    // Full steps for one 360° turn
const int MAX_ANGLE_STEPS      = 256;    // Steps for 45° jaw angle
const int MAX_DISP_STEPS       = 256;    // Steps for 10 mm displacement
const int MAX_PATIENT_ANGLE    = 45;     // degrees
const int MAX_PATIENT_DISP     = 10;     // mm

// Step speed: lower = faster.  8 ms per step is safe for 28BYJ-48
const int STEP_DELAY_MS = 8;

// ============================================================
//  HALF-STEP SEQUENCE TABLE FOR 28BYJ-48
// ============================================================
/*
 *  The 28BYJ-48 uses a 4-phase coil arrangement.
 *  A "half-step" sequence gives smoother motion with 8 states.
 *
 *  Each row below is one step:  { IN1, IN2, IN3, IN4 }
 *  Energise coils in this order → shaft rotates FORWARD.
 *  Reverse the order            → shaft rotates BACKWARD.
 */
const int STEP_SEQ[8][4] = {
  {1, 0, 0, 0},   // Step 0
  {1, 1, 0, 0},   // Step 1
  {0, 1, 0, 0},   // Step 2
  {0, 1, 1, 0},   // Step 3
  {0, 0, 1, 0},   // Step 4
  {0, 0, 1, 1},   // Step 5
  {0, 0, 0, 1},   // Step 6
  {1, 0, 0, 1},   // Step 7
};
const int NUM_HALF_STEPS = 8;  // total states in the sequence

// ============================================================
//  MOTOR POSITION TRACKING
// ============================================================
// We track the current position (in steps) of each motor
// so we can calculate how many steps to move and in which direction.
int currentAngleSteps = 0;      // current position of Motor-1
int currentDispSteps  = 0;      // current position of Motor-2

// ============================================================
//  NETWORK / SOCKET OBJECTS
// ============================================================
Preferences    preferences;
SocketIOclient socketIO;

String server_ip;
String server_port;
bool   shouldSaveConfig = false;

// Callback from WiFiManager when settings need saving
void saveConfigCallback() {
    Serial.println("[Portal] Saving new server config.");
    shouldSaveConfig = true;
}

// ============================================================
//  LOW-LEVEL STEPPER HELPERS
// ============================================================

/*
 *  stepMotor1() – drives Motor-1 (Angle) one half-step.
 *  stepIndex : which row of STEP_SEQ to use (0-7)
 */
void stepMotor1(int stepIndex) {
    digitalWrite(M1_IN1, STEP_SEQ[stepIndex][0]);
    digitalWrite(M1_IN2, STEP_SEQ[stepIndex][1]);
    digitalWrite(M1_IN3, STEP_SEQ[stepIndex][2]);
    digitalWrite(M1_IN4, STEP_SEQ[stepIndex][3]);
}

/*
 *  stepMotor2() – drives Motor-2 (Displacement) one half-step.
 */
void stepMotor2(int stepIndex) {
    digitalWrite(M2_IN1, STEP_SEQ[stepIndex][0]);
    digitalWrite(M2_IN2, STEP_SEQ[stepIndex][1]);
    digitalWrite(M2_IN3, STEP_SEQ[stepIndex][2]);
    digitalWrite(M2_IN4, STEP_SEQ[stepIndex][3]);
}

/*
 *  deenergiseMotor1() – turns off all coils of Motor-1.
 *  Always call this after the motor finishes moving to prevent
 *  overheating (28BYJ-48 draws current even when stationary).
 */
void deenergiseMotor1() {
    digitalWrite(M1_IN1, 0);
    digitalWrite(M1_IN2, 0);
    digitalWrite(M1_IN3, 0);
    digitalWrite(M1_IN4, 0);
}

void deenergiseMotor2() {
    digitalWrite(M2_IN1, 0);
    digitalWrite(M2_IN2, 0);
    digitalWrite(M2_IN3, 0);
    digitalWrite(M2_IN4, 0);
}

/*
 *  moveMotor1(targetSteps)
 *  -----------------------
 *  Moves Motor-1 from its CURRENT position to targetSteps.
 *  Automatically chooses direction (CW / CCW).
 *
 *  targetSteps : absolute position in the range [0, MAX_ANGLE_STEPS]
 */
void moveMotor1(int targetSteps) {
    int stepsToMove = targetSteps - currentAngleSteps; // + = forward, - = backward

    if (stepsToMove == 0) return;  // already there

    int direction = (stepsToMove > 0) ? 1 : -1;
    int absSteps  = abs(stepsToMove);

    Serial.printf("[Motor1-Angle] Moving %d half-steps %s\n",
                  absSteps, direction > 0 ? "FORWARD" : "BACKWARD");

    // Walk through the half-step sequence
    for (int i = 0; i < absSteps; i++) {
        // Calculate the current sequence index (wrap with modulo)
        currentAngleSteps += direction;
        int seqIndex = ((currentAngleSteps % NUM_HALF_STEPS) + NUM_HALF_STEPS) % NUM_HALF_STEPS;
        stepMotor1(seqIndex);
        delay(STEP_DELAY_MS);
    }

    deenergiseMotor1();   // Save power & prevent overheating
    Serial.printf("[Motor1-Angle] Reached position %d steps\n", currentAngleSteps);
}

/*
 *  moveMotor2(targetSteps)
 *  Same logic as moveMotor1 but for the displacement motor.
 */
void moveMotor2(int targetSteps) {
    int stepsToMove = targetSteps - currentDispSteps;

    if (stepsToMove == 0) return;

    int direction = (stepsToMove > 0) ? 1 : -1;
    int absSteps  = abs(stepsToMove);

    Serial.printf("[Motor2-Disp] Moving %d half-steps %s\n",
                  absSteps, direction > 0 ? "FORWARD" : "BACKWARD");

    for (int i = 0; i < absSteps; i++) {
        currentDispSteps += direction;
        int seqIndex = ((currentDispSteps % NUM_HALF_STEPS) + NUM_HALF_STEPS) % NUM_HALF_STEPS;
        stepMotor2(seqIndex);
        delay(STEP_DELAY_MS);
    }

    deenergiseMotor2();
    Serial.printf("[Motor2-Disp] Reached position %d steps\n", currentDispSteps);
}

// ============================================================
//  HIGH-LEVEL ARTICULATOR CONTROLLER
// ============================================================
/*
 *  controlPhysicalArticulator(angle, displacement)
 *  -----------------------------------------------
 *  Called every time a valid 'articulator_cmd' arrives.
 *
 *  angle       : protrusive jaw angle   (float, in degrees, 0-45)
 *  displacement: protrusive jaw offset  (float, in mm,      0-10)
 *
 *  The function maps these real-world values to motor steps using
 *  linear proportional scaling:
 *
 *    angleSteps = (angle / MAX_ANGLE°) × MAX_STEPS
 *    dispSteps  = (disp  / MAX_DISP_mm) × MAX_STEPS
 */
void controlPhysicalArticulator(float angle, float displacement) {
    // Clamp inputs to valid physical ranges
    angle        = constrain(angle, 0.0, (float)MAX_PATIENT_ANGLE);
    displacement = constrain(displacement, 0.0, (float)MAX_PATIENT_DISP);

    // Map to step targets (linear scaling)
    int targetAngleSteps = (int)((angle        / MAX_PATIENT_ANGLE) * MAX_ANGLE_STEPS);
    int targetDispSteps  = (int)((displacement / MAX_PATIENT_DISP)  * MAX_DISP_STEPS);

    Serial.printf("[Articulator] Angle=%.2f° → %d steps | Disp=%.2f mm → %d steps\n",
                  angle, targetAngleSteps, displacement, targetDispSteps);

    // Move both motors to their target positions
    moveMotor1(targetAngleSteps);
    moveMotor2(targetDispSteps);
}

// ============================================================
//  SOCKET.IO EVENT HANDLER
// ============================================================
/*
 *  handleSocketEvent() is called by the WebSockets library
 *  for every Socket.IO event received from the Flask server.
 *
 *  We look for the event named 'articulator_cmd' and extract
 *  the angle and displacement from the payload string:
 *       ["articulator_cmd","<12.50,3.20>"]
 */
void handleSocketEvent(socketIOmessageType_t type, uint8_t* payload, size_t length) {
    switch (type) {

        case sIOtype_DISCONNECT:
            Serial.println("[SocketIO] ❌ Disconnected from Flask Server.");
            break;

        case sIOtype_CONNECT:
            Serial.println("[SocketIO] ✅ Connected to Flask Server!");
            // Register this device as the physical articulator
            socketIO.sendEVENT("[\"register_articulator\",{}]");
            Serial.println("[SocketIO] Registered as Active Articulator.");
            break;

        case sIOtype_EVENT: {
            String eventStr = String((char*)payload);
            Serial.print("[SocketIO] Raw Event: ");
            Serial.println(eventStr);

            // Check if this event is for us
            if (eventStr.indexOf("\"articulator_cmd\"") == -1) break;

            // Parse the <angle,displacement> token
            // Example full string:  ["articulator_cmd","<12.50,3.20>"]
            int startAngle = eventStr.indexOf('<');
            int endAngle   = eventStr.indexOf('>');

            if (startAngle == -1 || endAngle == -1 || endAngle <= startAngle) {
                Serial.println("[SocketIO] ⚠️  Malformed payload – no <> brackets found.");
                break;
            }

            // Extract: "12.50,3.20"
            String inner     = eventStr.substring(startAngle + 1, endAngle);
            int    commaIdx  = inner.indexOf(',');

            if (commaIdx == -1) {
                Serial.println("[SocketIO] ⚠️  No comma separator in payload.");
                break;
            }

            float angle        = inner.substring(0, commaIdx).toFloat();
            float displacement = inner.substring(commaIdx + 1).toFloat();

            Serial.printf("[SocketIO] Parsed → Angle: %.2f°  Displacement: %.2f mm\n",
                          angle, displacement);

            // Drive the motors!
            controlPhysicalArticulator(angle, displacement);
            break;
        }

        default:
            break;
    }
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n====================================================");
    Serial.println("  SIMATS Jaw Articulator – Stepper Motor Edition");
    Serial.println("====================================================\n");

    // ── 1. Configure all stepper motor pins as OUTPUT ─────────
    int motor1Pins[] = { M1_IN1, M1_IN2, M1_IN3, M1_IN4 };
    int motor2Pins[] = { M2_IN1, M2_IN2, M2_IN3, M2_IN4 };

    for (int p : motor1Pins) { pinMode(p, OUTPUT); digitalWrite(p, 0); }
    for (int p : motor2Pins) { pinMode(p, OUTPUT); digitalWrite(p, 0); }
    Serial.println("[Setup] Stepper motor pins initialized.");

    // ── 2. Reset check via BOOT button (GPIO 0) ───────────────
    pinMode(0, INPUT_PULLUP);
    if (digitalRead(0) == LOW) {
        Serial.println("[Setup] BOOT button held. Hold 3 s to factory-reset...");
        delay(3000);
        if (digitalRead(0) == LOW) {
            WiFiManager wm;
            wm.resetSettings();
            preferences.begin("articulator", false);
            preferences.clear();
            preferences.end();
            Serial.println("[Setup] Settings cleared. Rebooting...");
            ESP.restart();
        }
    }

    // ── 3. Load saved server settings from NVS flash ──────────
    preferences.begin("articulator", false);
    server_ip   = preferences.getString("server_ip",   "180.235.121.245");
    server_port = preferences.getString("server_port",  "8068");
    Serial.printf("[Setup] Loaded → Server IP: %s  Port: %s\n",
                  server_ip.c_str(), server_port.c_str());

    // ── 4. WiFiManager captive portal ─────────────────────────
    WiFiManager wm;
    wm.setSaveConfigCallback(saveConfigCallback);

    WiFiManagerParameter custom_ip  ("server", "Flask Server IP",   server_ip.c_str(),   40);
    WiFiManagerParameter custom_port("port",   "Flask Server Port", server_port.c_str(),  6);
    wm.addParameter(&custom_ip);
    wm.addParameter(&custom_port);
    wm.setConfigPortalTimeout(180);

    Serial.println("[Setup] Attempting WiFi connection...");
    if (!wm.autoConnect("Articulator-Setup")) {
        Serial.println("[Setup] ❌ WiFi connection failed. Restarting...");
        delay(3000);
        ESP.restart();
    }
    Serial.println("[Setup] 🎉 WiFi connected!");
    Serial.print ("[Setup] Local IP: ");
    Serial.println(WiFi.localIP());

    if (shouldSaveConfig) {
        server_ip   = String(custom_ip.getValue());
        server_port = String(custom_port.getValue());
        preferences.putString("server_ip",   server_ip);
        preferences.putString("server_port", server_port);
        Serial.println("[Setup] 💾 New server settings saved to flash.");
    }
    preferences.end();

    // ── 5. Connect to Flask-SocketIO server ───────────────────
    int port_num = server_port.toInt();
    Serial.printf("[Setup] Connecting to Socket.IO at %s:%d ...\n",
                  server_ip.c_str(), port_num);
    socketIO.begin(server_ip, port_num, "/socket.io/?EIO=4");
    socketIO.onEvent(handleSocketEvent);

    Serial.println("[Setup] ✅ Setup complete. Waiting for commands...\n");
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
    // Keep the WebSocket connection alive and process incoming events
    socketIO.loop();

    // Note: moveMotor1/2 are blocking calls that happen inside
    // handleSocketEvent → controlPhysicalArticulator.
    // The socketIO.loop() will trigger them automatically.
}
