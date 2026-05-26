/*
 * ESP32 Articulator Controller (Physical) - Industry Standard Provisioning
 * 
 * This firmware establishes WiFi and Socket.IO connectivity dynamically.
 * - If no WiFi or Server IP is saved, it starts a local hotspot "Articulator-Setup" (IP: 192.168.4.1).
 * - Connect to this hotspot with your phone/PC to enter your local WiFi credentials and Flask Server IP.
 * - Saves settings to ESP32 permanent internal flash memory (Preferences) so you never re-flash!
 * - Hold the physical "BOOT" button (GPIO 0) on the ESP32 for 3 seconds at startup to reset all saved settings.
 * 
 * Required Arduino Libraries:
 *  1. WiFiManager by tzapu (https://github.com/tzapu/WiFiManager)
 *  2. WebSockets by links2004 (https://github.com/Links2004/arduinoWebSockets)
 *  3. ESP32Servo (https://github.com/madhephaestus/ESP32Servo)
 */

#include <WiFi.h>
#include <WebServer.h>
#include <WiFiManager.h>      // Dynamic Web Captive Portal
#include <Preferences.h>      // Internal Flash NVS Storage
#include <WebSocketsClient.h>
#include <SocketIOclient.h>
#include <ESP32Servo.h>

// ==========================================
// PIN DEFINITIONS & STORAGE
// ==========================================
const int PIN_BOOT_BUTTON = 0;   // Physical BOOT button on standard ESP32 boards
const int PIN_ANGLE_SERVO = 18;  // Servo representing Jaw Angle (GPIO 18)
const int PIN_DISP_SERVO  = 19;  // Servo representing Jaw Displacement (GPIO 19)

Preferences preferences;         // Non-volatile storage object

// Server connection variables
String server_ip;
String server_port;

Servo angleServo;
Servo dispServo;
SocketIOclient socketIO;

// Flag to check if we need to save custom parameters from the web portal
bool shouldSaveConfig = false;

// Callback notifying us that we entered config mode or parameters need saving
void saveConfigCallback() {
    Serial.println("[Portal] Config needs to be saved");
    shouldSaveConfig = true;
}

// ==========================================
// SOCKET.IO EVENT HANDLER & PARSER
// ==========================================
void handleSocketEvent(socketIOmessageType_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case sIOtype_DISCONNECT:
            Serial.println("[SocketIO] ❌ Disconnected from Flask Server");
            break;
            
        case sIOtype_CONNECT:
            Serial.println("[SocketIO] ✅ Connected to Flask Server");
            
            // Handshake: Register as the active physical articulator
            socketIO.sendEVENT("[\"register_articulator\",{}]");
            Serial.println("[SocketIO] Registered as Active Articulator");
            break;
            
        case sIOtype_EVENT: {
            String eventString = String((char*)payload);
            
            // Listen for 'articulator_cmd' containing "<angle,displacement>"
            int cmdEventIndex = eventString.indexOf("\"articulator_cmd\"");
            if (cmdEventIndex != -1) {
                int startToken = eventString.indexOf('<');
                int endToken = eventString.indexOf('>');
                
                if (startToken != -1 && endToken != -1 && endToken > startToken) {
                    String cleanPayload = eventString.substring(startToken, endToken + 1);
                    Serial.print("[SocketIO] Command: ");
                    Serial.println(cleanPayload);
                    
                    int commaIndex = cleanPayload.indexOf(',');
                    if (commaIndex != -1) {
                        String angleStr = cleanPayload.substring(1, commaIndex);
                        String dispStr = cleanPayload.substring(commaIndex + 1, cleanPayload.length() - 1);
                        
                        float angle = angleStr.toFloat();
                        float displacement = dispStr.toFloat();
                        
                        Serial.printf("   Joint Motion -> Angle: %.2f°, Displacement: %.2f mm\n", angle, displacement);
                        
                        // Control servos
                        controlPhysicalArticulator(angle, displacement);
                    }
                }
            }
            break;
        }
        default:
            break;
    }
}

void controlPhysicalArticulator(float angle, float displacement) {
    // Map patient opening angle (e.g. 0° to 45°) to servo motor angle (e.g. 0° to 180°)
    int targetAngleServoPos = map(constrain((int)angle, 0, 45), 0, 45, 0, 180);
    angleServo.write(targetAngleServoPos);
    
    // Map displacement (e.g. 0mm to 10mm) to servo position (e.g. 0° to 180°)
    int targetDispServoPos = map(constrain((int)displacement, 0, 10), 0, 10, 0, 180);
    dispServo.write(targetDispServoPos);
    
    Serial.printf("   Actuator Servo Outputs -> Angle: %d°, Displacement: %d°\n", targetAngleServoPos, targetDispServoPos);
}

// ==========================================
// SETUP & PROVISIONING
// ==========================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n=== SIMATS Industrial Physical Articulator ===");
    
    // 1. Check if user wants to reset settings by holding BOOT button
    pinMode(PIN_BOOT_BUTTON, INPUT_PULLUP);
    if (digitalRead(PIN_BOOT_BUTTON) == LOW) {
        Serial.println("⚠️ BOOT button detected on startup! Hold for 3 seconds to RESET all settings...");
        delay(3000);
        if (digitalRead(PIN_BOOT_BUTTON) == LOW) {
            Serial.println("💥 RESET COMMAND RECEIVED! Clearing saved WiFi and Server configuration...");
            
            // Reset WiFi credentials
            WiFiManager wm;
            wm.resetSettings();
            
            // Reset custom parameters
            preferences.begin("articulator", false);
            preferences.clear();
            preferences.end();
            
            Serial.println("Settings cleared. Rebooting...");
            ESP.restart();
        } else {
            Serial.println("Reset cancelled.");
        }
    }

    // 2. Attach Servo Motors
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    angleServo.setPeriodHertz(50);
    dispServo.setPeriodHertz(50);
    angleServo.attach(PIN_ANGLE_SERVO, 500, 2400);
    dispServo.attach(PIN_DISP_SERVO, 500, 2400);
    angleServo.write(0);
    dispServo.write(0);
    Serial.println("Servos calibrated and set to default rest position.");

    // 3. Load saved Server settings from Internal NVS Flash memory
    preferences.begin("articulator", false);
    server_ip = preferences.getString("server_ip", "172.25.81.101");
    server_port = preferences.getString("server_port", "5000");
    Serial.printf("Loaded Settings from Flash -> Server IP: %s, Port: %s\n", server_ip.c_str(), server_port.c_str());

    // 4. Configure WiFiManager Dynamic Portal
    WiFiManager wm;
    wm.setSaveConfigCallback(saveConfigCallback);

    // Create custom text fields for the portal config page
    // Parameters: custom_id, label, default_value, length
    WiFiManagerParameter custom_server_ip("server", "Flask Server IP Address", server_ip.c_str(), 40);
    WiFiManagerParameter custom_server_port("port", "Flask Server Port", server_port.c_str(), 6);
    
    wm.addParameter(&custom_server_ip);
    wm.addParameter(&custom_server_port);

    // Set connection timeout (seconds) so it doesn't wait forever if router is off
    wm.setConfigPortalTimeout(180); 

    // Launch portal or connect to stored network
    Serial.println("Attempting WiFi connection...");
    if (!wm.autoConnect("Articulator-Setup")) {
        Serial.println("❌ Failed to connect or timed out. Restarting...");
        delay(3000);
        ESP.restart();
    }

    Serial.println("\n🎉 WiFi connected successfully!");
    Serial.print("Local IP Address: ");
    Serial.println(WiFi.localIP());

    // Save custom settings if they were updated via the captive web portal
    if (shouldSaveConfig) {
        server_ip = String(custom_server_ip.getValue());
        server_port = String(custom_server_port.getValue());
        
        preferences.putString("server_ip", server_ip);
        preferences.putString("server_port", server_port);
        Serial.println("💾 Saved new Flask Server settings to NVS Flash memory!");
    }
    preferences.end(); // close preferences

    // 5. Connect to Socket.IO Server using resolved/configured IP and Port
    int port_num = server_port.toInt();
    Serial.printf("Connecting to Socket.IO Server on http://%s:%d ...\n", server_ip.c_str(), port_num);
    
    // In Flask-SocketIO, the default path is "/socket.io/?EIO=4"
    socketIO.begin(server_ip, port_num, "/socket.io/?EIO=4");
    socketIO.onEvent(handleSocketEvent);
}

void loop() {
    socketIO.loop();
}
