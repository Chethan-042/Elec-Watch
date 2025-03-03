#include <LiquidCrystal.h> 
#include <ESP8266WiFi.h> 
#include <PubSubClient.h> 

// WiFi and MQTT Setup
const char* ssid = "NSTL"; 
const char* password = "18931618";
const char* mqttServer = "your.mqtt.broker"; 
const int mqttPort = 1883; 
const char* mqttTopic = "openCircuitStatus"; 

// Current Sensor Pin
const int currentSensorPin = A0; 
const float sensitivity = 0.185; 
const float voltageSupply = 5.0; 

// LCD Pins
LiquidCrystal lcd(12, 11, 5, 4, 3, 2); 

// Threshold for fault detection
const float currentThreshold = 0.5; 

// MQTT Client
WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
    // Initialize Serial, WiFi, LCD and MQTT
    Serial.begin(115200);
    lcd.begin(16, 2); // Set up the LCD size

    // Connect to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi");

    // Connect to MQTT
    client.setServer(mqttServer, mqttPort);
    client.connect("OpenCircuitDetector");
}

void loop() {
    if (!client.connected()) {
        client.connect("OpenCircuitDetector");
    }
    client.loop();

    float currentReading = readCurrent(); 
    Serial.print("Current: ");
    Serial.println(currentReading);
    
    // Check for open circuit fault
    if (currentReading < currentThreshold) {
        lcd.setCursor(0, 0);
        lcd.print("Open Circuit Fault");
        client.publish(mqttTopic, "Fault Detected"); 
    } else {
        lcd.setCursor(0, 0);
        lcd.print("System Normal     ");
        client.publish(mqttTopic, "System Normal"); 
    }

    delay(2000); 
}

// Function to read current from ACS712 sensor
float readCurrent() {
    int sensorValue = analogRead(currentSensorPin);
    float voltage = (sensorValue / 1023.0) * voltageSupply; 
    float current = (voltage - (voltageSupply / 2)) / sensitivity; 
    return current;
}
