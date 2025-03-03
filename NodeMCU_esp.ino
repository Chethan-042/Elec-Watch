#include <ESP8266WiFi.h>  // Include ESP8266 library

const char* ssid = "auincubation";      // Replace with your Wi-Fi SSID
const char* password = "auic@XYZ123";  // Replace with your Wi-Fi password

void setup() {
  Serial.begin(115200);             // Start serial communication at baud rate 115200
  WiFi.begin(ssid, password);       // Connect to Wi-Fi network
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");              // Print dots while connecting
  }

  Serial.println("\nConnected to Wi-Fi");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());   // Print IP address
}

void loop() {
  // Keep the loop empty for testing connection
}
