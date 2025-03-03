#include <ESP8266WiFi.h>  // Include ESP8266WiFi library

const char* ssid = "Your_SSID";         // Replace with your Wi-Fi SSID
const char* password = "Your_PASSWORD"; // Replace with your Wi-Fi Password

void setup() {
  Serial.begin(115200);  // Initialize serial communication at 115200 baud rate
  delay(10);
  
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  // Connect to Wi-Fi network
  WiFi.begin(ssid, password);

  // Check Wi-Fi connection status
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());  // Print the local IP address
}

void loop() {
  // This code runs repeatedly after connecting to Wi-Fi
}



