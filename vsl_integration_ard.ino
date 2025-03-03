#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x3F, 16, 2); // LCD I2C address (0x27), 16 columns, 2 rows

// Define sensor pins
const int currentPin = A0; // ACS712 current sensor
const int voltagePin = A1; // ZMPT101B voltage sensor

// Calibration values
float voltageCalibration = 0.058; // Calibrate based on your setup
float currentCalibration = 0.1;   // Adjust this based on your ACS712 variant (30A, 20A, etc.)

void setup() {
  Serial.begin(9600);
  lcd.begin(16,2);
  lcd.backlight();
}

void loop() {
  // Reading voltage sensor
  int voltageRaw = analogRead(voltagePin);
  float voltage = voltageRaw * voltageCalibration; // Convert raw value to voltage
  
  // Reading current sensor
  int currentRaw = analogRead(currentPin);
  float current = (currentRaw - 512) * currentCalibration; // Adjust based on zero offset
  
  // Print on Serial Monitor
  Serial.print("Voltage: ");
  Serial.print(voltage);
  Serial.print(" V, Current: ");
  Serial.print(current);
  Serial.println(" A");
  
  // Display on LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Voltage: ");
  lcd.print(voltage);
  lcd.print(" V");
  lcd.setCursor(0, 1);
  lcd.print("Current: ");
  lcd.print(current);
  lcd.print(" A");
  
  delay(1000); // Update every second
}