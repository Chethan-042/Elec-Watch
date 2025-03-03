#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DHT.h>
#include <MPU6050_light.h>

// I2C LCD Address (typically 0x27 or 0x3F)
LiquidCrystal_I2C lcd(0x27, 16, 2); // Adjust if necessary

// Pin Definitions
#define DHTPIN 2                    // DHT22 data pin
#define DHTTYPE DHT22               // DHT22 sensor type
#define VOLTAGE_SENSOR_PIN A1       // ZMPT101B voltage sensor output
#define CURRENT_SENSOR_PIN A0       // ACS712 current sensor output
#define TRIG_PIN 3                  // HC-SR04 Trigger pin
#define ECHO_PIN 4                  // HC-SR04 Echo pin

// Initialize Sensors
DHT dht(DHTPIN, DHTTYPE);
MPU6050 mpu(Wire);

void setup() {
  Serial.begin(9600);

  // Initialize I2C and Sensors
  Wire.begin();
  dht.begin();
  mpu.begin();
  
  // Initialize LCD
  lcd.begin(16, 2);   // 16 columns, 2 rows
  lcd.backlight();    // Turn on the backlight

  // Ultrasonic Sensor Setup
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  lcd.setCursor(0, 0);
  lcd.print("Sensors Ready");
  delay(2000);
  lcd.clear();
}

void loop() {
  // Read DHT22 (Temperature & Humidity)
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // Read Voltage from ZMPT101B
  int voltageValue = analogRead(VOLTAGE_SENSOR_PIN);
  float voltage = (voltageValue / 1023.0) * 5.0 * 220; // Adjust scaling factor as needed

  // Read Current from ACS712
  int currentValue = analogRead(CURRENT_SENSOR_PIN);
  float current = (currentValue - 512) * 5.0 / 1023.0 / 0.066; // Adjust based on ACS712 sensitivity (30A version)

  // Read MPU6050 (Accelerometer & Gyroscope)
  mpu.update();
  float accelX = mpu.getAccX();
  float accelY = mpu.getAccY();
  float accelZ = mpu.getAccZ();

  // Read HC-SR04 (Ultrasonic Sensor) for Distance Measurement
  long duration, distance;
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  duration = pulseIn(ECHO_PIN, HIGH);
  distance = duration * 0.034 / 2; // Distance in cm

  // Output Sensor Data to Serial Monitor
  Serial.print("Temp: "); Serial.print(temperature); Serial.print(" C, ");
  Serial.print("Humidity: "); Serial.print(humidity); Serial.println(" %");
  Serial.print("Voltage: "); Serial.print(voltage); Serial.print(" V, ");
  Serial.print("Current: "); Serial.print(current); Serial.println(" A");
  Serial.print("AccelX: "); Serial.print(accelX);
  Serial.print(", AccelY: "); Serial.print(accelY);
  Serial.print(", AccelZ: "); Serial.println(accelZ);
  Serial.print("Distance: "); Serial.print(distance); Serial.println(" cm");
  Serial.println("----------");

  // Display Sensor Data on LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("T:"); lcd.print(temperature); lcd.print("C ");
  lcd.print("H:"); lcd.print(humidity); lcd.print("%");

  lcd.setCursor(0, 1);
  lcd.print("V:"); lcd.print(voltage); lcd.print("V ");
  lcd.print("C:"); lcd.print(current); lcd.print("A");

  delay(2000); // Delay for readability and to reduce flickering
}

