#include <esp_now.h>
#include <WiFi.h>
#include <Wire.h>
#include <SparkFun_BNO08x_Arduino_Library.h>

// Struct to match the Transmitter
typedef struct struct_message {
    float qw, qx, qy, qz;
    int flex[5];
} struct_message;

struct_message leftHandData;
BNO08x rightIMU;

const int flexPins[] = {34, 35, 32, 33, 39}; // ADC1 Pins for Right Hand

// Callback function when data is received from Left Hand
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  memcpy(&leftHandData, incomingData, sizeof(leftHandData));
}

void setup() {
  Serial.begin(115200);
  
  // 1. Initialize Wi-Fi and ESP-NOW
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW Init Failed");
    return;
  }
  esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv));

  // 2. Initialize Local I2C for Right Hand IMU
  Wire.begin(0x4B);
  if (!rightIMU.begin()) {
    Serial.println("Right BNO086 not detected");
    while (1);
  }
  rightIMU.enableRotationVector(20); // 50Hz updates
  
  Serial.println("System Ready: Receiving Left | Reading Right");
}

void loop() {
  // Check if Right Hand IMU has new data
  if (rightIMU.getSensorEvent()) {
    if (rightIMU.sensorValue.sensorId == SH2_ROTATION_VECTOR) {
      
      // --- PART 1: PRINT LEFT HAND (Received via ESP-NOW) ---
      Serial.print("L,");
      Serial.print(leftHandData.qw, 4); Serial.print(",");
      Serial.print(leftHandData.qx, 4); Serial.print(",");
      Serial.print(leftHandData.qy, 4); Serial.print(",");
      Serial.print(leftHandData.qz, 4); Serial.print(",");
      for(int i=0; i<5; i++) {
        Serial.print(leftHandData.flex[i]);
        if (i < 4) Serial.print(","); // This prevents the trailing comma
      }

      // --- PART 2: PRINT RIGHT HAND (Local Sensors) ---
      Serial.print("|R,");
      Serial.print(rightIMU.sensorValue.un.rotationVector.real, 4); Serial.print(",");
      Serial.print(rightIMU.sensorValue.un.rotationVector.i, 4); Serial.print(",");
      Serial.print(rightIMU.sensorValue.un.rotationVector.j, 4); Serial.print(",");
      Serial.print(rightIMU.sensorValue.un.rotationVector.k, 4); Serial.print(",");
      
      for(int i=0; i<5; i++) {
        int val = analogRead(flexPins[i]);
        Serial.print(val);
        if (i < 4) Serial.print(",");
      }

      // --- END OF FRAME ---
      Serial.println(); 
    }
  }
}