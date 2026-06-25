#include <esp_now.h>
#include <WiFi.h>
#include <Wire.h>
#include <SparkFun_BNO08x_Arduino_Library.h>

BNO08x myIMU;
// Update this with your actual Glove B MAC address
uint8_t broadcastAddress[] = {0xD4, 0xE9, 0xF4, 0x72, 0x22, 0x40};

typedef struct struct_message {
    float qw, qx, qy, qz;
    int flex[5];
} struct_message;

struct_message myData;
const int flexPins[] = {34, 35, 32, 33, 39};

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA); // Required for ESP-NOW

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }
  
  Wire.begin();
  if (!myIMU.begin(0x4B)) {
    Serial.println("BNO086 not detected");
    while(1);
  }
  myIMU.enableRotationVector(20);
}

void loop() {
  if (myIMU.getSensorEvent()) {
    if (myIMU.sensorValue.sensorId == SH2_ROTATION_VECTOR) {
      myData.qw = myIMU.sensorValue.un.rotationVector.real;
      myData.qx = myIMU.sensorValue.un.rotationVector.i;
      myData.qy = myIMU.sensorValue.un.rotationVector.j;
      myData.qz = myIMU.sensorValue.un.rotationVector.k;

      for (int i=0; i<5; i++) {
        myData.flex[i] = analogRead(flexPins[i]);
      }
      
      esp_now_send(broadcastAddress, (uint8_t *) &myData, sizeof(myData));
    }
  }
}