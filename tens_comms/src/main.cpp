#include <Arduino.h>

const int RELAY_CH0 = 7;   // Left
const int RELAY_CH1 = 8;   // Right
const int RELAY_CH2 = 9;   // Click
const int LED_PIN   = 13;

void stopAll() {
    digitalWrite(RELAY_CH0, HIGH);
    digitalWrite(RELAY_CH1, HIGH);
    digitalWrite(RELAY_CH2, HIGH);
    digitalWrite(LED_PIN, LOW);
    Serial.println("STOP");
}

void setup() {
    pinMode(RELAY_CH0, OUTPUT);
    pinMode(RELAY_CH1, OUTPUT);
    pinMode(RELAY_CH2, OUTPUT); 
    pinMode(LED_PIN, OUTPUT);
    stopAll();
    Serial.begin(115200);
    Serial.println("READY");
}

void loop() {
    if (Serial.available() > 0) {
        String incoming = Serial.readStringUntil('\n');
        incoming.trim();

        if (incoming == "L") {
            stopAll();
            digitalWrite(RELAY_CH0, LOW);
            digitalWrite(LED_PIN, HIGH);
            Serial.println("LEFT ON");
        }
        else if (incoming == "R") {
            stopAll();
            digitalWrite(RELAY_CH1, LOW);
            digitalWrite(LED_PIN, HIGH);
            Serial.println("RIGHT ON");
        }
        else if (incoming == "C") {
            stopAll();
            digitalWrite(RELAY_CH2, LOW);
            digitalWrite(LED_PIN, HIGH);
            Serial.println("CLICK ON");
        }
        else if (incoming == "N") {
            stopAll();
        }
        else {
            Serial.println("ERR: unknown command");
        }
    }
}