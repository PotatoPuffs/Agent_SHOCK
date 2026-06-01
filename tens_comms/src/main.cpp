#include <Arduino.h>

const int TENS_PIN = 9;

void sendPulses(int count, int pulseWidthMs);

void sendPulses(int count, int pulseWidthMs) {
    for (int i = 0; i < count; i++) {
        digitalWrite(TENS_PIN, HIGH);
        delay(pulseWidthMs);
        digitalWrite(TENS_PIN, LOW);
        delay(pulseWidthMs);
    }
}

void setup() {
    Serial.begin(9600);
    pinMode(TENS_PIN, OUTPUT);
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.println("AGENT SHOCK IS READY");
}

void loop() {
    if (Serial.available() > 0) {
        String incoming = Serial.readStringUntil('\n');
        incoming.trim();

        int pulseCount = incoming.toInt();
        if (pulseCount > 0 && pulseCount <= 100) {
            Serial.print("Firing ");
            Serial.print(pulseCount);
            Serial.println(" pulses");

            // Flash LED same number of times as pulses
            for (int i = 0; i < pulseCount; i++) {
                digitalWrite(LED_BUILTIN, HIGH);
                delay(100);
                digitalWrite(LED_BUILTIN, LOW);
                delay(100);
            }

            sendPulses(pulseCount, 10);
        } else {
            Serial.println("ERR: invalid value");
        }
    }
}