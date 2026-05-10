/*
  ems_controller.ino — Arduino/MCU firmware for EMS actuation

  Receives serial commands from Python middleware:
    Format:  C<channel>D<duration_ms>\n
    Example: C0D50\n  → fire channel 0 for 50ms
             C1D80\n  → fire channel 1 for 80ms

  Hardware:
    - Channel 0 relay: pin 7  (left forearm muscle — pronator / wrist flexor)
    - Channel 1 relay: pin 8  (right forearm muscle — supinator / wrist extensor)
    - TENS machine output wired through each relay
    - Safety: max 200ms per pulse, 500ms cooldown enforced in firmware

  Wiring:
    Arduino pin 7 → Relay IN1 → TENS CH1 (left)
    Arduino pin 8 → Relay IN2 → TENS CH2 (right)
    Relay COM → TENS electrode output
    Relay NC  → electrode wire to skin

  SAFETY NOTES:
    - Set TENS intensity LOW before testing (start at 5-10mA)
    - Never exceed 200ms pulse duration
    - Keep cooldown at ≥500ms between same-channel pulses
    - Add hardware emergency stop button that cuts relay power
    - Never stimulate near heart, head, or across the chest
*/

// ── Pin definitions ────────────────────────────────────────────────────────────
const int RELAY_CH0 = 7;   // Left  (move cursor left)
const int RELAY_CH1 = 8;   // Right (move cursor right)
const int LED_PIN   = 13;  // Built-in LED mirrors any active pulse

// ── Safety limits ──────────────────────────────────────────────────────────────
const unsigned long MAX_PULSE_MS  = 200;   // Hard cap on pulse duration
const unsigned long COOLDOWN_MS   = 500;   // Minimum gap between pulses (same channel)

// ── State tracking ─────────────────────────────────────────────────────────────
unsigned long lastFireTime[2] = {0, 0};
bool channelActive[2] = {false, false};
unsigned long channelOffTime[2] = {0, 0};

// ── Setup ──────────────────────────────────────────────────────────────────────
void setup() {
  pinMode(RELAY_CH0, OUTPUT);
  pinMode(RELAY_CH1, OUTPUT);
  pinMode(LED_PIN,   OUTPUT);

  // Relays are active-LOW on most boards: HIGH = OFF
  digitalWrite(RELAY_CH0, HIGH);
  digitalWrite(RELAY_CH1, HIGH);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(115200);
  Serial.println("EMS controller ready. Format: C<0|1>D<ms>");
}

// ── Parse incoming command ─────────────────────────────────────────────────────
void parseCommand(String cmd) {
  cmd.trim();
  if (cmd.length() < 4) return;
  if (cmd[0] != 'C') return;

  int dIdx = cmd.indexOf('D');
  if (dIdx < 0) return;

  int channel  = cmd.substring(1, dIdx).toInt();
  int duration = cmd.substring(dIdx + 1).toInt();

  if (channel < 0 || channel > 1) {
    Serial.println("ERR: channel must be 0 or 1");
    return;
  }

  // Clamp duration
  duration = constrain(duration, 1, (int)MAX_PULSE_MS);

  unsigned long now = millis();

  // Enforce cooldown
  if (now - lastFireTime[channel] < COOLDOWN_MS) {
    Serial.print("COOLDOWN channel ");
    Serial.println(channel);
    return;
  }

  // Don't fire if channel still active (shouldn't happen, but be safe)
  if (channelActive[channel]) {
    Serial.println("ERR: channel still active");
    return;
  }

  fireChannel(channel, (unsigned long)duration);
}

// ── Fire a single pulse ────────────────────────────────────────────────────────
void fireChannel(int ch, unsigned long duration_ms) {
  int pin = (ch == 0) ? RELAY_CH0 : RELAY_CH1;

  channelActive[ch] = true;
  channelOffTime[ch] = millis() + duration_ms;
  lastFireTime[ch] = millis();

  digitalWrite(pin, LOW);    // Relay ON (active-LOW)
  digitalWrite(LED_PIN, HIGH);

  Serial.print("FIRE C");
  Serial.print(ch);
  Serial.print(" D");
  Serial.println(duration_ms);
}

// ── Turn off expired channels ──────────────────────────────────────────────────
void updateChannels() {
  unsigned long now = millis();
  bool anyActive = false;

  for (int ch = 0; ch < 2; ch++) {
    if (channelActive[ch] && now >= channelOffTime[ch]) {
      int pin = (ch == 0) ? RELAY_CH0 : RELAY_CH1;
      digitalWrite(pin, HIGH);  // Relay OFF
      channelActive[ch] = false;
      Serial.print("OFF C");
      Serial.println(ch);
    }
    if (channelActive[ch]) anyActive = true;
  }

  digitalWrite(LED_PIN, anyActive ? HIGH : LOW);
}

// ── Main loop ──────────────────────────────────────────────────────────────────
String inputBuffer = "";

void loop() {
  // Non-blocking serial read
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      parseCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
      if (inputBuffer.length() > 32) inputBuffer = "";  // Overflow guard
    }
  }

  // Always check whether pulses need to be turned off
  updateChannels();
}
