import serial  # type: ignore[import]
import time

PORT = 'COM9'
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)

ready = ser.readline().decode().strip()
print(f"Arduino: {ready}")

def send_pulses(count: int):
    ser.write(f"{count}\n".encode())
    response = ser.readline().decode().strip()
    print(f"Arduino: {response}")

print("Enter a pulse count (1-100), or Ctrl+C to quit")
while True:
    try:
        raw = input("> ").strip()
        send_pulses(int(raw))
    except ValueError:
        print("Numbers only (1-100)")
    except KeyboardInterrupt:
        break

ser.close()
print("Connection closed")