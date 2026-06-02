import serial
import time
import threading
from pynput import keyboard

PORT = 'COM10'
BAUD = 9600

FIRE_INTERVAL = 0.5
PULSE_DURATION = 0.1

ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)

ready = ser.readline().decode().strip()
print(f"Arduino: {ready}")

running = False  # loop starts paused
stop_program = False

def send_command(cmd: str):
    ser.write(f"{cmd}\n".encode())
    time.sleep(0.1)
    while ser.in_waiting:
        response = ser.readline().decode().strip()
        if response:
            print(f"Arduino: {response}")

def on_press(key):
    global running, stop_program
    try:
        if key == keyboard.Key.space:
            running = not running
            print("Loop STARTED" if running else "Loop PAUSED")
        elif hasattr(key, 'char') and key.char and key.char.upper() == 'Q':
            stop_program = True
            running = False
            return False  # stop listener
    except AttributeError:
        pass

# Start keyboard listener in background
listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Spacebar = toggle loop on/off, Q = quit")

sequence = ['L', 'R', 'C']  # C = click
i = 0

try:
    while not stop_program:
        if running:
            cmd = sequence[i % len(sequence)]
            send_command(cmd)
            time.sleep(PULSE_DURATION)
            send_command('N')  # stop after each pulse
            time.sleep(FIRE_INTERVAL)
            i += 1
        else:
            time.sleep(0.1)  # idle, wait for spacebar

except KeyboardInterrupt:
    pass

send_command('S')
listener.stop()
ser.close()
print("Connection closed")