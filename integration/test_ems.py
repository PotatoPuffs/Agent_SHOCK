# test_ems.py — run this to verify EMS hardware works via ems_controller.py
from ems_controller import RealEMSController
import time

ems = RealEMSController(
    port="/dev/cu.usbmodem12301",  # ← your actual port from Step 2
    baud=9600                     # ← your actual baud rate
)

print("Testing LEFT...")
ems.send_action("left")
time.sleep(0.5)
ems.send_action("none")
time.sleep(0.5)

print("Testing RIGHT...")
ems.send_action("right")
time.sleep(0.5)
ems.send_action("none")
time.sleep(0.5)

print("Testing CLICK...")
ems.send_action("click")
time.sleep(0.3)
ems.send_action("none")

print("Done:", ems.status())
ems.close()