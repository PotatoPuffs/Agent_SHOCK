"""
hardware/ems_controller.py — Real EMS controller via Arduino serial.

Implements BaseEMSController. Called every frame by run.py's deploy_loop().

Communication protocol (ASCII over USB serial):
    Python  →  Arduino :  "L\n"  |  "R\n"  |  "N\n"
                          left      right     none/click (no pulse)

send_action() BLOCKS until "OK\n" is received so that run.py's loop
only advances to the next observation after the hand has finished moving.

Arduino sketch requirements (what your EMS teammate writes):
    - Read one character ('L', 'R', or 'N') from Serial
    - 'L': close LEFT relay for PULSE_DURATION_MS, open, send "OK\\n"
    - 'R': close RIGHT relay for PULSE_DURATION_MS, open, send "OK\\n"
    - 'N': send "OK\\n" immediately (no pulse)
    - Baud rate must match BAUD_RATE below (default 115200)

Dependencies:
    pip install pyserial

Finding your Arduino port:
    Windows : "COM3", "COM4", etc — check Device Manager → Ports
    macOS   : "/dev/cu.usbmodem*" — run `ls /dev/cu.*` in terminal
    Linux   : "/dev/ttyACM0" or "/dev/ttyUSB0" — run `ls /dev/tty*`
"""

import time
import serial
import serial.tools.list_ports
from integration.interfacing import BaseEMSController

# ── Serial configuration ──────────────────────────────────────────────────────
SERIAL_PORT  = "COM3"       # ❗ REAL VALUE: your Arduino's port (see docstring above)
BAUD_RATE    = 115200       # must match Arduino Serial.begin(115200)
# ─────────────────────────────────────────────────────────────────────────────

# Map action strings to the single-character command the Arduino expects
_ACTION_TO_CMD = {
    "left":  b"L\n",
    "right": b"R\n",
    "click": b"C\n",   # click = momentary pulse on click relay
    "none":  b"N\n",   # stop — open all relays
}


class RealEMSController(BaseEMSController):
    """
    Sends stimulation commands to the Arduino over USB serial (fire-and-forget).

    Each send_action() call:
        1. Checks if command differs from last command
        2. If different: writes command to Arduino and returns immediately
        3. If same: does nothing (relay stays in its current state)
        4. Returns None (cursor position tracked by CNN, not EMS)

    Relays remain in their state (open or closed) until a new command is sent.
    CNN is the source of truth for hand position — EMS just triggers the relay.
    """

    def __init__(
        self,
        port: str            = SERIAL_PORT,
        baud: int            = BAUD_RATE,
        auto_detect: bool    = False,
    ):
        """
        Args:
            port       : serial port the Arduino is on (e.g. "COM3")
            baud       : baud rate — must match Arduino sketch
            auto_detect: if True, ignore port and scan for the first
                         available Arduino automatically (useful for dev)
        """
        self._last_command = None  # Track last sent command for deduplication
        self._command_count = 0    # Diagnostics

        if auto_detect:
            port = self._find_arduino_port()

        print(f"[RealEMS] Connecting to Arduino on {port} at {baud} baud...")
        try:
            self._serial = serial.Serial(port, baud, timeout=1.0)
        except serial.SerialException as e:
            raise RuntimeError(
                f"[RealEMS] Could not open serial port {port!r}.\n"
                f"  Check the port name, that the Arduino is plugged in,\n"
                f"  and that no other program (e.g. Arduino IDE Serial Monitor)\n"
                f"  has the port open.\n  Original error: {e}"
            )

        # Arduino resets on serial connect — give it time to boot
        time.sleep(2.0)
        self._serial.reset_input_buffer()
        print(f"[RealEMS] Connected. Fire-and-forget mode (no blocking on response).")

    # ── Arduino auto-detection ────────────────────────────────────────────────

    @staticmethod
    def _find_arduino_port() -> str:
        """Scan serial ports and return the first one that looks like an Arduino."""
        candidates = []
        for port_info in serial.tools.list_ports.comports():
            desc = (port_info.description or "").lower()
            mfr  = (port_info.manufacturer or "").lower()
            if any(k in desc or k in mfr for k in ("arduino", "ch340", "ftdi", "usb serial")):
                candidates.append(port_info.device)

        if not candidates:
            raise RuntimeError(
                "[RealEMS] auto_detect=True but no Arduino found.\n"
                "Plug in the Arduino or set port manually."
            )
        print(f"[RealEMS] Auto-detected Arduino on: {candidates[0]}")
        return candidates[0]

    # ── Core send ─────────────────────────────────────────────────────────────

    def send_action(self, action: str) -> None:
        """
        Send command to Arduino if it differs from last command (fire-and-forget).

        Args:
            action : 'left' | 'right' | 'click' | 'none'

        Returns:
            None — relay states are maintained until next command. Cursor position
            comes from CNN observer, not from EMS.

        Raises:
            ValueError if action is not recognized.
        """
        cmd = _ACTION_TO_CMD.get(action)
        if cmd is None:
            raise ValueError(f"[RealEMS] Unknown action: {action!r}. "
                             f"Expected 'left', 'right', 'click', or 'none'.")

        # Only send if command changed — avoid redundant serial writes
        if cmd == self._last_command:
            return

        # Send the command and update state
        try:
            self._serial.write(cmd)
            self._serial.flush()
            self._last_command = cmd
            self._command_count += 1
        except serial.SerialException as e:
            raise RuntimeError(
                f"[RealEMS] Serial write failed for action={action!r}: {e}"
            )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return a status snapshot — useful for logging."""
        return {
            "port":           self._serial.port,
            "command_count":  self._command_count,
            "last_command":   self._last_command.decode() if self._last_command else None,
            "port_open":      self._serial.is_open,
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Send 'N' to open all relays, then close serial.
        Always called in run.py's finally block.
        """
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(b"N\n")
                self._serial.flush()
                time.sleep(0.1)
            except Exception:
                pass
            self._serial.close()
        print(f"[RealEMS] Closed. Commands sent: {self._command_count}")