"""Alert outputs: buzzer + seat/steering vibration motor.

NOT YET TESTED ON HARDWARE. The vibration intensities below are placeholders
until a user study (docs/ROADMAP.md, phase 3) establishes what wakes drivers
without startling them. The 5 Hz pulse rate comes from the original project
report (whole-body vibration near seat resonance) and is also unvalidated.

Hardware notes
--------------
* Never drive a motor or buzzer directly from a GPIO pin (a pin sources only
  a few mA). Use a logic-level MOSFET or a transistor/relay board and a
  flyback diode across the motor.
* Do not use GPIO2/GPIO3 for outputs: they carry the I2C pull-up resistors.
  The defaults here are GPIO18 (PWM-capable) and GPIO23.
* gpiozero is used instead of RPi.GPIO because it also works on Raspberry Pi 5.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .detector import AlertLevel, Output, Status


@dataclass(frozen=True)
class AlertProfile:
    vibration_duty: float   # 0..1 intensity of the vibration motor while pulsing
    pulse_hz: float         # on/off rate of the vibration envelope
    buzzer: bool
    message: str


PROFILES: Dict[AlertLevel, AlertProfile] = {
    AlertLevel.NORMAL: AlertProfile(0.0, 0.0, False, "OK"),
    AlertLevel.ADVISORY: AlertProfile(0.35, 5.0, False, "STAY ALERT"),
    AlertLevel.WARNING: AlertProfile(0.65, 5.0, True, "WAKE UP"),
    AlertLevel.CRITICAL: AlertProfile(1.00, 5.0, True, "PULL OVER"),
}


class Actuator:
    """Applies the profile for a detector output; acts only when something changes."""

    def __init__(self) -> None:
        self._key: Optional[Tuple[AlertLevel, Status]] = None

    def apply(self, out: Output) -> bool:
        key = (out.level, out.status)
        if key == self._key:
            return False
        self._key = key
        self._set_profile(PROFILES[out.level], out)
        return True

    def _set_profile(self, profile: AlertProfile, out: Output) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockActuator(Actuator):
    """Records what would have happened. Used by tests and by laptop runs."""

    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self.events: List[Tuple[float, AlertLevel, AlertProfile, str]] = []
        self.verbose = verbose

    def _set_profile(self, profile: AlertProfile, out: Output) -> None:
        self.events.append((out.t, out.level, profile, out.reason))
        if self.verbose:
            print(f"[{out.t:8.1f}s] {out.level.name:<8} {profile.message:<11} ({out.reason})")


class GpioActuator(Actuator):
    """Raspberry Pi outputs through gpiozero (vibration motor via MOSFET, buzzer)."""

    def __init__(self, vibration_pin: int = 18, buzzer_pin: int = 23, pwm_carrier_hz: int = 1000) -> None:
        super().__init__()
        from gpiozero import OutputDevice, PWMOutputDevice  # imported lazily: only exists on the Pi

        self._motor = PWMOutputDevice(vibration_pin, frequency=pwm_carrier_hz)
        self._buzzer = OutputDevice(buzzer_pin)
        self._profile = PROFILES[AlertLevel.NORMAL]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._envelope_loop, daemon=True)
        self._thread.start()

    def _set_profile(self, profile: AlertProfile, out: Output) -> None:
        self._profile = profile
        self._buzzer.value = 1 if profile.buzzer else 0

    def _envelope_loop(self) -> None:
        while not self._stop.is_set():
            prof = self._profile
            if prof.vibration_duty <= 0 or prof.pulse_hz <= 0:
                self._motor.off()
                self._stop.wait(0.05)
                continue
            half = 0.5 / prof.pulse_hz
            self._motor.value = prof.vibration_duty
            if self._stop.wait(half):
                break
            self._motor.off()
            if self._stop.wait(half):
                break
        self._motor.off()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._buzzer.off()
        self._motor.close()
        self._buzzer.close()
