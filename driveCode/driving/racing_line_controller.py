import math
import threading

from config import (
    RACING_CENTER_REF_X,
    RACING_CORNER_ENTER,
    RACING_CORNER_EXIT,
    RACING_CORNER_HOLD_SEC,
    RACING_FIXED_REF_X_BY_CORNER,
    RACING_LANE_WIDTH_PX,
    RACING_LAP_REARM_SEC,
    RACING_LINE_GAIN,
    RACING_MIN_CORNER_DPSI,
    RACING_PROFILE_TIMING_BY_CORNER,
    RACING_PSI_K,
    RACING_REFX_C2_TAU_SEC,
    RACING_REFX_DELTA_LIMIT,
    RACING_T1_VISION_ENTER,
    RACING_T1_VISION_HOLD_SEC,
    RACING_T4_MIN_EXIT_PROGRESS,
    RACING_TRACK_CORNERS,
    RACING_USE_PROFILE_BY_CORNER,
)


def _clip(value, low, high):
    return min(high, max(low, float(value)))


def _smootherstep01(value):
    x = _clip(value, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _blend_c2(start, end, progress):
    weight = _smootherstep01(progress)
    return float(start + (end - start) * weight)


class C2ReferenceFilter:
    """Three cascaded first-order filters; reference position is C2-smooth."""

    def __init__(self, initial=0.0, tau=0.12):
        self.tau = float(tau)
        self.reset(initial)

    def reset(self, value=0.0):
        self.x1 = self.x2 = self.x3 = float(value)

    def update(self, target, dt):
        dt = _clip(dt, 0.0, 0.5)
        if dt <= 0.0:
            return self.x3
        alpha = 1.0 - math.exp(-dt / max(self.tau, 1e-4))
        self.x1 += alpha * (float(target) - self.x1)
        self.x2 += alpha * (self.x1 - self.x2)
        self.x3 += alpha * (self.x2 - self.x3)
        return self.x3


class RacingLineController:
    """Continuous-lap v15 racing-line state shared by vision and 100 Hz control."""

    def __init__(self):
        self._lock = threading.RLock()
        self._reference_filter = C2ReferenceFilter(0.0, RACING_REFX_C2_TAU_SEC)
        self.reset()

    def reset(self):
        with self._lock:
            self.psi = 0.0
            self.corner_idx = -1
            self.in_corner = False
            self.psi_entry = 0.0
            self.completed_laps = 0
            self._enter_hold = 0.0
            self._enter_sign = 0
            self._vision_enter_hold = 0.0
            self._exit_hold = 0.0
            self._t4_hold = 0.0
            self._landmark_set = False
            self._t1_armed = True
            self._lap_rearm_hold = 0.0
            self.ref_x = float(RACING_CENTER_REF_X)
            self.target_fraction = 0.5
            self.last_fraction = 0.5
            self._reference_filter.reset(0.0)

    @staticmethod
    def _t4_index():
        return next((i for i, corner in enumerate(RACING_TRACK_CORNERS) if corner[2] > 0), -1)

    @staticmethod
    def _fixed_fraction(corner_name):
        return 0.5 + (
            float(RACING_FIXED_REF_X_BY_CORNER[corner_name]) - RACING_CENTER_REF_X
        ) / RACING_LANE_WIDTH_PX

    @staticmethod
    def _profile_fraction(curve, progress):
        name, _, _, entry, apex, exit_value = curve
        apex_at, exit_at = RACING_PROFILE_TIMING_BY_CORNER[name]
        p = _clip(progress, 0.0, 1.0)
        if p <= apex_at:
            return _blend_c2(entry, apex, p / max(apex_at, 1e-6))
        if p <= exit_at:
            return _blend_c2(apex, exit_value, (p - apex_at) / max(exit_at - apex_at, 1e-6))
        return float(exit_value)

    def _progress_unlocked(self):
        if not self.in_corner or not (0 <= self.corner_idx < len(RACING_TRACK_CORNERS)):
            return None
        dpsi = RACING_TRACK_CORNERS[self.corner_idx][1]
        if dpsi == 0:
            return 0.0
        return _clip((self.psi - self.psi_entry) / dpsi, 0.0, 1.0)

    def progress(self):
        with self._lock:
            return self._progress_unlocked()

    def _prepare_next_lap_unlocked(self):
        self.psi = 0.0
        self.corner_idx = -1
        self.in_corner = False
        self.psi_entry = 0.0
        self._enter_hold = 0.0
        self._enter_sign = 0
        self._vision_enter_hold = 0.0
        self._exit_hold = 0.0
        self._t4_hold = 0.0
        self._landmark_set = False
        # Do not mistake residual T5 left steering for the next T1.
        self._t1_armed = False
        self._lap_rearm_hold = 0.0

    def update(self, mapped, dt, commanded_steer=None):
        """Update track position from physical wheel angle at the 100 Hz rate."""
        dt = float(dt)
        if dt <= 0.0 or dt > 0.5:
            return
        with self._lock:
            mapped = float(mapped)
            self.psi += RACING_PSI_K * mapped * dt
            angle_magnitude = abs(mapped)
            t4 = self._t4_index()

            # T4 is the only right turn. It may pull an earlier state forward to
            # T4, but must never roll T5 or a later state back to T4.
            if t4 >= 0 and mapped > RACING_CORNER_ENTER:
                self._t4_hold += dt
                if self._t4_hold >= RACING_CORNER_HOLD_SEC and self.corner_idx < t4:
                    self.corner_idx = t4
                    self.in_corner = True
                    self.psi_entry = self.psi
                    self._landmark_set = True
                    self._exit_hold = 0.0
            else:
                self._t4_hold = 0.0

            if self.corner_idx == -1 and not self._t1_armed:
                if angle_magnitude < RACING_CORNER_EXIT:
                    self._lap_rearm_hold += dt
                    if self._lap_rearm_hold >= RACING_LAP_REARM_SEC:
                        self._t1_armed = True
                else:
                    self._lap_rearm_hold = 0.0

            if not self.in_corner and self.corner_idx == -1 and self._t1_armed:
                if commanded_steer is not None and commanded_steer < -RACING_T1_VISION_ENTER:
                    self._vision_enter_hold += dt
                    if self._vision_enter_hold >= RACING_T1_VISION_HOLD_SEC:
                        self.corner_idx = 0
                        self.in_corner = True
                        self.psi_entry = self.psi
                        self._landmark_set = False
                        self._vision_enter_hold = 0.0
                        self._enter_hold = 0.0
                        self._exit_hold = 0.0
                else:
                    self._vision_enter_hold = 0.0

            if not self.in_corner:
                # After lap reset, wait for a neutral straight before accepting
                # either vision or wheel-angle evidence for the next T1.
                if self.corner_idx == -1 and not self._t1_armed:
                    return
                sign = 1 if mapped > 0 else -1
                if angle_magnitude > RACING_CORNER_ENTER and sign == self._enter_sign:
                    self._enter_hold += dt
                    if self._enter_hold >= RACING_CORNER_HOLD_SEC:
                        next_idx = self.corner_idx + 1
                        if next_idx < len(RACING_TRACK_CORNERS):
                            self.corner_idx = next_idx
                            self.in_corner = True
                            self.psi_entry = self.psi
                            self._landmark_set = False
                        self._enter_hold = 0.0
                        self._exit_hold = 0.0
                elif angle_magnitude > RACING_CORNER_ENTER:
                    self._enter_sign = sign
                    self._enter_hold = dt
                else:
                    self._enter_hold = 0.0
                return

            progress = self._progress_unlocked()
            t4_exit_ready = self.corner_idx != t4 or (
                progress is not None and progress >= RACING_T4_MIN_EXIT_PROGRESS
            )
            if angle_magnitude < RACING_CORNER_EXIT and t4_exit_ready:
                self._exit_hold += dt
                if self._exit_hold >= RACING_CORNER_HOLD_SEC:
                    completed_idx = self.corner_idx
                    turn_amount = abs(self.psi - self.psi_entry)
                    self.in_corner = False
                    self._exit_hold = 0.0
                    self._enter_hold = 0.0
                    if (
                        not self._landmark_set
                        and turn_amount < RACING_MIN_CORNER_DPSI
                        and self.corner_idx >= 0
                    ):
                        self.corner_idx -= 1
                    elif completed_idx >= len(RACING_TRACK_CORNERS) - 1:
                        self.completed_laps += 1
                        self._prepare_next_lap_unlocked()
            else:
                self._exit_hold = 0.0

    def _corner_target_unlocked(self, curve, progress=None):
        name = curve[0]
        if RACING_USE_PROFILE_BY_CORNER[name]:
            desired = curve[3] if progress is None else self._profile_fraction(curve, progress)
            return 0.5 + RACING_LINE_GAIN * (desired - 0.5)
        return self._fixed_fraction(name)

    def compute_ref_x(self, dt, lane_ok=True):
        """Return the current racing reference without blocking the vision loop."""
        with self._lock:
            if not lane_ok:
                target_fraction = 0.5
            elif self.in_corner and 0 <= self.corner_idx < len(RACING_TRACK_CORNERS):
                curve = RACING_TRACK_CORNERS[self.corner_idx]
                target_fraction = self._corner_target_unlocked(curve, self._progress_unlocked())
            else:
                next_idx = self.corner_idx + 1
                if next_idx < len(RACING_TRACK_CORNERS):
                    target_fraction = self._corner_target_unlocked(RACING_TRACK_CORNERS[next_idx])
                else:
                    target_fraction = 0.5

            self.target_fraction = float(target_fraction)
            raw_delta = (self.target_fraction - 0.5) * RACING_LANE_WIDTH_PX
            raw_delta = _clip(raw_delta, -RACING_REFX_DELTA_LIMIT, RACING_REFX_DELTA_LIMIT)
            filtered_delta = self._reference_filter.update(raw_delta, dt)
            self.ref_x = RACING_CENTER_REF_X + filtered_delta
            self.last_fraction = 0.5 + filtered_delta / RACING_LANE_WIDTH_PX
            return self.ref_x

    def snapshot(self):
        with self._lock:
            if self.in_corner and 0 <= self.corner_idx < len(RACING_TRACK_CORNERS):
                segment = RACING_TRACK_CORNERS[self.corner_idx][0]
            else:
                next_idx = self.corner_idx + 1
                segment = "S>" + RACING_TRACK_CORNERS[next_idx][0] if next_idx < len(RACING_TRACK_CORNERS) else "S>FIN"
            return {
                "lap": self.completed_laps + 1,
                "completed_laps": self.completed_laps,
                "segment": segment,
                "progress": self._progress_unlocked(),
                "ref_x": self.ref_x,
            }
