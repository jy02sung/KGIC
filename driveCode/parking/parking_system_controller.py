import json, os, threading, time
from datetime import datetime
import cv2, keyboard, numpy as np
from config import *
from flight_recorder import FlightRecorder


class ParkingSystemController:
    def __init__(self, image, sonic, motor):
        self.image, self.sonic, self.motor = image, sonic, motor
        self.mode, self.running, self.stop_requested = "parking", False, False
        self.state, self.maneuver_active = "IDLE", False
        self.phase_started = self.run_started = self.drive_started = None
        self.front_cm = self.rear_cm = ULTRASONIC_FAR_DISTANCE_CM
        self.confirm_frames = self.lost_frames = self.frame_index = 0
        self.center_ema, self.last_steering = None, 0.0
        self.manual_steering_target = 0.0
        self.manual_updated_at = time.monotonic()
        self.last_keys, self.state_log, self.recorder = {}, [], None
        self.lock = threading.RLock()
        self.thread = threading.Thread(target=self._control_loop, daemon=True)

    def _once(self, key):
        down, old = keyboard.is_pressed(key), self.last_keys.get(key, False)
        self.last_keys[key] = down
        return down and not old

    def _log_state(self, kind):
        self.state_log.append({"time": time.monotonic(), "state": self.state, "kind": kind,
            "front_cm": self.front_cm, "rear_cm": self.rear_cm,
            "left_speed": self.motor.left_speed, "right_speed": self.motor.right_speed,
            "steer_target": self.motor.steering_angle})

    def _enter(self, state, kind="maneuver"):
        self.state, self.phase_started = state, time.monotonic(); self._log_state(kind)
        print(f"[PARK] phase -> {state}")

    def _start(self):
        self.state, self.maneuver_active = "WAIT_FRONT_HIGH_REAR_LOW", False
        self.run_started = self.drive_started = self.phase_started = time.monotonic()
        self.confirm_frames = self.lost_frames = 0; self.center_ema = None; self.last_steering = 0.0
        self.image.lane_follow_side, self.image.last_lane_center = "right", None
        self.state_log = []; self._log_state("space_start")
        self.motor.set_command(DRIVE_SPEED, DRIVE_SPEED, 0, True)
        if RECORD_ENABLE: self.recorder = FlightRecorder()
        print("[PARK] started")

    def _stop(self, message="Drive: stop"):
        self.running, self.maneuver_active, self.state = False, False, "IDLE"
        self.manual_steering_target = 0.0
        self.manual_updated_at = time.monotonic()
        self.motor.emergency_stop(); recorder, self.recorder = self.recorder, None
        if recorder: recorder.close()
        print(message)

    def _steer_ready(self, angle):
        return abs(float(np.clip(angle+STEER_TRIM, -20, 20))-self.motor.last_status["mapped"]) < STEER_DEADZONE

    def _save_log(self):
        if not self.state_log or self.run_started is None: return
        os.makedirs(PARKING_LOG_DIR, exist_ok=True)
        records = [dict(item, elapsed_seconds=round(item["time"]-self.run_started, 3)) for item in self.state_log]
        payload = {"final_state": self.state, "duration_seconds": round(time.monotonic()-self.run_started, 3),
                   "saved_at": datetime.now().isoformat(timespec="seconds"), "records": records}
        path = os.path.join(PARKING_LOG_DIR, datetime.now().strftime("parking_%Y%m%d_%H%M%S.json"))
        with open(path, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[PARK LOG] saved: {path}")

    def _safe_stop(self):
        self._enter("PARK_SAFE_STOP"); self.motor.emergency_stop()
        self.running, self.maneuver_active = False, False; self._save_log()
        print("[PARK] safe stop: steering timeout")

    def _update_state(self):
        now = time.monotonic(); command = self.motor.set_command
        if not self.maneuver_active:
            self.front_cm, self.rear_cm = self.sonic.read_pair()
            if self.front_cm == ULTRASONIC_INVALID_DISTANCE_CM or self.rear_cm == ULTRASONIC_INVALID_DISTANCE_CM:
                self.confirm_frames = 0; return
            fh, rh = self.front_cm < ULTRASONIC_MAX_DISTANCE_CM, self.rear_cm < ULTRASONIC_MAX_DISTANCE_CM
            transitions = {"WAIT_FRONT_HIGH_REAR_LOW": (fh and not rh, "FRONT_HIGH_REAR_LOW_SEEN"),
                "FRONT_HIGH_REAR_LOW_SEEN": (not fh and rh, "FRONT_LOW_REAR_HIGH_SEEN"),
                "FRONT_LOW_REAR_HIGH_SEEN": (not rh, "PARK_STOP_SETTLE")}
            if self.state not in transitions: return
            condition, nxt = transitions[self.state]
            if not condition: self.confirm_frames = 0; return
            self.confirm_frames += 1
            if self.confirm_frames < ULTRASONIC_PHASE_CONFIRM_FRAMES: return
            self.confirm_frames = 0; self._enter(nxt, "sensor")
            if nxt == "FRONT_HIGH_REAR_LOW_SEEN": command(70, 70, self.motor.steering_angle)
            elif nxt == "FRONT_LOW_REAR_HIGH_SEEN": command(40, 40, self.motor.steering_angle)
            else: self.maneuver_active = True; command(100, 100, 0)
            return
        elapsed = now-self.phase_started
        if self.state == "PARK_STOP_SETTLE":
            command(100,100,0)
            if elapsed >= .1: self._enter("PARK_LEFT_STEER_PREP")
        elif self.state == "PARK_LEFT_STEER_PREP":
            command(100,100,PARK_LEFT_STEER)
            if self._steer_ready(PARK_LEFT_STEER): self._enter("PARK_LEFT_FORWARD")
            elif elapsed >= PARK_STEER_TIMEOUT_SECONDS: self._safe_stop()
        elif self.state == "PARK_LEFT_FORWARD":
            command(100,100,PARK_LEFT_STEER)
            if elapsed >= PARK_LEFT_FORWARD_SECONDS: self._enter("PARK_FORWARD_BRAKE")
        elif self.state == "PARK_FORWARD_BRAKE":
            command(0,0,PARK_LEFT_STEER)
            if self.motor.drive_stopped() and elapsed >= PARK_STOP_SETTLE_SECONDS: self._enter("PARK_PRE_REVERSE_STRAIGHT_STEER")
        elif self.state == "PARK_PRE_REVERSE_STRAIGHT_STEER":
            command(0,0,0)
            if self._steer_ready(0): self._enter("PARK_PRE_RIGHT_STRAIGHT_REVERSE")
            elif elapsed >= PARK_STEER_TIMEOUT_SECONDS: self._safe_stop()
        elif self.state == "PARK_PRE_RIGHT_STRAIGHT_REVERSE":
            command(PARK_REVERSE_SPEED,PARK_REVERSE_SPEED,0)
            if elapsed >= PARK_PRE_RIGHT_STRAIGHT_REVERSE_SECONDS: self._enter("PARK_RIGHT_STEER_PREP")
        elif self.state == "PARK_RIGHT_STEER_PREP":
            command(PARK_REVERSE_SPEED,PARK_REVERSE_SPEED,PARK_RIGHT_STEER)
            if self._steer_ready(PARK_RIGHT_STEER): self._enter("PARK_RIGHT_REVERSE")
            elif elapsed >= PARK_STEER_TIMEOUT_SECONDS: self._safe_stop()
        elif self.state == "PARK_RIGHT_REVERSE":
            command(PARK_REVERSE_SPEED,PARK_REVERSE_SPEED,PARK_RIGHT_STEER)
            if elapsed >= PARK_RIGHT_REVERSE_SECONDS: self._enter("PARK_FINAL_STRAIGHT_STEER_PREP")
        elif self.state == "PARK_FINAL_STRAIGHT_STEER_PREP":
            command(PARK_REVERSE_SPEED,PARK_REVERSE_SPEED,0)
            if self._steer_ready(0): self._enter("PARK_FINAL_STRAIGHT_REVERSE")
            elif elapsed >= PARK_STEER_TIMEOUT_SECONDS: self._safe_stop()
        elif self.state == "PARK_FINAL_STRAIGHT_REVERSE":
            command(PARK_REVERSE_SPEED,PARK_REVERSE_SPEED,0)
            if elapsed >= PARK_STRAIGHT_REVERSE_SECONDS: self._enter("PARK_HOLD")
        elif self.state == "PARK_HOLD":
            command(0,0,0)
            if elapsed >= PARK_HOLD_SECONDS:
                command(EXIT_STRAIGHT_FORWARD_SPEED,EXIT_STRAIGHT_FORWARD_SPEED,0,True); self._enter("EXIT_STRAIGHT_FORWARD","exit")
        elif self.state == "EXIT_STRAIGHT_FORWARD":
            command(EXIT_STRAIGHT_FORWARD_SPEED,EXIT_STRAIGHT_FORWARD_SPEED,0)
            if elapsed >= EXIT_STRAIGHT_FORWARD_SECONDS: self._enter("EXIT_RIGHT_STEER_PREP","exit")
        elif self.state == "EXIT_RIGHT_STEER_PREP":
            command(EXIT_RIGHT_FORWARD_SPEED,EXIT_RIGHT_FORWARD_SPEED,EXIT_RIGHT_STEER)
            if self._steer_ready(EXIT_RIGHT_STEER): self._enter("EXIT_RIGHT_FORWARD","exit")
            elif elapsed >= PARK_STEER_TIMEOUT_SECONDS: self._safe_stop()
        elif self.state == "EXIT_RIGHT_FORWARD":
            command(EXIT_RIGHT_FORWARD_SPEED,EXIT_RIGHT_FORWARD_SPEED,EXIT_RIGHT_STEER)
            if elapsed >= EXIT_RIGHT_FORWARD_SECONDS:
                self.maneuver_active = False; self.center_ema = None; self.lost_frames = 0
                command(EXIT_STRAIGHT_OUT_SPEED,EXIT_STRAIGHT_OUT_SPEED,0)
                self._enter("EXIT_STRAIGHT_OUT","exit"); self._save_log()
                print("[EXIT] driving straight out; Space stops")

    def _control_loop(self):
        while not self.stop_requested:
            with self.lock:
                if self.running and self.mode == "parking": self._update_state()
                self.motor.control_once()
            time.sleep(1/CONTROL_HZ)

    def _manual(self):
        now = time.monotonic()
        elapsed = min(now-self.manual_updated_at, 0.1)
        self.manual_updated_at = now
        speed = MANUAL_DRIVE_SPEED if keyboard.is_pressed("w") and not keyboard.is_pressed("s") else (-MANUAL_DRIVE_SPEED if keyboard.is_pressed("s") and not keyboard.is_pressed("w") else 0)
        left = keyboard.is_pressed("a")
        right = keyboard.is_pressed("d")
        if left and not right:
            self.manual_steering_target -= MANUAL_STEER_RATE*elapsed
        elif right and not left:
            self.manual_steering_target += MANUAL_STEER_RATE*elapsed
        self.manual_steering_target = float(np.clip(
            self.manual_steering_target, -MANUAL_STEER_TARGET, MANUAL_STEER_TARGET
        ))
        self.motor.set_command(speed,speed,self.manual_steering_target)

    def _keys(self):
        if self._once("1"): self._stop("Mode: parking (stopped)"); self.mode = "parking"
        if self._once("2"): self._stop("Mode: manual (stopped)"); self.mode = "manual"
        if self._once("space"):
            if self.running: self._stop()
            else:
                self.running = True
                if self.mode == "parking": self._start()
                else:
                    self.manual_steering_target = 0.0
                    self.manual_updated_at = time.monotonic()
                    print("Drive: start (manual)")
        if self._once("q"): self.stop_requested = True; self.motor.emergency_stop()

    def _vision(self, frame):
        center, visual = self.image.process_frame(frame)
        if self.state == "EXIT_STRAIGHT_OUT":
            self.last_steering = 0; self.motor.set_command(EXIT_STRAIGHT_OUT_SPEED,EXIT_STRAIGHT_OUT_SPEED,0)
            return center, visual
        if not self.maneuver_active:
            if time.monotonic()-self.drive_started < START_STRAIGHT_SECONDS: self.last_steering=0; self.center_ema=None; self.lost_frames=0
            elif center is None:
                self.lost_frames += 1
                if self.lost_frames > LOST_HOLD_FRAMES: self.last_steering=0; self.center_ema=None
            else:
                self.lost_frames=0; self.center_ema=float(center) if self.center_ema is None else CENTER_EMA*center+(1-CENTER_EMA)*self.center_ema
                self.last_steering=float(np.clip((self.center_ema-REF_X)*STEER_GAIN,-20,20))
            self.motor.set_command(self.motor.left_speed,self.motor.right_speed,self.last_steering)
        return center, visual

    def run(self, camera_index=0):
        cap=cv2.VideoCapture(camera_index); cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAPTURE_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAPTURE_H)
        if not cap.isOpened(): raise RuntimeError("Camera could not be opened")
        self.thread.start(); print("Modes: 1 parking, 2 manual\nKeys: Space start/stop, W/A/S/D manual, Q quit")
        try:
            while not self.stop_requested:
                self._keys()
                if not self.running: time.sleep(.01); continue
                if self.mode == "manual": self._manual(); time.sleep(.005); continue
                ok, frame=cap.read()
                if not ok: raise RuntimeError("Camera frame read failed")
                center, visual=self._vision(frame); status=self.motor.last_status
                if self.recorder: self.recorder.record(visual,{"t":round(time.time(),3),"i":self.frame_index,"state":self.state,"front_cm":round(self.front_cm,2),"rear_cm":round(self.rear_cm,2),"center":center,"lost":self.lost_frames,"steer":round(self.motor.steering_angle,2),"target":round(status["target"],2),"mapped":round(status["mapped"],2),"adc":status["adc"],"cmd":status["cmd"],"duty":round(status["duty"],3),"exec_ms":round(self.image.last_exec_ms,1),"boxes":self.image.last_boxes,"scores":self.image.last_scores})
                if self.frame_index%HEARTBEAT_EVERY==0: print(f"[PARK] {self.state} F={self.front_cm:.1f} R={self.rear_cm:.1f} center={center}")
                self.frame_index+=1
        finally:
            self.stop_requested=True; self.motor.emergency_stop(); self.thread.join(timeout=1); cap.release()
            if self.recorder: self.recorder.close()
            self.motor.close()
