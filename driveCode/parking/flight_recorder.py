import json
import os
import queue
import threading
from datetime import datetime

import cv2

from config import RECORD_BASE_DIR, RECORD_JPEG_QUALITY


class FlightRecorder:
    def __init__(self, base_dir=RECORD_BASE_DIR, jpeg_quality=RECORD_JPEG_QUALITY):
        self.run_dir = os.path.join(base_dir, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(self.run_dir, "frames"), exist_ok=True)
        self.jpeg_quality = jpeg_quality
        self.queue = queue.Queue(maxsize=120)
        self.dropped = 0
        self._stop = False
        self._file = open(os.path.join(self.run_dir, "telemetry.jsonl"), "w", encoding="utf-8")
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        print(f"[REC] recording: {self.run_dir}")

    def record(self, image, telemetry):
        try:
            self.queue.put_nowait((None if image is None else image.copy(), telemetry))
        except queue.Full:
            self.dropped += 1

    def _write_loop(self):
        while not (self._stop and self.queue.empty()):
            try:
                image, telemetry = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if image is not None:
                filename = f"frames/{telemetry['i']:06d}.jpg"
                cv2.imwrite(os.path.join(self.run_dir, filename), image,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                telemetry["frame_file"] = filename
            self._file.write(json.dumps(telemetry) + "\n")

    def close(self):
        self._stop = True
        self._writer.join(timeout=10.0)
        self._file.flush()
        self._file.close()
        print(f"[REC] closed: {self.run_dir} (dropped={self.dropped})")
