import cv2
import time
import serial                         
from ultralytics import YOLO


CAM_INDEX = 2
PAYLOAD_WIDTH = 13.0
TARGET_WIDTH = 19.5
TRESHOLD = 3.0
MAX_LOST_FRAMES = 15

COLOR_PAYLOAD = (0,   200, 255)          # orange
COLOR_TARGET  = (180,  60, 255)          # ungu
COLOR_ORIGIN  = (0,   255, 255)          # cyan

STATE_COLORS = {
    "HOME":              (180, 180, 180),
    "SEARCHING":         (0,   220, 255),
    "MOVING_TO_PAYLOAD": (0,   200, 255),
    "GRIPPING":          (100, 100, 255),
    "MOVING_TO_TARGET":  (180,  60, 255),
    "DONE":              (80,  255,  80),
}
# ==========================================


arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)   
time.sleep(2)                                              

def serial_send(command: str):
    arduino.write((command + '\n').encode())            
    print(f"[SERIAL] >> {command}")                         

def serial_read() -> str:
    # if arduino.in_waiting:                                 
        # return arduino.readline().decode().strip()        
    return ""


class StateMachine:
    HOME           = "HOME"
    SEARCHING      = "SEARCHING"
    MOVING_TO_PAYLOAD = "MOVING_TO_PAYLOAD"
    GRIPPING       = "GRIPPING"
    MOVING_TO_TARGET  = "MOVING_TO_TARGET"
    DONE           = "DONE"

    def __init__(self):
        self.state = self.HOME
        self.saved_payload = None
        self.saved_target  = None
        self.lost_counter  = 0
        self.command_sent = False

    def update(self, payload, target):
        s = self.state

        if s == self.HOME:
            #tunggu data dari Arduino
            self.setState(self.SEARCHING)

        elif s == self.SEARCHING:
            if payload and target:
                self.saved_payload = payload
                self.saved_target  = target
                print(f"[SEARCHING] Payload: X={payload[0]:.2f} Y={payload[1]:.2f} cm")
                print(f"[SEARCHING] Target:  X={target[0]:.2f}  Y={target[1]:.2f}  cm")
                self.setState(self.MOVING_TO_PAYLOAD)

        elif s == self.MOVING_TO_PAYLOAD:
            if payload:
                self.lost_counter = 0
                if self.isCentered(payload):
                    print("[MOVING_TO_PAYLOAD] Centered -> GRIPPING")
                    self.setState(self.GRIPPING)
                else:
                    if not self.command_sent:
                        serial_send(f"MOVE_PAYLOAD,{payload[0]:.2f},{payload[1]:.2f}")
                        self.command_sent = True
            else:
                self.lost_counter += 1
                if self.lost_counter > MAX_LOST_FRAMES:
                    print("[MOVING_TO_PAYLOAD] set to SEARCHING")
                    self._reset_saved()
                    self.setState(self.SEARCHING)

        elif s == self.GRIPPING:
            print("[GRIPPING] Gripping payload...")
            #kirim data buat gripper
            
            time.sleep(1)                
            self.lost_counter = 0
            self.setState(self.MOVING_TO_TARGET)

        elif s == self.MOVING_TO_TARGET:
            if target:
                self.lost_counter = 0
                if self.isCentered(target):
                    print("[MOVING_TO_TARGET] Centered -> DONE")
                    self.setState(self.DONE)
                else:
                    serial_send(f"{target[0]:.2f},{target[1]:.2f}")
            else:
                self.lost_counter += 1

        elif s == self.DONE:
            print("[DONE]")
            #drop payload ke target dan reset
            
            time.sleep(1)               
            self._reset_saved()
            self.setState(self.SEARCHING)

    def setState(self, new_state):
        print(f"[STATE] {self.state} -> {new_state}")
        self.state = new_state

    def isCentered(self, coord):
        x, y = coord
        return abs(x) < TRESHOLD and abs(y) < TRESHOLD

    def _reset_saved(self):
        self.saved_payload = None
        self.saved_target  = None
        self.lost_counter  = 0


class GantrySystem:
    def __init__(self):
        self.model  = YOLO("./Programming/runs/detect/train2/weights/best.pt")
        self.cap    = cv2.VideoCapture(CAM_INDEX)
        self.sm     = StateMachine()
        self.origin = None

        if not self.cap.isOpened():
            raise RuntimeError("nggak bisa buka kamera")

    def detect(self, frame):
        results  = self.model(frame, verbose=False)
        detected = []

        if len(results[0].boxes) > 0:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                area = (x2 - x1) * (y2 - y1)
                detected.append((area, x1, y1, x2, y2))

        if len(detected) >= 2:
            detected.sort(key=lambda d: d[0])            
            return detected[0][1:], detected[-1][1:]    
        if len(detected) == 1:
            return detected[0][1:], None
        return None, None

    def to_cm(self, bbox, ref_width_cm):
        x1, y1, x2, y2 = bbox
        bbox_w_px = x2 - x1
        if bbox_w_px == 0:
            return None
        scale = ref_width_cm / bbox_w_px
        x_cm  = ((x1 + x2) / 2 - self.origin[0]) * scale
        y_cm  = (self.origin[1] - (y1 + y2) / 2) * scale
        return (x_cm, y_cm)

    def draw_object(self, img, bbox, label, coord, color, ref_width_cm):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # crosshair
        cv2.line(img, (cx - 12, cy), (cx + 12, cy), color, 1)
        cv2.line(img, (cx, cy - 12), (cx, cy + 12), color, 1)
        cv2.circle(img, (cx, cy), 4, color, -1)
        
        # label dan koordinat
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if coord:
            cv2.putText(img, f"X:{coord[0]:.1f} Y:{coord[1]:.1f}cm",
                        (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # lingkaran threshold
        bbox_w_px = x2 - x1
        if bbox_w_px > 0:
            r = int(TRESHOLD * bbox_w_px / ref_width_cm)
            cv2.circle(img, (cx, cy), r, color, 1)

    def draw_hud(self, img):
        sm = self.sm
        color = STATE_COLORS.get(sm.state, (255, 255, 255))
        cv2.putText(img, f"STATE: {sm.state}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y = 36
        if sm.saved_payload:
            cv2.putText(img,
                f"[P] X:{sm.saved_payload[0]:.1f} Y:{sm.saved_payload[1]:.1f}cm",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_PAYLOAD, 1)
            y += 15
        if sm.saved_target:
            cv2.putText(img,
                f"[T] X:{sm.saved_target[0]:.1f} Y:{sm.saved_target[1]:.1f}cm",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TARGET, 1)
            y += 15
        if sm.state in (sm.MOVING_TO_PAYLOAD, sm.MOVING_TO_TARGET) and sm.lost_counter > 0:
            cv2.putText(img, f"lost:{sm.lost_counter}/{MAX_LOST_FRAMES}",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 255), 1)
            y += 15

    def run(self):
        print("Initializing...")
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("frame gagal dibaca")
                break

            if self.origin is None:
                h, w, _ = frame.shape
                self.origin = (w // 2, h // 2)

            payload_bbox, target_bbox = self.detect(frame)
            payload = self.to_cm(payload_bbox, PAYLOAD_WIDTH) if payload_bbox else None
            target  = self.to_cm(target_bbox,  TARGET_WIDTH)  if target_bbox  else None

            self.sm.update(payload, target)

            annotated = frame.copy()
            if payload_bbox:
                self.draw_object(annotated, payload_bbox, "PAYLOAD",
                                 payload, COLOR_PAYLOAD, PAYLOAD_WIDTH)
            if target_bbox:
                self.draw_object(annotated, target_bbox, "TARGET",
                                 target, COLOR_TARGET, TARGET_WIDTH)

            ox, oy = self.origin
            cv2.line(annotated, (ox - 12, oy), (ox + 12, oy), COLOR_ORIGIN, 1)
            cv2.line(annotated, (ox, oy - 12), (ox, oy + 12), COLOR_ORIGIN, 1)

            self.draw_hud(annotated)

            cv2.imshow("", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GantrySystem().run()