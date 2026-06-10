import cv2
import pickle
import numpy as np
import json
import time
import threading
import urllib.request
import os
from ultralytics import YOLO
from cloud_uploader import push_async

MODEL_PATH    = "yolov8x.pt"
CONF          = 0.25
CLASSES       = [2, 5, 7]
JSON_INTERVAL = 1.0
MARGIN        = 5


def find_all_parks():
    parks      = []
    parklar_dir = "parklar"
    if not os.path.exists(parklar_dir):
        print(f"HATA: {parklar_dir} klasörü bulunamadı!")
        return parks
    for name in os.listdir(parklar_dir):
        folder   = os.path.join(parklar_dir, name)
        cfg_path = os.path.join(folder, "config.json")
        pkl_path = os.path.join(folder, "parking_positions.pkl")
        if os.path.isdir(folder) and os.path.exists(cfg_path) and os.path.exists(pkl_path):
            parks.append(name)
    return sorted(parks)


def load_profile(park_name):
    folder   = os.path.join("parklar", park_name)
    cfg_path = os.path.join(folder, "config.json")
    pkl_path = os.path.join(folder, "parking_positions.pkl")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(pkl_path, "rb") as f:
        polygons = pickle.load(f)
    if not polygons or not isinstance(polygons[0], np.ndarray):
        print(f"[{park_name}] HATA: Geçersiz polygon formatı!")
        return None, None
    return cfg, polygons


class IPCamera:
    def __init__(self, park_id, url_template, interval):
        self.park_id   = park_id
        self._url      = url_template
        self._interval = interval
        self._frame    = None
        self._lock     = threading.Lock()
        self._running  = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                url  = self._url.format(int(time.time() * 1000))
                data = urllib.request.urlopen(url, timeout=5).read()
                arr  = np.frombuffer(data, dtype=np.uint8)
                img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    with self._lock:
                        self._frame = img
            except Exception as e:
                print(f"[{self.park_id}] Kamera hatası: {e}")
            time.sleep(self._interval)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False


class VideoCamera:
    def __init__(self, park_id, path, interval):
        self.park_id   = park_id
        self._cap      = cv2.VideoCapture(path)
        self._interval = interval
        self._frame    = None
        self._lock     = threading.Lock()
        self._running  = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self._lock:
                self._frame = frame
            time.sleep(self._interval)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self._cap.release()


def get_contact_point(x1, y1, x2, y2):
    return float((x1 + x2) / 2), float(y1 + (y2 - y1) * 0.85)


def check_occupancy(detections, polygons):
    results = []
    for idx, poly in enumerate(polygons):
        poly_f32 = poly.astype(np.float32)
        occupied = False
        for (x1, y1, x2, y2, _) in detections:
            cx, cy = get_contact_point(x1, y1, x2, y2)
            if cv2.pointPolygonTest(poly_f32, (cx, cy), True) >= -MARGIN:
                occupied = True
                break
        results.append({"id": idx + 1, "status": "occupied" if occupied else "empty"})
    return results


def draw(frame, polygons, results, park_name):
    overlay = frame.copy()
    for res, poly in zip(results, polygons):
        color = (0, 0, 180) if res["status"] == "occupied" else (0, 180, 0)
        cv2.fillPoly(overlay, [poly], color)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    for res, poly in zip(results, polygons):
        color = (0, 0, 255) if res["status"] == "occupied" else (0, 255, 100)
        label = "DOLU" if res["status"] == "occupied" else "BOS"
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(frame, str(res["id"]), (cx - 8, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, label, (cx - 18, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    total    = len(results)
    empty    = sum(1 for r in results if r["status"] == "empty")
    occ_rate = round((total - empty) / total * 100) if total else 0
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (15, 15, 15), -1)
    cv2.putText(frame,
                f"[{park_name}]  BOS:{empty}  DOLU:{total-empty}  TOPLAM:{total}  %{occ_rate}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
    cv2.putText(frame, time.strftime("%H:%M:%S"),
                (frame.shape[1] - 90, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    return frame


def main():
    park_names = find_all_parks()
    if not park_names:
        print("HATA: Hiç otopark profili bulunamadı!")
        return

    print(f"Bulunan otoparklar: {park_names}")

    profiles = {}
    cameras  = {}
    for name in park_names:
        cfg, polygons = load_profile(name)
        if cfg is None:
            continue
        profiles[name] = {"cfg": cfg, "polygons": polygons}

        source_type = cfg.get("source_type", "ip")
        fetch_int   = cfg.get("fetch_interval", 0.5)

        if source_type == "video":
            video_path = cfg.get("video_path", "")
            if not os.path.isabs(video_path):
                video_path = os.path.join("parklar", name, video_path)
            cameras[name] = VideoCamera(name, video_path, fetch_int)
        else:
            cameras[name] = IPCamera(name, cfg["camera_url"], fetch_int)

        print(f"[{name}] yüklendi — {cfg.get('name', name)}")

    print(f"\nModel yükleniyor: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model hazır!\n")

    last_json = {name: 0.0 for name in profiles}

    print("Tüm otoparklar aktif. Çıkmak için 'q'.")

    while True:
        for park_id, profile in profiles.items():
            cfg       = profile["cfg"]
            polygons  = profile["polygons"]
            camera    = cameras[park_id]
            park_name = cfg.get("name", park_id)

            frame = camera.read()
            if frame is None:
                continue

            yolo_res   = model(frame, conf=CONF, classes=CLASSES, verbose=False)
            detections = []
            for r in yolo_res:
                for box, c in zip(r.boxes.xyxy, r.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    detections.append((x1, y1, x2, y2, float(c)))

            slot_results = check_occupancy(detections, polygons)

            now = time.time()
            if now - last_json[park_id] >= JSON_INTERVAL:
                push_async(park_id, cfg, slot_results)
                last_json[park_id] = now

            annotated = draw(frame, polygons, slot_results, park_name)
            cv2.imshow(f"{park_name}", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cam in cameras.values():
        cam.stop()
    cv2.destroyAllWindows()
    print("Kapatıldı.")


if __name__ == "__main__":
    main()
