import cv2
import pickle
import numpy as np
import json
import time
import threading
import urllib.request
import os
import argparse
from ultralytics import YOLO
from cloud_uploader import push_async

MODEL_PATH    = "yolov8x.pt"
CONF          = 0.25
CLASSES       = [2, 5, 7]
JSON_INTERVAL = 1.0
MARGIN        = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--park", required=True, help="Otopark profil adı")
    return p.parse_args()


class IPCamera:
    def __init__(self, url_template, interval):
        self._url      = url_template
        self._interval = interval
        self._frame    = None
        self._lock     = threading.Lock()
        self._running  = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("Kameraya bağlanılıyor...", end="", flush=True)
        for _ in range(20):
            time.sleep(0.5)
            with self._lock:
                if self._frame is not None:
                    print(" ✓")
                    return
        print(" ZAMAN AŞIMI!")

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
                print(f"\n[Kamera Hatası] {e}")
            time.sleep(self._interval)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False


class VideoCamera:
    def __init__(self, path, interval):
        if not os.path.exists(path):
            print(f"HATA: Video bulunamadı → {path}")
            exit()
        self._cap      = cv2.VideoCapture(path)
        self._interval = interval
        self._frame    = None
        self._lock     = threading.Lock()
        self._running  = True
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"Video yüklendi → {path} ✓")

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
class YoutubeCamera:
    def __init__(self, url, interval):
        self._interval = interval
        self._frame    = None
        self._lock     = threading.Lock()
        self._running  = True
        self._stream_url = self._get_stream_url(url)
        if not self._stream_url:
            print("HATA: YouTube stream URL alınamadı!")
            exit()
        threading.Thread(target=self._loop, daemon=True).start()
        print("YouTube kamerasına bağlanılıyor...", end="", flush=True)
        for _ in range(20):
            time.sleep(0.5)
            with self._lock:
                if self._frame is not None:
                    print(" ✓")
                    return
        print(" ZAMAN AŞIMI!")

    def _get_stream_url(self, url):
        try:
            import yt_dlp
            ydl_opts = {"format": "best[ext=mp4]/best", "quiet": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info["url"]
        except Exception as e:
            print(f"\n[YouTube] URL alınamadı: {e}")
            return None

    def _loop(self):
        cap = cv2.VideoCapture(self._stream_url)
        while self._running:
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(self._stream_url)
                continue
            with self._lock:
                self._frame = frame
            time.sleep(self._interval)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False

def load_profile(park_name):
    folder   = os.path.join("parklar", park_name)
    cfg_path = os.path.join(folder, "config.json")
    pkl_path = os.path.join(folder, "parking_positions.pkl")

    if not os.path.exists(cfg_path):
        print(f"HATA: {cfg_path} bulunamadı!")
        exit()
    if not os.path.exists(pkl_path):
        print(f"HATA: {pkl_path} bulunamadı!")
        print(f"  → Önce: python parking_space_picker.py --park {park_name}")
        exit()

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(pkl_path, "rb") as f:
        polygons = pickle.load(f)

    if not polygons or not isinstance(polygons[0], np.ndarray):
        print("HATA: Geçersiz polygon formatı. Yeniden picker çalıştır.")
        exit()

    return cfg, polygons, folder


def get_contact_point(x1, y1, x2, y2):
    cx = float((x1 + x2) / 2)
    cy = float(y1 + (y2 - y1) * 0.85)
    return cx, cy


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
        results.append({
            "id":     idx + 1,
            "status": "occupied" if occupied else "empty",
        })
    return results


def draw(frame, polygons, results, detections, park_name):
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

    for (x1, y1, x2, y2, _) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 1)
        tcx, tcy = get_contact_point(x1, y1, x2, y2)
        cv2.circle(frame, (int(tcx), int(tcy)), 5, (255, 0, 255), -1)
        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 3, (0, 0, 255), -1)

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


def save_json(results, cfg, park_id, output_path):
    total = len(results)
    empty = sum(1 for r in results if r["status"] == "empty")

    all_data = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                all_data = json.load(f)
        except:
            pass

    all_data[park_id] = {
        "name":           cfg.get("name", park_id),
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_slots":    total,
        "empty_slots":    empty,
        "full_slots":     total - empty,
        "occupancy_rate": round((total - empty) / total * 100, 1) if total else 0,
        "slots":          results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    cfg, polygons, folder = load_profile(args.park)

    park_name   = cfg.get("name", args.park)
    fetch_int   = cfg.get("fetch_interval", 0.5)
    source_type = cfg.get("source_type", "ip")
    output_json = "parking_status.json"

    print(f"Profil    : {park_name}")
    print(f"Kaynak    : {source_type}")
    print(f"Park yeri : {len(polygons)}")

    model = YOLO(MODEL_PATH)

    if source_type == "video":
        video_path = cfg.get("video_path", "")
        if not os.path.isabs(video_path):
            video_path = os.path.join(folder, video_path)
        camera = VideoCamera(video_path, fetch_int)
    else:
        camera = IPCamera(cfg["camera_url"], fetch_int)

    last_json = 0.0
    print("Detector başladı. Çıkmak için 'q'.")

    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.1)
            continue

        yolo_res = model(frame, conf=CONF, classes=CLASSES, verbose=False)
        detections = []
        for r in yolo_res:
            for box, c in zip(r.boxes.xyxy, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2, float(c)))

        slot_results = check_occupancy(detections, polygons)

        now = time.time()
        if now - last_json >= JSON_INTERVAL:
            save_json(slot_results, cfg, args.park, output_json)
            push_async(args.park, cfg, slot_results)
            last_json = now

        annotated = draw(frame, polygons, slot_results, detections, park_name)
        cv2.imshow(f"Otopark — {park_name}", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()