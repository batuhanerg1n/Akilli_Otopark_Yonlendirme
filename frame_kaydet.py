import argparse
import urllib.request
import numpy as np
import cv2
import json
import os
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--park", required=True, help="Otopark profil adı (parklar/ altındaki klasör)")
    return p.parse_args()


def main():
    args     = parse_args()
    folder   = os.path.join("parklar", args.park)
    cfg_path = os.path.join(folder, "config.json")

    if not os.path.exists(cfg_path):
        print(f"HATA: {cfg_path} bulunamadı!")
        exit()

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    source_type = cfg.get("source_type", "ip")

    if source_type == "video":
        video_path = cfg.get("video_path", "")
        if not os.path.isabs(video_path):
            video_path = os.path.join(folder, video_path)
        print(f"[{cfg['name']}] Videodan frame çekiliyor → {video_path}")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"HATA: Video okunamadı → {video_path}")
            exit()
    else:
        url = cfg["camera_url"].format(int(time.time() * 1000))
        print(f"[{cfg['name']}] Kameradan frame çekiliyor...")
        print(f"  URL: {url}")
        try:
            data  = urllib.request.urlopen(url, timeout=8).read()
            arr   = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"HATA: {e}")
            exit()

    if frame is None:
        print("HATA: Frame alınamadı!")
        exit()

    out_path = os.path.join(folder, "carParkImg.png")
    cv2.imwrite(out_path, frame)
    print(f"Frame kaydedildi → {out_path}  ({frame.shape[1]}x{frame.shape[0]})")


if __name__ == "__main__":
    main()