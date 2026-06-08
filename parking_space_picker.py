
import cv2
import pickle
import numpy as np
import os
import json
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--park", required=True, help="Otopark profil adı")
    return p.parse_args()


POINTS_PER_SLOT = 4
COLOR_DONE    = (0, 255, 100)
COLOR_PREVIEW = (0, 200, 255)
COLOR_POINT   = (0, 0, 255)
COLOR_LABEL   = (255, 255, 255)

polygons: list[np.ndarray] = []
current_pts: list[tuple]   = []
SAVE_PATH  = ""
PARK_NAME  = ""


def save():
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(polygons, f)


def mouse_click(event, x, y, flags, param):
    global current_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append((x, y))
        if len(current_pts) == POINTS_PER_SLOT:
            poly = np.array(current_pts, dtype=np.int32)
            polygons.append(poly)
            current_pts = []
            save()
            print(f"[{PARK_NAME}] Park yeri #{len(polygons)} eklendi.")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if polygons:
            polygons.pop()
            save()
            print(f"[{PARK_NAME}] Son park yeri silindi. Kalan: {len(polygons)}")


def draw(img):
    overlay = img.copy()
    for poly in polygons:
        cv2.fillPoly(overlay, [poly], color=(0, 255, 100))
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    for i, poly in enumerate(polygons):
        cv2.polylines(img, [poly], isClosed=True, color=COLOR_DONE, thickness=2)
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(img, str(i + 1), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_LABEL, 2)

    for pt in current_pts:
        cv2.circle(img, pt, 6, COLOR_POINT, -1)
    if len(current_pts) > 1:
        for i in range(len(current_pts) - 1):
            cv2.line(img, current_pts[i], current_pts[i + 1], COLOR_PREVIEW, 2)
    if current_pts:
        kalan = POINTS_PER_SLOT - len(current_pts)
        cv2.putText(img, f"{kalan} nokta daha...",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_PREVIEW, 2)

    cv2.rectangle(img, (0, 0), (img.shape[1], 55), (15, 15, 15), -1)
    cv2.putText(img,
                "Sol tik: nokta  |  Sag tik: son alani sil  |  z: geri al  |  c: temizle  |  q: cik",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(img,
                f"[{PARK_NAME}]  Park yeri: {len(polygons)}   Nokta: {len(current_pts)}/{POINTS_PER_SLOT}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1)
    return img


def main():
    global polygons, current_pts, SAVE_PATH, PARK_NAME

    args   = parse_args()
    folder = os.path.join("parklar", args.park)

    # Config yükle
    cfg_path = os.path.join(folder, "config.json")
    if not os.path.exists(cfg_path):
        print(f"HATA: {cfg_path} bulunamadı!")
        exit()
    with open(cfg_path) as f:
        cfg = json.load(f)
    PARK_NAME = cfg.get("name", args.park)

    # Görüntü yükle
    img_path = os.path.join(folder, "carParkImg.png")
    if not os.path.exists(img_path):
        print(f"HATA: {img_path} bulunamadı!")
        print(f"  → Önce: python frame_kaydet.py --park {args.park}")
        exit()
    img_orig = cv2.imread(img_path)

    # Mevcut pozisyonları yükle
    SAVE_PATH = os.path.join(folder, "parking_positions.pkl")
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "rb") as f:
            data = pickle.load(f)
        if data and isinstance(data[0], np.ndarray):
            polygons = data
        print(f"[{PARK_NAME}] {len(polygons)} park yeri yüklendi.")

    print(f"\n[{PARK_NAME}] Picker başlatıldı.")
    print("Sol tık: nokta koy (4 = 1 alan) | Sağ tık: son alanı sil | z: geri al | c: temizle | q: çık\n")

    cv2.namedWindow(f"Polygon Picker — {PARK_NAME}", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(f"Polygon Picker — {PARK_NAME}", mouse_click)

    while True:
        cv2.imshow(f"Polygon Picker — {PARK_NAME}", draw(img_orig.copy()))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("z"):
            if current_pts:
                current_pts.pop()
        elif key == ord("c"):
            polygons.clear()
            current_pts.clear()
            save()
            print(f"[{PARK_NAME}] Temizlendi.")

    cv2.destroyAllWindows()
    print(f"[{PARK_NAME}] {len(polygons)} park yeri kaydedildi → {SAVE_PATH}")


if __name__ == "__main__":
    main()
