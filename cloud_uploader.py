

import urllib.request
import urllib.error
import json
import threading

# ── 'URL' ──
BACKEND_URL = "https://web-production-ccdd52.up.railway.app"


_lock = threading.Lock()


def push_to_cloud(park_id: str, cfg: dict, slot_results: list) -> bool:
    """
    Otopark verilerini Railway backend'e POST atar.
    Hata olursa sessizce False doner, sistemi durdurmaz.
    """
    total = len(slot_results)
    empty = sum(1 for r in slot_results if r["status"] == "empty")

    payload = {
    "park_id":        park_id,
    "name":           cfg.get("name", park_id),
    "total_slots":    total,
    "empty_slots":    empty,
    "full_slots":     total - empty,
    "occupancy_rate": round((total - empty) / total * 100, 1) if total else 0,
    "slots":          slot_results,
    "lat":            cfg.get("lat", None),
    "lng":            cfg.get("lng", None),
    "info":           cfg.get("info", None),
}

    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            f"{BACKEND_URL}/update",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[Cloud] Gonderim hatasi: {e}")
        return False


def push_async(park_id: str, cfg: dict, slot_results: list):
    """Ana donguyu bloke etmemek icin thread'de gonder."""
    t = threading.Thread(
        target=push_to_cloud,
        args=(park_id, cfg, slot_results),
        daemon=True
    )
    t.start()
