# Akıllı Otopark Yönlendirme Sistemi

Mevcut güvenlik kameraları ve YOLOv8x derin öğrenme modeli kullanılarak geliştirilen gerçek zamanlı otopark doluluk tespit ve yönlendirme sistemi.

## Özellikler

- YOLOv8x ile gerçek zamanlı araç tespiti
- Polygon tabanlı park alanı işaretleme
- Lastik temas noktası algoritması ile hassas doluluk tespiti
- IP kamera ve video dosyası desteği
- Çoklu otopark profil yönetimi
- FastAPI backend ile bulut entegrasyonu
- React web uygulaması ile gerçek zamanlı yönlendirme

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
# Otoparklar -config dosyalarını düzenle
# Görüntü çıkar
python frame_kaydet.py --park otopark1 
# Park yerlerini işaretle
python parking_space_picker.py --park otopark1
# Detector çalıştır
python detector.py --park otopark1
```

## Teknolojiler

- Python 3.12
- YOLOv8x (Ultralytics)
- OpenCV
- FastAPI
- React.js
- Leaflet.js
- Railway (Backend)
- Vercel (Frontend)

## Proje Yapısı

canlı-2/
├── detector.py
├── parking_space_picker.py
├── frame_kaydet.py
├── cloud_uploader.py
├── requirements.txt
└── parklar/
 ├── otopark1/
 │   └── config.json
 └── otopark2/
 └── config.json
