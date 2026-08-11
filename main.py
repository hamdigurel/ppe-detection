"""
Video Tabanlı Personel Takibi ve Koruyucu Ekipman Kontrol Sistemi
-------------------------------------------------------------------
Kullanim:
    python main.py --video test_video.mp4

Kamera kullanilmaz, sistem kayitli video dosyalari uzerinden calisir.
"""

import argparse
import os
import time
import cv2

from detectors.ppe_detector import PPEDetector
from tracking.person_tracker import PersonTracker
from services.video_service import VideoService
from services.violation_service import ViolationService
from utils.geometry import box_center, upper_body_region, point_in_box
from utils.drawing import draw_person_box, draw_info_panel

# ---- SABIT AYARLAR ----
PERSON_MODEL_PATH = "yolov8s.pt"                          # COCO pretrained
PPE_MODEL_PATH = "runs/detect/runs/ppe_model/weights/best.pt"  # kendi fine-tune modelin
PERSON_CONF = 0.35
PPE_CONF = 0.35
VIOLATION_FRAME_THRESHOLD = 15                             # art arda kac kare -> ihlal


def build_person_status(tracked_people, ppe_boxes, ppe_class_names, ppe_confs):
    """
    Takip edilen kisiler ile tespit edilen PPE kutularini esler.
    Esleştirme yontemi: ekipman kutusunun merkez noktasi, kisinin
    ust govde bolgesi (upper_body_region) icinde mi -> nesne merkez noktasi yontemi.
    """
    person_status = []

    for person in tracked_people:
        pbox = person["box"]
        pid = person["id"]
        upper_region = upper_body_region(pbox)

        status = {
            "id": pid,
            "box": pbox,
            "helmet": None,
            "vest": None,
            "helmet_conf": 0.0,
            "vest_conf": 0.0,
        }

        for ebox, cls_name, conf in zip(ppe_boxes, ppe_class_names, ppe_confs):
            center = box_center(ebox)
            if not point_in_box(center, upper_region):
                continue

            name_lower = cls_name.lower()
            if "helmet" in name_lower:
                if "no" in name_lower:
                    status["helmet"] = "Missing"
                    status["helmet_conf"] = float(conf)
                elif status["helmet"] is None:
                    status["helmet"] = "Present"
                    status["helmet_conf"] = float(conf)
            elif "vest" in name_lower:
                if "no" in name_lower:
                    status["vest"] = "Missing"
                    status["vest_conf"] = float(conf)
                elif status["vest"] is None:
                    status["vest"] = "Present"
                    status["vest_conf"] = float(conf)

        if status["helmet"] is None:
            status["helmet"] = "Unknown"
        if status["vest"] is None:
            status["vest"] = "Unknown"

        person_status.append(status)

    return person_status


def main():
    parser = argparse.ArgumentParser(description="PPE Kontrol Sistemi")
    parser.add_argument("--video", type=str, required=True, help="Girdi video dosyasi yolu")
    parser.add_argument("--output", type=str, default=None, help="Cikti video yolu (belirtilmezse otomatik olusturulur)")
    parser.add_argument("--show", action="store_true", help="Isleme sirasinda pencerede goster")
    parser.add_argument("--person-conf", type=float, default=PERSON_CONF, help="Kisi tespiti guven esigi (varsayilan 0.35)")
    parser.add_argument("--ppe-conf", type=float, default=PPE_CONF, help="PPE tespiti guven esigi (varsayilan 0.35)")
    parser.add_argument("--violation-frames", type=int, default=VIOLATION_FRAME_THRESHOLD, help="Ihlal icin gereken art arda kare sayisi (varsayilan 15)")
    parser.add_argument("--debug", action="store_true", help="PPE tespit detaylarini terminalde goster")
    args = parser.parse_args()

    # Her video kendi alt klasorune yazar (video1 -> outputs/video1/, video2 -> outputs/video2/)
    # boylece farkli videolari sirayla islerken onceki sonuclar silinmez/uzerine yazilmaz
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = os.path.join("outputs", video_name)
    os.makedirs(output_dir, exist_ok=True)

    output_video_path = args.output if args.output else os.path.join(output_dir, "result.mp4")

    # ---- SERVISLERI BASLAT ----
    video_service = VideoService(args.video, output_video_path)
    person_tracker = PersonTracker(model_path=PERSON_MODEL_PATH, conf=args.person_conf)
    ppe_detector = PPEDetector(model_path=PPE_MODEL_PATH, conf=args.ppe_conf)
    violation_service = ViolationService(
        fps=video_service.fps,
        output_dir=output_dir,
        violation_frame_threshold=args.violation_frames,
    )

    print(f"Video aciliyor: {args.video} ({video_service.width}x{video_service.height} @ {video_service.fps:.1f} FPS)")
    print(f"Ciktilar kaydedilecek: {output_dir}/")
    print(f"Ayarlar -> person_conf: {args.person_conf} | ppe_conf: {args.ppe_conf} | violation_frames: {args.violation_frames}")
    print("PPE model siniflari:", ppe_detector.names)

    frame_count = 0
    fps_smooth = 0.0

    while True:
        ret, frame = video_service.read()
        if not ret:
            print("Video bitti.")
            break

        frame_count += 1
        t0 = time.time()

        # ---- TESPIT + TAKIP ----
        tracked_people = person_tracker.track(frame)
        ppe_boxes, ppe_class_names, ppe_confs = ppe_detector.detect(frame)

        # DEBUG: PPE modelinin o karede gercekte ne buldugunu gosterir (her 30 karede bir)
        if args.debug and frame_count % 30 == 0:
            print(f"Frame {frame_count}: PPE tespitleri -> {ppe_class_names} | confs: {[round(c, 2) for c in ppe_confs]}")

        # ---- KISI - EKIPMAN ESLESTIRME ----
        person_status = build_person_status(tracked_people, ppe_boxes, ppe_class_names, ppe_confs)

        # ---- IHLAL KONTROLU (15 kare kurali) ----
        violation_service.update(frame, frame_count, person_status)

        # ---- CIZIM ----
        annotated = frame.copy()
        for p in person_status:
            draw_person_box(annotated, p["id"], p["box"], p["helmet"], p["vest"])

        elapsed = time.time() - t0
        instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        fps_smooth = instant_fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * instant_fps)

        helmet_violations, vest_violations = violation_service.counts()
        draw_info_panel(
            annotated,
            frame_count=frame_count,
            total_person=len(person_status),
            helmet_violations=helmet_violations,
            vest_violations=vest_violations,
            fps=fps_smooth,
        )

        # ---- KAYDET ----
        video_service.write(annotated)

        if args.show:
            cv2.imshow("PPE Detection - 'q' ile kapat", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Kullanici tarafindan durduruldu.")
                break

    # ---- TEMIZLIK ----
    video_service.release()
    cv2.destroyAllWindows()
    json_path = violation_service.save_json()

    print(f"\nToplam {frame_count} kare islendi.")
    print(f"Ortalama FPS: {fps_smooth:.1f}")
    print(f"Cikti video: {output_video_path}")
    print(f"Ihlal kayitlari: {json_path}")


if __name__ == "__main__":
    main()
