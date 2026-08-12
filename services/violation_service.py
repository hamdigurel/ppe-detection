"""
İhlal oluşturma ve kayıt servisi.

Kural: Bir kişi art arda en az VIOLATION_FRAME_THRESHOLD (varsayılan 15) kare
boyunca ekipmansız (Missing) görünürse, bu bir ihlal olarak kaydedilir.
Tek bir karede eksik görünmek doğrudan ihlal sayılmaz -> yanlış pozitifleri azaltır.

Her ihlal için:
- outputs/violations.json dosyasına kayıt eklenir
- outputs/violations/ klasörüne ilgili kişinin ekran görüntüsü kaydedilir
"""

import os
import json
import cv2


class ViolationService:
    def __init__(self, fps, output_dir="outputs", violation_frame_threshold=15):
        self.fps = fps if fps > 0 else 25.0
        self.threshold = violation_frame_threshold

        self.output_dir = output_dir
        self.violations_dir = os.path.join(output_dir, "violations")
        os.makedirs(self.violations_dir, exist_ok=True)

        # her kişi için art arda eksik görülen kare sayacı
        self._missing_streak = {"helmet": {}, "vest": {}}
        # aynı ihlal döneminde tekrar tekrar kayıt açmamak için
        self._already_logged = {"helmet": set(), "vest": set()}

        self.records = []

    def _frame_to_time_str(self, frame_count):
        total_seconds = frame_count / self.fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        hours = int(minutes // 60)
        minutes = minutes % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _register_status(self, kind, person_id, is_missing):
        """kind: 'helmet' ya da 'vest'"""
        streak = self._missing_streak[kind]
        logged = self._already_logged[kind]

        if is_missing:
            streak[person_id] = streak.get(person_id, 0) + 1
        else:
            streak[person_id] = 0
            logged.discard(person_id)

        # eşiği tam o karede geçtiyse ve bu ihlal dönemi için daha önce kayıt açılmadıysa
        just_crossed_threshold = streak.get(person_id, 0) == self.threshold
        not_logged_yet = person_id not in logged
        return just_crossed_threshold and not_logged_yet

    def update(self, frame, frame_count, person_status_list):
        """
        Her kare için çağrılır. person_status_list elemanları:
        {"id": int, "box": (x1,y1,x2,y2), "helmet": "Present"/"Missing"/"Unknown",
         "vest": "Present"/"Missing"/"Unknown",
         "helmet_conf": float, "vest_conf": float}
        """
        for p in person_status_list:
            pid = p["id"]

            helmet_missing = p["helmet"] == "Missing"
            vest_missing = p["vest"] == "Missing"

            if self._register_status("helmet", pid, helmet_missing):
                self._log_violation(frame, frame_count, pid, p["box"], "helmet_missing", p.get("helmet_conf", 0.0))
                self._already_logged["helmet"].add(pid)

            if self._register_status("vest", pid, vest_missing):
                self._log_violation(frame, frame_count, pid, p["box"], "vest_missing", p.get("vest_conf", 0.0))
                self._already_logged["vest"].add(pid)

    def _log_violation(self, frame, frame_count, person_id, box, violation_type, confidence):
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        crop = frame[y1:y2, x1:x2]
        image_filename = f"id_{person_id}_{violation_type}_{frame_count}.jpg"
        image_path = os.path.join(self.violations_dir, image_filename)

        if crop.size > 0:
            cv2.imwrite(image_path, crop)

        record = {
            "person_id": person_id,
            "violation": violation_type,
            "video_time": self._frame_to_time_str(frame_count),
            "confidence": round(float(confidence), 2),
            "image_path": os.path.join("violations", image_filename).replace("\\", "/"),
        }
        self.records.append(record)
        print(f"[IHLAL] ID:{person_id} {violation_type} @ {record['video_time']}")

    def counts(self):
        """Şu ana kadar tespit edilen toplam helmet/vest ihlal sayısını döndürür."""
        helmet_count = sum(1 for r in self.records if r["violation"] == "helmet_missing")
        vest_count = sum(1 for r in self.records if r["violation"] == "vest_missing")
        return helmet_count, vest_count

    def save_json(self):
        json_path = os.path.join(self.output_dir, "violations.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=4, ensure_ascii=False)
        print(f"İhlal kayıtları kaydedildi: {json_path}")
        return json_path
