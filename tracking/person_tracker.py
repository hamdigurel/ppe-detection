"""
Kişi takibi modülü.

Ultralytics'in gömülü ByteTrack algoritması kullanılır (model.track ile).
Her kişiye benzersiz bir takip numarası (track ID) atanır ve bu numara,
kişi birkaç kare boyunca başka bir nesnenin arkasında kalsa bile korunur.

Neden ByteTrack:
- Ultralytics'e doğrudan entegre, ekstra kurulum gerektirmiyor
- Düşük güven skorlu tespitleri de ikinci aşamada değerlendirdiği için
  kısa süreli kayıplarda (oklüzyon) ID sürekliliğini daha iyi koruyor
"""

from ultralytics import YOLO
import numpy as np

COCO_PERSON_CLASS_ID = 0


class PersonTracker:
    def __init__(self, model_path="yolov8s.pt", conf=0.35, tracker_cfg="tracking/bytetrack_custom.yaml"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.tracker_cfg = tracker_cfg

    def track(self, frame):
        """
        Verilen karede kişileri tespit edip takip eder.
        Dönüş: liste of dict -> {"id": int, "box": (x1,y1,x2,y2)}
        """
        results = self.model.track(
            source=frame,
            conf=self.conf,
            classes=[COCO_PERSON_CLASS_ID],
            persist=True,
            tracker=self.tracker_cfg,
            verbose=False,
        )

        tracked_people = []
        boxes_obj = results[0].boxes

        if boxes_obj is None or boxes_obj.id is None:
            # henüz ID atanmamış tespitler (ilk karelerde olabilir) yok sayılır
            return tracked_people

        boxes = boxes_obj.xyxy.cpu().numpy()
        ids = boxes_obj.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            tracked_people.append({"id": int(track_id), "box": box})

        return tracked_people
