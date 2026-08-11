"""
Kişi tespiti modülü.
COCO pretrained YOLO modeli kullanılarak sadece 'person' sınıfı tespit edilir.
Bu model, geniş ve çeşitli COCO veri setiyle eğitildiği için kişi tespitinde
kendi fine-tune ettiğimiz PPE modelinden daha güvenilirdir.
"""

from ultralytics import YOLO
import numpy as np

COCO_PERSON_CLASS_ID = 0  # COCO veri setinde 'person' sınıfının index'i her zaman 0'dır


class PersonDetector:
    def __init__(self, model_path="yolov8s.pt", conf=0.35):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Verilen karede kişileri tespit eder.
        Dönüş: Nx4 boyutunda numpy array (x1, y1, x2, y2)
        """
        results = self.model.predict(
            source=frame, conf=self.conf, classes=[COCO_PERSON_CLASS_ID], verbose=False
        )
        if len(results[0].boxes) == 0:
            return np.array([])
        return results[0].boxes.xyxy.cpu().numpy()
