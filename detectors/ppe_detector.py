"""
PPE (Personal Protective Equipment) tespiti modülü.
Proje kapsamında fine-tune edilmiş model ile helmet, no_helmet, vest, no_vest
sınıfları tespit edilir.
"""

from ultralytics import YOLO
import numpy as np


class PPEDetector:
    def __init__(self, model_path, conf=0.35):
        self.model = YOLO(model_path)
        self.conf = conf
        self.names = self.model.names

    def detect(self, frame):
        """
        Verilen karede helmet/vest ilişkili nesneleri tespit eder.
        Dönüş: (boxes, class_names, confidences)
        """
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)
        boxes_obj = results[0].boxes

        if len(boxes_obj) == 0:
            return np.array([]), [], np.array([])

        boxes = boxes_obj.xyxy.cpu().numpy()
        classes = boxes_obj.cls.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        class_names = [self.names[int(c)] for c in classes]

        return boxes, class_names, confs
