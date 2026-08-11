"""
Video okuma/yazma servisi.
Şartnamenin 3.1 maddesindeki gereksinimi karşılar: video okunamazsa
anlaşılır bir hata mesajı verir.
"""

import cv2
import os


class VideoService:
    def __init__(self, video_path, output_path="outputs/result.mp4"):
        self.video_path = video_path
        self.output_path = output_path

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(
                f"Video acilamadi: '{video_path}'. "
                f"Dosya yolunu ve formatini (MP4/AVI/MOV) kontrol edin."
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

    def read(self):
        return self.cap.read()

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.cap.release()
        self.writer.release()
