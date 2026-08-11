"""
Geometri yardımcı fonksiyonları.
Kişi kutuları ile ekipman (helmet/vest) kutularını eşleştirmek için kullanılır.
"""

import numpy as np


def box_center(box):
    """Bir kutunun (x1, y1, x2, y2) merkez noktasını döndürür."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def upper_body_region(person_box, ratio=0.9):
    """
    Kişinin üst gövde/kafa bölgesini döndürür.
    Helmet ve vest kontrolü bu bölgeye göre yapılır.

    ratio: kutunun üstten ne kadarının 'üst gövde' sayılacağı (0.6 = kutunun ilk %60'ı)
    """
    x1, y1, x2, y2 = person_box
    upper_y2 = y1 + (y2 - y1) * ratio
    return (x1, y1, x2, upper_y2)


def point_in_box(point, box):
    """Bir noktanın verilen kutunun içinde olup olmadığını kontrol eder."""
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def iou(box_a, box_b):
    """
    İki kutu arasındaki IoU (Intersection over Union) değerini hesaplar.
    Alternatif bir eşleştirme yöntemi olarak kullanılabilir.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area
