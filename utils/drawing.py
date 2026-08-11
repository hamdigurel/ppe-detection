"""
Görüntü üzerine çizim yapan yardımcı fonksiyonlar.
Kutu, etiket ve genel bilgi paneli bu modülde üretilir.
"""

import cv2


def status_color(helmet_status, vest_status):
    """Kişinin ekipman durumuna göre kutu rengini belirler (BGR formatında)."""
    if helmet_status == "Missing" or vest_status == "Missing":
        return (0, 0, 255)      # kırmızı - ihlal
    if helmet_status == "Unknown" or vest_status == "Unknown":
        return (0, 255, 255)    # sarı - belirsiz
    return (0, 255, 0)          # yeşil - uygun


def draw_person_box(frame, person_id, box, helmet_status, vest_status):
    """Bir kişi için kutu ve durum etiketini çizer."""
    x1, y1, x2, y2 = [int(v) for v in box]
    color = status_color(helmet_status, vest_status)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID:{person_id} Helmet:{helmet_status} Vest:{vest_status}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    # etiket okunabilirliği için arka plan kutusu
    cv2.rectangle(frame, (x1, max(y1 - text_h - 12, 0)), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


def draw_info_panel(frame, frame_count, total_person, helmet_violations, vest_violations, fps):
    """Ekranın sağ üst köşesine genel bilgi panelini çizer."""
    height, width = frame.shape[:2]
    lines = [
        f"Frame: {frame_count}",
        f"Detected Persons: {total_person}",
        f"Helmet Violations: {helmet_violations}",
        f"Vest Violations: {vest_violations}",
        f"FPS: {fps:.1f}",
    ]

    panel_width = 260
    panel_height = 20 * len(lines) + 20
    x0 = width - panel_width - 10
    y0 = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_width, y0 + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        y = y0 + 25 + i * 20
        cv2.putText(frame, line, (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
