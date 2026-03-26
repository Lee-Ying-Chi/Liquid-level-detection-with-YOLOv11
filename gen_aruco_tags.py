import cv2
import numpy as np
import io

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# ========= Settings =========
DICT = cv2.aruco.DICT_4X4_50
TAG_IDS = [0, 1, 2, 3, 4]                 # 你要印的ID
TAG_SIZES_MM = [20, 30]     # 要測的尺寸
MARKER_BORDER_BITS = 1              # 外白框厚度（建議 1 或 2）
PDF_NAME = "aruco_tags_A4.pdf"
# ============================

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)

def make_marker_png_bytes(tag_id: int, marker_px: int, border_px: int):
    marker = cv2.aruco.generateImageMarker(aruco_dict, tag_id, marker_px)

    if border_px > 0:
        marker2 = cv2.copyMakeBorder(
            marker, border_px, border_px, border_px, border_px,
            cv2.BORDER_CONSTANT, value=255
        )
    else:
        marker2 = marker

    ok, buf = cv2.imencode(".png", marker2)
    if not ok:
        raise RuntimeError("PNG encode failed")

    total_px = marker_px + 2 * border_px
    return buf.tobytes(), marker_px, total_px


def main():
    c = canvas.Canvas(PDF_NAME, pagesize=A4)
    page_w, page_h = A4

    title_y = page_h - 15 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(15 * mm, title_y, "ArUco Tags (Print at 100% / No scaling)")

    c.setFont("Helvetica", 10)
    c.drawString(15 * mm, title_y - 6 * mm, f"Dict: DICT_4X4_50 | Border: {MARKER_BORDER_BITS} | IDs: {TAG_IDS}")

    # Layout parameters
    start_x = 15 * mm
    start_y = page_h - 30 * mm
    row_gap = 45 * mm
    col_gap = 45 * mm

    # Header row for sizes
    c.setFont("Helvetica-Bold", 10)
    c.drawString(start_x, start_y + 10 * mm, "ID \\ Size(mm)")
    for j, sz in enumerate(TAG_SIZES_MM):
        c.drawString(start_x + (j + 1) * col_gap, start_y + 10 * mm, f"{sz}mm")

    # Draw tags
    for i, tag_id in enumerate(TAG_IDS):
        y = start_y - i * row_gap

        c.setFont("Helvetica-Bold", 12)
        c.drawString(start_x, y, f"ID {tag_id}")

        for j, sz_mm in enumerate(TAG_SIZES_MM):
            # Make a marker at sufficient resolution (for print quality),
            # but embed into PDF with exact physical size sz_mm.
            px = 800  # high-res source; physical size is controlled by PDF placement
            png_bytes, marker_px, total_px = make_marker_png_bytes(tag_id, marker_px=800, border_px=80)
            img = ImageReader(io.BytesIO(png_bytes))

            draw_mm = sz_mm * (total_px / marker_px)
            
            x = start_x + (j + 1) * col_gap
            # Place image at exact mm size
            c.drawImage(
                img,
                x,
                y - draw_mm * mm * 0.5,
                width=draw_mm * mm,
                height=draw_mm * mm,
                mask='auto'
            )

            # draw a thin measurement frame (exact size)
            c.setLineWidth(0.3)
            c.rect(x, y - sz_mm * mm * 0.5, sz_mm * mm, sz_mm * mm)

    # Print reminder
    c.setFont("Helvetica-Bold", 12)
    c.drawString(15 * mm, 15 * mm, "IMPORTANT: Print at 100% (disable Fit to page / Scale to paper).")

    c.showPage()
    c.save()
    print("Saved:", PDF_NAME)

if __name__ == "__main__":
    main()

