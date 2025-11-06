from tkinter import Image
import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL import Image as PilImage

# ---------- 2) חישוב זווית יישור ----------
def deskew_by_hough(mask_bin, hough_thresh=120):
    edges = cv2.Canny(mask_bin, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=hough_thresh)
    if lines is None:
        return None
    angles = []
    for l in lines:
        rho, theta = l[0]
        angle_deg = (theta * 180.0 / np.pi) - 90.0  # מיישר לאופקי
        angles.append(angle_deg)
    return float(np.median(angles))

def fallback_angle_min_area_rect(mask_bin):
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), angle = cv2.minAreaRect(c)
    if w < h:
        angle += 90.0
    return float(angle)

def rotate_image_and_mask(image_bgr, mask_bin, angle_deg):
    H, W = mask_bin.shape[:2]
    M = cv2.getRotationMatrix2D((W/2, H/2), angle_deg, 1.0)
    rot_mask = cv2.warpAffine(
        mask_bin, M, (W, H),
        flags=cv2.INTER_NEAREST,                # לשמור בינאריות 0/255
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    rot_img  = cv2.warpAffine(
        image_bgr, M, (W, H),
        flags=cv2.INTER_LINEAR,                 # איכות טובה לתמונה צבעונית
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
    )
    return rot_img, rot_mask, M

# ---------- 3) יצירת מסכה מ-YOLO ----------
def get_seg_mask_from_results(results, target_names=("flat","slope")):
    r = results[0]
    if r.masks is None:
        raise ValueError("Model output has no masks. Use a YOLO *-seg model.")
    img_h, img_w = r.orig_img.shape[:2]
    id2name = r.names

    target_ids = {cid for cid, cname in id2name.items() if cname in set(target_names)}
    combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # מאחדים רק את המסכות של המחלקות הרצויות
    for i, cls_id in enumerate(r.boxes.cls.int().cpu().tolist()):
        if cls_id in target_ids:
            m = r.masks.data[i].cpu().numpy()       # [Hm, Wm], float 0..1
            m = (m > 0.5).astype(np.uint8) * 255    # בינארי
            m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, m)

    return combined_mask  # לא שומר לקובץ כאן—הפונקציה מחזירה את המסכה

# ---------- 4) פריסת פאנלים על מסכה (מיושרת) ----------
def layout_panels_on_mask(mask_bin, pixels_per_meter=20, panel_w_m=1.0, panel_h_m=1.7,
                          fill_ratio=0.95, out_path=None):
    _, mask = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)

    panel_w = int(panel_w_m * pixels_per_meter)
    panel_h = int(panel_h_m * pixels_per_meter)

    output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panel_count = 0

    for y in range(0, mask.shape[0] - panel_h, panel_h):
        for x in range(0, mask.shape[1] - panel_w, panel_w):
            roi = mask[y:y+panel_h, x:x+panel_w]
            if np.sum(roi == 255) > fill_ratio * panel_w * panel_h:
                cv2.rectangle(output, (x, y), (x+panel_w, y+panel_h), (0, 255, 0), 2)
                panel_count += 1

    # שמירה רק אם נתיב סופק
    if out_path:
        cv2.imwrite(out_path, output)

    return panel_count, output

# ---------- 5) צינור מלא ----------

def process_image(results, image, pixels_per_meter=20):
    # --- נרמול קלט: PIL.Image -> NumPy BGR ---
    if isinstance(image, PilImage.Image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        img_bgr = np.array(image)[:, :, ::-1]  # RGB -> BGR
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image
    else:
        raise TypeError("image must be PIL.Image or numpy.ndarray")

    # --- ב) הפקת מסכה מתוצאות YOLO ---
    mask = get_seg_mask_from_results(results, target_names=("flat", "slope"))
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_bin = mask_bin.astype(np.uint8)

    # --- ג) חישוב זווית יישור ---
    angle = deskew_by_hough(mask_bin)
    if angle is None:
        angle = fallback_angle_min_area_rect(mask_bin)

    # --- ד) סיבוב תמונה ומסכה ---
    rot_img, rot_mask, M = rotate_image_and_mask(img_bgr, mask_bin, angle)

    # --- ה) פריסת פאנלים על המסכה המיושרת ---
    panel_count, image_mask = layout_panels_on_mask(
        rot_mask,
        pixels_per_meter=pixels_per_meter,
        panel_w_m=1.0,
        panel_h_m=1.7,
        fill_ratio=0.95,
        out_path=None  # לא שומרים כלום
    )

    # מחזירים רק את התמונה (image_mask)
    return int(panel_count), image_mask
# # ---- הדגמה ----
# if __name__ == "__main__":
#     image_path = r"C:\Users\ofekp\AI lerning\solar_panels_project\Image 006.png"
#     info = process_image(image_path, pixels_per_meter=20)
#     print(info)
