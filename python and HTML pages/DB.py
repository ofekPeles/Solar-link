import mysql.connector
from mysql.connector import Error
import hashlib
import mimetypes
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import io


def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="your password",
        database="your database"
    )

def convert_image_to_binary(image_obj, filename_hint="image"):
    # image_obj יכול להיות bytes, PIL.Image, או numpy.ndarray
    if isinstance(image_obj, bytes):
        blob = image_obj
        fname = f"{filename_hint}.png"
        mime = "image/png"
    elif isinstance(image_obj, Image.Image):
        image_obj = ImageOps.exif_transpose(image_obj).convert("RGB")
        bio = io.BytesIO()
        image_obj.save(bio, format="PNG")
        blob = bio.getvalue()
        fname = f"{filename_hint}.png"
        mime = "image/png"
    elif isinstance(image_obj, np.ndarray):
        ok, buf = cv2.imencode(".png", image_obj)
        if not ok:
            raise RuntimeError("Failed to encode numpy image to PNG")
        blob = buf.tobytes()
        fname = f"{filename_hint}.png"
        mime = "image/png"
    else:
        raise TypeError("image_obj must be bytes, PIL.Image, or numpy.ndarray")

    sha = hashlib.sha256(blob).hexdigest()
    size = len(blob)
    return (fname, mime, size, sha, blob)

def insert_full_project(
    Adress, PanelsCount, InstallCostNIS, YearlyProfitNIS, PaybackYear,
    lat, lon,
    image_initial, image_mask,
    monthly_income_list,   # list of (month_index:int, profit_nis:float)
    annual_income_list, city     # list of (year_index:int, income_nis:float) – אצלך זו טבלת הצטברות/יתרה; אם זו יתרה, שנה שם העמודה בהתאם
):
    conn = get_connection()
    cur = conn.cursor()
    try:
        # 1) צור פרויקט (בלי ProjectID בעמודות – שדה אוטו אינקרמנט)
        cur.execute("""
            INSERT INTO SolarProject
              (Adress, City, PanelsCount, InstallCostNIS, YearlyProfitNIS, PaybackYears, lat, lon)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (Adress, city, PanelsCount, InstallCostNIS, YearlyProfitNIS, PaybackYear, lat, lon))
        project_id = cur.lastrowid

        # 2) תמונות
        fn, mime, size, sha, blob = convert_image_to_binary(image_initial, "initial")
        cur.execute("""
            INSERT INTO SolarProjectImage_initial
              (ProjectID, FileName, MimeType, FileSizeBytes, Sha256Hex, ImageData)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (project_id, fn, mime, size, sha, blob))

        fn, mime, size, sha, blob = convert_image_to_binary(image_mask, "mask")
        cur.execute("""
            INSERT INTO SolarProjectImage_mask
              (ProjectID, FileName, MimeType, FileSizeBytes, Sha256Hex, ImageData)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (project_id, fn, mime, size, sha, blob))

        # 3) חודשי (bulk)
        if monthly_income_list:
            cur.executemany("""
                INSERT INTO SolarMonthlyProfit (ProjectID, MonthIndex, ProfitNIS)
                VALUES (%s, %s, %s)
            """, [(project_id, m, p) for (m, p) in monthly_income_list])

        # 4) שנתי (bulk)
        if annual_income_list:
            cur.executemany("""
                INSERT INTO SolarAnnualIncome (ProjectID, YearIndex, IncomeNIS)
                VALUES (%s, %s, %s)
            """, [(project_id, y, inc) for (y, inc) in annual_income_list])

        conn.commit()
        return project_id
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
        
