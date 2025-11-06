import pandas as pd
import numpy as np
import collections
import ultralytics
from ultralytics.utils import SETTINGS
from ultralytics import YOLO
import cv2
import base64
from io import BytesIO
import requests
import geopy
from geopy.geocoders import Nominatim
import json
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
from Roof_processing import process_image
from DP import insert_full_project
import hashlib
from PIL import Image as PilImage, ImageOps
from Get_Data_From_DB import output_dataset_as_arrays

solar_panels_data = pd.read_csv(r"C:\Users\ofekp\AI lerning\solar_panels_project\Data_solar_panels.csv")

def Model_predict(path):
    model = YOLO(r"C:\Users\ofekp\AI lerning\solar_panels_project\best_YOLO_MODEL.pt")
    return model.predict(source = path)

def pixels_surface_size(results):
    mask = results[0].masks.data[0].cpu().numpy() 
    unique, counts = np.unique(mask, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    print("מספר פיקסלים לכל מחלקה:")
    for class_id, count in pixel_counts.items():
        print(f"מחלקה {class_id}: {count} פיקסלים")
        
    class_1, class_2 = pixel_counts.items()
    surface = class_1[1] - class_2[1]
    print(surface)
    return(surface)

def location(city,country):
     geolocator = Nominatim(user_agent="geoapi")
     location = geolocator.geocode(f"{city}, {country}")
     return((location.latitude, location.longitude))


def get_monthly_income(electricity_price,panels,lat,lon,price_per_month):
    peak_kw = panels * 0.4
    price_per_kwh = electricity_price
    url = f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?lat={lat}&lon={lon}&peakpower={peak_kw}&loss=14&outputformat=json"
    data = requests.get(url).json()

    monthly = data["outputs"]["monthly"]["fixed"]
    monthly_income = []
    sum=0

    for month in monthly:
        energy_kwh = (month["E_m"])
        income = energy_kwh * price_per_kwh
        sum+=income
        monthly_income.append((month["month"], round(income, 2)))
    income_per_year = round(sum,2)
    return (monthly_income,income_per_year)

def GSD(h, Sw, F, ImageW, ImageH):
    print(f"{ImageW} ImageW")
    # המרות ליחידות אחידות
    Sw_m = Sw / 10000  # חיישן למטר
    F_m = F / 10000    # אורך מוקד למטר
    # חישוב GSD
    GSD = (h * Sw_m) / (F_m * ImageW)
    return(GSD)

class solar_panels:
    def __init__(self, country_location: str, units: int, income_per_year: float):
        self.country = country_location
        self.units = units
        self.income_per_year = income_per_year  # ← תיקון השם

    def cost(self) -> int:
        # הערכה גסה: 0.4kW לפאנל * 4500 ₪ לקילוואט מותקן
        return int(4500 * round(self.units * 0.4))

    def electricity_cost(self) -> int:
        return int(self.income_per_year)

    def payback_period(self) -> float:
        # שנים → החזר שנתי (שמור על 2 ספרות)
        if self.income_per_year <= 0:
            return 3
        return round(self.cost() / self.income_per_year, 2)

import pdf2image
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import re

from re import match
import re
from google import genai
from google.genai import types
import pathlib
import os
from dotenv import load_dotenv
import json

def create_data_time_period(cost,time):
    if time == 0 or time == "None":
        return [(0, -cost), (1, -cost)]
    m = cost/time
    pay_back_time_array = []
    for i in range(15):
        pay_back_time_array.append((i+1, round(m * (i+1)-cost, 2)))
    return pay_back_time_array

# #BERT Model

with open(r"C:\Users\ofekp\AI lerning\solar_panels_project\BERT Model\intent_response_map (1).json", "r", encoding="utf-8") as f:
    intent_map = json.load(f)
with open(r"C:\Users\ofekp\AI lerning\solar_panels_project\BERT Model\solar_intents_expanded_final_30.json", "r", encoding="utf-8") as f:
    data = json.load(f)

class IntentClassifier:
    def __init__(self, model_path, tokenizer_path):
        # Use AutoTokenizer and AutoModel if you get version errors
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = TFBertForSequenceClassification.from_pretrained(model_path, from_pt=False)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding=True)
        outputs = self.model(**inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
        predicted_label_id = np.argmax(probs)
        confidence = probs[predicted_label_id]
        return predicted_label_id, confidence
    
response = IntentClassifier(
        r"C:\Users\ofekp\AI lerning\solar_panels_project\BERT Model\Model",
        r"C:\Users\ofekp\AI lerning\solar_panels_project\BERT Model\Tokenizer"
    )

def get_intent(text):
    label_id, confidence = response.predict(text)
    return label_id, confidence

def get_bot_response(text):
    label_id, confidence = get_intent(text)
    intent_entry = intent_map[label_id]
    responses = intent_entry.get("responses", [])
    if not responses:
        return "אין תגובה זמינה."
    return np.random.choice(responses)

import io, base64
from PIL import Image as PilImage

def image_to_base64(pil_img: PilImage) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def base64_to_image(b64: str) -> PilImage:
    return PilImage.open(io.BytesIO(base64.b64decode(b64)))
#Server
import flask
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import os
import os

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index_5.html")

@app.route("/predict", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image_file = request.files["image"]
    city = request.form.get("city")
    local_monthly_electricity_price = request.form.get("electricity_price", type=float)
    initial_image = Image.open(image_file.stream)
    results = Model_predict(initial_image)
    number_panels, mask_image = process_image(results, initial_image, pixels_per_meter=20)
    lat,lon = location(city,"Israel")
    per_monthly_income, incom_per_year = get_monthly_income(0.25, number_panels, lat, lon, local_monthly_electricity_price)
    roof = solar_panels("Israel", number_panels, incom_per_year)
    pay_back_time_array = create_data_time_period(roof.cost(), roof.payback_period())
    cost = roof.cost()
    payback = roof.payback_period()
    array_monthly = per_monthly_income
    array_annual = pay_back_time_array
    mask_pil = Image.fromarray(mask_image) if isinstance(mask_image, np.ndarray) else mask_image

    return jsonify({"number_panels": number_panels, "monthly_income": array_monthly, "annual_income": array_annual, "cost": cost, "payback": payback, "income_per_year": incom_per_year, "initial_image": image_to_base64(initial_image), "mask_image": image_to_base64(mask_pil), "lat": lat, "lon": lon})


@app.route("/save_in_database", methods=["POST"])
def save_in_database():
    data = request.get_json()

    Adress         = data.get("Adress", "Solar Roof")
    city           = data.get("city", "Unknown")
    number_panels  = int(data.get("number_panels"))
    income_per_year= float(data.get("income_per_year"))
    array_monthly  = data.get("monthly_income", [])   # list of [month, value]
    array_annual   = data.get("annual_income", [])    # list of [year, value] (כיום זו טבלת החזר מצטברת)
    cost           = float(data.get("cost"))
    payback        = float(data.get("payback"))
    lat            = float(data.get("lat"))
    lon            = float(data.get("lon"))
    # קבל תמונות מהקליינט (מומלץ) – או השתמש בגלובלים אם אתה חייב
    initial_b64 = data.get("initial_image_base64")
    mask_b64    = data.get("mask_image_base64")

    if initial_b64 and mask_b64:
         img_initial = base64_to_image(initial_b64)
         img_mask    = base64_to_image(mask_b64)

     # המרות לרשימות טאפלים
    monthly_list = [(int(m), float(v)) for (m, v) in array_monthly]
    annual_list  = [(int(y), float(v)) for (y, v) in array_annual]

    project_id = insert_full_project(
         Adress=Adress,
         PanelsCount=number_panels,
         InstallCostNIS=int(cost),
         YearlyProfitNIS=int(income_per_year),
         PaybackYear=payback,
         lat=lat, lon=lon,
         image_initial=img_initial,
         image_mask=img_mask,
         monthly_income_list=monthly_list,
         annual_income_list=annual_list,
         city=city
     )

    return jsonify({"status": "success", "project_id": project_id})

@app.route("/search")
def search():
    return render_template("search.html")

@app.route("/get_data_by_address", methods=["POST"])
def get_data_by_address():
    data = request.get_json()
    street = data.get("street")
    city = data.get("city")
    # Fetch data from the database based on the address
    project_data = output_dataset_as_arrays(street, city)
    array_annual = project_data["array_annual"]
    array_monthly = project_data["array_monthly"]
    Cost = project_data["Cost"]
    PaybackYears = project_data["PaybackYears"]
    YearIncome = project_data["YearIncome"]
    PanelsCount = project_data["PanelsCount"]
    src_initial = project_data["initial_image"]["src"] if project_data["initial_image"] else None
    src_mask = project_data["mask_image"]["src"] if project_data["mask_image"] else None

    return jsonify({"status": "success", "array_annual": array_annual, "array_monthly": array_monthly, "Cost": Cost, "PaybackYears": PaybackYears, "YearIncome": YearIncome, "PanelsCount": PanelsCount, "initial_image": src_initial, "mask_image": src_mask, "street": street})

if __name__ == "__main__":
    app.run(debug=True)
    