from flask import Flask, request, jsonify, send_from_directory
import uuid
import os
from PIL import Image
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from flask_cors import CORS
from urllib.parse import quote_plus
import cv2
import numpy as np
import threading
from functools import wraps
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from werkzeug.security import check_password_hash, generate_password_hash
from promptpay import qrcode
import base64
from io import BytesIO
import hashlib
import json
import requests
import pytesseract
import re
from flask import Flask, redirect, url_for
from flask_dance.contrib.google import make_google_blueprint, google
from dotenv import load_dotenv
from flask import render_template
import smtplib
import secrets
from email.message import EmailMessage
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_mail import Mail, Message
import random
from pymongo.server_api import ServerApi
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from zoneinfo import ZoneInfo
import shutil
import jwt
from dateutil.relativedelta import relativedelta
import torch
import concurrent.futures
from ocr_slip import AdvancedSlipOCR


# การตั้งค่า Flask
app = Flask(__name__)
CORS(app)

load_dotenv()

# key พิเศษ
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")

app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv("EMAIL_USER"),
    MAIL_PASSWORD=os.getenv("EMAIL_PASS"),
    MAIL_DEFAULT_SENDER="Phurinsukman3@gmail.com",  # ตั้งค่าอีเมลผู้ส่ง
)

mail = Mail(app)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# key พิเศษ
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")

# เชื่อมต่อ MongoDB
MONGO_URI = "mongodb://localhost:27017"  # เปลี่ยนตามการตั้งค่าของคุณ
client = MongoClient(MONGO_URI)
db = client["api_database"]
users_collection = db["users"]
api_keys_collection = db["api_keys"]
orders_collection = db["orders"]
otp_collection = db["otp_reset"]
uploaded_files_collection = db["uploaded_files"]
uploaded_files_collection.create_index([("created_at", 1)], expireAfterSeconds=3600)


# หน้าแรก
@app.route("/")
def home():
    # ใช้ relative path ไปยังโฟลเดอร์ 'home page'
    return send_from_directory(os.path.join(os.getcwd(), "homepage"), "index.html")


# เพิ่ม route สำหรับไฟล์อื่นๆ ที่อยู่ในโปรเจกต์
@app.route("/<path:filename>")
def serve_other_files(filename):
    # ให้ Flask สามารถเข้าถึงไฟล์จากทุกโฟลเดอร์ในโปรเจกต์
    return send_from_directory(os.getcwd(), filename)


# เพิ่ม route สำหรับไฟล์ CSS, JS ที่อยู่ในโฟลเดอร์ 'home page'
@app.route("/homepage/<path:filename>")
def serve_home_page_files(filename):
    # ให้ Flask สามารถเข้าถึงไฟล์ CSS และ JS ในโฟลเดอร์ 'home page'
    return send_from_directory(os.path.join(os.getcwd(), "homepage"), filename)


# สร้าง jwt
def generate_token(email):
    payload = {"email": email, "exp": datetime.utcnow() + timedelta(hours=1)}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    # ✅ พิมพ์ token ที่สร้างออกมา
    print(f"🔐 Generated token for {email}: {token}")
    return token


# ฟังก์ชันสำหรับตรวจสอบ JWT
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if "Authorization" in request.headers:
            bearer = request.headers.get("Authorization")  # Bearer <token>
            parts = bearer.split()
            if len(parts) == 2 and parts[0] == "Bearer":
                token = parts[1]

        if not token:
            return jsonify({"error": "Token is missing"}), 401

        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = users_collection.find_one({"email": data["email"]})
            if not current_user:
                return jsonify({"error": "User not found"}), 404
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(current_user, *args, **kwargs)

    return decorated


# ฟังก์ชันสำหรับสมัครสมาชิก
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json  # รับข้อมูล JSON
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")

    # ตรวจสอบว่าอีเมล, ชื่อผู้ใช้, และรหัสผ่านไม่ว่าง
    if not email or not username or not password:
        return jsonify({"message": "All fields are required"}), 400

    # ตรวจสอบว่าอีเมลนี้เคยลงทะเบียนแล้วหรือไม่
    if users_collection.find_one({"email": email}):
        return jsonify({"message": "Email already exists"}), 400

    # แฮชรหัสผ่าน
    hashed_password = generate_password_hash(
        password, method="pbkdf2:sha256", salt_length=8
    )

    # เพิ่มข้อมูลผู้ใช้ใหม่
    users_collection.insert_one(
        {"email": email, "username": username, "password": hashed_password}
    )

    return jsonify({"message": "Signup successful"}), 201


# ฟังก์ชันสำหรับล็อกอิน
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    if user.get("password") is None:
        # บัญชีนี้ล็อกอินด้วย Google เท่านั้น
        return (
            jsonify(
                {
                    "error": "This account uses Google login only. Please login with Google."
                }
            ),
            400,
        )

    if not check_password_hash(user["password"], password):
        return jsonify({"error": "Incorrect password"}), 400

    token = generate_token(email)
    return jsonify({"message": "Login successful", "token": token}), 200


# โฟลเดอร์สำหรับอัปโหลด
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ฟังก์ชันตรวจสอบประเภทไฟล์ (รองรับทุกประเภท)
def allowed_file(filename):
    return "." in filename  # ตรวจสอบว่ามี "." ในชื่อไฟล์


# ฟังก์ชันตรวจสอบว่าเป็นไฟล์ภาพจริง
def is_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False


# ฟังก์ชันแปลง .jfif เป็น .jpg
def convert_jfif_to_jpg(input_path):
    output_path = input_path.rsplit(".", 1)[0] + ".jpg"
    with Image.open(input_path) as img:
        img.convert("RGB").save(output_path, "JPEG")
    os.remove(input_path)  # ลบไฟล์เดิม
    return output_path


# ฟังก์ชันสำหรับลบไฟล์
def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Error deleting file: {e}")


# ฟังก์ชันสำหรับลบไฟล์ทุกไฟลืใน folder upload
def delete_all_files_in_upload_folder():
    folder = app.config["UPLOAD_FOLDER"]
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")


# ✅ โหลดโมเดลจาก local
def load_model(model_name):
    return YOLO(os.path.join("models", model_name))  # # แก้ไข


models = {
    "porn": load_model("best-porn.pt"),
    "weapon": load_model("best-weapon.pt"),
    "cigarette": load_model("best-cigarette.pt"),
    "violence": load_model("best-violence.pt"),
}


# ✅ วิเคราะห์ภาพโดยใช้ YOLO จากเครื่อง
def analyze_model_np(image_np, model, threshold):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # # เพิ่มตรวจสอบ CUDA
    results = model.predict(
        source=image_np,
        imgsz=640,
        device=device,
        conf=threshold,
        verbose=False,
        save=False,
        stream=False,
    )
    detections = []
    for result in results:
        if not hasattr(result, "boxes") or result.boxes is None:
            continue
        for box in result.boxes:
            confidence = float(box.conf)
            if confidence >= threshold:
                label_name = model.names[int(box.cls)].lower()
                x1, y1, x2, y2 = box.xyxy[0]
                bbox = [round(float(coord), 2) for coord in [x1, y1, x2, y2]]
                detections.append(
                    {
                        "label": label_name,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                    }
                )
    return detections


# ✅ วาดกรอบ bounding box
def draw_bounding_boxes_from_array(image_np, detections):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        label = detection["label"]
        confidence = detection["confidence"]

        image_height, image_width = image_np.shape[:2]
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({confidence:.2f})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            image_np,
            (x1, y1 - text_size[1] - 10),
            (x1 + text_size[0], y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            image_np,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
    return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))


# ✅ เบลอเฉพาะจุดที่ตรวจพบ
def blur_detected_areas(image_np, detections, blur_ksize=(51, 51)):
    blurred_image = image_np.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        h, w = blurred_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        roi = blurred_image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi_blurred = cv2.GaussianBlur(roi, blur_ksize, 0)
        blurred_image[y1:y2, x1:x2] = roi_blurred
    return Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))


# ✅ ประมวลผลหลายโมเดลพร้อมกัน พร้อมดัก Exception
def process_selected_models(image, model_types, thresholds):
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def run_model(model_type):
        model = models[model_type]
        threshold = thresholds.get(model_type, 0.5)
        return model_type, analyze_model_np(image_np, model, threshold)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_model, m) for m in model_types]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    all_detections = []
    for model_type, detections in results:
        for d in detections:
            d["model_type"] = model_type
        all_detections.extend(detections)

    output_image_bbox = draw_bounding_boxes_from_array(image_np.copy(), all_detections)
    output_image_blur = blur_detected_areas(image_np.copy(), all_detections)
    json_data = json.dumps(all_detections, indent=4, ensure_ascii=False)

    return output_image_bbox, output_image_blur, json_data


# ✅ API เรียกใช้งาน local YOLO
@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    try:
        api_key = request.headers.get("x-api-key")
        api_key_data = api_keys_collection.find_one({"api_key": api_key})
        if not api_key_data:
            return jsonify({"error": "Invalid API Key"}), 401

        expires_at = api_key_data.get("expires_at")
        if expires_at:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expires_at:
                return jsonify({"error": "API Key expired"}), 401

        quota = int(api_key_data["quota"])
        if quota != -1 and quota <= 0:
            return jsonify({"error": "Quota exceeded"}), 400

        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        ext = file.filename.rsplit(".", 1)[-1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # ✅ บันทึกชื่อไฟล์ลง DB เพื่อติดตามเวลา
        uploaded_files_collection.insert_one(
            {"filename": filename, "created_at": datetime.utcnow()}
        )

        if not is_image(file_path):
            if os.path.exists(file_path):
                os.remove(file_path)
                # ลบจาก DB ด้วยถ้าไฟล์ invalid
                uploaded_files_collection.delete_one({"filename": filename})
            return jsonify({"error": "Invalid image"}), 400

        analysis_types = api_key_data.get("analysis_types")
        if not analysis_types:
            analysis_types_json = request.form.get("analysis_types")
            if analysis_types_json:
                analysis_types = json.loads(analysis_types_json)
            else:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    uploaded_files_collection.delete_one({"filename": filename})
                return jsonify({"error": "No analysis_types provided"}), 400

        thresholds = {}
        key_thresholds = api_key_data.get("thresholds", {})
        if key_thresholds:
            thresholds = {k: float(v) for k, v in key_thresholds.items()}
        else:
            thresholds_json = request.form.get("thresholds")
            if thresholds_json:
                thresholds = {
                    k: float(v) for k, v in json.loads(thresholds_json).items()
                }
            else:
                thresholds = {mt: 0.5 for mt in analysis_types}

        image = Image.open(file_path).convert("RGB")
        output_image, blurred_output, detection_json = process_selected_models(
            image, analysis_types, thresholds
        )
        detection_data = json.loads(detection_json)

        processed_filename = f"processed_{uuid.uuid4()}.jpg"
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)
        output_image.save(processed_path)
        # ✅ บันทึก processed file
        uploaded_files_collection.insert_one(
            {"filename": processed_filename, "created_at": datetime.utcnow()}
        )

        blurred_filename = f"blurred_{uuid.uuid4()}.jpg"
        blurred_path = os.path.join(app.config["UPLOAD_FOLDER"], blurred_filename)
        blurred_output.save(blurred_path)
        # ✅ บันทึก blurred file
        uploaded_files_collection.insert_one(
            {"filename": blurred_filename, "created_at": datetime.utcnow()}
        )

        image_url = url_for(
            "uploaded_file", filename=processed_filename, _external=True
        )
        blurred_image_url = url_for(
            "uploaded_file", filename=blurred_filename, _external=True
        )

        status = "passed"
        for d in detection_data:
            threshold = float(thresholds.get(d["model_type"], 0.5))
            if d["confidence"] > threshold:
                status = "failed"
                break

        if os.path.exists(file_path):
            os.remove(file_path)

        if quota != -1:
            api_keys_collection.update_one(
                {"api_key": api_key}, {"$set": {"quota": quota - 1}}
            )

        return jsonify(
            {
                "status": status,
                "detections": detection_data,
                "processed_image_url": image_url,
                "processed_blurred_image_url": blurred_image_url,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API สำหรับขอ API Key
@app.route("/request-api-key", methods=["POST"])
@token_required
def request_api_key(current_user):  # << รับ current_user จาก decorator
    data = request.get_json()
    analysis_types = data.get("analysis_types", [])
    quota = data.get("quota", 100)
    thresholds = data.get("thresholds", {})  # รับค่า thresholds จากฟรอนต์
    plan = data.get("plan", "free")

    if not analysis_types:
        return jsonify({"error": "At least one analysis type is required"}), 400

    email = current_user["email"]  # << ดึง email จาก token ที่ถอดแล้ว

    if plan == "free":
        existing_free_key = api_keys_collection.find_one(
            {"email": email, "plan": "free"}
        )
        if existing_free_key:
            return jsonify({"error": "คุณได้ขอ API Key ฟรีไปแล้ว"}), 400

    # สร้าง API Key ใหม่
    api_key = str(uuid.uuid4())

    # บันทึกข้อมูลใหม่
    api_keys_collection.insert_one(
        {
            "email": email,
            "api_key": api_key,
            "analysis_types": analysis_types,
            "thresholds": thresholds,
            "quota": quota,
            "plan": plan,
        }
    )

    return jsonify({"apiKey": api_key})


# API สำหรับรายงานปัญหา
@app.route("/report-issue", methods=["POST"])
def report_issue():
    issue = request.json.get("issue")
    category = request.json.get("category")

    print(f"Received issue: {issue}, category: {category}")

    if issue and category:
        # สร้างอีเมล
        subject = f"[รายงานปัญหา] หมวดหมู่: {category}"
        body = f"หมวดหมู่: {category}\nรายละเอียดปัญหา: {issue}"

        try:
            msg = Message(
                subject=subject,
                recipients=["Phurinsukman3@gmail.com"],  # เปลี่ยนเป็นอีเมลผู้รับ
                body=body,
            )
            mail.send(msg)
            return jsonify({"success": True}), 200
        except Exception as e:
            print(f"Error sending email: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": False}), 400


# ดูข้อมูล database
@app.route("/get-api-keys", methods=["GET"])
@token_required
def get_api_keys(current_user):
    email = current_user["email"]

    if not email:
        return jsonify({"error": "Email is required"}), 400

    try:
        user = api_keys_collection.find({"email": email})
        api_keys = list(user)
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    if not api_keys:
        return jsonify({"error": "No API keys found for this email"}), 404

    # ✅ แปลง expires_at จาก datetime เป็น string
    for key in api_keys:
        if "expires_at" in key and isinstance(key["expires_at"], datetime):
            key["expires_at"] = key["expires_at"].isoformat()

    # ส่งคืนข้อมูล API Keys ทั้งหมดของผู้ใช้ พร้อม threshold
    return jsonify(
        {
            "api_keys": [
                {
                    "api_key": key.get("api_key", "ไม่พบ API Key"),
                    "analysis_types": key.get("analysis_types", []),
                    "quota": key.get("quota", 0),
                    "thresholds": key.get(
                        "thresholds", 0.5
                    ),  # เพิ่มตรงนี้ (ค่า default = 0.5 ถ้าไม่มีใน DB)
                    "expires_at": key.get("expires_at"),
                }
                for key in api_keys
            ]
        }
    )


@app.route("/get-username", methods=["GET"])
@token_required
def get_username(current_user):
    email = current_user["email"]
    if not email:
        return jsonify({"error": "Missing email parameter"}), 400

    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"username": user.get("username")}), 200


# API สำหรับดาวน์โหลดเอกสารคู่มือ
@app.route("/manual")
def download_manual():
    # ระบุเส้นทางไฟล์ PDF ที่เก็บอยู่ใน root directory
    file_path = os.path.join(app.root_path, "manual.pdf")  # 'manual.pdf' คือชื่อไฟล์
    return send_from_directory(app.root_path, "manual.pdf")


# ให้บริการไฟล์ที่อัปโหลด
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"], filename, as_attachment=False
    )


# API สำหรับสร้าง QR Code
def generate_qr_code(promptpay_id, amount=0):
    # สร้าง payload ด้วยหมายเลข PromptPay และจำนวนเงิน
    if amount > 0:
        payload = qrcode.generate_payload(promptpay_id, amount)
    else:
        payload = qrcode.generate_payload(promptpay_id)

    # สร้าง QR Code จาก payload
    img = qrcode.to_image(payload)

    # แปลงภาพ QR Code เป็น Base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


# เพิ่มตอนสร้าง QR ให้สร้าง ref_code และบันทึก order
@app.route("/generate_qr", methods=["POST"])
@token_required
def generate_qr(current_user):
    # 🔥 ตรวจสอบก่อนว่ามี order ที่ยังไม่ชำระอยู่หรือไม่
    existing_unpaid_order = orders_collection.find_one(
        {"email": current_user["email"], "paid": False}
    )

    if existing_unpaid_order:
        # ถ้ายังมี order ที่ยังไม่ชำระ → ให้ใช้ order นั้นแทนการสร้างใหม่
        ref_code = existing_unpaid_order[
            "ref_code"
        ]  # ยังใช้ ref_code เดิม (ถึงแม้จะไม่ได้ใช้ในการตรวจสอบแล้ว)
        amount = existing_unpaid_order["amount"]
        promptpay_id = "66882884744"  # หรือดึงจาก config

        # สร้าง QR ใหม่จาก order เดิม
        qr_base64 = generate_qr_code(promptpay_id, amount)

        return jsonify(
            {
                "qr_code_url": qr_base64,
                "ref_code": ref_code,
                "message": "ใช้งานคำสั่งซื้อเดิมที่ยังไม่ชำระ",
            }
        )

    # --- ถ้าไม่มี order ที่ยังไม่ชำระ → สร้างใหม่ ---
    data = request.get_json()
    amount = float(data.get("amount", 0))
    promptpay_id = data.get("promptpay_id", "66882884744")
    email = current_user["email"]
    quota = int(data.get("quota", 100))
    plan = data.get("plan", "paid")
    analysis_types = data.get("analysis_types", [])
    thresholds = data.get("thresholds", {})
    duration = int(data.get("duration", 1))

    # เวลาประเทศไทย
    thai_time = datetime.now(ZoneInfo("Asia/Bangkok"))
    current_time = thai_time.strftime("%d/%m/%Y %H:%M:%S")
    timestamp = thai_time.strftime("%Y%m%d%H%M%S")
    random_str = secrets.token_hex(4).upper()
    ref_code = f"{current_time} {timestamp}{random_str}"

    # บันทึกออร์เดอร์ลงฐานข้อมูล
    orders_collection.insert_one(
        {
            "ref_code": ref_code,
            "email": email,
            "amount": amount,
            "quota": quota,
            "plan": plan,
            "duration": duration,
            "analysis_types": analysis_types,
            "thresholds": thresholds,
            "paid": False,
            "created_at": current_time,
            "created_time": datetime.now(timezone.utc),
        }
    )

    # สร้าง QR
    qr_base64 = generate_qr_code(promptpay_id, amount)

    return jsonify({"qr_code_url": qr_base64, "ref_code": ref_code})


# TTL index
orders_collection.create_index([("email", 1), ("paid", 1), ("created_time", -1)])


# ฟังก์ชันตรวจสอบว่ามี QR Code หรือไม่
def check_qrcode(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False
    detector = cv2.QRCodeDetector()
    retval, points = detector.detect(image)  # ใช้ detect() แทน detectAndDecode()

    if retval:  # ถ้าคืนค่า True แสดงว่ามี QR code ในภาพ
        return True
    return False


# ฟังก์ชันตรวจสอบรหัสอ้างอิงและเวลาในสลิป
@app.route("/upload-receipt", methods=["POST"])
@token_required
def upload_receipt(current_user):
    save_path = None  # ประกาศไว้ด้านบนสุด
    try:
        if "receipt" not in request.files:
            return jsonify({"error": "No receipt file provided"}), 400

        file = request.files["receipt"]
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        print(f"📥 Saved file: {save_path}")  # ✅ แทน logger.info

        if not is_image(save_path):
            return jsonify({"error": "ไฟล์ไม่ใช่รูปภาพ"}), 400

        if not check_qrcode(save_path):
            return jsonify({"error": "รูปเเบบใบเสร็จไม่ถูกต้อง"}), 400

        # Initialize OCR
        try:
            ocr_engine = AdvancedSlipOCR()
        except Exception as e:
            print(f"Failed to initialize OCR engine: {e}")  # ✅ แทน logger.error
            return jsonify({"error": "ระบบ OCR ล้มเหลว"}), 500

        # Process image
        try:
            image = Image.open(save_path).convert("RGB")
            ocr_data = ocr_engine.extract_info(image)
        except Exception as e:
            print(f"Failed to process image: {e}")  # ✅ แทน logger.error
            return jsonify({"error": "ไม่สามารถประมวลผลรูปภาพได้"}), 500

        # ✅ แสดงผล OCR ทั้งหมดก่อนตรวจสอบ — เพื่อดูว่าอะไรเป็น None
        text = ocr_data.get("full_text", "ไม่มีข้อความ")
        date_text = ocr_data.get("date", "ไม่มีวันที่")
        time_ocr = ocr_data.get("time", "ไม่มีเวลา")
        amount = ocr_data.get("amount", "ไม่มีจำนวนเงิน")
        full_name = ocr_data.get("full_name", "ไม่มีชื่อ")
        time_receipts = ocr_data.get("time_receipts", "ไม่มีเวลา")

        print("=== OCR RESULT ===")
        print("OCR Full Text: ", text)
        print("Date from OCR: ", date_text)
        print("Time from OCR: ", time_ocr)
        print("Amount from OCR: ", amount)
        print("full_name: ", full_name)
        print("time_receipts: ", time_receipts)
        print("==================")

        # Validate required fields
        required_fields = [
            "full_text",
            "date",
            "time",
            "amount",
            "full_name",
            "time_receipts",
        ]
        for field in required_fields:
            if not ocr_data.get(field):
                print(f"❌ Missing field: {field}")  # ✅ เพิ่ม log นี้
                return jsonify({"error": f"ข้อมูล {field} ขาดหายไปหรือเป็นค่าว่าง"}), 400

        # Log extracted data
        text = ocr_data["full_text"]
        date_text = ocr_data["date"]
        time_ocr = ocr_data["time"]
        amount = ocr_data["amount"]
        full_name = ocr_data["full_name"]
        time_receipts = ocr_data["time_receipts"]

        print("OCR Full Text: ", text)
        print("Date from OCR: ", date_text)
        print("Time from OCR: ", time_ocr)
        print("Amount from OCR: ", amount)
        print("full_name: ", full_name)
        print("time_receipts: ", time_receipts)

        # Find unpaid order
        matched_order = orders_collection.find_one(
            {"email": current_user["email"], "paid": False},
            sort=[("created_time", -1)],
        )

        if not matched_order:
            return jsonify({"error": "ไม่พบคำสั่งซื้อที่ยังไม่ชำระเงินสำหรับคุณ"}), 404

        # Validate recipient name
        allowed_names = ["ภูรินทร์สุขมั่น", "ภูรินทร์", "สุขมั่น", "ภูรินทร์ สุขมั่น"]
        full_name_clean = full_name.strip().replace(" ", "").lower()
        allowed_names_clean = [name.replace(" ", "").lower() for name in allowed_names]

        if not any(name in full_name_clean for name in allowed_names_clean):
            return jsonify({"error": "ชื่อผู้รับเงินไม่ถูกต้อง"}), 400

        # Validate date
        try:
            created_datetime = datetime.strptime(
                matched_order["created_at"], "%d/%m/%Y %H:%M:%S"
            )
        except Exception as e:
            print(f"Error parsing created_at: {e}")  # ✅ แทน logger.error
            return jsonify({"error": "ข้อมูลวันที่ในฐานข้อมูลผิดพลาด"}), 500

        if date_text:
            try:
                date_from_ocr = datetime.strptime(date_text, "%d/%m/%Y").date()
                if date_from_ocr != created_datetime.date():
                    return jsonify({"error": "วันที่ในสลิปไม่ตรงกับวันที่สร้างออร์เดอร์"}), 400
            except Exception as e:
                print(f"Error parsing OCR date: {e}")  # ✅ แทน logger.error
                return jsonify({"error": "รูปแบบวันที่ในสลิปผิด"}), 400

        # Validate time
        if time_receipts:
            try:
                time_from_ocr = datetime.strptime(time_receipts, "%H:%M")
                time_from_ocr_full = datetime.combine(
                    created_datetime.date(), time_from_ocr.time()
                )
                time_diff = abs((created_datetime - time_from_ocr_full).total_seconds())
                if time_diff > 300:
                    return jsonify({"error": "เวลาในสลิปห่างกันเกิน 5 นาที"}), 400
            except Exception as e:
                print(f"Error parsing OCR time: {e}")  # ✅ แทน logger.error
                return jsonify({"error": "รูปแบบเวลาในสลิปผิด"}), 400

        # Validate amount
        if amount:
            try:
                amount_clean = float(amount.replace(",", ""))
                if float(matched_order.get("amount", 0)) != amount_clean:
                    return jsonify({"error": "ยอดเงินไม่ตรงกัน"}), 400
            except Exception as e:
                print(f"Error parsing amount: {e}")  # ✅ แทน logger.error
                return jsonify({"error": "ยอดเงินไม่สามารถแปลงได้"}), 400

        # Update order & generate API key
        orders_collection.update_one(
            {"_id": matched_order["_id"]},
            {
                "$set": {
                    "paid": True,
                    "paid_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                }
            },
        )

        api_key = str(uuid.uuid4())
        plan = matched_order.get("plan", "paid")
        insert_data = {
            "email": matched_order.get("email", ""),
            "api_key": api_key,
            "analysis_types": matched_order.get("analysis_types", []),
            "thresholds": matched_order.get("thresholds", {}),
            "quota": -1 if plan == "monthly" else matched_order.get("quota", 100),
            "plan": plan,
            "created_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        }

        if plan == "monthly":
            duration = matched_order.get("duration", 1)
            insert_data["expires_at"] = datetime.now(timezone.utc) + relativedelta(
                months=+duration
            )

        api_keys_collection.insert_one(insert_data)
        orders_collection.delete_one({"_id": matched_order["_id"]})

        print("✅ Upload receipt completed successfully")  # ✅ แทน logger.info
        return (
            jsonify(
                {
                    "success": True,
                    "message": "อัปโหลดสำเร็จ",
                    "api_key": api_key,
                    "ocr_data": {
                        "date": date_text,
                        "time": time_ocr,
                        "amount": amount,
                        "fullname": full_name,
                        "full_text": text,
                    },
                }
            ),
            200,
        )

    except Exception as e:
        print(f"❌ Unexpected error: {e}")  # ✅ แทน logger.error
        return jsonify({"error": "เกิดข้อผิดพลาดภายในระบบ"}), 500

    finally:
        # ลบไฟล์ชั่วคราว — ถ้ามี และยังอยู่
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
                print(f"🧹 Deleted temporary file: {save_path}")  # ✅ แทน logger.info
            except Exception as e:
                print(f"⚠️ Failed to delete {save_path}: {e}")  # ✅ แทน logger.error


# TTL Index สำหรับ API Key ที่หมดอายุ
api_keys_collection.create_index([("expires_at", 1)], expireAfterSeconds=0)


@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["image"]
        analysis_types = request.form.get("analysis_types")  # JSON string
        thresholds = request.form.get("thresholds")  # JSON string

        files = {"image": (file.filename, file.stream, file.mimetype)}

        data = {"analysis_types": analysis_types, "thresholds": thresholds}

        response = requests.post(
            "https://objexify.dpdns.org/analyze-image",
            headers={"x-api-key": API_KEY},
            files=files,
            data=data,
        )

        return (response.text, response.status_code, response.headers.items())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/auth/google")
def auth_google():
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={GOOGLE_REDIRECT_URI}&"
        f"response_type=code&"
        f"scope=openid email profile"
    )
    return redirect(google_auth_url)


@app.route("/auth/google/callback")
def google_callback():
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Authorization code not found"}), 400

    # แลกเปลี่ยน code เป็น access token
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    token_response = requests.post(token_url, data=token_data)
    token_json = token_response.json()

    access_token = token_json.get("access_token")

    # ดึงข้อมูลโปรไฟล์ผู้ใช้
    user_info_url = "https://www.googleapis.com/oauth2/v1/userinfo"
    user_info_response = requests.get(
        user_info_url, headers={"Authorization": f"Bearer {access_token}"}
    )
    user_info = user_info_response.json()

    email = user_info.get("email")
    user = users_collection.find_one({"email": email})
    if not user:
        users_collection.insert_one(
            {
                "email": email,
                "username": user_info.get("name"),
                "password": None,  # ไม่มีรหัสผ่านเพราะล็อกอินด้วย Google
            }
        )

    # สร้าง JWT token ให้ user
    token = generate_token(email)

    # ส่ง token กลับไปหน้า frontend ผ่าน query string
    return redirect(f"{API_BASE_URL}/apikey/view-api-keys.html?token={token}")


# สร้าง OTP และส่งอีเมล
@app.route("/reset-request", methods=["POST"])
@token_required
def reset_request():
    email = request.json.get("email")
    if not users_collection.find_one({"email": email}):
        return jsonify({"message": "ไม่พบอีเมลนี้"}), 404

    otp = str(random.randint(100000, 999999))
    expiration = datetime.utcnow() + timedelta(minutes=5)

    otp_collection.update_one(
        {"email": email},
        {"$set": {"otp": otp, "otp_expiration": expiration, "used": False}},
        upsert=True,
    )

    msg = Message("OTP สำหรับรีเซ็ตรหัสผ่าน", recipients=[email])
    msg.body = f"รหัส OTP ของคุณคือ: {otp}"
    mail.send(msg)

    return jsonify({"message": "ส่ง OTP แล้ว"}), 200


# ตรวจสอบ OTP
@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data = request.json
    email = data.get("email")
    otp = data.get("otp")

    record = otp_collection.find_one({"email": email, "otp": otp, "used": False})
    if not record:
        return jsonify({"message": "OTP ไม่ถูกต้อง"}), 400

    if record["otp_expiration"] < datetime.utcnow():
        return jsonify({"message": "OTP หมดอายุแล้ว"}), 400

    return jsonify({"message": "OTP ถูกต้อง"}), 200


# ตั้งรหัสผ่านใหม่
@app.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.json
    email = data.get("email")
    otp = data.get("otp")
    password = data.get("password")
    confirm_password = data.get("confirm_password")

    if password != confirm_password:
        return jsonify({"message": "รหัสผ่านไม่ตรงกัน"}), 400

    record = otp_collection.find_one({"email": email, "otp": otp, "used": False})
    if not record or record["otp_expiration"] < datetime.utcnow():
        return jsonify({"message": "OTP ไม่ถูกต้องหรือหมดอายุ"}), 400

    # แฮชรหัสผ่านใหม่ก่อนอัปเดตในฐานข้อมูล
    hashed_password = generate_password_hash(
        password, method="pbkdf2:sha256", salt_length=8
    )

    users_collection.update_one(
        {"email": email}, {"$set": {"password": hashed_password}}
    )
    otp_collection.update_one({"email": email}, {"$set": {"used": True}})

    return jsonify({"message": "รีเซ็ตรหัสผ่านเรียบร้อยแล้ว"}), 200


def cleanup_expired_files():
    """ลบไฟล์ที่หมดอายุแล้วจาก disk"""
    folder = app.config["UPLOAD_FOLDER"]
    try:
        current_files = set(os.listdir(folder))
        active_files = set(doc["filename"] for doc in uploaded_files_collection.find())
        expired_files = current_files - active_files
        for fname in expired_files:
            try:
                os.remove(os.path.join(folder, fname))
                print(f"🧹 Deleted expired file: {fname}")
            except Exception as e:
                print(f"❌ Error deleting {fname}: {e}")
    except Exception as e:
        print(f"Cleanup system error: {e}")


def start_cleanup_scheduler():
    import threading
    import time

    def run():
        while True:
            cleanup_expired_files()
            time.sleep(300)  # ทำงานทุก 5 นาที

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


start_cleanup_scheduler()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
