import cv2
import numpy as np
import easyocr
import re
from fuzzywuzzy import fuzz
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
import logging
import pytesseract
 
# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
class AdvancedSlipOCR:
    def __init__(self):
        """เริ่มต้นระบบ OCR ที่ทันสมัย"""
        try:
            # เริ่มต้น EasyOCR สำหรับภาษาไทยและอังกฤษ
            self.easyocr_reader = easyocr.Reader(["th", "en"], gpu=True)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
 
        # ตั้งค่าเดือนไทย (เพิ่มเติม)
        self.thai_months = {
    "ม.ค.": "01", "ม.ค": "01", "มค": "01",
    "ก.พ.": "02", "ก.พ": "02", "กพ": "02",
    "มี.ค.": "03", "มี.ค": "03", "มีค": "03",
    "เม.ย.": "04", "เม.ย": "04", "เมย": "04",
    "พ.ค.": "05", "พ.ค": "05", "พค": "05",
    "มิ.ย.": "06", "มิ.ย": "06", "มิย": "06",
    "ก.ค.": "07", "ก.ค": "07", "กค": "07",
    "ส.ค.": "08", "ส.ค": "08", "สค": "08",  # ← เพิ่ม "ส.ค." ตรงนี้!
    "ก.ย.": "09", "ก.ย": "09", "กย": "09",
    "ต.ค.": "10", "ต.ค": "10", "ตค": "10",
    "พ.ย.": "11", "พ.ย": "11", "พย": "11",
    "ธ.ค.": "12", "ธ.ค": "12", "ธค": "12",
    }
 
    def preprocess_image_advanced(self, image: np.ndarray) -> List[np.ndarray]:
        """ประมวลผลภาพขั้นสูงด้วยหลายวิธี"""
        processed_images = []
 
        # แปลงเป็น grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
 
        # วิธีที่ 1: ปรับความคมชัด
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        processed_images.append(sharpened)
 
        # วิธีที่ 2: ลดสัญญาณรบกวน
        denoised = cv2.fastNlMeansDenoising(gray)
        processed_images.append(denoised)
 
        # วิธีที่ 3: ปรับความสว่างและคอนทราสต์
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed_images.append(enhanced)
 
        # วิธีที่ 4: ใช้ morphological operations
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_morph)
        processed_images.append(morph)
 
        # วิธีที่ 5: ใช้ adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(adaptive)
 
        # วิธีที่ 6: ใช้ Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(otsu)
 
        return processed_images
 
    def extract_text_with_easyocr(self, image: np.ndarray) -> List[Tuple]:
        """ดึงข้อความด้วย EasyOCR"""
        if self.easyocr_reader is None:
            return []
 
        try:
            # แปลง BGR เป็น RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
 
            # ดึงข้อความด้วย EasyOCR
            results = self.easyocr_reader.readtext(image_rgb)
            return results
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return []
 
    def extract_text_with_pytesseract(self, image: np.ndarray) -> str:
        """ดึงข้อความด้วย Tesseract (backup)"""
        try:
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
            text = pytesseract.image_to_string(image, lang="tha+eng", config="--psm 6")
            return text
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return ""
 
    def extract_time(self, text: str) -> Optional[str]:
        """ดึงเวลาจากข้อความ"""
        # รูปแบบเวลา: HH:MM, HH.MM, HH-MM
        time_patterns = [
            r"(\d{1,2})[:\.\-](\d{2})",  # HH:MM, HH.MM, HH-MM
            r"(\d{1,2})[:\.\-](\d{2})[:\.\-](\d{2})",  # HH:MM:SS
            r"(\d{1,2})[:\.\-](\d{2})\s*(น\.|นาที|min|hr|hour)",  # HH:MM นาที
        ]
 
        valid_times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2:
                    try:
                        hour, minute = int(match[0]), int(match[1])
                        if 0 <= hour <= 23 and 0 <= minute <= 59:
                            valid_times.append(f"{hour:02d}:{minute:02d}")
                    except ValueError:
                        continue
 
        # เลือกเวลาที่มีความเป็นไปได้สูงสุด (ไม่ใช่ 00:00 หรือ 01:00)
        for time_str in sorted(valid_times, reverse=True):
            if time_str not in ["00:00", "01:00"]:
                return time_str
 
        # ถ้าไม่มี ให้คืนเวลาสุดท้ายที่พบ
        return valid_times[-1] if valid_times else None
 
    def extract_date(self, text: str) -> Optional[str]:
 
        # รูปแบบวันที่: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        date_patterns = [
            r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})",  # DD/MM/YYYY
            r"(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})",  # DD MM YYYY
            r"(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})",  # YYYY/MM/DD
        ]
 
        # รูปแบบเดือนไทย (รองรับทั้ง 68 และ 2568)
        thai_date_patterns = [
            r"(\d{1,2})\s*([ก-๙]{1,3})\.?\s*(\d{2,4})",  # 1 ก.ย. 2568 หรือ 1 ก.ย. 68
            r"(\d{1,2})([ก-๙]{1,3})\.(\d{2,4})",  # 1ก.ย.2568
            r"(\d{1,2})\s*([ก-๙]{2,})\s*(\d{2,4})",  # 1 กันยายน 2568
            r"(\d{1,2})\s*([ก-๙]{1,10})\.?\s*(\d{2,4})",  # วัน + เดือนย่อ/เต็ม + ปี
            r"(\d{1,2})\s*([ก-๙]{1,}\.?)\s*(\d{2,4})",  # 01 ก.ย. 2568
            r"\d{2}\s(?:ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)\s\d{4}",
            r"(\d{1,2})\s*([ก-๙\.]{2,5})\s*(\d{2,4})",  # 01 ก.ย 2568   
        ]
 
        # ตรวจสอบวันที่แบบตัวเลขก่อน
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:
                    if self.is_valid_date(match):
                        return f"{match[0]}/{match[1]}/{match[2]}"
 
        # ตรวจสอบวันที่แบบเดือนไทย
        for pattern in thai_date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:
                    day, thai_month, year_str = match
                    # ทำความสะอาดเดือนไทย
                    clean_month = re.sub(r"[^\u0E00-\u0E7F\.]", "", thai_month)
                    # หาเดือนใน dictionary
                    month_num = None
                    for key, value in self.thai_months.items():
                        if key in clean_month or clean_month.startswith(
                            key.replace(".", "")
                        ):
                            month_num = value
                            break
 
                    if month_num:
                        # ✅ แปลงปี พ.ศ. → ค.ศ. อย่างถูกต้อง
                        year_int = int(year_str)
                        if year_int < 100:  # เช่น 68
                            year_ad = year_int + 1957  # 68 + 1957 = 2025
                        elif year_int >= 2500:  # เช่น 2568
                            year_ad = year_int - 543  # 2568 - 543 = 2025
                        else:  # ถ้าเป็นปี ค.ศ. อยู่แล้ว เช่น 2025
                            year_ad = year_int
 
                        date_tuple = (day, month_num, str(year_ad))
                        if self.is_valid_date(date_tuple):
                            logger.info(f"Extracted Thai date: {day} {thai_month} {year_str} -> {int(day):02d}/{month_num}/{year_ad}")
                            return f"{int(day):02d}/{month_num}/{year_ad}"
    
        return None
 
    def is_valid_date(self, date_parts: Tuple[str, str, str]) -> bool:
 
        try:
            day, month, year = map(int, date_parts)
 
            # ตรวจสอบช่วงวันที่ที่สมเหตุสมผล
            if not (1 <= month <= 12):
                return False
 
            # ตรวจสอบวันในเดือน
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if month == 2 and ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0):
                days_in_month[1] = 29
 
            if not (1 <= day <= days_in_month[month - 1]):
                return False
 
            return True
        except:
            return False
 
    def extract_amount(self, text: str) -> Optional[str]:
        """ดึงจำนวนเงินจากข้อความ"""
        # รูปแบบจำนวนเงิน
        amount_patterns = [
            r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(บาท|฿|THB|baht)",
            r"(?:ราคา|จำนวน|ยอด|รวม|Total|Amount|Price)[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
            r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:บาท|฿|THB|baht)?",
        ]
 
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        amount = match[0]
                    else:
                        amount = match
 
                    # ทำความสะอาดจำนวนเงิน
                    amount_clean = re.sub(r"[^\d,\.]", "", amount)
                    if amount_clean:
                        return amount_clean
 
        return None
 
    def extract_name(
        self, text: str, expected_names: List[str] = None
    ) -> Optional[str]:
        """ดึงชื่อจากข้อความ"""
        if expected_names is None:
            expected_names = ["ภูรินทร์สุขมั่น", "ภูรินทร์", "สุขมั่น"]
 
        # แยกคำที่เป็นภาษาไทย
        thai_words = re.findall(r"[ก-๙]{2,}", text)
 
        best_match = None
        best_score = 0
 
        for word in thai_words:
            for expected_name in expected_names:
                # ใช้ fuzzy matching
                score = fuzz.ratio(word, expected_name)
                partial_score = fuzz.partial_ratio(word, expected_name)
                token_score = fuzz.token_sort_ratio(word, expected_name)
 
                # คำนวณคะแนนรวม
                total_score = (score + partial_score + token_score) / 3
 
                if total_score > best_score and total_score > 70:
                    best_score = total_score
                    best_match = expected_name
 
        return best_match
 
    def extract_receipt_number(self, text: str) -> Optional[str]:
        """ดึงเลขที่ใบเสร็จ"""
        receipt_patterns = [
            r"(?:เลขที่|Receipt|Ref|Ref\s*No|No\.?)[:\s]*([A-Z0-9\-]+)",
            r"([A-Z]{2,}\d{4,})",  # รูปแบบ: AB12345
            r"(\d{4,})",  # เลข 4 หลักขึ้นไป
        ]
 
        for pattern in receipt_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
 
        return None
 
    def extract_merchant_info(self, text: str) -> Optional[str]:
        """ดึงข้อมูลร้านค้า"""
        merchant_patterns = [
            r"(?:ร้าน|บริษัท|Company|Store|Shop)[:\s]*([ก-๙A-Za-z\s]+)",
            r"([ก-๙A-Za-z\s]{3,})",  # คำที่ยาว 3 ตัวอักษรขึ้นไป
        ]
 
        for pattern in merchant_patterns:
            matches = re.findall(pattern, text)
            if matches:
                merchant = matches[0].strip()
                if len(merchant) >= 3:
                    return merchant
 
        return None
 
    def process_image(
        self, image: np.ndarray, expected_names: List[str] = None
    ) -> Dict:
        """ประมวลผลภาพและดึงข้อมูลทั้งหมด"""
        try:
            # ประมวลผลภาพขั้นสูง
            processed_images = self.preprocess_image_advanced(image)
 
            # รวมข้อความจากทุกวิธี
            all_text = ""
            all_easyocr_results = []
 
            # ใช้ EasyOCR กับภาพที่ประมวลผลแล้ว
            for processed_img in processed_images:
                easyocr_results = self.extract_text_with_easyocr(processed_img)
                all_easyocr_results.extend(easyocr_results)
 
                # รวมข้อความ
                for bbox, text, confidence in easyocr_results:
                    if confidence > 0:  # ลด threshold ให้จับได้มากขึ้น
                        all_text += " " + text
 
            # ใช้ Tesseract เป็น backup
            tesseract_text = self.extract_text_with_pytesseract(image)
            all_text += " " + tesseract_text
 
            # ทำความสะอาดข้อความ
            all_text = re.sub(r"\s+", " ", all_text).strip()
 
            # ดึงข้อมูลต่างๆ
            result = {
                "time": self.extract_time(all_text),
                "date": self.extract_date(all_text),
                "amount": self.extract_amount(all_text),
                "full_name": self.extract_name(all_text, expected_names),
                "receipt_number": self.extract_receipt_number(all_text),
                "merchant": self.extract_merchant_info(all_text),
                "full_text": all_text,
                "confidence_scores": [conf for _, _, conf in all_easyocr_results],
                "average_confidence": (
                    np.mean([conf for _, _, conf in all_easyocr_results])
                    if all_easyocr_results
                    else 0
                ),
            }
 
            logger.info(
                f"OCR processing completed. Confidence: {result['average_confidence']:.2f}"
            )
            return result
 
        except Exception as e:
            logger.error(f"Error in process_image: {e}")
            return {
                "error": str(e),
                "full_text": "",
                "time": None,
                "date": None,
                "amount": None,
                "full_name": None,
                "receipt_number": None,
                "merchant": None,
                "confidence_scores": [],
                "average_confidence": 0,
            }
 
    def save_processed_images(
        self, image: np.ndarray, output_dir: str = "processed_images"
    ):
        """บันทึกภาพที่ประมวลผลแล้วเพื่อตรวจสอบ"""
        os.makedirs(output_dir, exist_ok=True)
 
        processed_images = self.preprocess_image_advanced(image)
 
        for i, processed_img in enumerate(processed_images):
            filename = os.path.join(output_dir, f"processed_{i+1}.jpg")
            cv2.imwrite(filename, processed_img)
            logger.info(f"Saved processed image: {filename}")
 
    def extract_info(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
        result = self.process_image(image)
 
        formatted_result = {
            "uuids": (
                [result.get("receipt_number")] if result.get("receipt_number") else []
            ),
            "time": result.get("time"),
            "date": result.get("date"),
            "amount": result.get("amount"),
            "full_name": result.get("full_name"),
            "time_receipts": result.get("time"),
            "full_text": result.get("full_text", ""),
        }
 
        return formatted_result
 
 
def main():
    """ฟังก์ชันหลักสำหรับทดสอบ"""
    # สร้าง instance ของ OCR
    ocr = AdvancedSlipOCR()
    kasikor = r"C:\Users\khongkaphan\Downloads\สื่อ (12) (1).jpeg"
    krungthai = r"C:\Users\khongkaphan\Downloads\สื่อ (1).jpg"
    aomson = r"C:\Users\khongkaphan\Downloads\สื่อ (5).jpg"
    krungthep = r"C:\Users\khongkaphan\Downloads\สื่อ (3).jpg"
    krungsri = r"C:\Users\khongkaphan\Downloads\kma-transfer-promptpay-step-06.webp"
    thaipanit = r"C:\Users\khongkaphan\Downloads\สื่อ (8).jpg"
    
    
    

    # ตัวอย่างการใช้งาน
    image_path = thaipanit  # เปลี่ยนเป็นไฟล์ที่ใช้งานจริง

    if os.path.exists(image_path):
        # อ่านภาพด้วย OpenCV ก่อน
        image = cv2.imread(image_path)

        # ถ้าอ่านไม่ได้ (เช่น ไฟล์ชื่อไทย) → fallback ไปใช้ Pillow
        if image is None:
            try:
                image = np.array(Image.open(image_path))
                print("อ่านภาพด้วย Pillow สำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถเปิดไฟล์ด้วย Pillow ได้: {e}")
                image = None

        if image is not None:
            # ประมวลผลภาพ
            result = ocr.process_image(image)

            # แสดงผลลัพธ์
            print("=== ผลการประมวลผล OCR ===")
            print(f"เวลา: {result.get('time', 'ไม่พบ')}")
            print(f"วันที่: {result.get('date', 'ไม่พบ')}")
            print(f"จำนวนเงิน: {result.get('amount', 'ไม่พบ')}")
            print(f"ชื่อ: {result.get('full_name', 'ไม่พบ')}")
            print(f"เลขที่ใบเสร็จ: {result.get('receipt_number', 'ไม่พบ')}")
            print(f"ร้านค้า: {result.get('merchant', 'ไม่พบ')}")
            print(f"ความเชื่อมั่นเฉลี่ย: {result.get('average_confidence', 0):.2f}")
            print(f"\nข้อความทั้งหมด:\n{result.get('full_text', 'ไม่พบ')}")

            # บันทึกภาพที่ประมวลผลแล้ว
            ocr.save_processed_images(image)
        else:
            print("ไม่สามารถอ่านภาพได้ (cv2 + Pillow ล้มเหลว)")
    else:
        print(f"ไม่พบไฟล์ภาพ: {image_path}")
 
 
if __name__ == "__main__":
    main()