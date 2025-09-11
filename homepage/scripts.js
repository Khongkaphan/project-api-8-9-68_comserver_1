// ฟังก์ชันสำหรับอัปโหลดภาพเมื่อคลิกปุ่ม "อัปโหลดรูปภาพ"
async function uploadImage() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';

  input.onchange = async () => {
    const file = input.files[0];
    if (!file) return;

    const selectedModels = [];
    const modelThresholds = {};

    document.querySelectorAll('input[name="analysis"]:checked').forEach(checkbox => {
      const model = checkbox.value;
      selectedModels.push(model);
      const thresholdInput = document.getElementById(`${model}-threshold`);
      const thresholdValue = parseFloat(thresholdInput.value) || 0.5;
      modelThresholds[model] = thresholdValue;
    });

    const formData = new FormData();
    formData.append('image', file);
    formData.append('analysis_types', JSON.stringify(selectedModels));
    formData.append('thresholds', JSON.stringify(modelThresholds));

    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultText = document.getElementById('resultText');
    const imagePreview = document.getElementById('imagePreview');
    const processedImage = document.getElementById('processedImage');

    loadingSpinner.style.display = 'block';
    resultText.textContent = '';
    processedImage.style.display = 'none';

    const reader = new FileReader();
    reader.onload = () => {
      imagePreview.src = reader.result;
      imagePreview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    try {
      const response = await fetch(`${window.API_BASE_URL}/analyze-image`, {
        method: 'POST',
        headers: {
          'x-api-key': window.API_KEY  // <--- เพิ่ม API Key ที่ถูกต้องของคุณตรงนี้
        },
        body: formData,
      });

      const data = await response.json();
      loadingSpinner.style.display = 'none';

      if (response.ok) {
        const detections = data.detections;

        if (detections.length > 0) {
          resultText.textContent = 'ผลลัพธ์: ไม่ผ่านการทดสอบ';
          resultText.style.color = 'red';
        } else {
          resultText.textContent = 'ผลลัพธ์: ผ่านการทดสอบ';
          resultText.style.color = 'green';
        }

        if (data.processed_image_url) {
          processedImage.src = data.processed_image_url;
          processedImage.style.display = 'block';
        }
      } else {
        resultText.textContent = `ข้อผิดพลาด: ${data.error || 'เกิดข้อผิดพลาด'}`;
        resultText.style.color = 'red';
      }
    } catch (error) {
      loadingSpinner.style.display = 'none';
      resultText.textContent = 'ข้อผิดพลาด: ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์';
      resultText.style.color = 'red';
    }
  };

  input.click();
}

// ฟังก์ชันสำหรับดาวน์โหลดเอกสารคู่มือ
function downloadManual() {
  const url = `${window.API_BASE_URL}/manual`;
  window.location.href = url;
}

function toggleAdvanced() {
  const advancedSection = document.getElementById("advanced-settings");
  advancedSection.style.display = (advancedSection.style.display === "none") ? "block" : "none";
  }
