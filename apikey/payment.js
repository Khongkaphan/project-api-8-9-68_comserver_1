document.addEventListener('DOMContentLoaded', function () {
  const token = localStorage.getItem("token");
  const amount = sessionStorage.getItem("selectedAmount");
  const quota = sessionStorage.getItem("selectedQuota");
  const analysisTypes = JSON.parse(sessionStorage.getItem("selectedAnalysis"));
  const thresholds = JSON.parse(sessionStorage.getItem("selectedThresholds"));
  const amountSpan = document.getElementById('amount');
  const qrCodeImage = document.getElementById('qrCodeImage');
  const paymentStatus = document.getElementById('paymentStatus'); // ต้องมีใน HTML!
  const countdownDisplay = document.getElementById('countdown');
  const confirmPaymentBtn = document.getElementById('confirmPaymentBtn');

  if (!token) {
    alert("กรุณาล็อกอินก่อน");
    return;
  }

  // แสดงจำนวนเงิน
  if (amountSpan) {
    amountSpan.innerText = amount;
  }

  // ถ้ามี QR Code อยู่แล้วใน sessionStorage ให้ใช้เลย
  const existingQrUrl = sessionStorage.getItem("qr_code_url");

  if (existingQrUrl) {
    qrCodeImage.src = existingQrUrl;
  } else {
    // เรียก API เพื่อสร้าง QR Code ใหม่
    fetch(`${window.API_BASE_URL}/generate_qr`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          amount: parseFloat(amount),
          quota: parseInt(quota),
          analysis_types: analysisTypes,
          thresholds: thresholds
        })
      })
      .then(res => res.json())
      .then(data => {
        if (data.qr_code_url) {
          qrCodeImage.src = data.qr_code_url;
          sessionStorage.setItem("qr_code_url", data.qr_code_url);
        } else {
          paymentStatus.textContent = "ไม่สามารถสร้าง QR Code ได้";
        }
      })
      .catch(err => {
        console.error("Error:", err);
        paymentStatus.textContent = "เกิดข้อผิดพลาดในการเชื่อมต่อเซิร์ฟเวอร์";
      });
  }

  // ============ ตรวจสอบและอัปเดทเวลาถอยหลังทุกครั้งที่โหลดหน้า ============
  let countdownInterval;
  const FIVE_MINUTES = 300; // 5 นาที = 300 วินาที

  function startCountdown(duration) {
    let timer = duration;
    countdownInterval = setInterval(() => {
      const minutes = Math.floor(timer / 60);
      const seconds = timer % 60;
      countdownDisplay.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;

      if (timer <= 0) {
        clearInterval(countdownInterval);
        countdownDisplay.textContent = "หมดเวลา!";
        countdownDisplay.style.color = "red";
        paymentStatus.textContent = "คำสั่งซื้อของคุณถูกยกเลิกเนื่องจากเลยเวลาชำระเงิน";
        confirmPaymentBtn.disabled = true;
        confirmPaymentBtn.style.backgroundColor = "#ccc";
        confirmPaymentBtn.style.cursor = "not-allowed";
        sessionStorage.removeItem("qr_code_url");
        sessionStorage.removeItem("countdown_start_time");
      }

      timer--;
    }, 1000);
  }

  // ✅ ตรวจสอบเวลาที่เหลือทุกครั้งที่โหลดหน้า — ไม่ว่าจะรีเฟรชหรือกลับมาจากหน้าอื่น
  const startTime = sessionStorage.getItem("countdown_start_time");

  if (startTime) {
    const elapsed = Math.floor((Date.now() - parseInt(startTime)) / 1000);
    const remaining = FIVE_MINUTES - elapsed;

    if (remaining > 0) {
      // ✅ อัปเดท UI ทันที + เริ่มนับถอยหลังใหม่
      const minutes = Math.floor(remaining / 60);
      const seconds = remaining % 60;
      countdownDisplay.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
      startCountdown(remaining); // เริ่มนับต่อจากเวลาที่เหลือ
    } else {
      // หมดเวลาแล้ว
      clearInterval(countdownInterval);
      countdownDisplay.textContent = "หมดเวลา!";
      countdownDisplay.style.color = "red";
      paymentStatus.textContent = "คำสั่งซื้อของคุณถูกยกเลิกเนื่องจากเลยเวลาชำระเงิน";
      confirmPaymentBtn.disabled = true;
      confirmPaymentBtn.style.backgroundColor = "#ccc";
      confirmPaymentBtn.style.cursor = "not-allowed";
      sessionStorage.removeItem("qr_code_url");
      sessionStorage.removeItem("countdown_start_time");
    }
  } else {
    // ไม่มีเวลาเริ่มต้น → สร้างใหม่
    sessionStorage.setItem("countdown_start_time", Date.now().toString());
    startCountdown(FIVE_MINUTES);
  }
  // ✅ เพื่ม visibility — อัปเดทเวลาเมื่อกลับมาที่แท็บ
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) {
      const startTime = sessionStorage.getItem("countdown_start_time");
      if (startTime) {
        const elapsed = Math.floor((Date.now() - parseInt(startTime)) / 1000);
        const remaining = FIVE_MINUTES - elapsed;
        if (remaining > 0) {
          const minutes = Math.floor(remaining / 60);
          const seconds = remaining % 60;
          countdownDisplay.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
        }
      }
    }
  });
});