document.addEventListener('DOMContentLoaded', function () {
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const receiptImage = document.getElementById('receiptImage');
    const apiKeyDisplay = document.getElementById('api_key');
    const spinner = document.querySelector('.spinner');
    const container = document.querySelector('.container');
    const countdownDisplay = document.getElementById('countdown'); // ✅ เลือก countdown

    // ============ ตัวนับเวลาถอยหลัง ============
    if (countdownDisplay) {
        const FIVE_MINUTES = 300; // 5 นาที = 300 วินาที
        let countdownInterval;

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
                    uploadStatus.textContent = "คุณไม่ได้อัปโหลดสลิปภายในเวลาที่กำหนด ระบบยกเลิกคำสั่งซื้อแล้ว";
                    uploadBtn.disabled = true;
                    uploadBtn.style.backgroundColor = "#ccc";
                    uploadBtn.style.cursor = "not-allowed";
                    sessionStorage.removeItem("qr_code_url");
                    sessionStorage.removeItem("countdown_start_time");
                }

                timer--;
            }, 1000);
        }

        // ตรวจสอบเวลาที่เหลือจาก sessionStorage
        const startTime = sessionStorage.getItem("countdown_start_time");
        if (startTime) {
            const elapsed = Math.floor((Date.now() - parseInt(startTime)) / 1000);
            const remaining = FIVE_MINUTES - elapsed;
            if (remaining > 0) {
                startCountdown(remaining);
            } else {
                clearInterval(countdownInterval);
                countdownDisplay.textContent = "หมดเวลา!";
                countdownDisplay.style.color = "red";
                uploadStatus.textContent = "คุณไม่ได้อัปโหลดสลิปภายในเวลาที่กำหนด ระบบยกเลิกคำสั่งซื้อแล้ว";
                uploadBtn.disabled = true;
                uploadBtn.style.backgroundColor = "#ccc";
                uploadBtn.style.cursor = "not-allowed";
                sessionStorage.removeItem("qr_code_url");
                sessionStorage.removeItem("countdown_start_time");
            }
        } else {
            // ถ้าไม่มีเวลาเริ่มต้น — ให้แสดงข้อความแจ้ง
            countdownDisplay.textContent = "ไม่มีเวลาเหลือ";
            countdownDisplay.style.color = "gray";
        }
    }

    // ============ ฟังก์ชันอัปโหลดสลิป ============
    if (uploadBtn && uploadStatus && receiptImage && apiKeyDisplay && spinner) {
        uploadBtn.addEventListener('click', async function () {
            const file = receiptImage.files[0];
            if (!file) {
                uploadStatus.textContent = 'กรุณาเลือกไฟล์ใบเสร็จ';
                return;
            }

            const token = localStorage.getItem('token');
            if (!token) {
                uploadStatus.textContent = '⚠️ กรุณาเข้าสู่ระบบก่อน';
                return;
            }

            // ✅ เริ่มโหลด — แสดง spinner + ซ่อนปุ่ม
            container.classList.add('loading');
            uploadStatus.textContent = 'กำลังประมวลผล...';
            apiKeyDisplay.textContent = '';

            const formData = new FormData();
            formData.append('receipt', file);

            try {
                const response = await fetch(`${window.API_BASE_URL}/upload-receipt`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`
                    },
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    uploadStatus.textContent = '✅ อัปโหลดสำเร็จ!';
                    apiKeyDisplay.textContent = data.api_key;
                    // ลบเวลาเมื่ออัปโหลดสำเร็จ
                    sessionStorage.removeItem("countdown_start_time");
                    sessionStorage.removeItem("qr_code_url");
                } else {
                    uploadStatus.textContent = '❌ เกิดข้อผิดพลาด: ' + (data.error || 'ไม่ทราบสาเหตุ');
                }
            } catch (error) {
                console.error('เกิดข้อผิดพลาด:', error);
                uploadStatus.textContent = "ควย5555";
            } finally {
                // ✅ จบโหลด — ซ่อน spinner + แสดงปุ่ม
                container.classList.remove('loading');
            }
        });
    }
});