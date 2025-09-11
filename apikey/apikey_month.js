async function generateApiKey() {
    const token = localStorage.getItem('token');
    if (!token) {
        alert("กรุณาล็อกอินก่อน");
        return;
    }

    let analysisTypes = [];
    document.querySelectorAll('.analysis-option:checked').forEach(option => {
        analysisTypes.push(option.value);
    });

    let duration = parseInt(document.getElementById("duration").value); // <-- เปลี่ยนจาก quota
    if (analysisTypes.length === 0 || isNaN(duration) || duration < 1) {
        alert("กรุณากรอกข้อมูลทั้งหมดให้ถูกต้อง");
        return;
    }

    let thresholds = {};
    analysisTypes.forEach(type => {
        let slider = document.getElementById(type + "-threshold");
        if (slider) {
            thresholds[type] = parseFloat(slider.value);
        }
    });

    let amount = duration * 30;

    try {
        const response = await fetch(`${window.API_BASE_URL}/generate_qr`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({
                amount: amount,
                plan: "monthly",
                duration: duration, 
                analysis_types: analysisTypes,
                thresholds: thresholds
            })
        });

        const data = await response.json();

        if (!response.ok) {
            alert("เกิดข้อผิดพลาดในการสร้าง QR Code");
            return;
        }

        sessionStorage.setItem("selectedAmount", amount);
        sessionStorage.setItem("selectedAnalysis", JSON.stringify(analysisTypes));
        sessionStorage.setItem("selectedThresholds", JSON.stringify(thresholds));
        sessionStorage.setItem("qr_code_url", data.qr_code_url);
        sessionStorage.setItem("ref_code", data.ref_code);

        window.location.href = "payment.html";

    } catch (err) {
        console.error("Error:", err);
        alert("ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้");
    }
}
