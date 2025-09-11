function generateApiKey() {
    // ดึง token แทน email
    let token = localStorage.getItem('token');
    if (!token) {
        alert("กรุณาล็อกอินก่อน");
        return;
    }

    let analysisTypes = [];
    let thresholds = {};
    document.querySelectorAll('.analysis-option:checked').forEach(option => {
        analysisTypes.push(option.value);
        let threshold = document.getElementById(option.value + '-threshold').value;
        thresholds[option.value] = parseFloat(threshold);
    });

    let quota = document.getElementById("quota").value;

    if (analysisTypes.length === 0 || !quota || isNaN(quota) || quota <= 0) {
        alert("กรุณากรอกข้อมูลทั้งหมดให้ถูกต้อง");
        return;
    }

    fetch(`${window.API_BASE_URL}/request-api-key`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}` // <<-- แนบ token แทน email
        },
        body: JSON.stringify({
            analysis_types: analysisTypes,
            thresholds: thresholds,
            quota: quota,
            plan: 'free'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else if (data.apiKey) {
            document.getElementById("apiKey").textContent = data.apiKey;
        } else {
            alert("เกิดข้อผิดพลาดในการสร้าง API Key");
        }
    })
    .catch(error => {
        console.error("เกิดข้อผิดพลาดในการเชื่อมต่อ:", error);
        alert("เกิดข้อผิดพลาดในการเชื่อมต่อกับ server");
    });
}
