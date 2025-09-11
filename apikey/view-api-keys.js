// ฟังก์ชันดึง token จาก URL (กรณี login ด้วย Google แล้ว redirect มากับ ?token=...)
(function () {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('token');
    if (token) {
        localStorage.setItem('token', token);

        // ลบ token จาก URL หลังเก็บ
        const newUrl = window.location.origin + window.location.pathname;
        window.history.replaceState({}, document.title, newUrl);
    }
})();

// ฟังก์ชันดึงชื่อผู้ใช้จาก token
async function fetchUsername() {
    const token = localStorage.getItem('token');
    if (!token) {
        document.getElementById("usernameDisplay").textContent = "⚠️ กรุณาเข้าสู่ระบบ";
        return;
    }

    try {
        const res = await fetch(`${window.API_BASE_URL}/get-username`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        const data = await res.json();
        if (res.ok && data.username) {
            document.getElementById("usernameDisplay").textContent = `👤 สวัสดีคุณ: ${data.username}`;
        } else {
            document.getElementById("usernameDisplay").textContent = `👤 ไม่พบชื่อผู้ใช้`;
        }
    } catch (err) {
        console.error("Error fetching username:", err);
        document.getElementById("usernameDisplay").textContent = `👤 ดึงชื่อผู้ใช้ไม่สำเร็จ`;
    }
}

// ฟังก์ชันในการดึงข้อมูล API Keys ของผู้ใช้
function fetchApiKeys() {
    const token = localStorage.getItem('token');
    if (!token) {
        document.getElementById("apiKeysList").innerHTML = "<p>⚠️ กรุณาเข้าสู่ระบบก่อน</p>";
        return;
    }

    fetch(`${window.API_BASE_URL}/get-api-keys`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("apiKeysList").innerHTML = `<p>${data.error}</p>`;
        } else {
            let apiKeysHtml = "";
            data.api_keys.forEach(key => {
                apiKeysHtml += `
                    <div class="api-key">
                        <p><strong>API Key:</strong> ${key.api_key}</p>
                        <p><strong>Analysis Types:</strong> ${key.analysis_types.join(", ")}</p>
                        <p><strong>Quota:</strong> ${key.quota === -1 ? 'ไม่จำกัดการใช้งาน' : key.quota}</p>
                        <p><strong>Threshold:</strong> ${Object.entries(key.thresholds).map(([type, val]) => `${type}: ${val}`).join(", ")}</p>
                        ${key.expires_at ? `<p><strong>Expires At:</strong> ${new Date(key.expires_at).toLocaleString("th-TH")}</p>` : ""}
                    </div>
                `;
            });
            document.getElementById("apiKeysList").innerHTML = apiKeysHtml;
        }
    })
    .catch(error => {
        console.error("Error fetching API keys:", error);
        document.getElementById("apiKeysList").innerHTML = "<p>เกิดข้อผิดพลาดในการดึงข้อมูล API Keys</p>";
    });
}

// โหลดข้อมูลเมื่อเปิดหน้า
window.onload = async function () {
    const token = localStorage.getItem('token');
    if (!token) {
        document.getElementById("usernameDisplay").textContent = "⚠️ กรุณาเข้าสู่ระบบ";
        return;
    }

    await fetchUsername();  // ✅ โหลดชื่อผู้ใช้
    fetchApiKeys();         // ✅ โหลด API keys
};

// ฟังก์ชันในการออกจากระบบ
function logout() {
    localStorage.removeItem('token'); // 🔒 ลบ token
    window.location.href = "../login-singup/login.html"; // 🔁 กลับไปหน้า login
}
