# Hệ Thống Điểm Danh AI (AI Attendance Dashboard)

Hệ thống điểm danh sinh viên tự động sử dụng MediaPipe (nhận diện khuôn mặt) và InceptionResnetV1 (trích xuất đặc trưng), kết hợp FastAPI cho backend.

## 1. Yêu cầu hệ thống
- Python 3.9 trở lên.

## 2. Cài đặt môi trường
Tạo môi trường ảo và cài đặt các thư viện cần thiết:
```bash
python -m venv venv
# Active môi trường ảo (Windows)
venv\Scripts\activate
# Active môi trường ảo (Linux/Mac)
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
3. Cấu hình hệ thống
Đổi tên tệp .env.example thành .env.

Thay đổi các thông số bên trong (như SESSION_SECRET, TEACHER_INVITE_CODE) để tăng cường bảo mật. Đặt COOKIE_SECURE=1 nếu chạy qua HTTPS.

4. Chuẩn bị Trọng số AI
Tải tệp trọng số tĩnh model_v3_p_r1_0_r2_6_r3_3_weights.pt (do dung lượng lớn nên không có sẵn trên kho lưu trữ) và đặt vào thư mục gốc của dự án.

Lưu ý: Tệp detector.tflite của MediaPipe sẽ được hệ thống tự động tải về trong lần chạy đầu tiên.

5. Khởi động máy chủ
Khởi chạy hệ thống bằng uvicorn:

Bash
uvicorn main:app --host 0.0.0.0 --port 8000
Nếu triển khai trên máy chủ sản xuất, hãy đổi 8000 thành cổng mong muốn.

Đặc biệt lưu ý: Để tính năng Camera Realtime trên trình duyệt hoạt động, máy chủ phải được cấu hình chứng chỉ HTTPS. Nếu chạy sau Nginx hoặc Cloudflare, hãy chạy lệnh với cờ proxy:

Bash
uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips