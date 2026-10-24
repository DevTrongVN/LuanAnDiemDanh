
"""
security.py - Các tiện ích bảo mật dùng chung cho main.py
(xác thực session, rate limit, kiểm tra file upload, chống path traversal, giới hạn body).
"""
import io
import os
import re
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

from PIL import Image
from starlette.requests import Request
from starlette.responses import JSONResponse

# ==========================================================
# CẤU HÌNH (đọc từ biến môi trường, có mặc định an toàn)
# ==========================================================
SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE", str(8 * 3600)))        # 8 giờ
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"                    # bật =1 khi chạy HTTPS
TEACHER_INVITE_CODE = os.getenv("TEACHER_INVITE_CODE", "")                # rỗng = khóa đăng ký GV
ADMIN_USERS = {u.strip() for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip()}

MAX_MEDIA_BYTES = int(os.getenv("MAX_MEDIA_BYTES", str(300 * 1024 * 1024)))   # 300MB
MAX_EXCEL_BYTES = int(os.getenv("MAX_EXCEL_BYTES", str(5 * 1024 * 1024)))     # 5MB
MAX_FACE_IMG_BYTES = 8 * 1024 * 1024
CHUNK_BYTES = 5 * 1024 * 1024        # PHẢI khớp CHUNK_SIZE trong teacher_dashboard.html
MAX_CHUNKS = 200

MAX_REALTIME_USERS = int(os.getenv("MAX_REALTIME_USERS", "1"))   # tracker là biến global -> 1 là đúng
MAX_CONCURRENT_ANALYSIS = int(os.getenv("MAX_CONCURRENT_ANALYSIS", "2"))
MAX_RT_FRAMES = 30
MAX_RT_FRAME_BYTES = 1536 * 1024   # frame realtime giờ là ảnh độ phân giải cao (<= 1920px)
MAX_RT_SIDE = 1920

MEDIA_EXTS = {".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png"}
EXCEL_EXTS = {".xls", ".xlsx"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ==========================================================
# VALIDATORS
# ==========================================================
STUDENT_ID_RE = re.compile(r"^[0-9]{1,15}$")
TEACHER_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
MSSV_RE = re.compile(r"^[0-9A-Za-z]{1,20}$")
FULLNAME_RE = re.compile(r"^[\w \-]{2,60}$")          # \w = chữ Unicode (tiếng Việt) + số + _
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
EXPORT_NAME_RE = re.compile(r"^KetQua_[A-Za-z0-9_]{1,60}\.xlsx$")
TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def safe_ext(filename, allowed):
    """Lấy đuôi file từ tên do client gửi; chỉ trả về nếu nằm trong whitelist."""
    ext = Path(os.path.basename(filename or "")).suffix.lower()
    return ext if ext in allowed else None


def magic_ok(ext, head: bytes) -> bool:
    """Kiểm tra byte đầu file có khớp đúng định dạng khai báo không (không tin đuôi file)."""
    if ext in (".jpg", ".jpeg"):
        return head[:3] == b"\xff\xd8\xff"
    if ext == ".png":
        return head[:8] == b"\x89PNG\r\n\x1a\n"
    if ext in (".mp4", ".mov"):
        return head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip")
    if ext == ".avi":
        return head[:4] == b"RIFF" and head[8:12] == b"AVI "
    if ext == ".xlsx":
        return head[:4] == b"PK\x03\x04"
    if ext == ".xls":
        return head[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    return False


def resolve_inside(base_dir, rel):
    """Ghép base_dir + rel; trả None nếu kết quả thoát ra ngoài base_dir (../, đường dẫn tuyệt đối...)."""
    try:
        base = Path(base_dir).resolve()
        target = (base / rel).resolve()
    except (ValueError, OSError):
        return None
    return target if target.is_relative_to(base) else None


def decode_jpeg_frame(data: bytes):
    """Kiểm tra header JPEG + kích thước TRƯỚC khi decode (chống decompression bomb). Trả ndarray BGR hoặc None."""
    import cv2
    import numpy as np
    if len(data) > MAX_RT_FRAME_BYTES:
        return None
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format != "JPEG" or im.width > MAX_RT_SIDE or im.height > MAX_RT_SIDE:
                return None
    except Exception:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def check_image_upload(data: bytes):
    """Ảnh đăng ký khuôn mặt: JPEG/PNG thật, kích thước hợp lý."""
    if len(data) > MAX_FACE_IMG_BYTES:
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.format in ("JPEG", "PNG") and im.width * im.height <= 40_000_000
    except Exception:
        return False


# ==========================================================
# SECRET KEY CHO SESSION
# ==========================================================
def load_secret_key(path="database/.session_secret"):
    env = os.getenv("SESSION_SECRET")
    if env:
        return env
    p = Path(path)
    if p.exists():
        return p.read_text().strip()
    key = secrets.token_urlsafe(48)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(key)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return key


# ==========================================================
# XÁC THỰC / PHÂN QUYỀN
# ==========================================================
class NotAuthenticated(Exception):
    pass


class Forbidden(Exception):
    pass


def current_user(request: Request):
    user = request.session.get("user")
    if not user:
        raise NotAuthenticated()
    return user


def require_teacher(request: Request):
    user = current_user(request)
    if user.get("role") != "teacher":
        raise Forbidden()
    return user


def require_student(request: Request):
    user = current_user(request)
    if user.get("role") != "student":
        raise Forbidden()
    return user


def require_admin(request: Request):
    user = require_teacher(request)
    if ADMIN_USERS and user["username"] not in ADMIN_USERS:
        raise Forbidden()
    return user


# ==========================================================
# RATE LIMIT (trong RAM, đủ cho 1 tiến trình uvicorn)
# ==========================================================
class RateLimiter:
    def __init__(self, max_calls, per_seconds):
        self.max_calls, self.per = max_calls, per_seconds
        self.hits = defaultdict(deque)

    def _prune(self, key, now):
        dq = self.hits[key]
        while dq and now - dq[0] > self.per:
            dq.popleft()
        if not dq:
            self.hits.pop(key, None)
            return self.hits[key]
        return dq

    def blocked(self, key):
        return len(self._prune(key, time.monotonic())) >= self.max_calls

    def hit(self, key):
        self.hits[key].append(time.monotonic())

    def allow(self, key):
        """Kiểm tra + ghi nhận 1 lần gọi. False nếu vượt hạn mức."""
        now = time.monotonic()
        if len(self._prune(key, now)) >= self.max_calls:
            return False
        self.hits[key].append(now)
        return True

    def clear(self, key):
        self.hits.pop(key, None)


class RealtimeGate:
    """Chỉ cho tối đa N người quét realtime cùng lúc và mỗi người chỉ 1 request đang xử lý."""

    def __init__(self, max_users=1, idle_seconds=15):
        self.max_users, self.idle = max_users, idle_seconds
        self.active = {}        # username -> lần hoạt động cuối
        self.inflight = set()

    def acquire(self, username):
        now = time.monotonic()
        for u in [u for u, t in self.active.items() if now - t > self.idle and u not in self.inflight]:
            del self.active[u]
        if username not in self.active and len(self.active) >= self.max_users:
            return "busy"
        if username in self.inflight:
            return "inflight"
        self.active[username] = now
        self.inflight.add(username)
        return "ok"

    def release(self, username):
        self.inflight.discard(username)

    def leave(self, username):
        self.inflight.discard(username)
        self.active.pop(username, None)


# ==========================================================
# GIỚI HẠN KÍCH THƯỚC BODY (ASGI middleware, chặn trước khi đọc dữ liệu)
# ==========================================================
class BodyLimitMiddleware:
    def __init__(self, app, limits, default=1 * 1024 * 1024):
        self.app, self.limits, self.default = app, limits, default

    def _limit_for(self, path):
        for prefix, lim in self.limits.items():
            if path.startswith(prefix):
                return lim
        return self.default

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT", "PATCH"):
            headers = {k.lower(): v for k, v in scope["headers"]}
            cl = headers.get(b"content-length")
            if cl is None and b"transfer-encoding" in headers:
                resp = JSONResponse({"success": False, "detail": "Thiếu Content-Length."}, status_code=411)
                return await resp(scope, receive, send)
            if cl is not None:
                try:
                    too_big = int(cl) > self._limit_for(scope["path"])
                except ValueError:
                    resp = JSONResponse({"success": False, "detail": "Content-Length không hợp lệ."}, status_code=400)
                    return await resp(scope, receive, send)
                if too_big:
                    resp = JSONResponse({"success": False, "detail": "Dữ liệu gửi lên quá lớn."}, status_code=413)
                    return await resp(scope, receive, send)
        await self.app(scope, receive, send)
