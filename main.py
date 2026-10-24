
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import shutil
import os
import re
import uuid
import sqlite3
import time
import asyncio
import concurrent.futures
import logging
import unicodedata
import json
import io
import math
import bcrypt
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image

import cv2
import numpy as np

from pipeline_MediapipeInceptionV3 import (
    process_attendance,
    register_student_face_from_images,
    load_face_bank_to_ram,
    GLOBAL_FACE_BANK,
    GLOBAL_STUDENT_MAP,
    remove_vietnamese_accents,
    reset_realtime_tracker,
    PIPELINE_CONFIG
)
import security as sec
from security import (
    NotAuthenticated, Forbidden, current_user, require_teacher, require_student, require_admin,
    RateLimiter, RealtimeGate, BodyLimitMiddleware
)

# ==========================================
# CẤU HÌNH LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)d] | %(message)s",
    handlers=[
        logging.FileHandler("system_events.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Tắt /docs, /redoc, /openapi.json: không để lộ danh sách API cho người lạ
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("database/registered_faces", exist_ok=True)
os.makedirs("uploads/temp_chunks", exist_ok=True)
os.makedirs("uploads/detected_faces", exist_ok=True)

# ==========================================
# MIDDLEWARE (add sau = chạy ngoài cùng)
# ==========================================
app.add_middleware(
    SessionMiddleware,
    secret_key=sec.load_secret_key(),
    session_cookie="attendance_session",
    max_age=sec.SESSION_MAX_AGE,
    same_site="lax",                 # chặn CSRF từ site khác với các POST
    https_only=sec.COOKIE_SECURE,    # đặt COOKIE_SECURE=1 khi chạy sau HTTPS
)
app.add_middleware(
    BodyLimitMiddleware,
    limits={
        "/api/realtime_stream": 12 * 1024 * 1024,   # batch 10 frame độ phân giải cao (<= 1920px)
        "/api/diem_danh/upload_chunk": 6 * 1024 * 1024,      # chunk 5MB + phần form
        "/api/dang_ky_khuon_mat": 25 * 1024 * 1024,
        "/api/upload_realtime_class_list": 6 * 1024 * 1024,
    },
    default=1 * 1024 * 1024,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.exception_handler(NotAuthenticated)
async def not_authenticated_handler(request: Request, exc: NotAuthenticated):
    if request.url.path.startswith(("/api/", "/uploads/", "/faces/")):
        return JSONResponse({"success": False, "detail": "Chưa đăng nhập hoặc phiên đã hết hạn."}, status_code=401)
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@app.exception_handler(Forbidden)
async def forbidden_handler(request: Request, exc: Forbidden):
    return JSONResponse({"success": False, "detail": "Bạn không có quyền truy cập."}, status_code=403)


# ==========================================
# GIỚI HẠN TỐC ĐỘ / TÀI NGUYÊN
# ==========================================
# Cả lớp thường dùng chung 1 IP (wifi trường) nên hạn mức theo IP phải rộng hơn hạn mức theo tài khoản
login_ip_limiter = RateLimiter(max_calls=40, per_seconds=300)    # 40 lần SAI / 5 phút / IP
login_user_limiter = RateLimiter(max_calls=8, per_seconds=300)   # 8 lần SAI / 5 phút / tài khoản
register_limiter = RateLimiter(max_calls=60, per_seconds=3600)   # 60 lần đăng ký / giờ / IP
face_reg_limiter = RateLimiter(max_calls=10, per_seconds=3600)  # 10 lần đăng ký mặt / giờ / user
rt_limiter = RateLimiter(max_calls=15, per_seconds=1)           # ~15 request/giây / user
chunk_limiter = RateLimiter(max_calls=60, per_seconds=60)
realtime_gate = RealtimeGate(max_users=sec.MAX_REALTIME_USERS)
_analysis_running = 0


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


MAX_WORKERS = os.cpu_count() or 4
ai_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
realtime_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)


@app.on_event("startup")
async def startup_event():
    logger.info("Khởi động Server. Đang nạp RAM Cache...")
    load_face_bank_to_ram()
    if not sec.TEACHER_INVITE_CODE:
        logger.warning("TEACHER_INVITE_CODE chưa đặt: đăng ký tài khoản giảng viên đang bị KHÓA.")
    if not sec.COOKIE_SECURE:
        logger.warning("COOKIE_SECURE=0: cookie phiên sẽ gửi được qua HTTP thường. Hãy đặt =1 khi chạy HTTPS.")


DB_PATH = "database/users.db"
DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt()).decode()   # chống dò tài khoản qua thời gian phản hồi


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        fullname TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()
init_db()


def login_error(request, msg, code=200):
    return templates.TemplateResponse(request=request, name="login.html", context={"error": msg}, status_code=code)


# ==========================================
# ĐĂNG NHẬP / ĐĂNG KÝ / ĐĂNG XUẤT
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    user = request.session.get("user")
    if user:
        return RedirectResponse(url="/teacher" if user["role"] == "teacher" else "/student",
                                status_code=status.HTTP_303_SEE_OTHER)
    return login_error(request, "")


@app.post("/login")
async def handle_login(request: Request, username: str = Form(...), password: str = Form(...)):
    ip_key, user_key = f"ip:{client_ip(request)}", f"user:{username.strip().lower()}"
    if login_ip_limiter.blocked(ip_key) or login_user_limiter.blocked(user_key):
        return login_error(request, "Thử đăng nhập quá nhiều lần. Vui lòng đợi vài phút.", 429)

    username = username.strip()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password, role, fullname FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    conn.close()

    ok = False
    try:
        if user:
            ok = bcrypt.checkpw(password.encode("utf-8"), user[0].encode("utf-8"))
        else:
            bcrypt.checkpw(password.encode("utf-8"), DUMMY_HASH.encode())   # tốn thời gian như thật
    except ValueError:   # mật khẩu > 72 byte
        ok = False

    if ok:
        _, role, fullname = user
        login_user_limiter.clear(user_key)
        request.session.clear()                  # chống session fixation
        request.session["user"] = {"username": username, "role": role, "fullname": fullname}
        logger.info(f"ĐĂNG NHẬP: {role} '{fullname}' (tài khoản: {username}) đã đăng nhập.")
        target = "/teacher" if role == "teacher" else "/student"
        return RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)

    login_ip_limiter.hit(ip_key)
    login_user_limiter.hit(user_key)
    return login_error(request, "Sai tài khoản hoặc mật khẩu!")


@app.post("/logout")
async def handle_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/register_user")
async def handle_register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    fullname: str = Form(...),
    role: str = Form(...),
    confirm_password: Optional[str] = Form(None),
    invite_code: str = Form("")
):
    if not register_limiter.allow(client_ip(request)):
        return login_error(request, "Bạn đăng ký quá nhiều lần. Vui lòng thử lại sau.", 429)

    username = username.strip()
    fullname = unicodedata.normalize("NFC", fullname.strip())

    if role not in ("student", "teacher"):
        return login_error(request, "Vai trò không hợp lệ!")
    if role == "teacher":
        # Trước đây ai cũng tự đăng ký được quyền giảng viên -> nay cần mã mời
        import secrets
        if not sec.TEACHER_INVITE_CODE or not secrets.compare_digest(invite_code, sec.TEACHER_INVITE_CODE):
            return login_error(request, "Mã mời giảng viên không đúng hoặc đăng ký giảng viên đang bị khóa.")
        if not sec.TEACHER_ID_RE.fullmatch(username):
            return login_error(request, "Tên tài khoản GV chỉ gồm chữ, số, . _ - (3-30 ký tự).")
    else:
        if not sec.STUDENT_ID_RE.fullmatch(username):
            return login_error(request, "Mã số sinh viên chỉ được chứa số!")
    if not sec.FULLNAME_RE.fullmatch(fullname):
        return login_error(request, "Họ tên chỉ gồm chữ cái, khoảng trắng và dấu gạch ngang (2-60 ký tự).")
    if confirm_password is not None and password != confirm_password:
        return login_error(request, "Mật khẩu xác nhận không trùng khớp!")
    if len(password) < 8 or len(password.encode("utf-8")) > 72:
        return login_error(request, "Mật khẩu phải từ 8 ký tự và không quá 72 byte.")

    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, role, fullname) VALUES (?, ?, ?, ?)",
                       (username, hashed_password, role, fullname))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return login_error(request, "Tài khoản đã tồn tại!")
    conn.close()
    logger.info(f"ĐĂNG KÝ: tài khoản {role} '{username}' được tạo.")
    return login_error(request, "Đăng ký thành công! Hãy đăng nhập.")


# ==========================================
# TRANG DASHBOARD (danh tính lấy từ SESSION, không còn từ query string)
# ==========================================
@app.get("/teacher", response_class=HTMLResponse)
async def teacher_dashboard(request: Request, user: dict = Depends(require_teacher)):
    # Thông số detect realtime lấy từ PIPELINE_CONFIG["realtime"] (nguồn duy nhất) để frontend không bị lệch cấu hình
    rt = PIPELINE_CONFIG["realtime"]
    rt_cfg = {k: rt[k] for k in ("det_thresh", "scale", "detect_max_side", "grid", "padding_ratio",
                                 "nms_iou", "full_frame_pass", "edge_margin_px")}
    return templates.TemplateResponse(request=request, name="teacher_dashboard.html",
                                      context={"name": user["fullname"], "rt_cfg": rt_cfg})


@app.get("/student", response_class=HTMLResponse)
async def student_dashboard(request: Request, user: dict = Depends(require_student)):
    mssv, name = user["username"], user["fullname"]
    is_registered = False
    registered_images = []

    safe_name = remove_vietnamese_accents(name)
    folder_name = f"{mssv}_{safe_name}"
    folder_path = sec.resolve_inside(os.path.join("database", "registered_faces"), folder_name)

    if folder_path and folder_path.is_dir():
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                registered_images.append(f"/faces/{folder_name}/{file}?t={int(time.time())}")
        if len(registered_images) > 0:
            is_registered = True
            registered_images.sort()

    if not is_registered and GLOBAL_FACE_BANK is not None and not GLOBAL_FACE_BANK.empty:
        db_mssv = GLOBAL_FACE_BANK['MSSV'].astype(str).str.replace(r'\.0$', '', regex=True).values
        if str(mssv).strip() in db_mssv:
            is_registered = True

    return templates.TemplateResponse(request=request, name="student_dashboard.html", context={
        "mssv": mssv, "name": name,
        "is_registered": is_registered,
        "registered_images": registered_images
    })


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, user: dict = Depends(require_admin)):
    logs = []
    log_path = "system_events.log"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            logs = f.readlines()
    logs.reverse()
    return templates.TemplateResponse(request=request, name="admin_dashboard.html", context={"logs": logs[:200]})


# ==========================================
# PHỤC VỤ ẢNH (thay cho app.mount công khai)
# ==========================================
@app.get("/uploads/{subpath:path}")
async def serve_upload(subpath: str, user: dict = Depends(require_teacher)):
    """Chỉ giáo viên đã đăng nhập, chỉ file ảnh, không đụng vào temp_chunks."""
    if subpath.split("/")[0] == "temp_chunks":
        raise HTTPException(status_code=404)
    p = sec.resolve_inside("uploads", subpath)
    if not p or not p.is_file() or p.suffix.lower() not in sec.IMAGE_EXTS:
        raise HTTPException(status_code=404)
    return FileResponse(p, headers={"Cache-Control": "private, max-age=300"})


@app.get("/faces/{subpath:path}")
async def serve_face(subpath: str, user: dict = Depends(current_user)):
    """Giáo viên xem được tất cả; sinh viên chỉ xem được thư mục khuôn mặt của chính mình."""
    parts = subpath.split("/")
    if len(parts) != 2:
        raise HTTPException(status_code=404)
    folder, _ = parts
    if user["role"] == "student" and not folder.startswith(f"{user['username']}_"):
        raise HTTPException(status_code=403)
    p = sec.resolve_inside(os.path.join("database", "registered_faces"), subpath)
    if not p or not p.is_file() or p.suffix.lower() not in sec.IMAGE_EXTS:
        raise HTTPException(status_code=404)
    return FileResponse(p, headers={"Cache-Control": "private, max-age=300"})


# ==========================================
# UPLOAD CHUNK (whitelist đuôi, kiểm tra magic byte, giới hạn dung lượng, tên file do server đặt)
# ==========================================
def chunk_path(session_id: str, file_type: str, filename: str):
    allowed = sec.MEDIA_EXTS if file_type == "media" else sec.EXCEL_EXTS
    ext = sec.safe_ext(filename, allowed)
    if not sec.SESSION_ID_RE.fullmatch(session_id) or file_type not in ("media", "excel") or not ext:
        return None, None
    return f"uploads/temp_chunks/{session_id}_{file_type}{ext}", ext


@app.post("/api/diem_danh/upload_chunk")
async def upload_attendance_chunk(
    session_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file_type: str = Form(...),
    original_filename: str = Form(...),
    chunk_file: UploadFile = File(...),
    user: dict = Depends(require_teacher)
):
    if not chunk_limiter.allow(user["username"]):
        raise HTTPException(status_code=429, detail="Gửi quá nhanh, vui lòng thử lại.")

    path, ext = chunk_path(session_id, file_type, original_filename)
    if not path:
        raise HTTPException(status_code=400, detail="Tên phiên, loại file hoặc định dạng file không hợp lệ.")
    if not (1 <= total_chunks <= sec.MAX_CHUNKS) or not (0 <= chunk_index < total_chunks):
        raise HTTPException(status_code=400, detail="Số thứ tự chunk không hợp lệ.")

    chunk_bytes = await chunk_file.read(sec.CHUNK_BYTES + 1)
    if len(chunk_bytes) > sec.CHUNK_BYTES:
        raise HTTPException(status_code=413, detail="Chunk quá lớn.")

    limit = sec.MAX_MEDIA_BYTES if file_type == "media" else sec.MAX_EXCEL_BYTES
    offset = chunk_index * sec.CHUNK_BYTES
    if offset + len(chunk_bytes) > limit:
        raise HTTPException(status_code=413, detail="File vượt dung lượng tối đa cho phép.")

    if chunk_index == 0:
        if not sec.magic_ok(ext, chunk_bytes[:16]):
            raise HTTPException(status_code=400, detail="Nội dung file không khớp với định dạng khai báo.")
        mode = "wb"
    else:
        if not os.path.exists(path) or offset > os.path.getsize(path):
            raise HTTPException(status_code=409, detail="Chunk gửi sai thứ tự.")
        mode = "r+b"

    # Ghi theo offset -> gửi lại cùng 1 chunk (retry) không làm hỏng file
    with open(path, mode) as f:
        f.seek(offset)
        f.write(chunk_bytes)
        if chunk_index == total_chunks - 1:
            f.truncate()
    return {"success": True, "session_id": session_id, "chunk_index": chunk_index,
            "is_last": (chunk_index == total_chunks - 1)}


@app.post("/api/diem_danh/finalize")
async def finalize_attendance(
    session_id: str = Form(...),
    excel_filename: Optional[str] = Form(None),
    media_filename: str = Form(...),
    user: dict = Depends(require_teacher)
):
    global _analysis_running
    temp_media, media_ext = chunk_path(session_id, "media", media_filename)
    if not temp_media or not os.path.exists(temp_media):
        raise HTTPException(status_code=400, detail="Không tìm thấy file media đã tải lên.")

    if _analysis_running >= sec.MAX_CONCURRENT_ANALYSIS:
        raise HTTPException(status_code=429, detail="Máy chủ đang bận phân tích. Vui lòng thử lại sau ít phút.")

    _analysis_running += 1
    try:
        timestamp = int(time.time())
        uid = uuid.uuid4().hex[:8]
        final_media_path = f"uploads/media_{timestamp}_{uid}{media_ext}"   # tên do server đặt, giữ đuôi whitelist
        shutil.move(temp_media, final_media_path)

        final_excel_path = None
        if excel_filename and excel_filename.strip():
            temp_excel, excel_ext = chunk_path(session_id, "excel", excel_filename)
            if temp_excel and os.path.exists(temp_excel):
                final_excel_path = f"uploads/danh_sach_{timestamp}_{uid}{excel_ext}"
                shutil.move(temp_excel, final_excel_path)

        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(ai_pool, process_attendance, final_media_path, final_excel_path)

        return {
            "success": True,
            "excel_url": f"/api/download_excel?filename={os.path.basename(res['excel_path'])}",
            "media_url": f"/{res['media_result_path']}",
            "is_image": res["is_image"],
            "stats": {
                "total": res["total"],
                "present": res["present"],
                "absent": res["absent"],
                "unregistered": res["unregistered"],
                "pct_present": res["pct_present"],
                "pct_absent": res["pct_absent"],
                "pct_unregistered": res["pct_unregistered"]
            },
            "detected_crops": res["detected_crops"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LỖI PHÂN TÍCH: {e}")
        raise HTTPException(status_code=500, detail="Lỗi khi phân tích. Vui lòng kiểm tra lại file.")
    finally:
        _analysis_running -= 1


# ==========================================
# DANH SÁCH LỚP + XUẤT EXCEL REALTIME
# ==========================================
@app.post("/api/upload_realtime_class_list")
async def upload_realtime_class_list(excel_file: UploadFile = File(...), user: dict = Depends(require_teacher)):
    try:
        import pandas as pd
        ext = sec.safe_ext(excel_file.filename, sec.EXCEL_EXTS)
        if not ext:
            return {"success": False, "error": "Chỉ chấp nhận file .xls hoặc .xlsx."}
        contents = await excel_file.read(sec.MAX_EXCEL_BYTES + 1)
        if len(contents) > sec.MAX_EXCEL_BYTES:
            return {"success": False, "error": "File Excel quá lớn (tối đa 5MB)."}
        if not sec.magic_ok(ext, contents[:16]):
            return {"success": False, "error": "Nội dung file không phải Excel hợp lệ."}

        # Xóa file cũ đi để giải phóng dung lượng
        for f in os.listdir("database"):
            if f.startswith("current_realtime_class_list."):
                os.remove(os.path.join("database", f))

        tmp_excel = f"database/current_realtime_class_list{ext}"      # đuôi lấy từ whitelist, không từ client
        with open(tmp_excel, "wb") as f:
            f.write(contents)

        try:
            df_raw = pd.read_excel(tmp_excel, header=None)
        except Exception:
            df_raw = pd.read_excel(tmp_excel, header=None, engine='xlrd')

        header_row_idx = None
        for idx, row in df_raw.iterrows():
            if row.astype(str).str.contains('MSSV|Mã', case=False, na=False).any():
                header_row_idx = idx
                break

        if header_row_idx is None:
            raise ValueError("Không tìm thấy cột 'MSSV'.")

        df_list = pd.read_excel(tmp_excel, header=header_row_idx)
        mssv_col = [c for c in df_list.columns if 'MSSV' in str(c).upper() or 'MÃ' in str(c).upper()][0]
        mssv_list = df_list[mssv_col].dropna().astype(str).str.replace(r'\.0$', '', regex=True).str.strip().tolist()
        mssv_list = [m for m in mssv_list if m.isalnum()]

        return {"success": True, "mssv_list": mssv_list, "total": len(mssv_list)}
    except Exception as e:
        logger.error(f"LỖI NẠP DANH SÁCH LỚP: {e}")
        return {"success": False, "error": "Không đọc được file Excel. Hãy kiểm tra định dạng."}


class ExportRealtimeRequest(BaseModel):
    present_data: dict
    use_class_list: bool = False


def sanitize_present(raw: dict) -> dict:
    """Chỉ giữ cặp {MSSV hợp lệ: 'HH:MM:SS'} -> chặn rác, chuỗi dài và công thức Excel (=, +, @...)."""
    out = {}
    for k, v in list(raw.items())[:2000]:
        if isinstance(k, str) and isinstance(v, str) and sec.MSSV_RE.fullmatch(k) and sec.TIME_RE.fullmatch(v):
            out[k] = v
    return out


@app.post("/api/export_realtime_excel")
async def export_realtime_excel(req: ExportRealtimeRequest, user: dict = Depends(require_teacher)):
    import pandas as pd
    import pipeline_MediapipeInceptionV3 as pipe

    present_dict = sanitize_present(req.present_data)
    use_class_list = req.use_class_list

    def clean_mssv(val):
        if pd.isna(val) or str(val).strip() == "": return ""
        return str(val).replace('.0', '').strip()

    file_path = None
    if use_class_list:
        for f in os.listdir("database"):
            if f.startswith("current_realtime_class_list."):
                file_path = os.path.join("database", f)
                break

    if not use_class_list or not file_path:
        # KỊCH BẢN A: Không dùng danh sách lớp -> Lấy toàn bộ FaceBank
        if not pipe.GLOBAL_STUDENT_MAP:
            pipe.load_face_bank_to_ram()

        all_mssvs = list(pipe.GLOBAL_STUDENT_MAP.keys())
        data_rows = []
        for m in all_mssvs:
            status_txt = "✅ Có mặt" if m in present_dict else "❌ Vắng mặt"
            time_detected = present_dict[m] if m in present_dict else ""
            name = pipe.GLOBAL_STUDENT_MAP[m]["HoTen"]
            data_rows.append({"MSSV": m, "Họ và Tên": name, "Trạng thái AI": status_txt, "Thời gian": time_detected})

        df_list = pd.DataFrame(data_rows)
        out_name = f"KetQua_Realtime_ALL_{int(time.time())}_{uuid.uuid4().hex[:6]}.xlsx"
        df_list.to_excel(f"uploads/{out_name}", index=False)
        return {"success": True, "excel_url": f"/api/download_excel?filename={out_name}"}

    # KỊCH BẢN B: Có danh sách lớp tải lên
    try:
        try:
            df_raw = pd.read_excel(file_path, header=None)
        except Exception:
            df_raw = pd.read_excel(file_path, header=None, engine='xlrd')

        header_row_idx = None
        for idx, row in df_raw.iterrows():
            if row.astype(str).str.contains('MSSV|Mã', case=False, na=False).any():
                header_row_idx = idx
                break

        if header_row_idx is None:
            return {"success": False, "error": "Không tìm thấy cột 'MSSV' trong file."}

        df_list = pd.read_excel(file_path, header=header_row_idx)
        mssv_col = [c for c in df_list.columns if 'MSSV' in str(c).upper() or 'MÃ' in str(c).upper()][0]

        if not pipe.GLOBAL_STUDENT_MAP:
            pipe.load_face_bank_to_ram()

        def get_realtime_status(mssv_val):
            m_str = clean_mssv(mssv_val)
            if not m_str: return ""
            if m_str in present_dict:
                return "✅ Có mặt"
            elif m_str not in pipe.GLOBAL_STUDENT_MAP:
                return "⚠️ Chưa có dữ liệu mặt"
            else:
                return "❌ Vắng mặt"

        def get_realtime_time(mssv_val):
            m_str = clean_mssv(mssv_val)
            if not m_str: return ""
            return present_dict.get(m_str, "")

        df_list['Trạng thái AI'] = df_list[mssv_col].apply(get_realtime_status)
        df_list['Thời điểm phát hiện'] = df_list[mssv_col].apply(get_realtime_time)
        df_list = df_list.fillna('')

        out_name = f"KetQua_Realtime_CLASS_{int(time.time())}_{uuid.uuid4().hex[:6]}.xlsx"
        df_list.to_excel(f"uploads/{out_name}", index=False)
        return {"success": True, "excel_url": f"/api/download_excel?filename={out_name}"}
    except Exception as e:
        logger.error(f"LỖI XUẤT EXCEL: {e}")
        return {"success": False, "error": "Không xuất được file Excel."}


# ==========================================
# REALTIME
# ==========================================
MAX_RT_FACES_PER_FRAME = 40          # số mặt tối đa / khung
MAX_RT_TOTAL_CROPS = 400             # số crop tối đa / request
MAX_RT_CROP_BYTES = 64 * 1024        # crop 160x160 JPEG thường ~5-15KB
MAX_RT_META_BYTES = 256 * 1024


def parse_realtime_meta(meta_str: str, n_crops: int):
    """Kiểm tra metadata JSON do frontend gửi. Trả list khung đã chuẩn hóa hoặc None nếu sai định dạng."""
    if len(meta_str) > MAX_RT_META_BYTES:
        return None
    try:
        data = json.loads(meta_str)
    except ValueError:
        return None
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, list) or not (1 <= len(frames) <= sec.MAX_RT_FRAMES):
        return None

    def is_int(v):
        return isinstance(v, int) and not isinstance(v, bool)

    out, used = [], set()
    for fr in frames:
        if not isinstance(fr, dict):
            return None
        w, h, seq = fr.get("w"), fr.get("h"), fr.get("seq", 0)
        if not (is_int(w) and is_int(h) and 1 <= w <= sec.MAX_RT_SIDE and 1 <= h <= sec.MAX_RT_SIDE and is_int(seq)):
            return None
        faces = fr.get("faces", [])
        if not isinstance(faces, list) or len(faces) > MAX_RT_FACES_PER_FRAME:
            return None
        clean = []
        for f in faces:
            if not isinstance(f, dict):
                return None
            box, c = f.get("box"), f.get("c", -1)
            if not (isinstance(box, list) and len(box) == 4
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)):
                return None
            x1, y1, x2, y2 = (int(v) for v in box)
            if not (0 <= x1 <= w and 0 <= x2 <= w and 0 <= y1 <= h and 0 <= y2 <= h):
                return None
            if not is_int(c) or not (-1 <= c < n_crops):
                return None
            if c >= 0:
                if c in used:            # mỗi crop chỉ thuộc về 1 khuôn mặt
                    return None
                used.add(c)
            clean.append({"box": [x1, y1, x2, y2], "c": c})
        out.append({"seq": seq, "w": w, "h": h, "faces": clean})
    return out


def decode_face_crop(data: bytes):
    """Crop mặt từ frontend: JPEG thật, kích thước nhỏ. Trả ndarray RGB uint8 (đúng thứ tự kênh pipeline đang dùng) hoặc None."""
    if not data or len(data) > MAX_RT_CROP_BYTES:
        return None
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format != "JPEG" or not (16 <= im.width <= 512 and 16 <= im.height <= 512):
                return None
            return np.array(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


@app.post("/api/realtime_stream")
async def do_realtime_stream(
    meta: str = Form(...),
    crops: List[UploadFile] = File(default=[]),
    mssv_list_str: str = Form(""),
    client_send_time: str = Form("0"),  # Tham số mới
    user: dict = Depends(require_teacher)
):
    # ================= CODE KIỂM TRA DỮ LIỆU FRONTEND =================
    debug_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:4]}"
    
    # 1. Lưu toàn bộ chuỗi siêu dữ liệu (meta) ra file .json
    meta_path = f"uploads/detected_faces/debug_meta_{debug_id}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(meta)
        
    # 2. Tính toán và in ra dung lượng thực tế
    meta_size_kb = len(meta.encode('utf-8')) / 1024
    print(f"\n📦 --- GÓI DỮ LIỆU ĐẾN ---")
    print(f"📄 Chuỗi Meta (Toạ độ/JSON) : {meta_size_kb:.2f} KB (Đã lưu vào {meta_path})")
    print(f"🖼️ Tổng số lượng ảnh (Crops): {len(crops)} file")
    # Chốt thời gian ngay khi API bắt đầu xử lý
    recv_time = int(time.time() * 1000)
    username = user["username"]
    if not rt_limiter.allow(username):
        return JSONResponse({"success": False, "error": "Gửi khung hình quá nhanh."}, status_code=429)

    gate = realtime_gate.acquire(username)
    if gate == "busy":
        return JSONResponse({"success": False, "error": "Đang có giảng viên khác quét realtime."}, status_code=429)
    if gate == "inflight":
        return JSONResponse({"success": False, "error": "Yêu cầu trước vẫn đang xử lý."}, status_code=429)

    try:
        from pipeline_MediapipeInceptionV3 import process_realtime_faces_batch

        if len(crops) > MAX_RT_TOTAL_CROPS:
            return JSONResponse({"success": False, "error": "Quá nhiều khuôn mặt trong 1 yêu cầu."}, status_code=400)
        if len(mssv_list_str) > 20000:
            return JSONResponse({"success": False, "error": "Danh sách MSSV quá dài."}, status_code=400)

        frames_meta = parse_realtime_meta(meta, len(crops))
        if frames_meta is None:
            return JSONResponse({"success": False, "error": "Metadata khung hình không hợp lệ."}, status_code=400)

        crops_rgb, crops_raw = [], []
        for idx, f in enumerate(crops):
            data = await f.read(MAX_RT_CROP_BYTES + 1)
            
            # --- CODE THÊM VÀO ĐỂ LƯU ẢNH FRONTEND GỬI LÊN ---
            if data and len(data) <= MAX_RT_CROP_BYTES:
                # Tạo tên file có chứa timestamp và index để không bị trùng
                debug_filename = f"uploads/detected_faces/rt_front_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}_{idx}.jpg"
                with open(debug_filename, "wb") as debug_file:
                    debug_file.write(data)
            # ------------------------------------------------
            
            crops_rgb.append(decode_face_crop(data))
            crops_raw.append(data if len(data) <= MAX_RT_CROP_BYTES else b"")

        # Crop hỏng -> vẫn giữ box cho tracker (như cũ: box có nhưng không crop được) nhưng không bỏ phiếu
        for fr in frames_meta:
            for face in fr["faces"]:
                if face["c"] >= 0 and crops_rgb[face["c"]] is None:
                    face["c"] = -1

        target_mssvs = [m.strip() for m in mssv_list_str.split(",") if m.strip()][:500]
        target_mssvs = [m for m in target_mssvs if sec.MSSV_RE.fullmatch(m)]

        loop = asyncio.get_running_loop()
        detected_faces, snapshot_tracks = await loop.run_in_executor(
            realtime_pool, process_realtime_faces_batch, frames_meta, crops_rgb, crops_raw, target_mssvs
        )
        # snapshot_b64 giữ lại (None) để tương thích; ảnh toàn cảnh do frontend vẽ từ snapshot_tracks
        return {"success": True, "detected_faces": detected_faces,
                "snapshot_b64": None, "snapshot_tracks": snapshot_tracks}
    except Exception as e:
        logger.error(f"LỖI REALTIME: {e}")
        return {"success": False, "error": "Lỗi xử lý khung hình."}
    finally:
        realtime_gate.release(username)


@app.post("/api/reset_realtime")
async def reset_realtime_state(user: dict = Depends(require_teacher)):
    reset_realtime_tracker()
    realtime_gate.leave(user["username"])      # nhả "chỗ" quét realtime cho người khác
    return {"success": True, "message": "Đã xóa trí nhớ Tracker."}


# ==========================================
# TẢI FILE EXCEL (chỉ nhận TÊN file, khớp mẫu KetQua_*.xlsx)
# ==========================================
@app.get("/api/download_excel")
async def download_excel(filename: str, user: dict = Depends(require_teacher)):
    name = os.path.basename(filename)
    if name != filename or not sec.EXPORT_NAME_RE.fullmatch(name):
        raise HTTPException(status_code=404, detail="File không tồn tại.")
    for base in ("uploads", "database"):       # pipeline lưu KetQua_AI_*.xlsx trong database/
        p = sec.resolve_inside(base, name)
        if p and p.is_file():
            return FileResponse(
                path=p, filename=name,
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    raise HTTPException(status_code=404, detail="File không tồn tại.")


# ==========================================
# ĐĂNG KÝ KHUÔN MẶT
# ==========================================
@app.post("/api/dang_ky_khuon_mat")
async def do_register_face(
    request: Request,
    mssv: str = Form(""),
    name: str = Form(""),
    files: List[UploadFile] = File(...),
    user: dict = Depends(current_user)
):
    if user["role"] == "student":
        mssv, name = user["username"], user["fullname"]       # sinh viên chỉ đăng ký được cho CHÍNH MÌNH
    else:
        name = unicodedata.normalize("NFC", name.strip())
        if not sec.MSSV_RE.fullmatch(mssv) or not sec.FULLNAME_RE.fullmatch(name):
            return {"success": False, "message": "MSSV hoặc họ tên không hợp lệ."}

    if not face_reg_limiter.allow(user["username"]):
        return JSONResponse({"success": False, "message": "Bạn đăng ký quá nhiều lần. Vui lòng thử lại sau."}, status_code=429)
    if not (1 <= len(files) <= 6):
        return {"success": False, "message": "Cần gửi tối đa 6 ảnh."}

    tmp_paths = []
    try:
        for file in files:
            content = await file.read(sec.MAX_FACE_IMG_BYTES + 1)
            if not sec.check_image_upload(content):
                return {"success": False, "message": "Ảnh không hợp lệ (chỉ JPG/PNG, tối đa 8MB/ảnh)."}
            path = f"uploads/tmp_{uuid.uuid4().hex}.jpg"
            with open(path, "wb") as buffer:
                buffer.write(content)
            tmp_paths.append(path)

        loop = asyncio.get_running_loop()
        success, message = await loop.run_in_executor(
            ai_pool, register_student_face_from_images, tmp_paths, mssv, name
        )
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"LỖI EKYC: {str(e)}")
        return {"success": False, "message": "Lỗi hệ thống AI nội bộ. Vui lòng thử lại."}
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)
