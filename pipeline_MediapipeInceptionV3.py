
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3' 
os.environ['PYTHONWARNINGS'] = 'ignore' 
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
import time
import threading
import urllib.request
import numpy as np
import pandas as pd
import mediapipe as mp
import csv
import unicodedata
import base64
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.ao.quantization as quantization
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# ==========================================
# 0. CẤU HÌNH TỐI ĐA NĂNG LỰC CPU
# ==========================================
total_cpus = os.cpu_count() or 4
torch.set_num_threads(total_cpus)
os.environ["OMP_NUM_THREADS"] = str(total_cpus)
os.environ["MKL_NUM_THREADS"] = str(total_cpus)

device = torch.device('cpu')
DETECTOR_MODEL_PATH = "detector.tflite"
SAVED_WEIGHTS_PATH = "model_v3_p_r1_0_r2_6_r3_3_weights.pt"
DB_CSV_PATH = "database/face_bank_v3.csv"
SAME_PERSON_THRESHOLD = 0.70

def check_and_download_detector():
    if not os.path.exists(DETECTOR_MODEL_PATH):
        print(f"⏳ Đang tải {DETECTOR_MODEL_PATH}...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            DETECTOR_MODEL_PATH
        )

check_and_download_detector()

# ==========================================
# 1. STRUCTURAL PRUNING & INT8 QUANTIZATION
# ==========================================
def prune_inception_v1(model, drop_repeat1=0, drop_repeat2=6, drop_repeat3=3):
    if drop_repeat1 > 0 and len(model.repeat_1) > drop_repeat1:
        model.repeat_1 = model.repeat_1[:-drop_repeat1]
    if drop_repeat2 > 0 and len(model.repeat_2) > drop_repeat2:
        model.repeat_2 = model.repeat_2[:-drop_repeat2]
    if drop_repeat3 > 0 and len(model.repeat_3) > drop_repeat3:
        model.repeat_3 = model.repeat_3[:-drop_repeat3]
    return model

def build_pruned_v3_int8_model():
    torch.backends.quantized.engine = 'fbgemm'
    base_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    base_model = prune_inception_v1(base_model, drop_repeat1=0, drop_repeat2=6, drop_repeat3=3)

    dummy_input = torch.randn(1, 3, 160, 160)
    qconfig_dict = {"": quantization.get_default_qconfig('fbgemm')}
    prepared_model = prepare_fx(base_model, qconfig_dict, example_inputs=(dummy_input,))

    if os.path.exists(SAVED_WEIGHTS_PATH):
        try:
            print(f"⚡ [Model] Đang nạp trọng số nhanh từ: {SAVED_WEIGHTS_PATH}")
            quantized_model = convert_fx(prepared_model)
            start_load = time.time()
            quantized_model.load_state_dict(torch.load(SAVED_WEIGHTS_PATH, map_location=device))
            load_time_ms = (time.time() - start_load) * 1000
            quantized_model.eval()
            print(f"✅ Nạp mô hình Static INT8 thành công! (Thời gian: {load_time_ms:.2f} ms)")
            return quantized_model
        except Exception as e:
            print(f"⚠️ Nạp state_dict thất bại ({e}). Đang tạo mới...")

    print("⏳ Đang thực hiện Calibrate nhanh...")
    with torch.no_grad():
        for _ in range(10):
            calib_batch = torch.randn(8, 3, 160, 160)
            prepared_model(calib_batch)

    quantized_model = convert_fx(prepared_model)
    quantized_model.eval()
    try:
        torch.save(quantized_model.state_dict(), SAVED_WEIGHTS_PATH)
        print(f"✅ Đã lưu trọng số vào {SAVED_WEIGHTS_PATH}")
    except Exception as e:
        print(f"⚠️ Lưu trọng số thất bại: {e}")

    return quantized_model

RECOGNITION_MODEL = build_pruned_v3_int8_model()

facenet_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
realtime_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# ==========================================
# CẤU HÌNH PIPELINE ĐỘC LẬP
# ==========================================
PIPELINE_CONFIG = {
    "image": {
        "det_thresh": 0.5,
        "rec_thresh": 0.70,
        "nms_iou": 0.20,
        "grid": (3, 3),
        "padding_ratio": 0.15,
        "full_frame_pass": True 
    },
    "video": {
        "det_thresh": 0.70,         # Tăng lên 0.7 để lọc mặt ma
        "rec_thresh": 0.70,
        "nms_iou": 0.30,            # Tăng lên 0.3 để gộp box mượt hơn
        "grid": (3, 3),             # Cấu trúc lưới 3x3 chuẩn
        "padding_ratio": 0.30,      # Đệm 30% để chống đứt đôi khuôn mặt
        "frame_skip": 5,            
        "voting_window_frames": 12, 
        "min_vote_ratio": 0.6       
    },
    "realtime": {
        "det_thresh": 0.6,
        "rec_thresh": 0.7,
        "scale": 0.6,               # chỉ dùng cho lượt quét toàn khung
        "detect_max_side": 640,     # cạnh dài của ảnh thu nhỏ dành riêng cho MediaPipe; crop mặt luôn lấy từ frame gốc
        "grid": (1, 1),
        "padding_ratio": 0,      # vùng chồng lấn giữa các tile 0.30
        "nms_iou": 0,           #0.30
        "full_frame_pass": True,    # bắt mặt lớn vắt ngang ranh giới
        "edge_margin_px": 4,
        "voting_window_frames": 10,
        "min_vote_ratio": 0.6,
        "max_disappeared": 4        # Tăng lên 4 theo lưu ý để chống rớt track
    }
}

options = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    min_detection_confidence=0.4 
)

# ==========================================
# 2. BỘ NHỚ RAM INDEXED (O(1) LOOKUP)
# ==========================================
GLOBAL_FACE_BANK = None
GLOBAL_STUDENT_MAP = {} 
_REALTIME_MATRIX_CACHE = {}
_thread_local = threading.local()

def get_thread_face_detector():
    detector = getattr(_thread_local, "face_detector", None)
    if detector is None:
        detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        _thread_local.face_detector = detector
    return detector

def load_face_bank_to_ram(db_csv_path=DB_CSV_PATH):
    global GLOBAL_FACE_BANK, GLOBAL_STUDENT_MAP
    _REALTIME_MATRIX_CACHE.clear()
    start_time = time.time()
    if os.path.exists(db_csv_path):
        try:
            GLOBAL_FACE_BANK = pd.read_csv(db_csv_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            try:
                GLOBAL_FACE_BANK = pd.read_csv(db_csv_path, encoding='latin-1')
            except Exception:
                GLOBAL_FACE_BANK = pd.read_csv(db_csv_path, encoding='ISO-8859-1', errors='ignore')

        GLOBAL_FACE_BANK['MSSV'] = GLOBAL_FACE_BANK['MSSV'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

        embedding_cols = [f"emb_{i}" for i in range(512)]
        GLOBAL_STUDENT_MAP = {}
        for mssv, group in GLOBAL_FACE_BANK.groupby('MSSV'):
            ho_ten = group['HoTen'].iloc[0]
            vectors = group[embedding_cols].to_numpy(dtype=np.float32)
            GLOBAL_STUDENT_MAP[mssv] = {
                "HoTen": ho_ten,
                "vectors": vectors
            }
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"⚡ [RAM Cache] Đã lập chỉ mục {len(GLOBAL_STUDENT_MAP)} sinh viên vào RAM mất: {elapsed_ms:.2f} ms.")
    else:
        GLOBAL_FACE_BANK = pd.DataFrame()
        GLOBAL_STUDENT_MAP = {}
        print("⚠ [RAM Cache] CSDL trống. Chờ nạp vector sinh viên.")

def remove_vietnamese_accents(text):
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text).replace(" ", "_")

# ==========================================
# 3. TIỆN ÍCH XỬ LÝ HÌNH ẢNH & EMBEDDING
# ==========================================
def calculate_iou(box_a, box_b):
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    
    if area_a == 0 or area_b == 0:
        return 0.0
        
    union = area_a + area_b - intersection
    iou = intersection / union
    
    min_area = min(area_a, area_b)
    iom = intersection / min_area if min_area > 0 else 0.0
    
    return max(iou, iom)

def non_max_suppression(boxes, scores, iou_threshold=0.20):
    if len(boxes) == 0: return []
    boxes, scores = np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        current = order[0]
        keep.append(current)
        remaining = [idx for idx in order[1:] if calculate_iou(boxes[current], boxes[idx]) <= iou_threshold]
        order = np.array(remaining, dtype=np.int64)
    return keep

def clamp_box(x1, y1, x2, y2, frame_w, frame_h):
    return max(0, min(int(x1), frame_w - 1)), max(0, min(int(y1), frame_h - 1)), \
           max(0, min(int(x2), frame_w - 1)), max(0, min(int(y2), frame_h - 1))

def extract_embedding_from_crop(crop_rgb):
    try:
        pil_img = Image.fromarray(crop_rgb)
        tensor = facenet_transform(pil_img).unsqueeze(0).contiguous()
        with torch.inference_mode():
            emb = RECOGNITION_MODEL(tensor)
            if hasattr(emb, 'is_quantized') and emb.is_quantized:
                emb = emb.dequantize()
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        return None

def process_tile_detection(tile_info, det_thresh):
    tile_img, x_start, y_start, w, h = tile_info
    if tile_img.size == 0: return []

    detector = get_thread_face_detector()
    rgb_tile = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
    detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_tile))

    results = []
    if detection_result.detections:
        for detection in detection_result.detections:
            score = float(detection.categories[0].score)
            if score < det_thresh:
                continue
            
            bbox = detection.bounding_box
            rx1 = x_start + bbox.origin_x
            ry1 = y_start + bbox.origin_y
            rx2 = rx1 + bbox.width
            ry2 = ry1 + bbox.height
            
            results.append(([rx1, ry1, rx2, ry2], score))

    return results

def detect_faces_with_grid(frame, cfg):
    h, w = frame.shape[:2]
    grid_x, grid_y = cfg.get("grid", (3, 3))
    pad_ratio = cfg.get("padding_ratio", 0.10)
    
    step_x, step_y = w // grid_x, h // grid_y
    tiles_payload = []
    
    for i in range(grid_y):
        for j in range(grid_x):
            x1, y1 = j * step_x, i * step_y
            x2 = w if j == grid_x - 1 else (j + 1) * step_x
            y2 = h if i == grid_y - 1 else (i + 1) * step_y
            
            pad_w, pad_h = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
            px1, py1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            px2, py2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
            
            tile_img = frame[py1:py2, px1:px2]
            tiles_payload.append((tile_img, px1, py1, px2 - px1, py2 - py1))

    if cfg.get("full_frame_pass", False):
        tiles_payload.append((frame, 0, 0, w, h))

    all_boxes, all_scores = [], []
    with ThreadPoolExecutor(max_workers=total_cpus) as executor:
        futures = [executor.submit(process_tile_detection, payload, cfg["det_thresh"]) for payload in tiles_payload]
        for future in as_completed(futures):
            for box, score in future.result():
                all_boxes.append(box)
                all_scores.append(score)

    final_boxes = []
    if len(all_boxes) > 0:
        keep_indices = non_max_suppression(all_boxes, all_scores, iou_threshold=cfg.get("nms_iou", 0.20))
        for idx in keep_indices:
            final_boxes.append(all_boxes[idx])
            
    return final_boxes

class FaceTracker:
    def __init__(self, max_disappeared=4, iou_thresh=0.3):
        self.next_track_id = 1
        self.tracks = {}  
        self.max_disappeared = max_disappeared
        self.iou_thresh = iou_thresh

    def update(self, boxes):
        start_tracker = time.time()
        if len(boxes) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return []

        if len(self.tracks) == 0:
            active_tracks = []
            for b in boxes:
                self.tracks[self.next_track_id] = {"bbox": b, "disappeared": 0, "history": []}
                active_tracks.append((self.next_track_id, b))
                self.next_track_id += 1
            return active_tracks

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["bbox"] for tid in track_ids]
        
        iou_matrix = np.zeros((len(track_boxes), len(boxes)))
        for i, t_box in enumerate(track_boxes):
            for j, n_box in enumerate(boxes):
                iou_matrix[i, j] = calculate_iou(t_box, n_box)

        matched_tracks = set()
        matched_boxes = set()
        active_tracks = []

        while True:
            max_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]
            
            if max_iou < self.iou_thresh:
                break
                
            t_idx, b_idx = max_idx
            tid = track_ids[t_idx]
            
            self.tracks[tid]["bbox"] = boxes[b_idx]
            self.tracks[tid]["disappeared"] = 0
            active_tracks.append((tid, boxes[b_idx]))
            
            matched_tracks.add(t_idx)
            matched_boxes.add(b_idx)
            
            iou_matrix[t_idx, :] = -1
            iou_matrix[:, b_idx] = -1

        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
                else:
                    active_tracks.append((tid, self.tracks[tid]["bbox"]))

        for j, b in enumerate(boxes):
            if j not in matched_boxes:
                self.tracks[self.next_track_id] = {"bbox": b, "disappeared": 0, "history": []}
                active_tracks.append((self.next_track_id, b))
                self.next_track_id += 1
        tracker_ms = (time.time() - start_tracker) * 1000
        # In ra nếu mất hơn 2ms để tiện theo dõi, tránh spam console
        if tracker_ms > 2.0:
            print(f"⏱️ [FaceTracker] Xử lý gộp {len(boxes)} boxes mất: {tracker_ms:.2f} ms")
        return active_tracks


# ==========================================
# 4. ĐĂNG KÝ eKYC TỪ 6 ẢNH
# ==========================================
def register_student_face_from_images(image_paths, mssv, name, db_csv_path=DB_CSV_PATH):
    student_vectors = []
    action_names = ["1_TrucDien", "2_MimCuoi", "3_Trai", "4_Phai", "5_Len", "6_Xuong"]

    safe_name = remove_vietnamese_accents(name)
    save_dir = os.path.join("database", "registered_faces", f"{mssv}_{safe_name}")
    os.makedirs(save_dir, exist_ok=True)

    for f in os.listdir(save_dir):
        if f.endswith('.jpg'): os.remove(os.path.join(save_dir, f))

    error_logs = []  

    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        for idx, path in enumerate(image_paths):
            if idx >= len(action_names): break

            img = cv2.imread(path)
            if img is None: continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape

            # Phân nhánh: nếu ảnh đã được frontend crop sẵn chuẩn 160x160 -> bỏ qua detector, tránh crop 2 lần
            is_pre_cropped_160 = (w == 160 and h == 160)

            if is_pre_cropped_160:
                gray_crop = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                mean_brightness = np.mean(gray_crop)
                if mean_brightness < 40:
                    error_logs.append("Môi trường quá tối")
                    continue
                if mean_brightness > 230:
                    error_logs.append("Bị chói sáng/lóa")
                    continue

                vec = extract_embedding_from_crop(img_rgb)
                if vec is not None:
                    # Kiểm tra tính đồng nhất danh tính (Same-person Check) so với ảnh trực diện đầu tiên (ngưỡng Cosine Similarity >= 0.70)
                    if len(student_vectors) > 0:
                        sim_to_ref = float(np.dot(student_vectors[0], vec))
                        if sim_to_ref < SAME_PERSON_THRESHOLD:
                            error_logs.append(f"Góc {action_names[idx]} không khớp với ảnh tham chiếu (Cosine Similarity {sim_to_ref:.2f} < {SAME_PERSON_THRESHOLD:.2f})")
                            continue
                    student_vectors.append(vec)

                img_filename = os.path.join(save_dir, f"{action_names[idx]}.jpg")
                is_success, im_buf_arr = cv2.imencode(".jpg", img)
                if is_success:
                    im_buf_arr.tofile(img_filename)
                continue

            # Fallback tương thích ngược: ảnh toàn khung hình (Full Resolution) cần detector cắt mặt
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection_result = detector.detect(mp_image)

            if not detection_result.detections:
                error_logs.append("Không tìm thấy khuôn mặt")
                continue

            if len(detection_result.detections) > 1:
                error_logs.append("Có nhiều hơn 1 người trong khung hình")
                continue

            detection = detection_result.detections[0]
            bbox = detection.bounding_box
            xmin, ymin = max(0, int(bbox.origin_x)), max(0, int(bbox.origin_y))
            box_w, box_h = min(w - xmin, int(bbox.width)), min(h - ymin, int(bbox.height))

            if box_w < 60 or box_h < 60:
                error_logs.append("Khuôn mặt quá nhỏ/xa")
                continue
                
            if hasattr(detection, 'keypoints') and len(detection.keypoints) >= 3:
                leye = detection.keypoints[0]
                reye = detection.keypoints[1]
                nose = detection.keypoints[2]
                
                eye_dist = abs(reye.x - leye.x)
                if eye_dist > 0:
                    nose_ratio = abs(nose.x - leye.x) / eye_dist
                    if nose_ratio < 0.2 or nose_ratio > 0.8:
                        error_logs.append("Quay mặt quá sâu (>45 độ)")
                        continue

            crop_face = img_rgb[ymin:ymin+box_h, xmin:xmin+box_w]
            if crop_face.size > 0:
                gray_crop = cv2.cvtColor(crop_face, cv2.COLOR_RGB2GRAY)
                mean_brightness = np.mean(gray_crop)
                if mean_brightness < 40:
                    error_logs.append("Môi trường quá tối")
                    continue
                if mean_brightness > 230:
                    error_logs.append("Bị chói sáng/lóa")
                    continue

                vec = extract_embedding_from_crop(crop_face)
                if vec is not None:
                    # Kiểm tra cùng 1 người so với ảnh trực diện đầu tiên (ngưỡng Cosine Similarity >= 0.70)
                    if len(student_vectors) > 0:
                        sim_to_ref = float(np.dot(student_vectors[0], vec))
                        if sim_to_ref < SAME_PERSON_THRESHOLD:
                            error_logs.append(f"Góc {action_names[idx]} không khớp với ảnh tham chiếu (Cosine Similarity {sim_to_ref:.2f} < {SAME_PERSON_THRESHOLD:.2f})")
                            continue
                    student_vectors.append(vec)

                pad = 30
                y1, y2 = max(0, ymin - pad), min(h, ymin + box_h + pad)
                x1, x2 = max(0, xmin - pad), min(w, xmin + box_w + pad)
                face_img = img[y1:y2, x1:x2]
                if face_img.size > 0:
                    img_filename = os.path.join(save_dir, f"{action_names[idx]}.jpg")
                    is_success, im_buf_arr = cv2.imencode(".jpg", face_img)
                    if is_success:
                        im_buf_arr.tofile(img_filename)

    if len(student_vectors) < 4:
        unique_errors = list(set(error_logs))
        err_detail = " | ".join(unique_errors) if unique_errors else "Khuôn mặt không rõ ràng."
        return False, f"Chỉ hợp lệ {len(student_vectors)}/6 ảnh. Lỗi: {err_detail}. Vui lòng sử dụng cùng một người và quay góc vừa phải."

    avg_vector = np.mean(student_vectors, axis=0)
    avg_vector = avg_vector / np.linalg.norm(avg_vector)

    embedding_cols = [f"emb_{i}" for i in range(512)]
    file_exists = os.path.isfile(db_csv_path)

    clean_mssv = str(mssv).replace('.0', '').strip()
    if file_exists:
        df = pd.read_csv(db_csv_path, encoding='utf-8-sig')
        df['MSSV'] = df['MSSV'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        if clean_mssv in df['MSSV'].values:
            row_idx = df[df['MSSV'] == clean_mssv].index[0]
            df.loc[row_idx, 'HoTen'] = name
            for i, col in enumerate(embedding_cols):
                df.loc[row_idx, col] = float(avg_vector[i])
            df.to_csv(db_csv_path, index=False, encoding='utf-8-sig')
        else:
            with open(db_csv_path, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([clean_mssv, name] + avg_vector.tolist())
    else:
        with open(db_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['MSSV', 'HoTen'] + embedding_cols)
            writer.writerow([clean_mssv, name] + avg_vector.tolist())

    load_face_bank_to_ram(db_csv_path)
    return True, f"Thành công! Đã đăng ký khuôn mặt cho sinh viên {name}."

# ==========================================
# 5. XỬ LÝ ĐIỂM DANH: ẢNH HOẶC VIDEO
# ==========================================
os.makedirs("uploads/detected_faces", exist_ok=True)

def process_attendance(media_path, excel_path=None):
    global GLOBAL_STUDENT_MAP
    if not GLOBAL_STUDENT_MAP:
        raise FileNotFoundError("CSDL Face Bank trống. Hãy tạo Face Bank trước.")

    timestamp_base = int(time.time())

    if excel_path and os.path.exists(excel_path):
        try:
            df_raw = pd.read_excel(excel_path, header=None)
        except Exception:
            df_raw = pd.read_excel(excel_path, header=None, engine='xlrd')

        header_row_idx = None
        for idx, row in df_raw.iterrows():
            if row.astype(str).str.contains('MSSV', case=False, na=False).any():
                header_row_idx = idx
                break

        if header_row_idx is None:
            raise ValueError("Không tìm thấy cột 'MSSV' trong file danh sách.")

        df_list = pd.read_excel(excel_path, header=header_row_idx)
        mssv_col = [c for c in df_list.columns if 'MSSV' in str(c).upper()][0]
        class_mssv_list = df_list[mssv_col].dropna().astype(str).str.replace(r'\.0$', '', regex=True).str.strip().tolist()
        class_mssv_list = [m for m in class_mssv_list if m.isalnum()]
    else:
        class_mssv_list = list(GLOBAL_STUDENT_MAP.keys())
        mssv_col = "MSSV"
        df_list = pd.DataFrame({
            "MSSV": class_mssv_list,
            "Họ và Tên": [GLOBAL_STUDENT_MAP[m]["HoTen"] for m in class_mssv_list]
        })

    db_embeddings_list = []
    db_info_list = []
    registered_mssvs = set()

    for mssv in class_mssv_list:
        if mssv in GLOBAL_STUDENT_MAP:
            registered_mssvs.add(mssv)
            student_data = GLOBAL_STUDENT_MAP[mssv]
            for vec in student_data["vectors"]:
                db_embeddings_list.append(vec)
                db_info_list.append(f"{mssv} - {student_data['HoTen']}")

    unregistered_mssvs = [m for m in class_mssv_list if m not in registered_mssvs]
    has_registered = len(db_embeddings_list) > 0
    db_matrix = np.vstack(db_embeddings_list) if has_registered else np.empty((0, 512), dtype=np.float32)

    attendance_records = {}
    detected_crops = []

    is_image = media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    cfg = PIPELINE_CONFIG["image"] if is_image else PIPELINE_CONFIG["video"]

    def handle_detected_face(face_bgr, rx1, ry1, rx2, ry2, best_sim, student_info, time_str, is_known=True):
        pad = 20
        h_f, w_f = face_bgr.shape[:2]
        y1_c, y2_c = max(0, ry1 - pad), min(h_f, ry2 + pad)
        x1_c, x2_c = max(0, rx1 - pad), min(w_f, rx2 + pad)
        crop_img = face_bgr[y1_c:y2_c, x1_c:x2_c]

        face_uid = f"{int(time.time()*1000)}_{np.random.randint(100, 999)}"
        crop_filename = f"uploads/detected_faces/face_{face_uid}.jpg"
        if crop_img.size > 0:
            cv2.imwrite(crop_filename, crop_img)
        else:
            crop_filename = ""

        if is_known:
            parts = student_info.split(" - ", 1)
            st_mssv = parts[0]
            st_name = parts[1] if len(parts) > 1 else ""
            if st_mssv not in attendance_records:
                record = {
                    "type": "known",
                    "mssv": st_mssv,
                    "name": st_name,
                    "time": time_str,
                    "score": round(float(best_sim) * 100, 1),
                    "crop_url": f"/{crop_filename}" if crop_filename else ""
                }
                attendance_records[st_mssv] = record
                detected_crops.append(record)
        else:
            detected_crops.append({
                "type": "unknown",
                "mssv": "Unknown",
                "name": "Người lạ / Chưa ĐK",
                "time": time_str,
                "score": round(float(best_sim) * 100, 1),
                "crop_url": f"/{crop_filename}" if crop_filename else ""
            })

    def recognize_and_draw_faces(frame_target, boxes, time_string):
        drawn_frame = frame_target.copy()
        valid_faces_count = 0
        h_f, w_f = drawn_frame.shape[:2]
        
        for box in boxes:
            rx1, ry1, rx2, ry2 = box
            rx1, ry1, rx2, ry2 = clamp_box(rx1, ry1, rx2, ry2, w_f, h_f)
            if rx2 <= rx1 or ry2 <= ry1: continue
            
            crop_face = frame_target[int(ry1):int(ry2), int(rx1):int(rx2)]
            crop_face_rgb = cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB)
            
            vec = extract_embedding_from_crop(crop_face_rgb)
            if vec is None: continue

            valid_faces_count += 1
            best_sim = np.max(np.dot(db_matrix, vec)) if has_registered else 0.0
            best_idx = np.argmax(np.dot(db_matrix, vec)) if has_registered else -1

            if has_registered and best_sim >= cfg["rec_thresh"]:
                student_info = db_info_list[best_idx]
                box_color = (0, 255, 0)
                label_text = f"{student_info} ({best_sim:.2f})"
                handle_detected_face(drawn_frame, rx1, ry1, rx2, ry2, best_sim, student_info, time_string, is_known=True)
            else:
                box_color = (0, 0, 255)
                label_text = f"Unknown ({best_sim:.2f})"
                handle_detected_face(drawn_frame, rx1, ry1, rx2, ry2, best_sim, "Unknown", time_string, is_known=False)

            cv2.rectangle(drawn_frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), box_color, 2)
            cv2.putText(drawn_frame, label_text, (int(rx1) + 4, int(ry1) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)
            
        return drawn_frame, valid_faces_count

    if is_image:
        print(f"📸 [AI Service] Đang phân tích Ảnh toàn cảnh: {media_path}")
        frame = cv2.imread(media_path)
        if frame is None: raise ValueError("Không thể đọc ảnh tải lên.")

        final_boxes = detect_faces_with_grid(frame, cfg)
        
        current_date_time = time.strftime("%d/%m/%Y %H:%M:%S")
        annotated_frame, _ = recognize_and_draw_faces(frame, final_boxes, current_date_time)
        
        output_media_path = f"uploads/detected_result_{timestamp_base}.jpg"
        cv2.imwrite(output_media_path, annotated_frame)

    else:
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened(): raise ValueError("Không mở được video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps): fps = 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🎬 [AI Service] Bắt đầu quét Video bằng Grid: Tổng cộng {total_video_frames} frames (FPS: {fps:.1f})")

        output_media_path = f"uploads/detected_result_{timestamp_base}.jpg"
        representative_frame = None
        max_faces_in_single_frame = 0

        frame_count = 0
        frame_skip = cfg.get("frame_skip", 5)
        
        tracker = FaceTracker(max_disappeared=4, iou_thresh=0.2)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            if frame_count % frame_skip != 0: continue

            current_sec = frame_count / fps
            
            video_time_str = time.strftime("%H:%M:%S", time.gmtime(current_sec))
            current_date_time = time.strftime("%d/%m/%Y %H:%M:%S")
            timestamp_str = f"{current_date_time} (Video: {video_time_str})"

            if frame_count % (frame_skip * 4) == 0 or frame_count >= total_video_frames - frame_skip:
                pct = (frame_count / total_video_frames) * 100 if total_video_frames > 0 else 0
                print(f"⏳ [Đang xử lý Video] Frame {frame_count}/{total_video_frames} ({pct:.1f}%) | Đã phát hiện: {len(attendance_records)} SV")

            final_boxes = detect_faces_with_grid(frame, cfg)
            
            tracked_faces = tracker.update(final_boxes)
            
            current_faces_count = len(tracked_faces)
            annotated_current = frame.copy()
            h_f, w_f = annotated_current.shape[:2]

            window_size = cfg.get("voting_window_frames", 12)
            min_ratio = cfg.get("min_vote_ratio", 0.6)

            for track_id, box in tracked_faces:
                rx1, ry1, rx2, ry2 = box
                rx1, ry1, rx2, ry2 = clamp_box(rx1, ry1, rx2, ry2, w_f, h_f)
                if rx2 <= rx1 or ry2 <= ry1: continue
                
                if "confirmed_mssv" in tracker.tracks[track_id] and tracker.tracks[track_id]["confirmed_mssv"]:
                    top_candidate = tracker.tracks[track_id]["confirmed_mssv"]
                    if top_candidate != "Unknown":
                        box_color = (0, 255, 0)
                        display_label = f"ID:{track_id} | {top_candidate.split(' - ')[0]} \u2713"
                    else:
                        box_color = (0, 0, 255)
                        display_label = f"ID:{track_id} | Unknown"
                        
                    cv2.rectangle(annotated_current, (int(rx1), int(ry1)), (int(rx2), int(ry2)), box_color, 2)
                    cv2.putText(annotated_current, display_label, (int(rx1) + 4, int(ry1) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)
                    continue 

                crop_face = frame[int(ry1):int(ry2), int(rx1):int(rx2)]
                crop_face_rgb = cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB)
                
                vec = extract_embedding_from_crop(crop_face_rgb)
                if vec is None: continue

                best_sim = np.max(np.dot(db_matrix, vec)) if has_registered else 0.0
                best_idx = np.argmax(np.dot(db_matrix, vec)) if has_registered else -1

                predicted_mssv = "Unknown"
                if has_registered and best_sim >= cfg["rec_thresh"]:
                    predicted_mssv = db_info_list[best_idx]
                
                if "history" not in tracker.tracks[track_id]:
                     tracker.tracks[track_id]["history"] = []
                tracker.tracks[track_id]["history"].append(predicted_mssv)
                
                if "confirmed_mssv" not in tracker.tracks[track_id]:
                    tracker.tracks[track_id]["confirmed_mssv"] = None

                display_label = f"ID:{track_id} | {predicted_mssv.split(' - ')[0]}"
                box_color = (0, 255, 255)

                history = tracker.tracks[track_id]["history"]
                if len(history) >= window_size:
                    recent_votes = history[-window_size:]
                    
                    from collections import Counter
                    vote_counts = Counter(recent_votes)
                    top_candidate, top_count = vote_counts.most_common(1)[0]
                    
                    if top_candidate != "Unknown" and (top_count / window_size) >= min_ratio:
                        tracker.tracks[track_id]["confirmed_mssv"] = top_candidate
                        box_color = (0, 255, 0)
                        display_label = f"ID:{track_id} | {top_candidate.split(' - ')[0]} \u2713"
                        
                        handle_detected_face(annotated_current, rx1, ry1, rx2, ry2, best_sim, top_candidate, timestamp_str, is_known=True)
                    
                    elif top_candidate == "Unknown" and (top_count / window_size) >= min_ratio:
                        tracker.tracks[track_id]["confirmed_mssv"] = "Unknown"
                        box_color = (0, 0, 255)
                        display_label = f"ID:{track_id} | Unknown"
                        # Chỉ lưu và xuất ảnh "Unknown" 1 lần duy nhất cho mỗi Track ID
                        if not tracker.tracks[track_id].get("has_logged_unknown"):
                            handle_detected_face(annotated_current, rx1, ry1, rx2, ry2, best_sim, "Unknown", timestamp_str, is_known=False)
                            tracker.tracks[track_id]["has_logged_unknown"] = True

                cv2.rectangle(annotated_current, (int(rx1), int(ry1)), (int(rx2), int(ry2)), box_color, 2)
                cv2.putText(annotated_current, display_label, (int(rx1) + 4, int(ry1) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

            if current_faces_count > max_faces_in_single_frame:
                max_faces_in_single_frame = current_faces_count
                representative_frame = annotated_current

        cap.release()
        if representative_frame is not None:
            cv2.imwrite(output_media_path, representative_frame)
        else:
            cv2.imwrite(output_media_path, np.zeros((480, 640, 3), dtype=np.uint8))

        print(f"✅ [Hoàn tất Video] Đã duyệt {frame_count} frames! Nhận diện thành công {len(attendance_records)} SV.")

    total_students = len(class_mssv_list)
    count_present = len(attendance_records)
    count_unregistered = len([m for m in unregistered_mssvs if m not in attendance_records])
    count_absent = max(0, total_students - count_present - count_unregistered)

    pct_present = round((count_present / total_students) * 100, 1) if total_students > 0 else 0
    pct_absent = round((count_absent / total_students) * 100, 1) if total_students > 0 else 0
    pct_unregistered = round((count_unregistered / total_students) * 100, 1) if total_students > 0 else 0

    def get_ai_status(mssv):
        if pd.isna(mssv) or str(mssv).strip() == "": return ""
        m_str = str(mssv).replace('.0', '').strip()
        if m_str in attendance_records: return "✅ Có mặt"
        if m_str in unregistered_mssvs: return "⚠️ Chưa có dữ liệu mặt" 
        return "❌ Vắng mặt"

    def get_detected_time(mssv):
        if pd.isna(mssv) or str(mssv).strip() == "": return ""
        m_str = str(mssv).replace('.0', '').strip()
        if m_str in attendance_records:
            return attendance_records[m_str]["time"]
        return ""

    df_list['Trạng thái AI'] = df_list[mssv_col].apply(get_ai_status)
    df_list['Thời điểm phát hiện'] = df_list[mssv_col].apply(get_detected_time)
    df_list = df_list.fillna('')

    output_excel_path = f"database/KetQua_AI_{timestamp_base}.xlsx"
    df_list.to_excel(output_excel_path, index=False)

    return {
        "excel_path": output_excel_path,
        "media_result_path": output_media_path,
        "is_image": is_image,
        "total": total_students,
        "present": count_present,
        "absent": count_absent,
        "unregistered": count_unregistered,
        "pct_present": pct_present,
        "pct_absent": pct_absent,
        "pct_unregistered": pct_unregistered,
        "detected_crops": detected_crops
    }

# ==========================================
# 6. REAL-TIME INFERENCE TỪ CAMERA TRÌNH DUYỆT
# ==========================================
def _get_realtime_db_matrix(class_mssv_list):
    key = tuple(sorted(class_mssv_list)) if class_mssv_list else ()
    if len(_REALTIME_MATRIX_CACHE) > 16:
            _REALTIME_MATRIX_CACHE.clear()
    cached = _REALTIME_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    db_embeddings_list = []
    db_info_list = []

    if key:
        for mssv in key:
            student_data = GLOBAL_STUDENT_MAP.get(mssv)
            if student_data:
                for vec in student_data["vectors"]:
                    db_embeddings_list.append(vec)
                    db_info_list.append(f"{mssv} - {student_data['HoTen']}")
    else:
        for mssv, student_data in GLOBAL_STUDENT_MAP.items():
            for vec in student_data["vectors"]:
                db_embeddings_list.append(vec)
                db_info_list.append(f"{mssv} - {student_data['HoTen']}")

    has_reg = len(db_embeddings_list) > 0
    db_mat = np.vstack(db_embeddings_list) if has_reg else np.empty((0, 512), dtype=np.float32)

    result = (db_mat, db_info_list, has_reg)
    _REALTIME_MATRIX_CACHE[key] = result
    return result

realtime_tracker = FaceTracker(max_disappeared=4, iou_thresh=0.2)

def reset_realtime_tracker():
    """Xóa sạch bộ nhớ của Realtime Tracker"""
    global realtime_tracker
    realtime_tracker = FaceTracker(max_disappeared=4, iou_thresh=0.2)

def process_realtime_frame_streamlined(frame_bgr, class_mssv_list=None):
    global GLOBAL_STUDENT_MAP
    if not GLOBAL_STUDENT_MAP or frame_bgr is None:
        return [], None

    db_mat, db_info_list, has_reg = _get_realtime_db_matrix(class_mssv_list or [])

    h, w = frame_bgr.shape[:2]
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    detector = get_thread_face_detector()
    cfg = PIPELINE_CONFIG["realtime"]
    scale = cfg.get("scale", 0.6)
    
    if scale < 1.0:
        small_rgb = cv2.resize(rgb_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        small_rgb = rgb_frame

    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb))
    inv_scale = 1.0 / scale

    detected_faces = []
    annotated_snapshot_b64 = None

    raw_boxes = []
    if res.detections:
        for d in res.detections:
            if float(d.categories[0].score) < cfg["det_thresh"]:
                continue
            bbox = d.bounding_box
            xmin = max(0, int(bbox.origin_x * inv_scale))
            ymin = max(0, int(bbox.origin_y * inv_scale))
            box_w = min(w - xmin, int(bbox.width * inv_scale))
            box_h = min(h - ymin, int(bbox.height * inv_scale))
            raw_boxes.append([xmin, ymin, xmin + box_w, ymin + box_h])

    tracked_faces = realtime_tracker.update(raw_boxes)
    
    should_create_snapshot = (len(tracked_faces) >= 2)
    snapshot_frame = frame_bgr.copy() if should_create_snapshot else None

    window_size = cfg.get("voting_window_frames", 6)
    min_ratio = cfg.get("min_vote_ratio", 0.6)

    for track_id, box in tracked_faces:
        rx1, ry1, rx2, ry2 = box
        rx1, ry1, rx2, ry2 = clamp_box(rx1, ry1, rx2, ry2, w, h)
        if rx2 <= rx1 or ry2 <= ry1: continue
        
        # NẾU ĐÃ KHÓA MỤC TIÊU -> TẮT INCEPTION
        if "confirmed_mssv" in realtime_tracker.tracks[track_id] and realtime_tracker.tracks[track_id]["confirmed_mssv"] is not None:
            top_candidate = realtime_tracker.tracks[track_id]["confirmed_mssv"]
            final_status = "known" if top_candidate != "Unknown" else "unknown"
            box_color = (0, 255, 0) if final_status == "known" else (0, 0, 255)
            display_label = f"ID:{track_id} | {top_candidate.split(' - ')[0]} \u2713" if final_status == "known" else f"ID:{track_id} | Unknown"
            
            if should_create_snapshot:
                cv2.rectangle(snapshot_frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), box_color, 2)
                cv2.putText(snapshot_frame, display_label, (int(rx1), max(20, int(ry1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)
            continue # Tắt Inception, không gửi gì thêm xuống Web

        # NẾU CHƯA KHÓA -> CHẠY INCEPTION
        pad = 10
        y1_c, y2_c = max(0, int(ry1) - pad), min(h, int(ry2) + pad)
        x1_c, x2_c = max(0, int(rx1) - pad), min(w, int(rx2) + pad)
        face_crop = frame_bgr[y1_c:y2_c, x1_c:x2_c]
        
        crop_rgb = rgb_frame[int(ry1):int(ry2), int(rx1):int(rx2)]
        if crop_rgb.size == 0 or (rx2-rx1) <= 0 or (ry2-ry1) <= 0:
            continue

        vec = extract_embedding_from_crop(crop_rgb)
        if vec is None:
            continue

        best_sim = np.max(np.dot(db_mat, vec)) if has_reg else 0.0
        best_idx = np.argmax(np.dot(db_mat, vec)) if has_reg else -1

        predicted_mssv = "Unknown"
        if has_reg and best_sim >= cfg["rec_thresh"]:
            predicted_mssv = db_info_list[best_idx]

        if "history" not in realtime_tracker.tracks[track_id]:
             realtime_tracker.tracks[track_id]["history"] = []
        realtime_tracker.tracks[track_id]["history"].append(predicted_mssv)
        
        if "confirmed_mssv" not in realtime_tracker.tracks[track_id]:
            realtime_tracker.tracks[track_id]["confirmed_mssv"] = None

        curr_time = time.strftime("%H:%M:%S")
        final_status = "waiting" 
        box_color = (0, 255, 255) 
        display_label = f"ID:{track_id} | Đang xac nhan..."

        history = realtime_tracker.tracks[track_id]["history"]
        if len(history) >= window_size:
            recent_votes = history[-window_size:]
            from collections import Counter
            vote_counts = Counter(recent_votes)
            top_candidate, top_count = vote_counts.most_common(1)[0]
            
            if top_candidate != "Unknown" and (top_count / window_size) >= min_ratio:
                realtime_tracker.tracks[track_id]["confirmed_mssv"] = top_candidate
                final_status = "known"
                box_color = (0, 255, 0)
                display_label = f"ID:{track_id} | {top_candidate.split(' - ')[0]} \u2713"
            elif top_candidate == "Unknown" and (top_count / window_size) >= min_ratio:
                realtime_tracker.tracks[track_id]["confirmed_mssv"] = "Unknown"
                final_status = "unknown"
                box_color = (0, 0, 255)
                display_label = f"ID:{track_id} | Unknown"

        # GỬI DATA XUỐNG UI ĐÚNG 1 LẦN VÀO KHOẢNH KHẮC "KHÓA MỤC TIÊU"
        if final_status != "waiting":
            if not realtime_tracker.tracks[track_id].get("has_sent_image", False):
                st_mssv = "Unknown"
                st_name = "Người lạ / Chưa ĐK"
                
                if final_status == "known":
                    st_mssv = realtime_tracker.tracks[track_id]["confirmed_mssv"].split(" - ")[0]
                    st_name = realtime_tracker.tracks[track_id]["confirmed_mssv"].split(" - ")[1]

                crop_b64 = ""
                if face_crop.size > 0:
                    _, crop_buf = cv2.imencode('.jpg', face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf).decode('utf-8')

                detected_faces.append({
                    "type": final_status, 
                    "mssv": st_mssv,
                    "name": st_name,
                    "score": round(float(best_sim) * 100, 1),
                    "crop_url": crop_b64,
                    "time": curr_time,
                    "track_id": track_id
                })
                
                realtime_tracker.tracks[track_id]["has_sent_image"] = True

        if should_create_snapshot:
            cv2.rectangle(snapshot_frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), box_color, 2)
            cv2.putText(snapshot_frame, display_label, (int(rx1), max(20, int(ry1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

    if should_create_snapshot and snapshot_frame is not None:
        _, snap_buf = cv2.imencode('.jpg', snapshot_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        annotated_snapshot_b64 = "data:image/jpeg;base64," + base64.b64encode(snap_buf).decode('utf-8')

    return detected_faces, annotated_snapshot_b64 

_realtime_lock = threading.Lock()   # tracker là biến global, tránh 2 batch chạy đè nhau

#def extract_embeddings_batch(crops_rgb):
#    """B khuôn mặt -> 1 lần forward (B,3,160,160). Trả về list vector hoặc None."""
#    if not crops_rgb:
#        return []
#    try:
#        batch = torch.stack([realtime_transform(Image.fromarray(c)) for c in crops_rgb]).contiguous()
#        with torch.inference_mode():
#            emb = RECOGNITION_MODEL(batch)
#            if hasattr(emb, 'is_quantized') and emb.is_quantized:
#                emb = emb.dequantize()
#            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
#        return list(emb.cpu().numpy().astype(np.float32))
#    except Exception:
#        return [None] * len(crops_rgb)

def extract_embeddings_batch(crops_rgb):
    """B khuôn mặt -> 1 lần forward (B,3,160,160). Trả về list vector hoặc None."""
    if not crops_rgb:
        return []
    try:
        # Bắt đầu đo thời gian
        start_time = time.time()
        
        batch = torch.stack([realtime_transform(Image.fromarray(c)) for c in crops_rgb]).contiguous()
        with torch.inference_mode():
            emb = RECOGNITION_MODEL(batch)
            if hasattr(emb, 'is_quantized') and emb.is_quantized:
                emb = emb.dequantize()
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            
        result = list(emb.cpu().numpy().astype(np.float32))
        
        # Kết thúc đo thời gian và in ra terminal
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"⏱️ [Inception CPU] Trích xuất {len(crops_rgb)} khuôn mặt mất: {elapsed_ms:.2f} ms")
        
        return result
    except Exception as e:
        print(f"⚠️ Lỗi Inception: {e}")
        return [None] * len(crops_rgb)

def _build_tiles(w, h, grid, pad_ratio):
    gx, gy = grid
    sx, sy = w // gx, h // gy
    tiles = []
    for i in range(gy):
        for j in range(gx):
            x1, y1 = j * sx, i * sy
            x2 = w if j == gx - 1 else (j + 1) * sx
            y2 = h if i == gy - 1 else (i + 1) * sy
            pw, ph = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
            tiles.append((max(0, x1 - pw), max(0, y1 - ph), min(w, x2 + pw), min(h, y2 + ph)))
    return tiles

def _detect_in_region(rgb, region, det_thresh, scale, drop_clipped, margin):
    """Detect trong 1 vùng, trả box theo TỌA ĐỘ TOÀN ẢNH."""
    fh, fw = rgb.shape[:2]
    px1, py1, px2, py2 = region
    crop = rgb[py1:py2, px1:px2]
    if crop.size == 0:
        return []
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    crop = np.ascontiguousarray(crop)
    res = get_thread_face_detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    inv = 1.0 / scale
    out = []
    for d in (res.detections or []):
        score = float(d.categories[0].score)
        if score < det_thresh:
            continue
        bb = d.bounding_box
        x1 = px1 + bb.origin_x * inv
        y1 = py1 + bb.origin_y * inv
        x2 = x1 + bb.width * inv
        y2 = y1 + bb.height * inv
        if drop_clipped:
            # Box dính sát biên TRONG của tile = khuôn mặt bị cắt dở -> bỏ, tile bên cạnh sẽ có bản đủ
            if (px1 > 0 and x1 - px1 <= margin) or (px2 < fw and px2 - x2 <= margin) \
               or (py1 > 0 and y1 - py1 <= margin) or (py2 < fh and py2 - y2 <= margin):
                continue
        out.append(([x1, y1, x2, y2], score))
    return out

_DETECT_POOL = ThreadPoolExecutor(max_workers=min(4, total_cpus), thread_name_prefix="rt-det")

def _detect_batch_boxes(frames_rgb, cfg):
    """Detect song song (khung x tile), gộp NMS theo từng khung. Trả list[list[box]] đúng thứ tự khung."""
    margin = cfg.get("edge_margin_px", 4)
    det_thresh = cfg["det_thresh"]
    jobs = []
    for fi, rgb in enumerate(frames_rgb):
        h, w = rgb.shape[:2]
        for t in _build_tiles(w, h, cfg.get("grid", (2, 2)), cfg.get("padding_ratio", 0.30)):
            jobs.append((fi, _DETECT_POOL.submit(_detect_in_region, rgb, t, det_thresh, 1.0, True, margin)))
        if cfg.get("full_frame_pass", True):
            jobs.append((fi, _DETECT_POOL.submit(
                _detect_in_region, rgb, (0, 0, w, h), det_thresh, cfg.get("scale", 0.6), False, margin)))
    per_frame = [[] for _ in frames_rgb]
    for fi, fut in jobs:
        per_frame[fi].extend(fut.result())
    merged = []
    for fi, dets in enumerate(per_frame):
        if not dets:
            merged.append([])
            continue
        h, w = frames_rgb[fi].shape[:2]
        boxes = [d[0] for d in dets]
        scores = [d[1] for d in dets]
        keep = non_max_suppression(boxes, scores, iou_threshold=cfg.get("nms_iou", 0.30))
        merged.append([[int(max(0, boxes[k][0])), int(max(0, boxes[k][1])),
                        int(min(w, boxes[k][2])), int(min(h, boxes[k][3]))] for k in keep])
    return merged

def process_realtime_batch_streamlined(frames_bgr, class_mssv_list=None):
    if not GLOBAL_STUDENT_MAP or not frames_bgr:
        return [], None
    with _realtime_lock:
        return _run_realtime_batch(frames_bgr, class_mssv_list or [])

def _make_detection_frame(frame_bgr, max_side):
    """Ảnh thu nhỏ RIÊNG cho MediaPipe. Không ghi đè frame gốc; nếu frame đã <= max_side thì trả chính nó (scale = 1)."""
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return frame_bgr
    r = max_side / float(longest)
    new_w, new_h = max(1, int(round(w * r))), max(1, int(round(h * r)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _map_box_to_original(box, small_shape, orig_shape):
    """Quy đổi box (x1,y1,x2,y2) từ hệ tọa độ pixel của ảnh thu nhỏ về pixel của frame gốc.
    Scale tính riêng từng trục theo kích thước THỰC của hai ảnh; x1,y1 làm tròn xuống, x2,y2 làm tròn lên
    để vùng cắt luôn bao trọn box gốc; kết quả kẹp trong [0, W] x [0, H] (chỉ số cắt kiểu slice)."""
    sh, sw = small_shape[:2]
    oh, ow = orig_shape[:2]
    sx, sy = ow / float(sw), oh / float(sh)
    x1, y1, x2, y2 = box
    X1 = max(0, min(int(np.floor(x1 * sx)), ow))
    Y1 = max(0, min(int(np.floor(y1 * sy)), oh))
    X2 = max(0, min(int(np.ceil(x2 * sx)), ow))
    Y2 = max(0, min(int(np.ceil(y2 * sy)), oh))
    return X1, Y1, X2, Y2

def _run_realtime_batch(frames_bgr, class_mssv_list):
    from collections import Counter

    tracker = realtime_tracker
    db_mat, db_info_list, has_reg = _get_realtime_db_matrix(class_mssv_list)
    cfg = PIPELINE_CONFIG["realtime"]
    window_size = cfg.get("voting_window_frames", 6)
    min_ratio = cfg.get("min_vote_ratio", 0.6)

    # ---- Pha 1: detect lưới 2x2 (song song) -> gộp -> track tuần tự theo thứ tự khung ----
    pending = []
    last_tracked = []
    # frames_bgr = frame GỐC (độ phân giải cao, giữ nguyên để cắt mặt); small_bgr_list = ảnh thu nhỏ chỉ để MediaPipe phát hiện
    small_bgr_list = [_make_detection_frame(f, cfg.get("detect_max_side", 640)) for f in frames_bgr]
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in small_bgr_list]
    boxes_per_frame = _detect_batch_boxes(frames_rgb, cfg)

    for frame_bgr, small_bgr, raw_boxes in zip(frames_bgr, small_bgr_list, boxes_per_frame):
        h, w = small_bgr.shape[:2]               # h, w = kích thước ảnh thu nhỏ (hệ tọa độ của box/tracker)
        tracked = tracker.update(raw_boxes)      # tracker làm việc trên tọa độ ảnh thu nhỏ (IoU không đổi theo scale đều)
        last_tracked = tracked

        for track_id, box in tracked:
            trk = tracker.tracks.get(track_id)
            if trk is None or trk.get("confirmed_mssv") is not None:
                continue                      # đã khóa -> không chạy Inception
            if trk.get("disappeared", 0) > 0:
                continue                      # box cũ (khung này mất mặt) -> không bỏ phiếu
            rx1, ry1, rx2, ry2 = clamp_box(*box, w, h)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            # Quy đổi box về frame gốc rồi cắt mặt từ frame độ phân giải cao
            ox1, oy1, ox2, oy2 = _map_box_to_original((rx1, ry1, rx2, ry2), small_bgr.shape, frame_bgr.shape)
            crop_hi = frame_bgr[oy1:oy2, ox1:ox2]
            if crop_hi.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop_hi, cv2.COLOR_BGR2RGB)
            pad = 10                              # pad tính theo ảnh thu nhỏ (như cũ), được quy đổi cùng box
            px1, py1, px2, py2 = _map_box_to_original((rx1 - pad, ry1 - pad, rx2 + pad, ry2 + pad), small_bgr.shape, frame_bgr.shape)
            face_crop = frame_bgr[py1:py2, px1:px2]
            pending.append((track_id, crop_rgb, face_crop))

    # ---- Pha 2: 1 lần forward cho toàn bộ crop, so khớp DB bằng 1 phép nhân ma trận ----
    detected_faces = []
    if pending:
        embs = extract_embeddings_batch([p[1] for p in pending])
        valid = [(p, e) for p, e in zip(pending, embs) if e is not None]

        per_track = {}   # tid -> [(pred, sim, face_crop)]
        if valid:
            if has_reg:
                sims = np.vstack([e for _, e in valid]) @ db_mat.T      # (N, M)
                best_idx, best_sim = sims.argmax(axis=1), sims.max(axis=1)

            for i, ((tid, _, face_crop), _) in enumerate(valid):
                trk = tracker.tracks.get(tid)
                if trk is None:
                    continue
                sim = float(best_sim[i]) if has_reg else 0.0
                pred = db_info_list[best_idx[i]] if (has_reg and sim >= cfg["rec_thresh"]) else "Unknown"
                trk.setdefault("history", []).append(pred)
                per_track.setdefault(tid, []).append((pred, sim, face_crop))

        # ---- Pha 3: voting, khóa mục tiêu ----
        curr_time = time.strftime("%H:%M:%S")
        for tid, items in per_track.items():
            trk = tracker.tracks[tid]
            hist = trk["history"]
            if len(hist) < window_size:
                continue
            top, cnt = Counter(hist[-window_size:]).most_common(1)[0]
            if cnt / window_size < min_ratio:
                continue

            trk["confirmed_mssv"] = top
            if trk.get("has_sent_image"):
                continue

            cands = [it for it in items if it[0] == top] or items
            _, sim, crop_bgr = max(cands, key=lambda it: it[1])

            if top != "Unknown":
                st_mssv, st_name = top.split(" - ", 1)
                ftype = "known"
            else:
                st_mssv, st_name, ftype = "Unknown", "Người lạ / Chưa ĐK", "unknown"

            crop_b64 = ""
            if crop_bgr.size > 0:
                _, buf = cv2.imencode('.jpg', crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

            detected_faces.append({
                "type": ftype, "mssv": st_mssv, "name": st_name,
                "score": round(sim * 100, 1), "crop_url": crop_b64,
                "time": curr_time, "track_id": tid
            })
            trk["has_sent_image"] = True

    # ---- Snapshot toàn cảnh từ khung cuối (chỉ khi >= 2 mặt) ----
    snapshot_b64 = None
    if len(last_tracked) >= 2:
        snap = small_bgr_list[-1].copy()      # vẽ trên ảnh thu nhỏ: cùng hệ tọa độ với box, kích thước phản hồi như cũ
        h, w = snap.shape[:2]
        for tid, box in last_tracked:
            conf = tracker.tracks.get(tid, {}).get("confirmed_mssv")
            if conf is None:
                color, label = (0, 255, 255), f"ID:{tid} | Dang xac nhan..."
            elif conf == "Unknown":
                color, label = (0, 0, 255), f"ID:{tid} | Unknown"
            else:
                color, label = (0, 255, 0), f"ID:{tid} | {conf.split(' - ')[0]} \u2713"
            x1, y1, x2, y2 = clamp_box(*box, w, h)
            cv2.rectangle(snap, (x1, y1), (x2, y2), color, 2)
            cv2.putText(snap, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        _, sb = cv2.imencode('.jpg', snap, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        snapshot_b64 = "data:image/jpeg;base64," + base64.b64encode(sb).decode('utf-8')

    return detected_faces, snapshot_b64


# ==========================================
# 7. REALTIME - NHẬN CROP KHUÔN MẶT TỪ FRONTEND (MediaPipe chạy ở trình duyệt)
# ==========================================
# Trình duyệt đã làm: detect lưới 2x2 + full-frame pass + NMS, quy đổi toạ độ, crop mặt 160x160 (RGB, JPEG).
# Backend chỉ còn: tracker (IoU) -> Inception -> so khớp Face Bank -> voting. Tracker/voting/ngưỡng GIỮ NGUYÊN
# như _run_realtime_batch; hàm cũ (process_realtime_batch_streamlined / _run_realtime_batch) vẫn được giữ lại.

_SNAP_STYLE = {
    "pending": "#ffff00",   # (0,255,255) BGR cũ
    "unknown": "#ff0000",   # (0,0,255) BGR cũ
    "known":   "#00ff00",   # (0,255,0) BGR cũ
}

def process_realtime_faces_batch(frames_meta, crops_rgb, crops_raw, class_mssv_list=None):
    """frames_meta: list[{"seq","w","h","faces":[{"box":[x1,y1,x2,y2],"c":int}]}] theo đúng thứ tự khung.
    crops_rgb[c]: ndarray RGB uint8 (đã giải mã từ JPEG); crops_raw[c]: bytes JPEG gốc (dùng làm ảnh hiển thị).
    Trả (detected_faces, snapshot_tracks)."""
    if not GLOBAL_STUDENT_MAP or not frames_meta:
        return [], None
    with _realtime_lock:
        return _run_realtime_faces_batch(frames_meta, crops_rgb, crops_raw, class_mssv_list or [])

def _run_realtime_faces_batch(frames_meta, crops_rgb, crops_raw, class_mssv_list):
    from collections import Counter

    tracker = realtime_tracker
    db_mat, db_info_list, has_reg = _get_realtime_db_matrix(class_mssv_list)
    cfg = PIPELINE_CONFIG["realtime"]
    window_size = cfg.get("voting_window_frames", 6)
    min_ratio = cfg.get("min_vote_ratio", 0.6)

    # ---- Pha 1: tracker tuần tự theo khung; box của frontend nằm trong hệ toạ độ ảnh detect (w, h) ----
    pending = []          # (track_id, crop_rgb, crop_jpeg_bytes, frame_seq, is_aligned)
    last_tracked, last_wh = [], (0, 0)
    for fr in frames_meta:
        w, h = fr["w"], fr["h"]
        faces = fr["faces"]
        raw_boxes = [list(f["box"]) for f in faces]
        tracked = tracker.update(raw_boxes)
        last_tracked, last_wh = tracked, (w, h)
        # tracker trả lại ĐÚNG đối tượng box vừa truyền vào cho track khớp/mới -> ghép box <-> crop và alignment_ok theo id()
        crop_of = {id(b): f["c"] for b, f in zip(raw_boxes, faces)}
        align_of = {id(b): bool(f.get("alignment_ok", False)) for b, f in zip(raw_boxes, faces)}

        for track_id, box in tracked:
            trk = tracker.tracks.get(track_id)
            if trk is None or trk.get("confirmed_mssv") is not None:
                continue                      # đã khóa -> không chạy Inception
            if trk.get("disappeared", 0) > 0:
                continue                      # box cũ (khung này mất mặt) -> không bỏ phiếu
            rx1, ry1, rx2, ry2 = clamp_box(*box, w, h)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            ci = crop_of.get(id(box), -1)
            if ci < 0 or crops_rgb[ci] is None:
                continue
            is_aligned = align_of.get(id(box), False)
            pending.append((track_id, crops_rgb[ci], crops_raw[ci], fr.get("seq"), is_aligned))

    # ---- Pha 2: 1 lần forward cho toàn bộ crop (Inception/preprocess y như cũ), so khớp DB bằng 1 phép nhân ma trận ----
    detected_faces = []
    if pending:
        embs = extract_embeddings_batch([p[1] for p in pending])
        valid = [(p, e) for p, e in zip(pending, embs) if e is not None]

        per_track = {}   # tid -> [(pred, sim, jpeg_bytes, frame_seq)]
        if valid:
            if has_reg:
                start_match = time.time() # Bắt đầu đo nhân ma trận
                sims = np.vstack([e for _, e in valid]) @ db_mat.T      # (N, M)
                best_idx, best_sim = sims.argmax(axis=1), sims.max(axis=1)
                match_ms = (time.time() - start_match) * 1000
                print(f"⏱️ [So khớp DB] Tìm kiếm {len(valid)} khuôn mặt trong CSDL mất: {match_ms:.2f} ms")

            for i, ((tid, _, raw, seq, is_aligned), _) in enumerate(valid):
                trk = tracker.tracks.get(tid)
                if trk is None:
                    continue
                # BẢO VỆ TUYỆT ĐỐI: Chỉ mẫu đạt chuẩn Face Alignment (màu xanh / alignment_ok=True) mới được tham gia voting!
                if not is_aligned:
                    continue
                sim = float(best_sim[i]) if has_reg else 0.0
                pred = db_info_list[best_idx[i]] if (has_reg and sim >= cfg["rec_thresh"]) else "Unknown"
                trk.setdefault("history", []).append(pred)
                per_track.setdefault(tid, []).append((pred, sim, raw, seq))

        # ---- Pha 3: voting, khóa mục tiêu (giữ nguyên logic cũ) ----
        curr_time = time.strftime("%H:%M:%S")
        for tid, items in per_track.items():
            trk = tracker.tracks[tid]
            hist = trk["history"]
            if len(hist) < window_size:
                continue
            top, cnt = Counter(hist[-window_size:]).most_common(1)[0]
            if cnt / window_size < min_ratio:
                continue

            trk["confirmed_mssv"] = top
            if trk.get("has_sent_image"):
                continue

            cands = [it for it in items if it[0] == top] or items
            _, sim, raw, seq = max(cands, key=lambda it: it[1])

            if top != "Unknown":
                st_mssv, st_name = top.split(" - ", 1)
                ftype = "known"
            else:
                st_mssv, st_name, ftype = "Unknown", "Người lạ / Chưa ĐK", "unknown"

            # Ảnh hiển thị = chính crop 160x160 frontend gửi lên (không còn frame gốc ở backend để cắt có pad)
            crop_b64 = "data:image/jpeg;base64," + base64.b64encode(raw).decode('utf-8') if raw else ""

            detected_faces.append({
                "type": ftype, "mssv": st_mssv, "name": st_name,
                "score": round(sim * 100, 1), "crop_url": crop_b64,
                "time": curr_time, "track_id": tid, "frame_seq": seq
            })
            trk["has_sent_image"] = True

    # ---- Snapshot: backend chỉ trả box/nhãn/màu của khung cuối (>= 2 mặt); frontend vẽ lên khung của nó ----
    snapshot_tracks = None
    if len(last_tracked) >= 2:
        w, h = last_wh
        snapshot_tracks = []
        for tid, box in last_tracked:
            conf = tracker.tracks.get(tid, {}).get("confirmed_mssv")
            if conf is None:
                state, label = "pending", f"ID:{tid} | Dang xac nhan..."
            elif conf == "Unknown":
                state, label = "unknown", f"ID:{tid} | Unknown"
            else:
                state, label = "known", f"ID:{tid} | {conf.split(' - ')[0]} ✓"
            x1, y1, x2, y2 = clamp_box(*box, w, h)
            snapshot_tracks.append({"track_id": tid, "box": [x1, y1, x2, y2],
                                    "label": label, "color": _SNAP_STYLE[state]})

    return detected_faces, snapshot_tracks
