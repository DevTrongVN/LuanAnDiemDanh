import os
import cv2
import time
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.ao.quantization as quantization
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THIẾT BỊ
# ==========================================
ROOT_DATA_DIR = r"D:\Khoa_Luan\Data"
OUTPUT_CSV = r"database/face_bank_v3.csv"
SAVED_WEIGHTS_PATH = "model_v3_p_r1_0_r2_6_r3_3_weights.pt"
DETECTOR_MODEL_PATH = "detector.tflite"

DUPLICATE_SIMILARITY_THRESHOLD = 0.95
TARGET_HEIGHT = 640

num_cores = min(2, os.cpu_count() or 2)
torch.set_num_threads(num_cores)
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

device = torch.device('cpu')

facenet_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH V3_P_R1_0_R2_6_R3_3
# ==========================================
def check_and_download_detector():
    if not os.path.exists(DETECTOR_MODEL_PATH):
        print(f"⏳ Đang tải {DETECTOR_MODEL_PATH}...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            DETECTOR_MODEL_PATH
        )

def prune_inception_v1(model, drop_repeat1=0, drop_repeat2=6, drop_repeat3=3):
    if drop_repeat1 > 0 and len(model.repeat_1) > drop_repeat1:
        model.repeat_1 = model.repeat_1[:-drop_repeat1]
    if drop_repeat2 > 0 and len(model.repeat_2) > drop_repeat2:
        model.repeat_2 = model.repeat_2[:-drop_repeat2]
    if drop_repeat3 > 0 and len(model.repeat_3) > drop_repeat3:
        model.repeat_3 = model.repeat_3[:-drop_repeat3]
    return model

def get_v3_pruned_int8_model():
    torch.backends.quantized.engine = 'fbgemm'
    base_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    base_model = prune_inception_v1(base_model, drop_repeat1=0, drop_repeat2=6, drop_repeat3=3)

    dummy_input = torch.randn(1, 3, 160, 160)
    qconfig_dict = {"": quantization.get_default_qconfig('fbgemm')}
    prepared_model = prepare_fx(base_model, qconfig_dict, example_inputs=(dummy_input,))

    if os.path.exists(SAVED_WEIGHTS_PATH):
        try:
            print(f"⚡ Đang nạp trọng số từ: {SAVED_WEIGHTS_PATH}")
            quantized_model = convert_fx(prepared_model)
            quantized_model.load_state_dict(torch.load(SAVED_WEIGHTS_PATH, map_location=device))
            quantized_model.eval()
            print("✅ Nạp thành công mô hình Static INT8!")
            return quantized_model
        except Exception as e:
            print(f"⚠️ Nạp state_dict thất bại ({e}). Đang calibrate lại...")

    print("⚙️ Khởi tạo & Calibrate mô hình V3 INT8...")
    with torch.no_grad():
        for _ in range(10):
            calib_batch = torch.randn(8, 3, 160, 160)
            prepared_model(calib_batch)

    quantized_model = convert_fx(prepared_model)
    quantized_model.eval()
    try:
        torch.save(quantized_model.state_dict(), SAVED_WEIGHTS_PATH)
        print(f"✅ Đã lưu trọng số vào: {SAVED_WEIGHTS_PATH}")
    except Exception as e:
        print(f"⚠️ Lưu trọng số thất bại: {e}")

    return quantized_model

# ==========================================
# 3. TRÍCH XUẤT CSDL KHUÔN MẶT (BỎ MaLop)
# ==========================================
def build_face_bank():
    check_and_download_detector()
    model = get_v3_pruned_int8_model()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.5
    )

    if not os.path.exists(ROOT_DATA_DIR):
        print(f"❌ Không tìm thấy thư mục: {ROOT_DATA_DIR}")
        return

    # Quét tất cả thư mục sinh viên (hỗ trợ cả lồng trong lớp hoặc phẳng)
    student_dirs = []
    for root, dirs, files in os.walk(ROOT_DATA_DIR):
        folder_name = os.path.basename(root)
        if "_" in folder_name and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            student_dirs.append(root)

    print(f"🚀 Tìm thấy {len(student_dirs)} thư mục sinh viên. Bắt đầu trích xuất...")

    database_records = []

    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        for s_dir in tqdm(student_dirs, desc="Trích xuất Face Bank"):
            folder_name = os.path.basename(s_dir)
            parts = folder_name.split("_", 1)
            mssv = str(parts[0]).replace('.0', '').strip()
            ho_ten = parts[1] if len(parts) > 1 else "Unknown"

            image_files = [f for f in os.listdir(s_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                continue

            student_vectors = []
            for img_name in image_files:
                img_path = os.path.join(s_dir, img_name)
                image_bgr = cv2.imread(img_path)
                if image_bgr is None:
                    continue

                h, w = image_bgr.shape[:2]
                scale = TARGET_HEIGHT / float(h)
                new_w = int(w * scale)
                resized_frame = cv2.resize(image_bgr, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
                image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                detection_result = detector.detect(mp_image)

                if detection_result.detections:
                    bbox = detection_result.detections[0].bounding_box
                    xmin = max(0, int(bbox.origin_x))
                    ymin = max(0, int(bbox.origin_y))
                    box_w = min(new_w - xmin, int(bbox.width))
                    box_h = min(TARGET_HEIGHT - ymin, int(bbox.height))

                    face_crop = image_rgb[ymin:ymin+box_h, xmin:xmin+box_w]
                    if face_crop.size == 0 or box_w <= 0 or box_h <= 0:
                        continue

                    tensor = facenet_transform(Image.fromarray(face_crop)).unsqueeze(0).contiguous()
                    with torch.inference_mode():
                        emb = model(tensor)
                        if hasattr(emb, 'is_quantized') and emb.is_quantized:
                            emb = emb.dequantize()
                        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                        embedding_512d = emb.squeeze(0).cpu().numpy().flatten().astype(np.float32)

                    # Khử trùng lặp vector trong cùng 1 sinh viên
                    # Tiền xử lý: Lọc bỏ các ảnh có góc mặt quá giống nhau (>= 0.95)
                    is_unique = True
                    for existing_vec in student_vectors:
                        if np.dot(existing_vec, embedding_512d) >= DUPLICATE_SIMILARITY_THRESHOLD:
                            is_unique = False
                            break

                    if is_unique:
                        student_vectors.append(embedding_512d)

            # --- SAU KHI QUÉT XONG TẤT CẢ ẢNH CỦA 1 SINH VIÊN ---
            if len(student_vectors) > 0:
                # 1. Tính trung bình cộng của tất cả các vector hợp lệ
                avg_vector = np.mean(student_vectors, axis=0)
                # 2. Chuẩn hóa L2 (L2 Normalization) để chiều dài vector = 1
                avg_vector = avg_vector / np.linalg.norm(avg_vector)
                
                # 3. Ghi duy nhất 1 dòng cho sinh viên này vào database
                row = {"MSSV": mssv, "HoTen": ho_ten}
                for idx in range(512):
                    row[f"emb_{idx}"] = float(avg_vector[idx])
                database_records.append(row)
            else:
                print(f"\n⚠️ Cảnh báo: Không tìm thấy khuôn mặt hợp lệ nào để tạo Face Bank cho SV {mssv} - {ho_ten}")

    df_db = pd.DataFrame(database_records)
    # Gắn MSSV làm index để tối ưu hóa truy xuất
    df_db.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎯 ĐÃ TẠO XONG FACE BANK THÀNH CÔNG!")
    print(f"📁 Tổng số vector: {len(database_records)}")
    print(f"📁 Đường dẫn lưu: {OUTPUT_CSV}")

if __name__ == '__main__':
    build_face_bank()