#!/usr/bin/env python3

import jetson.inference
import jetson.utils
import numpy as np
import cv2
import os
import json
import logging
import argparse
import time # 添加 time 模組，雖然在這個腳本目前沒用，但在其他模組可能用到
from typing import List, Dict, Any, Optional # 添加 Optional 類型提示

# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定義人臉相關模型名稱 (這是全局變數)
FACE_DETECTION_MODEL_NAME = "facenet"
FACE_EMBEDDING_MODEL_NAME = "resnet18-facenet"

# 定義最低信心度閾值和最低照片數量 (這些是全局變數，將在 main 函數中修改)
FACE_DETECTION_THRESHOLD = 0.9
MIN_PHOTOS_PER_PERSON = 3

# 這裡的 process_person_photos 函數需要訪問全局變數 FACE_DETECTION_THRESHOLD 和 MIN_PHOTOS_PER_PERSON
# 它們沒有在函數內部賦值，所以理論上不需要 global 聲明
# 但是為了確保它讀取到的是 main 函數修改後的全局值，而不是最初定義的值，
# 或者如果未來在函數內部需要賦值，提前添加 global 聲明是一種好的防禦性編程習慣。
# 這裡暫時不加 global，只在 main 函數中加，因為只有 main 函數對它們進行了賦值。
def process_person_photos(person_id: str, photo_paths: List[str],
                          face_detector: jetson.inference.detectNet,
                          face_embedder: jetson.inference.poseNet) -> Optional[np.ndarray]:
    # ... (函數內容與之前相同) ...
    embeddings: List[np.ndarray] = []
    logger.info(f"處理人物 '{person_id}' 的 {len(photo_paths)} 張照片...")

    for photo_path in photo_paths:
        if not os.path.exists(photo_path):
            logger.warning(f"照片檔案 '{photo_path}' 不存在，跳過。")
            continue

        try:
            img_np = cv2.imread(photo_path)
            if img_np is None:
                logger.warning(f"無法讀取影像檔案 '{photo_path}'，跳過。")
                continue

            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_cuda = jetson.utils.cudaFromNumpy(img_rgb)

            face_detections = face_detector.Detect(img_cuda)

            if not face_detections:
                logger.warning(f"在照片 '{photo_path}' 中未能偵測到人臉。")
                continue

            if not face_detections: # 重複檢查，可移除
                 logger.warning(f"在照片 '{photo_path}' 中未能偵測到人臉。")
                 continue

            best_face_det = max(face_detections, key=lambda det: det.Confidence)

            # 使用全局變數 FACE_DETECTION_THRESHOLD
            if best_face_det.Confidence < FACE_DETECTION_THRESHOLD:
                logger.warning(f"在照片 '{photo_path}' 偵測到的人臉信心度 ({best_face_det.Confidence:.2f}) 低於閾值 ({FACE_DETECTION_THRESHOLD:.2f})，跳過。")
                continue

            face_bbox = [int(best_face_det.Left), int(best_face_det.Top), int(best_face_det.Right), int(best_face_det.Bottom)]
            face_bbox[0] = max(0, face_bbox[0])
            face_bbox[1] = max(0, face_bbox[1])
            face_bbox[2] = min(img_np.shape[1], face_bbox[2])
            face_bbox[3] = min(img_np.shape[0], face_bbox[3])

            if face_bbox[2] <= face_bbox[0] or face_bbox[3] <= face_bbox[1]:
                 logger.warning(f"照片 '{photo_path}' 裁剪到的人臉區域無效，跳過。")
                 continue

            face_img_np = img_np[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
            face_img_rgb = cv2.cvtColor(face_img_np, cv2.COLOR_BGR2RGB)
            face_img_cuda = jetson.utils.cudaFromNumpy(face_img_rgb)

            embedding = face_embedder.Process(face_img_cuda)

            if embedding is not None:
                embeddings.append(np.asarray(embedding))
                logger.info(f"從照片 '{photo_path}' 成功提取人臉 Embedding。")
            else:
                 logger.warning(f"從照片 '{photo_path}' 提取人臉 Embedding 失敗。")

        except Exception as e:
            logger.error(f"處理照片 '{photo_path}' 時發生錯誤: {e}", exc_info=True)

    if not embeddings:
        logger.warning(f"人物 '{person_id}' 未能從任何照片中提取到有效 Embedding。")
        return None
    
    # 使用全局變數 MIN_PHOTOS_PER_PERSON
    if len(embeddings) < MIN_PHOTOS_PER_PERSON:
         logger.warning(f"人物 '{person_id}' 只提取到 {len(embeddings)} 個 Embedding，少於最低要求 {MIN_PHOTOS_PER_PERSON} 個。可能影響識別準確度。")
         # 這裡可以選擇返回 None 或強制計算平均值
         # return None # 如果希望強制要求最低照片數量

    average_embedding = np.mean(embeddings, axis=0)
    logger.info(f"人物 '{person_id}' 成功從 {len(embeddings)} 張照片計算平均 Embedding。")

    return average_embedding


def main():
    # 修正：將 global 聲明移動到函數的最開始
    global FACE_DETECTION_THRESHOLD, MIN_PHOTOS_PER_PERSON

    parser = argparse.ArgumentParser(description='Build known faces database from photos.')
    parser.add_argument('--photos_dir', type=str, required=True,
                        help='Directory containing subdirectories for each person. Subdirectory names are used as person IDs.')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Output JSON file path for the known faces database.')
    parser.add_argument('--face_det_model', type=str, default=FACE_DETECTION_MODEL_NAME, # 使用全局默認值
                        help=f'Jetson-inference face detection model name (default: {FACE_DETECTION_MODEL_NAME})')
    parser.add_argument('--face_emb_model', type=str, default=FACE_EMBEDDING_MODEL_NAME, # 使用全局默認值
                        help=f'Jetson-inference face embedding model name (default: {FACE_EMBEDDING_MODEL_NAME})')
    parser.add_argument('--det_threshold', type=float, default=FACE_DETECTION_THRESHOLD, # 使用全局默認值
                        help=f'Face detection confidence threshold (default: {FACE_DETECTION_THRESHOLD})')
    parser.add_argument('--min_photos', type=int, default=MIN_PHOTOS_PER_PERSON, # 使用全局默認值
                        help=f'Minimum number of photos required per person (default: {MIN_PHOTOS_PER_PERSON})')


    args = parser.parse_args()

    # 這些賦值將修改全局變數
    FACE_DETECTION_THRESHOLD = args.det_threshold
    MIN_PHOTOS_PER_PERSON = args.min_photos

    # 載入人臉偵測和特徵提取模型
    try:
        logger.info(f"載入人臉偵測模型: {args.face_det_model}, 閾值: {FACE_DETECTION_THRESHOLD}")
        face_detector = jetson.inference.detectNet(args.face_det_model, threshold=FACE_DETECTION_THRESHOLD)
        if face_detector is None:
             logger.error("人臉偵測模型載入失敗。請檢查模型名稱和安裝。")
             return

        logger.info(f"載入人臉特徵提取模型: {args.face_emb_model}")
        face_embedder = jetson.inference.poseNet(args.face_emb_model)
        if face_embedder is None:
             logger.error("人臉特徵提取模型載入失敗。請檢查模型名稱和安裝。")
             return

    except Exception as e:
        logger.error(f"載入模型時發生錯誤: {e}", exc_info=True)
        return

    known_faces_data: Dict[str, List[float]] = {}

    if not os.path.isdir(args.photos_dir):
        logger.error(f"輸入照片文件夾 '{args.photos_dir}' 不存在或不是一個文件夾。")
        return

    person_dirs = [d for d in os.listdir(args.photos_dir) if os.path.isdir(os.path.join(args.photos_dir, d))]

    if not person_dirs:
        logger.warning(f"在文件夾 '{args.photos_dir}' 中未找到任何人物子文件夾。")
        return

    logger.info(f"找到 {len(person_dirs)} 個人物文件夾。")

    for person_id in person_dirs:
        person_photo_dir = os.path.join(args.photos_dir, person_id)
        photo_paths = [os.path.join(person_photo_dir, f) for f in os.listdir(person_photo_dir) if os.path.isfile(os.path.join(person_photo_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not photo_paths:
            logger.warning(f"人物文件夾 '{person_id}' 中沒有找到圖片檔案，跳過。")
            continue

        average_embedding = process_person_photos(person_id, photo_paths, face_detector, face_embedder)

        if average_embedding is not None:
            known_faces_data[person_id] = average_embedding.tolist()
        else:
            logger.warning(f"未能為人物 '{person_id}' 生成 Embedding，將其從數據庫中排除。")

    if not known_faces_data:
        logger.warning("未生成任何有效的人臉 Embedding 數據。")
        return

    try:
        output_dir = os.path.dirname(args.output_json)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True)
             logger.info(f"創建輸出文件夾: {output_dir}")
        # 處理 args.output_json 是根目錄的情況，os.path.dirname('/') 是 '/' 或 ''
        if not output_dir: # 如果 output_json 是文件名，dirname 為空字串
             output_dir = '.' # 設置為當前目錄

        # 確保目標檔案可寫 (可能需要 sudo 權限，例如 /data/known_faces.json)
        # 或者建議將 output_json 指向用戶主目錄下的路徑
        if not os.access(output_dir, os.W_OK):
             logger.error(f"無權限寫入到輸出文件夾 '{output_dir}'。請檢查權限或更改輸出路徑。")
             return

        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(known_faces_data, f, indent=4)

        logger.info(f"已知人臉數據庫已成功保存到 '{args.output_json}'，包含 {len(known_faces_data)} 個人物。")

    except Exception as e:
        logger.error(f"保存已知人臉數據庫檔案時發生錯誤: {e}", exc_info=True)


if __name__ == "__main__":
    main()