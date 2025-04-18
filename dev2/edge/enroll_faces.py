#!/usr/bin/env python3

import jetson.inference
import jetson.utils
import numpy as np
import cv2
import os
import json
import logging
import argparse # 用於處理命令行參數
from typing import List, Dict, Any, Optional

# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定義人臉相關模型名稱
FACE_DETECTION_MODEL_NAME = "facenet"
FACE_EMBEDDING_MODEL_NAME = "resnet18-facenet"

# 定義最低信心度閾值
FACE_DETECTION_THRESHOLD = 0.9 # 人臉偵測信心度閾值，確保偵測到的是清晰的人臉
MIN_PHOTOS_PER_PERSON = 3 # 每個已知人物至少需要幾張照片來計算平均 Embedding

def process_person_photos(person_id: str, photo_paths: List[str],
                          face_detector: jetson.inference.detectNet,
                          face_embedder: jetson.inference.poseNet) -> Optional[np.ndarray]:
    """
    處理一個人物的多張照片，提取所有人臉 Embedding，並計算平均值。
    Args:
        person_id (str): 人物 ID。
        photo_paths (List[str]): 該人物所有照片的檔案路徑列表。
        face_detector (jetson.inference.detectNet): 已載入的人臉偵測模型實例。
        face_embedder (jetson.inference.poseNet): 已載入的人臉特徵提取模型實例。
    Returns:
        Optional[np.ndarray]: 該人物的平均人臉 Embedding，如果未能提取到足夠數量的有效 Embedding 則為 None。
    """
    embeddings: List[np.ndarray] = []
    logger.info(f"處理人物 '{person_id}' 的 {len(photo_paths)} 張照片...")

    for photo_path in photo_paths:
        if not os.path.exists(photo_path):
            logger.warning(f"照片檔案 '{photo_path}' 不存在，跳過。")
            continue

        try:
            # 讀取影像
            img_np = cv2.imread(photo_path)
            if img_np is None:
                logger.warning(f"無法讀取影像檔案 '{photo_path}'，跳過。")
                continue

            # 將 NumPy 影像轉換為 CUDA 影像
            # jetson.utils.cudaFromNumpy 需要 RGB 格式
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_cuda = jetson.utils.cudaFromNumpy(img_rgb)

            # 執行人臉偵測
            face_detections = face_detector.Detect(img_cuda)

            if not face_detections:
                logger.warning(f"在照片 '{photo_path}' 中未能偵測到人臉。")
                continue

            # 在偵測到的人臉中，選擇最可能是目標人物的人臉進行 Embedding 提取
            # 這裡簡化：選擇信心度最高的人臉
            best_face_det = max(face_detections, key=lambda det: det.Confidence)

            if best_face_det.Confidence < FACE_DETECTION_THRESHOLD:
                logger.warning(f"在照片 '{photo_path}' 偵測到的人臉信心度 ({best_face_det.Confidence:.2f}) 低於閾值 ({FACE_DETECTION_THRESHOLD:.2f})，跳過。")
                continue

            # 裁剪人臉區域 (NumPy 格式)
            face_bbox = [int(best_face_det.Left), int(best_face_det.Top), int(best_face_det.Right), int(best_face_det.Bottom)]
            # 確保裁剪範圍不超過影像邊界
            face_bbox[0] = max(0, face_bbox[0])
            face_bbox[1] = max(0, face_bbox[1])
            face_bbox[2] = min(img_np.shape[1], face_bbox[2])
            face_bbox[3] = min(img_np.shape[0], face_bbox[3])

            if face_bbox[2] <= face_bbox[0] or face_bbox[3] <= face_bbox[1]:
                 logger.warning(f"照片 '{photo_path}' 裁剪到的人臉區域無效，跳過。")
                 continue


            face_img_np = img_np[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]

            # 將裁剪的人臉 NumPy 影像轉換為 CUDA 影像
            # jetson.utils.cudaFromNumpy 期望 RGB
            face_img_rgb = cv2.cvtColor(face_img_np, cv2.COLOR_BGR2RGB)
            face_img_cuda = jetson.utils.cudaFromNumpy(face_img_rgb)

            # 執行人臉特徵提取
            embedding = face_embedder.Process(face_img_cuda) # poseNet.Process 返回 embedding

            if embedding is not None:
                embeddings.append(np.asarray(embedding))
                logger.info(f"從照片 '{photo_path}' 成功提取人臉 Embedding。")
            else:
                 logger.warning(f"從照片 '{photo_path}' 提取人臉 Embedding 失敗。")

        except Exception as e:
            logger.error(f"處理照片 '{photo_path}' 時發生錯誤: {e}", exc_info=True)

    # --------------------------------------------------------------------
    # 計算該人物的代表性 Embedding
    # --------------------------------------------------------------------
    if not embeddings:
        logger.warning(f"人物 '{person_id}' 未能從任何照片中提取到有效 Embedding。")
        return None
    
    if len(embeddings) < MIN_PHOTOS_PER_PERSON:
         logger.warning(f"人物 '{person_id}' 只提取到 {len(embeddings)} 個 Embedding，少於最低要求 {MIN_PHOTOS_PER_PERSON} 個。可能影響識別準確度。")
         # 這裡可以選擇返回 None 或強制計算平均值
         # return None # 如果希望強制要求最低照片數量

    # 計算所有有效 Embedding 的平均值
    average_embedding = np.mean(embeddings, axis=0)
    logger.info(f"人物 '{person_id}' 成功從 {len(embeddings)} 張照片計算平均 Embedding。")

    return average_embedding


def main():
    parser = argparse.ArgumentParser(description='Build known faces database from photos.')
    parser.add_argument('--photos_dir', type=str, required=True,
                        help='Directory containing subdirectories for each person. Subdirectory names are used as person IDs.')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Output JSON file path for the known faces database.')
    parser.add_argument('--face_det_model', type=str, default=FACE_DETECTION_MODEL_NAME,
                        help=f'Jetson-inference face detection model name (default: {FACE_DETECTION_MODEL_NAME})')
    parser.add_argument('--face_emb_model', type=str, default=FACE_EMBEDDING_MODEL_NAME,
                        help=f'Jetson-inference face embedding model name (default: {FACE_EMBEDDING_MODEL_NAME})')
    parser.add_argument('--det_threshold', type=float, default=FACE_DETECTION_THRESHOLD,
                        help=f'Face detection confidence threshold (default: {FACE_DETECTION_THRESHOLD})')
    parser.add_argument('--min_photos', type=int, default=MIN_PHOTOS_PER_PERSON,
                        help=f'Minimum number of photos required per person (default: {MIN_PHOTOS_PER_PERSON})')


    args = parser.parse_args()

    global FACE_DETECTION_THRESHOLD, MIN_PHOTOS_PER_PERSON
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
        # resnet18-facenet 載入到 poseNet
        face_embedder = jetson.inference.poseNet(args.face_emb_model)
        if face_embedder is None:
             logger.error("人臉特徵提取模型載入失敗。請檢查模型名稱和安裝。")
             return

    except Exception as e:
        logger.error(f"載入模型時發生錯誤: {e}", exc_info=True)
        return

    # 讀取人物照片文件夾結構
    # 假設 photos_dir 下有子文件夾，每個子文件夾是一個人物，文件夾名是 person_id
    # 例如： photos_dir/
    #         ├── employee_001/
    #         │   ├── photo1.jpg
    #         │   └── photo2.png
    #         └── visitor_A/
    #             ├── img_001.jpeg
    #             └── img_002.jpg
    
    known_faces_data: Dict[str, List[float]] = {} # 儲存最終的 {person_id: embedding_list}

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
        photo_paths = [os.path.join(person_photo_dir, f) for f in os.listdir(person_photo_dir) if os.path.isfile(os.path.join(person_photo_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))] # 篩選圖片檔案

        if not photo_paths:
            logger.warning(f"人物文件夾 '{person_id}' 中沒有找到圖片檔案，跳過。")
            continue

        average_embedding = process_person_photos(person_id, photo_paths, face_detector, face_embedder)

        if average_embedding is not None:
            known_faces_data[person_id] = average_embedding.tolist() # 將 NumPy 陣列轉換為列表以便保存為 JSON
        else:
            logger.warning(f"未能為人物 '{person_id}' 生成 Embedding，將其從數據庫中排除。")

    # 保存數據庫為 JSON 檔案
    if not known_faces_data:
        logger.warning("未生成任何有效的人臉 Embedding 數據。")
        return

    try:
        # 確保輸出文件夾存在
        output_dir = os.path.dirname(args.output_json)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True)
             logger.info(f"創建輸出文件夾: {output_dir}")

        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(known_faces_data, f, indent=4) # 使用 indent=4 使 JSON 檔案更易讀

        logger.info(f"已知人臉數據庫已成功保存到 '{args.output_json}'，包含 {len(known_faces_data)} 個人物。")

    except Exception as e:
        logger.error(f"保存已知人臉數據庫檔案時發生錯誤: {e}", exc_info=True)


if __name__ == "__main__":
    main()