# inference/face_recognizer.py

import logging
import jetson.utils
import numpy as np
import cv2
import time # 用於時間戳比較
from typing import List, Any, Optional, Tuple

# 引入人臉相關的推論器和數據庫
from .face_inferencers import FaceDetector, FaceEmbedder, FaceMatcher, estimate_sharpness
from .face_db import KnownFacesDB

# 引入 utils 和 capture_manager
from utils import cuda_utils, image_utils
# 這裡不直接引入 capture_manager，而是在方法調用時接收其提供的幀緩衝區


logger = logging.getLogger(__name__)

# 定義一個結構來存儲帶有元數據的幀
class FrameData:
    def __init__(self, frame_np: np.ndarray, frame_cuda: jetson.utils.cudaImage,
                 timestamp: float, detections_raw: List[jetson.inference.Detection]):
        self.frame_np = frame_np # NumPy 格式的原始幀 (用於裁剪等 OpenCV 操作)
        self.frame_cuda = frame_cuda # CUDA 格式的原始幀 (用於 jetson-inference 推論)
        self.timestamp = timestamp
        self.detections_raw = detections_raw # 這幀的物件偵測結果

class FaceRecognizer:
    """
    整合人臉偵測、特徵提取和比對，用於從幀緩衝區中識別人物。
    """
    def __init__(self, settings: dict, face_detector: FaceDetector,
                 face_embedder: FaceEmbedder, known_faces_db: KnownFacesDB):
        """
        初始化人臉識別器。
        Args:
            settings (dict): 人臉識別相關設定 (config.known_faces_db, config.detectors.person 中的部分)。
            face_detector (FaceDetector): 人臉偵測推論器實例。
            face_embedder (FaceEmbedder): 人臉特徵提取推論器實例。
            known_faces_db (KnownFacesDB): 已知人臉數據庫實例。
        """
        self.settings = settings
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.known_faces_db = known_faces_db

        # 從設定中獲取人臉識別閾值
        self.match_threshold = self.settings.get('known_faces_db', {}).get('match_threshold', 0.6)
        self.face_matcher = FaceMatcher(self.match_threshold) # 使用 FaceMatcher 類別

        # 從設定中獲取人臉偵測所需的最低信心度
        self.min_face_detection_confidence = self.settings.get('detectors', {}).get('person', {}).get('min_face_detection_confidence', 0.95)

        # 可選：人臉清晰度最低要求
        self.min_face_sharpness = self.settings.get('detectors', {}).get('person', {}).get('min_face_sharpness') # 預設為 None，不檢查清晰度


        logger.info("FaceRecognizer 初始化成功。")
        logger.info(f"人臉識別比對閾值: {self.match_threshold}")
        logger.info(f"人臉偵測最低信心度: {self.min_face_detection_confidence}")
        if self.min_face_sharpness is not None:
             logger.info(f"人臉區域最低清晰度 (Laplacian Variance): {self.min_face_sharpness}")


    def find_and_recognize_best_face(self, frame_buffer: List[FrameData], person_detection) -> \
            Tuple[Optional[str], Optional[float], Optional[float], Optional[np.ndarray], Optional[List[int]], Optional[float]]:
        """
        從幀緩衝區中尋找包含指定人物偵測結果的最佳幀，
        並在該幀上執行人臉偵測和識別。

        Args:
            frame_buffer (List[FrameData]): 包含最近幀數據的列表。
            person_detection (jetson_inference.Detection): 觸發識別人臉的當前幀中的人物偵測結果。

        Returns:
            Tuple[Optional[str], Optional[float], Optional[float], Optional[np.ndarray], Optional[List[int]], Optional[float]]:
                - person_id (Optional[str]): 識別到的人物 ID ('unknown' 或已知 ID)，如果未能識別則為 None。
                - face_detection_confidence (Optional[float]): **實際用於識別的人臉的**偵測信心度，如果未能找到人臉則為 None。
                - match_confidence (Optional[float]): 人臉**比對的**相似度信心度，如果未能識別為已知人物則為 None。
                - face_embedding (Optional[np.ndarray]): 提取到的人臉特徵向量，如果失敗則為 None。
                - face_bbox (Optional[List[int]]): 提取特徵使用的人臉 bounding box [x1, y1, x2, y2]，如果未能提取則為 None。
                - frame_timestamp (Optional[float]): 提取特徵使用的幀的時間戳，如果未能提取則為 None。
        """
        logger.debug(f"開始在 {len(frame_buffer)} 幀緩衝區中尋找最佳人臉進行識別...")

        best_frame_data: Optional[FrameData] = None
        best_face_detection: Optional[jetson.inference.Detection] = None
        best_face_sharpness: float = -1.0 # 用於記錄最佳人臉的清晰度

        # --------------------------------------------------------------------
        # 步驟 1: 從緩衝區中尋找最適合的幀和人臉
        # 這裡的邏輯可以有多種實現，例如：
        # - 簡單：只用 buffer 中最新的一幀
        # - 智能：遍歷 buffer，找到包含人物偵測框 (或與之重疊) 的幀，在這些幀上運行人臉偵測，
        #         然後根據人臉偵測信心度、人臉清晰度等指標選擇最佳人臉。
        # 這裡實現一個相對智能的邏輯：找到與人物偵測框中心點接近的幀中的人臉，並根據信心度和清晰度排序。
        # 為了簡化，我們遍歷 buffer，在每一幀上運行人臉偵測，找到符合條件的人臉，並選出最佳。
        # --------------------------------------------------------------------
        
        
        candidate_faces = [] # (face_det, frame_data, sharpness)

        # 遍歷緩衝區，從最新的幀開始往前找 (buffer 通常是按時間順序存儲)
        for frame_data in reversed(frame_buffer):
             # 在這幀上運行人臉偵測
             face_detections_in_frame = self.face_detector.detect_faces(frame_data.frame_cuda)

             for face_det in face_detections_in_frame:
                 # 檢查人臉偵測信心度是否足夠
                 if face_det.Confidence >= self.min_face_detection_confidence:
                     # 可選：裁剪人臉區域並評估清晰度
                     face_bbox_np = [int(face_det.Left), int(face_det.Top), int(face_det.Right), int(face_det.Bottom)]
                     face_image_np = frame_data.frame_np[face_bbox_np[1]:face_bbox_np[3], face_bbox_np[0]:face_bbox_np[2]]
                     current_sharpness = estimate_sharpness(face_image_np)

                     # 可選：檢查人臉區域是否清晰度足夠
                     if self.min_face_sharpness is None or current_sharpness >= self.min_face_sharpness:
                         # 將符合條件的人臉作為候選加入列表
                         candidate_faces.append((face_det, frame_data, current_sharpness))
                         logger.debug(f"找到一個候選人臉 (Conf: {face_det.Confidence:.2f}, Sharpness: {current_sharpness:.2f}) 在時間 {frame_data.timestamp}")
                     else:
                         logger.debug(f"找到人臉但清晰度不足 (Conf: {face_det.Confidence:.2f}, Sharpness: {current_sharpness:.2f} < {self.min_face_sharpness})")

        # 如果找到了候選人臉，根據某些指標排序並選擇最佳
        if not candidate_faces:
             logger.warning("在緩衝區中未找到符合條件的人臉進行識別。")
             return None, None, None, None, None, None

        # 排序候選人臉，例如按信心度降序，清晰度降序
        candidate_faces.sort(key=lambda x: (x[0].Confidence, x[2]), reverse=True)

        # 選擇最佳的人臉和其所在的幀
        best_face_detection, best_frame_data, best_face_sharpness = candidate_faces[0]
        logger.debug(f"從候選中選出最佳人臉 (Conf: {best_face_detection.Confidence:.2f}, Sharpness: {best_face_sharpness:.2f}) 在時間 {best_frame_data.timestamp}")


        # --------------------------------------------------------------------
        # 步驟 2: 從最佳人臉提取特徵向量
        # --------------------------------------------------------------------

        face_bbox_np = [int(best_face_detection.Left), int(best_face_detection.Top), int(best_face_detection.Right), int(best_face_detection.Bottom)]

        try:
            face_image_np = best_frame_data.frame_np[face_bbox_np[1]:face_bbox_np[3], face_bbox_np[0]:face_bbox_np[2]]
            face_cuda_image = cuda_utils.numpy_to_cuda(face_image_np)
        except Exception as e:
             logger.error(f"裁剪或轉換最佳人臉影像為 CUDA 時發生錯誤: {e}", exc_info=True)
             # 修正：如果提取特徵失敗，返回 unknown 和相關信息，但比對信心度和 embedding 為 None
             return "unknown", float(best_face_detection.Confidence) if best_face_detection else None, None, None, face_bbox_np, best_frame_data.timestamp


        # 提取特徵
        face_embedding = self.face_embedder.get_embedding(face_cuda_image)
        if face_embedding is None:
            logger.warning("未能提取人臉特徵向量。")
            # 修正：如果提取特徵失敗，返回 unknown 和相關信息，但比對信心度和 embedding 為 None
            return "unknown", float(best_face_detection.Confidence) if best_face_detection else None, None, None, face_bbox_np, best_frame_data.timestamp

        logger.debug(f"成功提取人臉特徵向量，維度: {face_embedding.shape}")

        # --------------------------------------------------------------------
        # 步驟 3: 與已知人臉數據庫比對
        # --------------------------------------------------------------------

        known_embeddings = self.known_faces_db.get_all_embeddings()
        best_match_id: Optional[str] = None
        best_match_similarity: float = self.match_threshold - 0.001 # 設置一個初始值略低於閾值

        if known_embeddings: # 只有當數據庫有內容時才進行比對
             for person_id, known_embedding in known_embeddings.items():
                 similarity = self.face_matcher.compare_embeddings(face_embedding, known_embedding)
                 if similarity > best_match_similarity:
                     best_match_similarity = similarity
                     best_match_id = person_id
        else:
            logger.warning("已知人臉數據庫為空，跳過人臉比對。")


        # --------------------------------------------------------------------
        # 步驟 4: 判斷識別結果並返回
        # --------------------------------------------------------------------

        final_person_id: str
        final_match_confidence: Optional[float] = None # 初始化最終比對信心度

        if best_match_id is not None and best_match_similarity >= self.match_threshold:
            final_person_id = best_match_id
            final_match_confidence = best_match_similarity
            logger.info(f"識別到已知人物: '{final_person_id}' (相似度: {final_match_confidence:.4f})")
        else:
            final_person_id = "unknown"
            final_match_confidence = best_match_similarity if best_match_id is not None else None # 如果沒有候選比對對象，則為 None
            logger.info(f"未能識別為已知人物") # 即使是 unknown，也記錄最佳相似度


        # 返回結果
        # 修正：確保返回值的順序和類型與簽名一致
        # person_id, face_detection_confidence, match_confidence, face_embedding, face_bbox, frame_timestamp
        return final_person_id, float(best_face_detection.Confidence) if best_face_detection else None, final_match_confidence, face_embedding, face_bbox_np if face_bbox_np else None, best_frame_data.timestamp if best_frame_data else None # 確保所有潛在的 None 都被處理