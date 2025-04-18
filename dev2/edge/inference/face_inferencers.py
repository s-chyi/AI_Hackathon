# inference/face_inferencers.py

import logging
import jetson.inference
import jetson.utils
import numpy as np
import cv2
from typing import List, Any, Optional, Dict # 引入類型提示

# 配置 logging
logger = logging.getLogger(__name__)

# 引入 utils 模組來使用圖像處理和 CUDA 轉換
from utils import cuda_utils, image_utils

# 從 jetson-inference 導入 PoseNet 和 detectNet
# from jetson.inference import poseNet # 根據 jetson-inference 版本和模型類型可能需要 poseNet
# from jetson.inference import detectNet # 人臉偵測使用 detectNet("facenet")
# 假設 ModelManager 已經載入了模型實例並傳遞進來


class FaceDetector:
    """
    使用 jetson-inference 執行人臉偵測。
    """
    def __init__(self, model: jetson.inference.detectNet):
        """
        初始化人臉偵測器。
        Args:
            model (jetson.inference.detectNet): 已載入的人臉偵測模型實例 (例如 "facenet")。
        """
        if not isinstance(model, jetson.inference.detectNet):
             raise TypeError("FaceDetector 需要 jetson.inference.detectNet 實例。")
        self.model = model
        logger.info("FaceDetector 初始化成功。")


    def detect_faces(self, frame_cuda: jetson_utils.cudaImage) -> List[jetson.inference.Detection]:
        """
        在給定的 CUDA 影像上執行人臉偵測。
        Args:
            frame_cuda (jetson_utils.cudaImage): CUDA 影像數據。
        Returns:
            List[jetson.inference.Detection]: 人臉偵測結果列表。
        """
        if frame_cuda is None:
            logger.warning("輸入的 CUDA 影像為 None，無法執行人臉偵測。")
            return []
        try:
            # 執行偵測
            detections = self.model.Detect(frame_cuda)
            # logger.debug(f"偵測到 {len(detections)} 張人臉。")
            return detections
        except Exception as e:
            logger.error(f"執行人臉偵測時發生錯誤: {e}", exc_info=True)
            return []

# Jetson-inference 的人臉特徵提取通常使用 resnet18-facenet 模型，載入到 poseNet 中
# 這裡假設使用 poseNet 載入 resnet18-facenet
class FaceEmbedder:
    """
    使用 jetson-inference 執行人臉特徵提取 (Embedding)。
    """
    def __init__(self, model: jetson.inference.poseNet): # 根據實際模型載入方式調整類型提示
        """
        初始化人臉特徵提取器。
        Args:
            model: 已載入的人臉特徵提取模型實例 (例如 resnet18-facenet 載入到 poseNet)。
        """
        # 根據實際情況調整檢查類型
        # if not isinstance(model, jetson.inference.poseNet):
        #      raise TypeError("FaceEmbedder 需要 jetson.inference.poseNet 實例 (用於 resnet18-facenet)。")
        self.model = model
        logger.info("FaceEmbedder 初始化成功。")

    # Jetson-inference 的 poseNet.Process 方法通常需要 CUDA 影像作為輸入
    def get_embedding(self, face_cuda_image: jetson_utils.cudaImage) -> Optional[np.ndarray]:
        """
        從裁剪好的人臉 CUDA 影像中提取特徵向量。
        Args:
            face_cuda_image (jetson_utils.cudaImage): 裁剪好的人臉 CUDA 影像。
        Returns:
            Optional[np.ndarray]: 人臉特徵向量 (NumPy 陣列)，如果失敗則為 None。
        """
        if face_cuda_image is None:
            logger.warning("輸入的人臉 CUDA 影像為 None，無法提取特徵。")
            return None
        try:
            # 執行特徵提取
            # poseNet.Process 返回的是關鍵點或 embedding，根據模型調整處理邏輯
            # 對於 resnet18-facenet，Process 方法直接返回 embedding 向量
            embedding = self.model.Process(face_cuda_image)
            if embedding is not None:
                 # 確保 embedding 是 NumPy 陣列 (如果 SDK 返回的是其他類型)
                 return np.asarray(embedding)
            else:
                 logger.warning("人臉特徵提取返回 None。")
                 return None

        except Exception as e:
            logger.error(f"執行人臉特徵提取時發生錯誤: {e}", exc_info=True)
            return None

# 可選：人臉對齊輔助函數 (如果需要，比較複雜)
# def align_face(frame_cuda, face_detection):
#    # 使用 openCV 或其他庫進行人臉對齊
#    pass

class FaceMatcher:
    """
    計算兩個特徵向量的相似度。
    """
    def __init__(self, match_threshold: float):
        """
        初始化人臉比對器。
        Args:
            match_threshold (float): 相似度閾值。
        """
        self.match_threshold = match_threshold
        logger.info(f"FaceMatcher 初始化成功，比對閾值: {match_threshold}")


    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        計算兩個特徵向量的餘弦相似度。
        Args:
            embedding1 (np.ndarray): 第一個特徵向量。
            embedding2 (np.ndarray): 第二個特徵向量。
        Returns:
            float: 餘弦相似度 (值介於 -1 到 1 之間)。
        """
        if embedding1 is None or embedding2 is None or embedding1.shape != embedding2.shape:
            logger.warning("特徵向量無效，無法比對。")
            return -1.0 # 返回一個表示無效或低相似度的值

        # 計算餘弦相似度
        # 餘弦相似度 = 向量點積 / (向量1 的範數 * 向量2 的範數)
        dot_product = np.dot(embedding1, embedding2)
        norm_a = np.linalg.norm(embedding1)
        norm_b = np.linalg.norm(embedding2)

        if norm_a == 0 or norm_b == 0:
            logger.warning("特徵向量範數為零，無法計算餘弦相似度。")
            return -1.0 # 返回一個表示無效的值

        similarity = dot_product / (norm_a * norm_b)
        # logger.debug(f"計算得到相似度: {similarity}")
        return float(similarity) # 返回 float 類型


    def is_match(self, embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
        """
        判斷兩個特徵向量是否匹配 (相似度是否超過閾值)。
        Args:
            embedding1 (np.ndarray): 第一個特徵向量。
            embedding2 (np.ndarray): 第二個特徵向量。
        Returns:
            bool: 如果相似度超過閾值則為 True，否則為 False。
        """
        similarity = self.compare_embeddings(embedding1, embedding2)
        return similarity >= self.match_threshold

# 簡單的人臉清晰度評估 (使用 Laplacian Variance)
def estimate_sharpness(image_np: np.ndarray) -> float:
    """
    估計影像清晰度，使用 Laplacian Variance 方法。
    Args:
        image_np (np.ndarray): OpenCV 格式的影像 (灰階或彩色)。
    Returns:
        float: Laplacian Variance 值，值越高通常越清晰。
    """
    if image_np is None or image_np.size == 0:
        return 0.0

    try:
        # 轉換為灰階 (如果不是)
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        # 計算 Laplacian 變異數
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # logger.debug(f"計算得到清晰度: {laplacian_var}")
        return float(laplacian_var)

    except Exception as e:
        logger.error(f"估計影像清晰度時發生錯誤: {e}", exc_info=True)
        return 0.0