# inference/inferencer.py

import jetson.utils
import jetson.inference
import logging
import numpy as np
from typing import List, Any # 引入類型提示

logger = logging.getLogger(__name__)

class BaseInferencer:
    """
    邊緣模型推論器的基類。
    """
    def __init__(self, model, class_mapping: dict = None):
        """
        初始化推論器。
        Args:
            model: 載入的模型實例 (如 jetson.inference.detectNet)。
            class_mapping (dict, optional): 模型原始類別 ID 到內部類別名稱的映射。Defaults to None.
        """
        if model is None:
            raise ValueError("Model cannot be None for Inferencer.")
        self.model = model
        self.class_mapping = class_mapping if class_mapping is not None else {}

    def infer(self, frame_cuda: jetson.utils.cudaImage) -> Any:
        """
        在給定的 CUDA 影像上執行模型推論。
        這個方法應由子類實現。
        Args:
            frame_cuda (jetson.utils.cudaImage): CUDA 影像數據。
        Returns:
            Any: 推論結果。
        """
        raise NotImplementedError("Subclass must implement abstract method 'infer'")

class ObjectDetector(BaseInferencer):
    """
    物件偵測模型推論器。
    """
    def infer(self, frame_cuda: jetson.utils.cudaImage) -> List:
        """
        執行物件偵測推論。
        Args:
            frame_cuda (jetson.utils.cudaImage): CUDA 影像數據。
        Returns:
            List: 偵測結果列表。
        """
        if not isinstance(self.model, jetson.inference.detectNet):
             logger.error("指定的模型不是 jetson.inference.detectNet 類型。")
             return []
        return self.model.Detect(frame_cuda) # 執行偵測並返回結果

# 可擴展其他推論器，例如：
# class Classifier(BaseInferencer):
#     """
#     影像分類模型推論器。
#     """
#     def infer(self, frame_cuda: jetson.utils.cudaImage) -> Any: # 根據模型輸出類型調整返回類型
#          # 執行分類推論的邏輯
#          pass
#
# class PoseEstimator(BaseInferencer):
#     """
#     姿勢估計模型推論器。
#     """
#     def infer(self, frame_cuda: jetson.utils.cudaImage) -> Any: # 根據模型輸出類型調整返回類型
#          # 執行姿勢估計推論的邏輯
#          pass