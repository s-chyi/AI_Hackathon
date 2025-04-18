# detectors/base_detector.py

import logging
from typing import List, Any
import numpy as np
import jetson.inference

# 引入需要的模組
from inference.inferencer import ObjectDetector # 假設主要使用 ObjectDetector
from events.event_manager import EventManager
from events.event_publisher import EventPublisher
from data_capture.capture_manager import CaptureManager

logger = logging.getLogger(__name__)

class BaseDetector:
    """
    所有邊緣偵測器的基類。
    """
    def __init__(self, settings: dict,
                 object_detector: ObjectDetector,
                 event_manager: EventManager,
                 event_publisher: EventPublisher,
                 capture_manager: CaptureManager):
        """
        初始化基本偵測器。
        Args:
            settings (dict): 此偵測器的特定設定。
            object_detector (ObjectDetector): 物件偵測推論器實例。
            event_manager (EventManager): 事件管理器實例。
            event_publisher (EventPublisher): 事件發布器實例。
            capture_manager (CaptureManager): 捕獲管理器實例。
        """
        self.settings = settings
        self.object_detector = object_detector
        self.event_manager = event_manager
        self.event_publisher = event_publisher
        self.capture_manager = capture_manager

        self.is_enabled = self.settings.get('enabled', False)
        if not self.is_enabled:
            logger.info(f"偵測器 '{self.__class__.__name__}' 已禁用。")

    def process(self, frame_cuda: jetson.utils.cudaImage, detections_raw: List):
        """
        處理單個影像幀和原始偵測結果。
        這是核心邏輯，應由子類實現。
        Args:
            frame_cuda (jetson.utils.cudaImage): 當前幀的 CUDA 影像數據。
            detections_raw (List): 物件偵測模型輸出的原始偵測結果列表。
        """
        if not self.is_enabled:
            return

        # 子類應在此處實現其特定的偵測邏輯
        # 例如：過濾出感興趣的物件，應用規則判斷，決定是否觸發事件

        # 範例：簡單地印出正在處理
        # logger.debug(f"偵測器 '{self.__class__.__name__}' 正在處理幀...")

    def _trigger_event(self, event_type: str, metadata: dict = None, cooldown_override: float = None):
        """
        內部方法，用於觸發一個事件。會先經過 EventManager 檢查冷卻時間。
        Args:
            event_type (str): 要觸發的事件類型 (string 或 EventType value)。
            metadata (dict, optional): 事件相關的元數據。Defaults to None.
            cooldown_override (float, optional): 此觸發的冷卻時間覆蓋值。Defaults to None.
        """
        if self.event_manager.should_trigger_event(event_type, cooldown_override):
            logger.info(f"事件 '{event_type}' 觸發。")
            # 捕獲相關影像並添加到 S3 上傳佇列
            s3_path = self.capture_manager.capture_and_upload_image(event_type, metadata)

            # 如果影像成功添加到上傳佇列 (即使尚未完成上傳)，發布事件訊息
            # 注意：這裡只檢查 s3_path 是否為 None，表示捕獲管理器是否成功創建了上傳任務
            # 上傳任務本身可能還在執行中或失敗
            if s3_path is not None:
                # 修正：將關鍵字參數名稱從 s3_path 改為 s3_image_path
                self.event_publisher.publish_event(event_type, s3_image_path=s3_path, metadata=metadata)
                self.event_manager.record_event_triggered(event_type) # 記錄觸發時間
            else:
                 # 如果 capture_and_upload_image 返回 None (表示捕獲或添加到佇列失敗)
                 # 這裡可以選擇是否仍然發布一個不包含影像路徑的事件，或者完全不發布
                 # 目前的邏輯是如果不包含路徑就不發布，可以根據需求調整
                 logger.warning(f"未能捕獲或添加到佇列影像用於事件 '{event_type}'，跳過發布事件訊息。")
                 # 如果需要即使沒有影像也發布事件，取消註釋下面一行
                 # self.event_publisher.publish_event(event_type, metadata=metadata)
                 # self.event_manager.record_event_triggered(event_type) # 記錄觸發時間 (如果決定發布事件)

# 可擴展其他基類方法，例如根據 ROI 判斷目標是否在區域內
# def is_in_roi(self, detection: jetson.inference.Detection, roi: list) -> bool:
#     """
#     檢查偵測到的物體是否在指定的 ROI 內。
#     Args:
#         detection (jetson.inference.Detection): 偵測結果。
#         roi (list): ROI 座標 [x1, y1, x2, y2]。
#     Returns:
#         bool: 如果在 ROI 內則為 True，否則為 False。
#     """
#     det_center_x = (detection.Left + detection.Right) / 2
#     det_center_y = (detection.Top + detection.Bottom) / 2
#     return roi[0] <= det_center_x <= roi[2] and roi[1] <= det_center_y <= roi[3]