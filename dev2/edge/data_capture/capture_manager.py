# data_capture/capture_manager.py

import cv2
import numpy as np
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List # 引入類型提示
import threading
import queue # 引入 queue 模組
import jetson.inference
import jetson.utils

# 引入 S3 上傳器和 FrameData 結構
from utils.s3_uploader import S3Uploader

logger = logging.getLogger(__name__)

class FrameData:
    def __init__(self, frame_np: np.ndarray, frame_cuda: jetson.utils.cudaImage,
                 timestamp: float, detections_raw: List):
        self.frame_np = frame_np # NumPy 格式的原始幀 (用於裁剪等 OpenCV 操作)
        self.frame_cuda = frame_cuda # CUDA 格式的原始幀 (用於 jetson-inference 推論)
        self.timestamp = timestamp
        self.detections_raw = detections_raw # 這幀的物件偵測結果 (預期是 jetson_inference.Detection 列表)

class CaptureManager:
    """
    管理事件觸發時的影像/短片捕獲和上傳。
    """
    def __init__(self, s3_uploader: S3Uploader, s3_settings: dict, capture_settings: dict):
        """
        初始化捕獲管理器。
        Args:
            s3_uploader (S3Uploader): S3 上傳器實例。
            s3_settings (dict): S3 相關設定，包含 bucket_name, upload_folder。
            capture_settings (dict): 捕獲相關設定，包含 frame_buffer_size。
        """
        self.s3_uploader = s3_uploader
        self.s3_settings = s3_settings
        self.capture_settings = capture_settings

        # 幀緩衝區 (使用列表實現簡單的循環緩衝)
        self._frame_buffer: List[FrameData] = []
        self._buffer_size = self.capture_settings.get('frame_buffer_size', 15) # 預設緩衝 15 幀

        # 保護緩衝區的鎖 (因為主循環和選幀邏輯可能同時訪問)
        self._buffer_lock = threading.Lock()

        logger.info(f"CaptureManager 初始化成功，幀緩衝區大小: {self._buffer_size}")

    def add_frame_to_buffer(self, frame_np: np.ndarray, frame_cuda: jetson.utils.cudaImage,
                           detections_raw: List):
        """
        將一幀影像數據添加到緩衝區。
        Args:
            frame_np (np.ndarray): OpenCV 格式的影像幀。
            frame_cuda (jetson.utils.cudaImage): CUDA 格式的影像幀。
            detections_raw (List[jetson.inference.Detection]): 這幀的物件偵測結果。
        """
        # 創建 FrameData 實例
        # 注意：深度複製影像可能消耗較多記憶體，對於邊緣設備需要謹慎。
        # 這裡為了示例，複製 NumPy 幀
        new_frame_data = FrameData(
            frame_np=frame_np.copy(), # 複製 NumPy 幀
            frame_cuda=frame_cuda, # CUDA 幀通常是設備內存，不需要複製
            timestamp=time.time(),
            detections_raw=detections_raw # detections_raw 已經是新列表
        )

        with self._buffer_lock:
            # 添加到緩衝區尾部
            self._frame_buffer.append(new_frame_data)
            # 如果緩衝區超過設定大小，移除最舊的幀 (頭部)
            while len(self._frame_buffer) > self._buffer_size:
                oldest_frame = self._frame_buffer.pop(0)
                # 如果 FrameData 中有需要顯式釋放的資源 (如 CUDA 內存)，在這裡處理
                # jetson_utils.cudaFromNumpy 創建的 CUDA 影像生命週期由它管理，通常不需要手動釋放
                del oldest_frame

            # logger.debug(f"幀已添加到緩衝區，當前大小: {len(self._frame_buffer)}")

    def get_frame_buffer(self) -> List[FrameData]:
        """
        獲取當前的幀緩衝區內容。
        返回緩衝區的拷貝，避免外部修改影響內部狀態。
        Returns:
            List[FrameData]: 幀緩衝區的拷貝。
        """
        with self._buffer_lock:
            return list(self._frame_buffer) # 返回列表的淺拷貝

    def capture_and_upload_image(self, event_type: str, frame_data: FrameData,
                                 metadata: Dict[str, Any] = None):
        """
        捕獲指定 FrameData 中的影像並添加到 S3 上傳佇列。
        Args:
            event_type (str): 觸發捕獲的事件類型。
            frame_data (FrameData): 要捕獲的特定幀數據。
            metadata (Dict[str, Any], optional): 與捕獲相關的元數據。Defaults to None.
        Returns:
            str | None: 如果成功添加到佇列，返回 S3 的目標 URL (包含 bucket)；否則返回 None。
        """
        if frame_data is None or frame_data.frame_np is None:
            logger.warning("指定的 FrameData 或影像數據為 None，無法捕獲。")
            return None

        # 使用 FrameData 中的 NumPy 影像進行處理
        frame_to_save = frame_data.frame_np

        # 生成 S3 檔案路徑
        timestamp_str = datetime.fromtimestamp(frame_data.timestamp).strftime("%Y%m%d%H%M%S%f") # 使用幀的時間戳
        s3_folder = self.s3_settings.get('upload_folder', 'uploads/')
        # 檔案命名可以包含事件類型和時間戳
        s3_key = f"icam-{timestamp_str}.jpg"

        # 將影像編碼為 JPG 格式的 Bytes
        try:
            ret, buffer = cv2.imencode('.jpg', frame_to_save)
            if not ret:
                logger.error(f"影像編碼失敗: {s3_key}")
                return None
            image_data = buffer.tobytes()
        except Exception as e:
            logger.error(f"影像編碼時發生錯誤: {e}", exc_info=True)
            return None

        # 將上傳任務添加到 S3 上傳器佇列
        try:
            self.s3_uploader.put_upload_task(image_data, s3_key)
            logger.info(f"已將影像捕獲任務添加到 S3 上傳佇列，S3 Key: {s3_key}")
            # 返回完整的 S3 URL 或 Key，供事件發布器使用
            bucket_name = self.s3_settings.get('bucket_name')
            if bucket_name:
                return f"s3://{bucket_name}/{s3_key}"
            else:
                 logger.error("S3 bucket_name 未設定。無法生成 S3 URL。")
                 return None
        except Exception as e:
            logger.error(f"添加上傳任務到佇列時發生錯誤: {e}", exc_info=True)
            return None


    # 可擴展實現短片捕獲邏輯 (需要維護一個幀緩衝區和一個視訊寫入器)
    # def start_clip_capture(self, duration_seconds: int):
    #     pass
    # def stop_clip_capture_and_upload(self, event_type: str, metadata: Dict[str, Any] = None):
    #     pass