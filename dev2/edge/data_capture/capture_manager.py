# data_capture/capture_manager.py

import cv2
import numpy as np
import time
import logging
from datetime import datetime
from typing import Dict, Any
from utils.s3_uploader import S3Uploader # 引入 S3 上傳器

logger = logging.getLogger(__name__)

class CaptureManager:
    """
    管理事件觸發時的影像/短片捕獲和上傳。
    """
    def __init__(self, s3_uploader: S3Uploader, s3_settings: dict):
        """
        初始化捕獲管理器。
        Args:
            s3_uploader (S3Uploader): S3 上傳器實例。
            s3_settings (dict): S3 相關設定，包含 bucket_name, upload_folder。
        """
        self.s3_uploader = s3_uploader
        self.s3_settings = s3_settings
        self._current_frame: np.ndarray | None = None # 儲存當前最新一幀影像
        self._frame_timestamp: float = 0 # 儲存最新一幀的時間戳

    def update_frame(self, frame: np.ndarray):
        """
        更新捕獲管理器中的當前影像。
        應在主循環中處理完一幀後調用。
        Args:
            frame (np.ndarray): 當前的 OpenCV 影像幀。
        """
        self._current_frame = frame
        self._frame_timestamp = time.time()

    def capture_and_upload_image(self, event_type: str, metadata: Dict[str, Any] = None) -> str | None:
        """
        捕獲當前影像並添加到 S3 上傳佇列。
        Args:
            event_type (str): 觸發捕獲的事件類型。
            metadata (Dict[str, Any], optional): 與捕獲相關的元數據。Defaults to None.
        Returns:
            str | None: 如果成功添加到佇列，返回 S3 的目標 Key；否則返回 None。
        """
        if self._current_frame is None:
            logger.warning("沒有當前影像可用，無法捕獲。")
            return None

        # 複製影像以防止在處理過程中被修改
        frame_to_save = self._current_frame.copy()

        # 生成 S3 檔案路徑
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # 包含毫秒
        s3_folder = self.s3_settings.get('upload_folder', 'uploads/')
        # 檔案命名可以包含事件類型等信息，方便在 S3 中組織
        s3_key = f"{s3_folder}{event_type.lower()}_{timestamp_str}.jpg"

        # 將影像編碼為 JPG 格式的 Bytes
        try:
            # 壓縮品質可調整 (e.g., cv2.IMWRITE_JPEG_QUALITY, 90)
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
            bucket_name = self.s3_settings.get('bucket_name', 'your-default-bucket') # 確保 bucket_name 可用
            return f"s3://{bucket_name}/{s3_key}"
        except Exception as e:
            logger.error(f"添加上傳任務到佇列時發生錯誤: {e}", exc_info=True)
            return None

    # 可擴展實現短片捕獲邏輯 (需要維護一個幀緩衝區和一個視訊寫入器)
    # def start_clip_capture(self, duration_seconds: int):
    #     pass
    # def stop_clip_capture_and_upload(self, event_type: str, metadata: Dict[str, Any] = None):
    #     pass