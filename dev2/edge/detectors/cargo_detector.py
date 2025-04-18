# detectors/cargo_detector.py

import logging
import time
from typing import List, Dict, Optional, Any
# 仍然需要 jetson_inference 獲取 Detection 類型
import jetson.inference
import jetson.utils # 需要 cudaImage 類型
import threading # 引入 threading

from events.event_types import EventType # 引入事件類型 (如果 CargoDetector 需要觸發事件)
from events.event_manager import EventManager # 引入 EventManager (如果 CargoDetector 需要觸發事件)
from events.event_publisher import EventPublisher # 引入 EventPublisher (如果 CargoDetector 需要發布事件)
from data_capture.capture_manager import CaptureManager, FrameData # 引入 CaptureManager 和 FrameData (如果 CargoDetector 需要捕獲影像)

from .base_detector import BaseDetector

from inference.inferencer import ObjectDetector # 引入 ObjectDetector 推論器類型

logger = logging.getLogger(__name__)

class CargoDetector(BaseDetector):
    """
    專注於偵測貨物和與貨物相關的事件，根據人員識別結果調整行為。
    """
    def __init__(self, settings: dict,
                 object_detector: ObjectDetector, # 主要物件偵測器 (偵測貨物和人)
                 event_manager: EventManager, # 如果需要觸發貨物事件
                 event_publisher: EventPublisher, # 如果需要發布貨物事件
                 capture_manager: CaptureManager, # 如果需要捕獲貨物事件影像
                 recognition_result_state: Dict[str, Any], # <-- 共享的最新識別結果狀態
                 recognition_result_lock: threading.Lock): # <-- 保護狀態的鎖
        """
        初始化貨物偵測器。
        Args:
            settings (dict): 此偵測器的特定設定 (config.detectors.cargo)。
            object_detector (ObjectDetector): 主要物件偵測推論器實例。
            event_manager (EventManager): 事件管理器實例 (如果需要觸發貨物事件)。
            event_publisher (EventPublisher): 事件發布器實例 (如果需要發布貨物事件)。
            capture_manager (CaptureManager): 捕獲管理器實例 (如果需要捕獲貨物事件影像)。
            recognition_result_state (Dict[str, Any]): 共享的最新雲端識別結果字典。
            recognition_result_lock (threading.Lock): 保護識別結果狀態的鎖。
        """
        # 這裡將所有依賴都傳給基類，即使 CargoDetector 可能只用到其中一部分
        # 在基類中，可以根據是否傳入相應實例來判斷功能是否可用
        super().__init__(settings, object_detector, event_manager, event_publisher, capture_manager)

        self.cargo_class_name = self.settings.get('class_name', 'cargo')
        self.cooldown_seconds = self.settings.get('cooldown_seconds', 30) # 貨物事件冷卻時間

        self.recognition_result_state = recognition_result_state # 儲存對共享狀態的引用
        self.recognition_result_lock = recognition_result_lock # 儲存鎖的引用

        logger.info("CargoDetector 初始化成功。")

    def process(self, frame_cuda: jetson.utils.cudaImage, detections_raw: List[Any]): # List[jetson_inference.Detection]
        """
        處理貨物偵測邏輯。
        Args:
            frame_cuda (jetson.utils.cudaImage): 當前幀的 CUDA 影像數據。
            detections_raw (List[Any]): 物件偵測模型輸出的原始偵測結果列表 (預期類型為 List[jetson_inference.Detection])。
        """
        if not self.is_enabled:
            return

        # 篩選出貨物偵測結果
        cargo_detections = [
            det for det in detections_raw
            if isinstance(det, jetson.inference.Detection) and self.object_detector.class_mapping.get(det.ClassID) == self.cargo_class_name
        ]

        # --------------------------------------------------------------------
        # 獲取最新的雲端識別結果
        # --------------------------------------------------------------------
        latest_person_id = "no_person"
        latest_result_timestamp = 0 # 收到結果的時間
        latest_original_event_timestamp = 0 # 觸發該結果的邊緣事件時間
        latest_match_confidence = None

        with self.recognition_result_lock:
            latest_person_id = self.recognition_result_state.get("person_id", "no_person")
            latest_result_timestamp = self.recognition_result_state.get("timestamp", 0)
            latest_original_event_timestamp = self.recognition_result_state.get("original_event_timestamp", 0)
            latest_match_confidence = self.recognition_result_state.get("match_confidence")


        # logger.debug(f"CargoDetector 獲取最新識別結果: {latest_person_id} (收到時間: {latest_result_timestamp})")


        # --------------------------------------------------------------------
        # 根據最新的識別結果，處理貨物偵測邏輯
        # --------------------------------------------------------------------

        # 判斷是否要處理貨物事件
        # 例如：只在識別到已知員工時才監控貨物狀態異常
        process_cargo_events = False
        if latest_person_id != "no_person" and latest_person_id != "unknown" and not latest_person_id.startswith("error_"):
            # 如果識別到已知員工
            # 您可能需要檢查收到結果的時間與當前時間的間隔，確保結果是「新鮮」的
            # 例如，如果結果是 30 秒前收到的，可能就過期了
            # 或者檢查邊緣事件時間戳與當前時間的間隔
            if (time.time() - latest_result_timestamp) < 30: # 假設結果 30 秒內有效
                process_cargo_events = True
                logger.debug(f"識別到已知人物 '{latest_person_id}'，啟用貨物事件處理。")
            else:
                logger.debug("識別結果已過期，跳過貨物事件處理。")
        elif latest_person_id == "unknown":
            # 如果偵測到未知人物，是否要特別監控貨物？
            # 根據需求調整
            # process_cargo_events = True # 例如，對未知人物加強監控
            logger.debug("偵測到未知人物，貨物事件處理未啟用。")
        else:
            logger.debug("未偵測到人物或識別失敗，貨物事件處理未啟用。")


        # 如果決定處理貨物事件
        if process_cargo_events and len(cargo_detections) > 0:
            pass
            # 實現具體的貨物偵測規則和事件觸發邏輯
            # 只有在有人物且人物符合特定條件時，才檢查貨物是否異常
            # 例如：
            # 規則範例 1: 偵測到貨物傾斜 (需要更複雜的邊緣模型或邏輯)
            # if self.settings.get('alert_on_tilt', False):
            #      for cargo_det in cargo_detections:
            #           if self._is_cargo_tilted(cargo_det): # 假設有判斷傾斜的方法
            #                event_type_cargo = EventType.CARGO_TILTED.value
            #                # 觸發事件，構建 metadata (包含貨物框，以及當前相關的人物 ID)
            #                metadata_cargo = {
            #                     "cargo_bbox": [int(cargo_det.Left), int(cargo_det.Top), int(cargo_det.Right), int(cargo_det.Bottom)],
            #                     "responsible_person_id": latest_person_id, # 附加當前最近的識別到的人物 ID
            #                     "person_match_confidence": latest_match_confidence # 附加信心度
            #                }
            #                # 檢查貨物事件冷卻時間
            #                if self.event_manager.should_trigger_event(event_type_cargo, cooldown_override=self.cooldown_seconds):
            #                     logger.info(f"事件 '{event_type_cargo}' 觸發 (與人物 {latest_person_id} 相關)。")
            #                     # 獲取當前幀用於捕獲
            #                     frame_buffer = self.capture_manager.get_frame_buffer()
            #                     current_frame_data = frame_buffer[-1] if frame_buffer else None
            #                     if current_frame_data:
            #                          s3_image_path = self.capture_manager.capture_and_upload_image(event_type_cargo, current_frame_data, metadata_cargo)
            #                          if s3_image_path:
            #                              self.event_publisher.publish_event(event_type_cargo, s3_image_path=s3_image_path, metadata=metadata_cargo)
            #                              self.event_manager.record_event_triggered(event_type_cargo)
            #                          else:
            #                              logger.warning(f"未能捕獲影像用於貨物傾斜事件 '{event_type_cargo}'。")
            #                     else:
            #                          logger.error("捕獲管理器緩衝區為空，無法捕獲影像用於貨物傾斜事件。")
            #                    break # 只觸發一次傾斜事件

            # 規則範例 2: 偵測到貨物在移動區域 (需要區域設定)
            # if self.settings.get('alert_on_movement_in_area', False):
            #      # ... 實現邏輯 ...
            #      pass

        # else:
        #      # 如果不處理貨物事件，則跳過所有貨物相關的偵測和觸發
        #      pass