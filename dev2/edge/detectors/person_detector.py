# detectors/person_detector.py

import os
import logging
import time # 添加 time 模組用於時間戳
from typing import List, Dict, Optional, Any
# 修正：將舊的導入方式改為新的帶底線的方式
# import jetson.inference # <-- 移除這行或註釋掉
import jetson.inference   # <-- 改為導入新的庫
import jetson.utils
from events.event_types import EventType
from events.event_manager import EventManager
from events.event_publisher import EventPublisher
from .base_detector import BaseDetector

# 引入 CaptureManager 和 FrameData
from data_capture.capture_manager import CaptureManager, FrameData # 引入 FrameData 從 capture_manager

# 引入 ObjectDetector 推論器類型
from inference.inferencer import ObjectDetector


logger = logging.getLogger(__name__)

class PersonDetector(BaseDetector):
    """
    專注於偵測人員和與人員相關的事件，並進行人臉識別。
    """
    def __init__(self, settings: dict,
                 object_detector: ObjectDetector, # 主要物件偵測器
                 event_manager: EventManager,
                 event_publisher: EventPublisher,
                 capture_manager: CaptureManager):
        """
        初始化人員偵測器。
        Args:
            settings (dict): 此偵測器的特定設定 (config.detectors.person)。
            object_detector (ObjectDetector): 主要物件偵測推論器實例。
            event_manager (EventManager): 事件管理器實例。
            event_publisher (EventPublisher): 事件發布器實例。
            capture_manager (CaptureManager): 捕獲管理器實例。
        """
        # 修正：移除 face_detector 參數
        super().__init__(settings, object_detector, event_manager, event_publisher, capture_manager)

        # 移除對 face_detector, face_recognizer 的引用
        # self.face_detector = face_detector # <-- 移除
        # self.face_recognizer = face_recognizer # <-- 移除

        self.person_class_name = self.settings.get('class_name', 'person')
        self.cooldown_seconds = self.settings.get('cooldown_seconds', 10) # 人物偵測事件冷卻時間

        # 新增：從設定中獲取是否在偵測到人物時觸發雲端識別事件
        self.alert_on_person_detection = self.settings.get('alert_on_person_detection', True)

        # 新增：獲取人臉識別影像的 S3 檔案夾前綴
        self.s3_face_recognition_folder = self.capture_manager.s3_settings.get('s3_face_recognition_folder')
        if not self.s3_face_recognition_folder:
             logger.error("settings.yaml 中未設定 aws.s3.s3_face_recognition_folder。人臉識別影像將無法正確上傳。")

        logger.info("PersonDetector 初始化成功 (觸發雲端人臉識別)。")

    # 修正：將 detections_raw 的類型提示從 List 改為 List[Any] 並在註釋中說明
    def process(self, frame_cuda: jetson.utils.cudaImage, detections_raw: List[Any]):
        """
        處理人員偵測邏輯。
        在偵測到人物後，觸發雲端進行人臉識別的事件。
        Args:
            frame_cuda (jetson.utils.cudaImage): 當前幀的 CUDA 影像數據。
            detections_raw (List[Any]): 物件偵測模型輸出的原始偵測結果列表 (預期類型為 List[jetson.inference.Detection])。
        """
        if not self.is_enabled:
            return

        # 篩選出人員偵測結果
        person_detections = [
            det for det in detections_raw
            if det and self.object_detector.class_mapping.get(det.ClassID) == self.person_class_name
        ]

        # --------------------------------------------------------------------
        # 規則範例 1: 偵測到至少一人，觸發雲端人臉識別事件
        # --------------------------------------------------------------------
        if self.alert_on_person_detection and len(person_detections) > 0:
            event_type = EventType.PERSON_FOR_IDENTIFICATION.value
            cooldown_key = event_type

            if self.event_manager.should_trigger_event(cooldown_key, cooldown_override=self.cooldown_seconds):
                logger.info(f"事件 '{event_type}' 觸發。")

                metadata: Dict[str, Any] = {
                    "person_count_in_frame": len(person_detections),
                    "person_detection_bbox": [int(person_detections[0].Left), int(person_detections[0].Top), int(person_detections[0].Right), int(person_detections[0].Bottom)] if person_detections else None,
                    "person_detection_confidence": float(person_detections[0].Confidence) if person_detections else None,
                    "frame_timestamp": time.time()
                }

                frame_buffer = self.capture_manager.get_frame_buffer()
                current_frame_data = frame_buffer[-1] if frame_buffer else None

                if current_frame_data:
                    # 修正：調用 capture_and_upload_image 時傳入人臉識別檔案夾前綴
                    if self.s3_face_recognition_folder:
                        s3_image_path = self.capture_manager.capture_and_upload_image(
                            event_type,
                            current_frame_data,
                            self.s3_face_recognition_folder, # <-- 傳入人臉識別檔案夾
                            metadata
                        )
                        if s3_image_path:
                            self.event_publisher.publish_event(event_type, s3_image_path=s3_image_path, metadata=metadata)
                            self.event_manager.record_event_triggered(cooldown_key)
                        else:
                            logger.warning(f"未能捕獲或添加到佇列影像用於事件 '{event_type}'。跳過發布事件訊息。")
                    else:
                        logger.error("未設定人臉識別影像 S3 檔案夾，跳過捕獲和發布事件。")
                else:
                     logger.error("捕獲管理器緩衝區為空，無法捕獲影像用於事件觸發。")

            # else:
            #      logger.debug(f"事件 '{event_type}' 仍在冷卻時間內，跳過觸發。")


        # --------------------------------------------------------------------
        # 可擴展其他人員相關規則 (如跌倒、靜止過久等，這些可能不需要人臉識別)
        # --------------------------------------------------------------------
        # 規則範例 2: 偵測到人員在特定區域 (需要額外的區域設定和檢查邏輯)
        # if self.settings.get('alert_in_restricted_area', False) and 'restricted_area' in self.settings:
        #     restricted_area_roi = self.settings.get('restricted_area')
        #     if restricted_area_roi:
        #          for person_det in person_detections:
        #               # 這裡使用基類中的 is_in_roi 方法，需要確保 BaseDetector 實現了這個方法
        #               if self._is_in_roi(person_det, restricted_area_roi):
        #                    # 這裡可以再決定是否對在限制區域內的人進行人臉識別
        #                    # 如果決定識別，走類似上面的邏輯，觸發 RestrictedArea_Known/Unknown 事件
        #                    # 如果不識別，只觸發一個 PERSON_IN_RESTRICTED_AREA 事件 (EventTypes 中需要定義)
        #                    event_type_restricted = "PERSON_IN_RESTRICTED_AREA" # 需要在 EventTypes 中定義
        #                    if self.event_manager.should_trigger_event(event_type_restricted): # 使用另一個冷卻
        #                         metadata_restricted = {
        #                              "person_bbox": [int(person_det.Left), int(person_det.Top), int(person_det.Right), int(person_det.Bottom)],
        #                              "restricted_area_roi": restricted_area_roi
        #                         }
        #                         # 捕獲當前幀用於此事件
        #                         current_frame_data = self.capture_manager.get_frame_buffer()[-1] # 獲取最新一幀
        #                         s3_path_restricted = self.capture_manager.capture_and_upload_image(event_type_restricted, current_frame_data, metadata_restricted)
        #                         if s3_path_restricted:
        #                              self.event_publisher.publish_event(event_type_restricted, s3_image_path=s3_path_restricted, metadata=metadata_restricted)
        #                              self.event_manager.record_event_triggered(event_type_restricted)
        #                         else:
        #                              logger.warning(f"未能捕獲影像用於限制區域事件 '{event_type_restricted}'。")
        #                    break # 找到一個在限制區域的人就處理一次

        # ... 其他規則範例 ...