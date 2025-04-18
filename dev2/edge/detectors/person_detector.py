# detectors/person_detector.py

import os
import logging
import time # 添加 time 模組用於時間戳
from typing import List, Dict, Optional, Any
# 修正：將舊的導入方式改為新的帶底線的方式
# import jetson.inference # <-- 移除這行或註釋掉
import jetson_inference   # <-- 改為導入新的庫
import jetson.utils
from events.event_types import EventType # 引入事件類型
from events.event_manager import EventManager
from events.event_publisher import EventPublisher
from .base_detector import BaseDetector # 引入基類

# 引入 FaceRecognizer 和 FrameData
from inference.face_recognizer import FaceRecognizer, FrameData
# 引入 CaptureManager (需要其類型提示)
from data_capture.capture_manager import CaptureManager
# 引入 ObjectDetector (需要其類型提示)
from inference.inferencer import ObjectDetector # 確保 ObjectDetector 可以被引用


logger = logging.getLogger(__name__)

class PersonDetector(BaseDetector):
    """
    專注於偵測人員和與人員相關的事件，並進行人臉識別。
    """
    def __init__(self, settings: dict,
                 object_detector: ObjectDetector, # 主要物件偵測器
                 face_recognizer: FaceRecognizer, # 人臉識別器
                 event_manager: EventManager,
                 event_publisher: EventPublisher,
                 capture_manager: CaptureManager): # 捕獲管理器 (用於獲取幀緩衝區)
        """
        初始化人員偵測器。
        Args:
            settings (dict): 此偵測器的特定設定 (config.detectors.person)。
            object_detector (ObjectDetector): 主要物件偵測推論器實例。
            face_recognizer (FaceRecognizer): 人臉識別器實例。
            event_manager (EventManager): 事件管理器實例。
            event_publisher (EventPublisher): 事件發布器實例。
            capture_manager (CaptureManager): 捕獲管理器實例。
        """
        # 將所有依賴傳遞給基類
        super().__init__(settings, object_detector, event_manager, event_publisher, capture_manager)

        self.face_recognizer = face_recognizer # 保存人臉識別器實例

        self.person_class_name = self.settings.get('class_name', 'person')
        self.cooldown_seconds = self.settings.get('cooldown_seconds', 10) # 人物事件冷卻時間 (現在可能按人物 ID 細分)

        # 從設定中獲取是否對已知/未知人物觸發事件
        self.alert_on_unknown_person = self.settings.get('alert_on_unknown_person', True)
        self.alert_on_known_person = self.settings.get('alert_on_known_person', False)

        # 可以擴展實現按人物 ID 的冷卻邏輯
        self._last_event_time_per_person: Dict[str, float] = {} # {person_id: timestamp}

        logger.info("PersonDetector 初始化成功。")

    # 修正：將 detections_raw 的類型提示從 List 改為 List[Any] 並在註釋中說明
    def process(self, frame_cuda: jetson.utils.cudaImage, detections_raw: List[Any]): # List[jetson_inference.Detection]
        """
        處理人員偵測邏輯。
        Args:
            frame_cuda (jetson.utils.cudaImage): 當前幀的 CUDA 影像數據。
            detections_raw (List[Any]): 物件偵測模型輸出的原始偵測結果列表 (預期類型為 List[jetson_inference.Detection])。
        """
        if not self.is_enabled:
            return

        # 篩選出人員偵測結果
        person_detections = [
            det for det in detections_raw
            if self.object_detector.class_mapping.get(det.ClassID) == self.person_class_name
        ]

        # --------------------------------------------------------------------
        # 實現具體的人員偵測規則和事件觸發邏輯
        # --------------------------------------------------------------------

        # 規則範例 1: 偵測到至少一人，嘗試進行人臉識別
        if len(person_detections) > 0:
            # 為了簡化，我們只對第一個偵測到的人員進行人臉識別嘗試
            first_person_detection = person_detections[0]

            # 從捕獲管理器獲取最近的幀緩衝區
            frame_buffer = self.capture_manager.get_frame_buffer()

            # 調用人臉識別器在緩衝區中尋找最佳人臉並進行識別
            # FaceRecognizer.find_and_recognize_best_face 返回值 (已修正):
            # person_id, face_detection_confidence, match_confidence, face_embedding, face_bbox, frame_timestamp
            person_id, face_detection_confidence_val, match_confidence_val, face_embedding, face_bbox, frame_timestamp = \
                self.face_recognizer.find_and_recognize_best_face(frame_buffer, first_person_detection)

            # 根據識別結果構建元數據
            metadata: Dict[str, Any] = {
                "person_count_in_frame": len(person_detections), # 這幀偵測到的人數
                # person_detection_bbox 是當前幀中物件偵測到的人員框
                "person_detection_bbox": [int(first_person_detection.Left), int(first_person_detection.Top), int(first_person_detection.Right), int(first_person_detection.Bottom)],
                # face_detection_confidence 是實際用於識別的人臉的偵測信心度
                "face_detection_confidence": float(face_detection_confidence_val) if face_detection_confidence_val is not None else None,
                "face_bbox": face_bbox, # 識別使用的人臉 bounding box (在識別使用的幀中的座標)
                # face_match_confidence 是人臉比對的相似度
                "face_match_confidence": float(match_confidence_val) if match_confidence_val is not None else None, # 修正：使用正確的變數名 match_confidence_val

                "recognized_frame_timestamp": frame_timestamp # 識別使用的幀的時間戳
                # "face_embedding": face_embedding.tolist() if face_embedding is not None else None # 再次確認：embedding 不發布到 IoT
            }

            # 判斷觸發哪種事件
            event_type: Optional[str] = None # 重新初始化 event_type

            if person_id is None:
                 # 未能找到符合最低條件的人臉進行識別 (find_and_recognize_best_face 返回的 person_id 為 None)
                 # 這可能是因為緩衝區中沒有符合最低人臉偵測信心度/清晰度的人臉
                 logger.warning("偵測到人物但未能找到適合的人臉進行識別。")
                 # 這裡可以選擇觸發一個表示“偵測到人物但無法識別人臉”的事件，或者完全不觸發人臉識別相關事件
                 # 例如，觸發原始的 PERSON_DETECTED 事件
                 # event_type = EventType.PERSON_DETECTED.value # 需要在 EventTypes 中定義這個事件
                 # 如果觸發原始事件，metadata 需要調整 (可能只需要 person_detection_bbox 和 count)
                 pass # 選擇不觸發特定事件，等待下一個循環

            elif person_id == "unknown" and self.alert_on_unknown_person:
                 # 偵測到人物但識別為未知 (人臉識別器返回的 person_id 為 "unknown")
                 event_type = EventType.UNKNOWN_PERSON_DETECTED.value
                 # 為未知人物設置一個獨特的 ID 或標記，可能結合時間戳或設備 ID
                 db_identifier = self.face_recognizer.known_faces_db.local_cache_path # 獲取本地路徑
                 if db_identifier:
                     db_identifier = os.path.basename(db_identifier).replace('.', '_') # 使用文件名作為標識的一部分
                 else:
                     db_identifier = "no_db" # 如果沒有本地緩存路徑
                 # 修正：未知人物的 ID 可以包含更多信息，如事件時間戳、設備 ID
                 metadata["person_id"] = f"unknown_{self.face_recognizer.known_faces_db.local_cache_path.split('/')[-1].split('.')[0]}_{int(time.time())}" # 範例：unknown_known_faces_<timestamp>
                 metadata["person_id"] = f"unknown_{db_identifier}_{int(time.time())}" # 使用更安全的 basename
                 # 也可以在雲端賦予 unknown person 持久的臨時 ID

                 logger.info(f"觸發事件: {event_type}")

            elif person_id != "unknown" and self.alert_on_known_person:
                 # 偵測到人物且識別為已知 (人臉識別器返回已知人物 ID)
                 event_type = EventType.KNOWN_PERSON_DETECTED.value
                 metadata["person_id"] = person_id # 附加識別到的已知人物 ID
                 logger.info(f"觸發事件: {event_type} for known person '{person_id}'")

            # 如果決定觸發事件 (event_type 已被設置)
            if event_type:
                # 檢查事件冷卻時間 (這裡可以擴展為按 person_id 檢查冷卻時間)
                # 為了簡單，先使用整個 PersonDetector 的冷卻時間
                if self.event_manager.should_trigger_event(event_type, cooldown_override=self.cooldown_seconds):
                    logger.info(f"觸發事件 '{event_type}' 通過冷卻時間檢查。")
                    # 觸發事件並捕獲使用的幀影像
                    # 需要先確定使用哪一幀影像進行上傳，這裡使用 find_and_recognize_best_face 返回的幀數據
                    recognized_frame_data = None
                    # 從緩衝區中找到識別時使用的那幀數據
                    # 修正：在 FaceRecognizer.find_and_recognize_best_face 返回 FrameData 實例本身會更高效和安全
                    # 但目前它是返回時間戳，所以我們需要遍歷緩衝區尋找匹配時間戳的幀數據
                    # 這裡應該檢查 frame_timestamp 是否為 None，如果為 None 說明識別失敗，無需尋找幀
                    if frame_timestamp is not None:
                         for fd in frame_buffer:
                              if fd.timestamp == frame_timestamp:
                                   recognized_frame_data = fd
                                   break
                         if recognized_frame_data is None:
                              logger.warning(f"在緩衝區中未能找到時間戳為 {frame_timestamp} 的幀。")


                    if recognized_frame_data:
                         # 捕獲並上傳識別時使用的那幀影像
                         # metadata 中已經包含人臉識別結果，會一起發布到 IoT
                         s3_image_path = self.capture_manager.capture_and_upload_image(event_type, recognized_frame_data, metadata)
                         # 如果影像成功添加到上傳佇列 (即使尚未完成上傳)，則發布事件訊息
                         if s3_image_path:
                             # EventPublisher 會自動將 s3_image_path 包含在 payload 中
                             self.event_publisher.publish_event(event_type, s3_image_path=s3_image_path, metadata=metadata)
                             # 記錄事件觸發時間 (用於冷卻)
                             self.event_manager.record_event_triggered(event_type) # 記錄總體事件類型冷卻時間
                         else:
                             # 如果 capture_and_upload_image 返回 None (表示捕獲或添加到佇列失敗)
                             logger.warning(f"未能捕獲或添加到佇列影像用於事件 '{event_type}'。跳過發布事件訊息。")
                             # 如果需要即使沒有影像也發布事件，取消註釋下面一行
                             # self.event_publisher.publish_event(event_type, metadata=metadata)
                             # self.event_manager.record_event_triggered(event_type) # 記錄觸發時間 (如果決定發布事件)
                    else:
                         # 這是未能找到與返回時間戳匹配的幀的情況
                         logger.error(f"未能找到用於人臉識別的幀數據 (timestamp: {frame_timestamp}) 在緩衝區中。無法捕獲影像或發布事件。")


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