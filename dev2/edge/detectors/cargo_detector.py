# detectors/cargo_detector.py

import logging
import time
from typing import List, Dict, Optional, Any
import jetson.inference # 需要 Detection 類型
import jetson.utils # 需要 cudaImage 類型
import threading

from events.event_types import EventType # 引入事件類型
from events.event_manager import EventManager
from events.event_publisher import EventPublisher
from data_capture.capture_manager import CaptureManager, FrameData

from .base_detector import BaseDetector

from inference.inferencer import ObjectDetector

# 引入 QR 掃描工具
from utils import qr_scanner
# 引入圖像工具，用於繪製 (可選)
from utils import image_utils
import cv2 # 引入 cv2 進行繪圖

logger = logging.getLogger(__name__)

class CargoDetector(BaseDetector):
    """
    專注於偵測貨物和與貨物相關的事件，根據人員識別結果調整行為，並提取貨物信息。
    """
    def __init__(self, settings: dict,
                 object_detector: ObjectDetector,
                 event_manager: EventManager,
                 event_publisher: EventPublisher,
                 capture_manager: CaptureManager,
                 recognition_result_state: Dict[str, Any],
                 recognition_result_lock: threading.Lock):
        """
        初始化貨物偵測器。
        Args:
            settings (dict): 此偵測器的特定設定 (config.detectors.cargo)。
            object_detector (ObjectDetector): 主要物件偵測推論器實例。
            event_manager (EventManager): 事件管理器實例。
            event_publisher (EventPublisher): 事件發布器實例。
            capture_manager (CaptureManager): 捕獲管理器實例。
            recognition_result_state (Dict[str, Any]): 共享的最新雲端識別結果字典。
            recognition_result_lock (threading.Lock): 保護識別結果狀態的鎖。
        """
        super().__init__(settings, object_detector, event_manager, event_publisher, capture_manager)

        # 貨物類別名稱 (可以是一個列表)
        self.cargo_class_names = self.settings.get('cargo_class_names')
        if not self.cargo_class_names:
            # 如果未指定具體貨物類別名稱，則將所有非人物的類別視為貨物
            all_classes = list(object_detector.class_mapping.values())
            person_class = self.settings.get('class_name', 'person') # 獲取人物類別名
            self.cargo_class_names = [cls for cls in all_classes if cls != person_class]
            logger.info(f"CargoDetector 未指定具體貨物類別，將以下類別視為貨物: {self.cargo_class_names}")
            if not self.cargo_class_names:
                logger.warning("CargoDetector 找不到任何非人物的類別映射，將無法偵測貨物。請檢查 class_mapping 設定。")


        self.cooldown_seconds = self.settings.get('cooldown_seconds', 30)

        self.recognition_result_state = recognition_result_state
        self.recognition_result_lock = recognition_result_lock

        self.allowed_person_ids = self.settings.get('allowed_person_ids', [])
        self.recognition_result_validity_sec = self.settings.get('recognition_result_validity_sec', 10)

        # 貨物偵測的感興趣區域 (ROI) - [x1, y1, x2, y2]
        self.cargo_roi = self.settings.get('cargo_roi')
        if self.cargo_roi:
             logger.info(f"CargoDetector 將僅在 ROI 區域 {self.cargo_roi} 內偵測貨物。")
        else:
             logger.warning("CargoDetector 未設定 cargo_roi，將偵測整個畫面中的貨物。")


        # 是否啟用 OCR 作為 QR Code 備案
        self.enable_ocr_fallback = self.settings.get('enable_ocr_fallback', True)
        if self.enable_ocr_fallback and qr_scanner.pyzbar is None:
             logger.warning("已啟用 OCR 備案，但 pyzbar 未安裝。QR 掃描將失敗，OCR 備案標記可能被設置。")
        elif not self.enable_ocr_fallback:
             logger.info("已禁用 OCR 備案。")


        # 定義貨物事件類型
        self.cargo_processing_event_type = EventType.CARGO_INFO_FOR_PROCESSING.value

        # 新增：獲取貨物入庫影像的 S3 檔案夾前綴
        self.s3_cargo_checkin_folder = self.capture_manager.s3_settings.get('s3_cargo_checkin_folder')
        if not self.s3_cargo_checkin_folder:
             logger.error("settings.yaml 中未設定 aws.s3.s3_cargo_checkin_folder。貨物入庫影像將無法正確上傳。")

        logger.info("CargoDetector 初始化成功。")


    def process(self, frame_cuda: jetson.utils.cudaImage, detections_raw: List[Any]):
        """
        處理貨物偵測邏輯。
        Args:
            frame_cuda (jetson.utils.cudaImage): 當前幀的 CUDA 影像數據。
            detections_raw (List[Any]): 物件偵測模型輸出的原始偵測結果列表 (預期類型為 List[jetson_inference.Detection])。
        """
        if not self.is_enabled or not self.cargo_class_names or not self.s3_cargo_checkin_folder:
            # 如果偵測器禁用、沒有貨物類別、或沒有設定 S3 檔案夾，則跳過處理
            return

        # --------------------------------------------------------------------
        # 獲取最新的雲端識別結果
        # --------------------------------------------------------------------
        latest_person_id = "no_person"
        latest_result_timestamp = 0
        latest_original_event_timestamp = 0
        latest_match_confidence = None
        latest_person_is_allowed = False

        with self.recognition_result_lock:
            latest_person_id = self.recognition_result_state.get("person_id", "no_person")
            latest_result_timestamp = self.recognition_result_state.get("timestamp", 0)
            latest_original_event_timestamp = self.recognition_result_state.get("original_event_timestamp", 0)
            latest_match_confidence = self.recognition_result_state.get("match_confidence")

            if latest_person_id != "no_person" and latest_person_id != "unknown" and not latest_person_id.startswith("error_"):
                if self.allowed_person_ids and latest_person_id in self.allowed_person_ids:
                    if (time.time() - latest_result_timestamp) < self.recognition_result_validity_sec:
                        latest_person_is_allowed = True


        # --------------------------------------------------------------------
        # 判斷是否根據識別結果處理貨物事件
        # --------------------------------------------------------------------
        # 只有當最新識別到的是允許的人物且結果有效時，才進行貨物處理
        if not latest_person_is_allowed:
            return


        # --------------------------------------------------------------------
        # 如果識別到允許的人物，則進行貨物偵測和處理
        # --------------------------------------------------------------------

        # 篩選出貨物偵測結果，並只考慮在 ROI 內的貨物
        cargo_detections = []
        cargo_detections_raw = [
            det for det in detections_raw
            if det and self.object_detector.class_mapping.get(det.ClassID) in self.cargo_class_names
        ]

        if self.cargo_roi:
            frame_buffer = self.capture_manager.get_frame_buffer()
            current_frame_data = frame_buffer[-1] if frame_buffer else None

            if current_frame_data:
                for det in cargo_detections_raw:
                    center_x = (det.Left + det.Right) / 2
                    center_y = (det.Top + det.Bottom) / 2
                    if self.cargo_roi[0] <= center_x <= self.cargo_roi[2] and \
                        self.cargo_roi[1] <= center_y <= self.cargo_roi[3]:
                        cargo_detections.append(det)
            else:
                logger.warning("捕獲管理器緩衝區為空，無法檢查貨物 ROI。")
                # 如果無法檢查 ROI，且有配置 ROI，則忽略所有貨物
                # 如果沒有配置 ROI，這個 else 塊不執行
                pass # 忽略所有貨物


        else: # 沒有設定 ROI，處理所有偵測到的貨物
            cargo_detections = cargo_detections_raw


        # 如果在 ROI 內偵測到貨物，則觸發貨物信息處理事件
        if len(cargo_detections) > 0:
            # 為了簡化，我們只對第一個偵測到的貨物進行處理
            first_cargo_detection = cargo_detections[0]

            event_type = self.cargo_processing_event_type
            # 冷卻鍵使用 事件類型_人物ID
            cooldown_key = f"{event_type}_{latest_person_id}"

            # 檢查貨物事件冷卻時間
            if self.event_manager.should_trigger_event(cooldown_key, cooldown_override=self.cooldown_seconds):
                logger.info(f"事件 '{event_type}' 觸發 (與人物 {latest_person_id} 相關)。")

                # 獲取當前幀數據用於捕獲和 QR 掃描 (從 capture_manager 獲取最新一幀)
                frame_buffer = self.capture_manager.get_frame_buffer()
                current_frame_data = frame_buffer[-1] if frame_buffer else None

                qr_data: Optional[str] = None
                needs_ocr_fallback = False

                if current_frame_data:
                    # --------------------------------------------------------------------
                    # 步驟：掃描 QR Code
                    # --------------------------------------------------------------------
                    # 裁剪出貨物 bounding box 區域進行 QR 掃描
                    cargo_bbox_np = [int(first_cargo_detection.Left), int(first_cargo_detection.Top), int(first_cargo_detection.Right), int(first_cargo_detection.Bottom)]

                    # 確保裁剪範圍不超過影像邊界
                    h, w = current_frame_data.frame_np.shape[:2]
                    x1, y1, x2, y2 = cargo_bbox_np
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    # 裁剪出貨物區域 NumPy 影像
                    cargo_image_np = current_frame_data.frame_np[y1:y2, x1:x2]

                    qr_data = qr_scanner.scan_qr_code(cargo_image_np)

                    # --------------------------------------------------------------------
                    # 步驟：判斷是否需要 OCR 備案
                    # --------------------------------------------------------------------
                    if qr_data is None and self.enable_ocr_fallback:
                        logger.warning("QR Code 掃描失敗，已啟用 OCR 備案，將標記需要雲端 OCR。")
                        needs_ocr_fallback = True
                    elif qr_data is None and not self.enable_ocr_fallback:
                        logger.warning("QR Code 掃描失敗，已禁用 OCR 備案。")
                        # qr_data 保持 None，needs_ocr_fallback 保持 False
                    elif qr_data is not None:
                        logger.info("QR Code 掃描成功。")


                else:
                    logger.error("捕獲管理器緩衝區為空，無法獲取當前幀進行 QR 掃描。")
                    # 如果無法獲取當前幀，是否需要 OCR 備案？根據需求決定
                    # 如果沒有影像，雲端也無法做 OCR，所以這裡不設置 needs_ocr_fallback = True
                    qr_data = None
                    needs_ocr_fallback = False


                # --------------------------------------------------------------------
                # 步驟：構建貨物事件元數據
                # --------------------------------------------------------------------
                metadata: Dict[str, Any] = {
                    "cargo_count_in_frame": len(cargo_detections),
                    "cargo_detection_bbox_edge": [int(first_cargo_detection.Left), int(first_cargo_detection.Top), int(first_cargo_detection.Right), int(first_cargo_detection.Bottom)],
                    "cargo_detection_confidence_edge": float(first_cargo_detection.Confidence),
                    "frame_timestamp_edge": time.time(), # 邊緣處理此幀的時間戳

                    "related_person_id": latest_person_id, # 附加相關的人物 ID
                    "person_recognition_time": latest_result_timestamp, # 附加收到識別結果的時間
                    "person_match_confidence": latest_match_confidence, # 附加人物比對信心度

                    "qr_code_data": qr_data, # 掃描到的 QR Code 數據 (None 如果失敗)
                    "needs_ocr_fallback": needs_ocr_fallback, # 是否需要雲端 OCR 備案

                    # 如果設置了 cargo_roi，可以添加到 metadata
                    "cargo_roi": self.cargo_roi if self.cargo_roi else None,
                    # 添加邊緣設備名稱，方便雲端識別來源
                    "edge_thing_name": self.event_publisher.thing_name # 從 event_publisher 獲取 thing_name
                }


                # --------------------------------------------------------------------
                # 步驟：捕獲影像並發布事件
                # --------------------------------------------------------------------
                if current_frame_data: # 確保有當前幀數據
                    # 可以在捕獲的影像上繪製貨物框、QR 框、OCR 狀態等 (用於調試或記錄)
                    frame_to_capture_np = current_frame_data.frame_np.copy()
                    # 繪製貨物框
                    cargo_bbox_np_for_drawing = [int(first_cargo_detection.Left), int(first_cargo_detection.Top), int(first_cargo_detection.Right), int(first_cargo_detection.Bottom)]
                    color = (0, 255, 255) # 黃色 (BGR 格式)
                    thickness = 2
                    cv2.rectangle(frame_to_capture_np, (cargo_bbox_np_for_drawing[0], cargo_bbox_np_for_drawing[1]), (cargo_bbox_np_for_drawing[2], cargo_bbox_np_for_drawing[3]), color, thickness)

                    # 繪製文本
                    info_text = f"QR: {qr_data if qr_data else 'None'}"
                    if needs_ocr_fallback:
                        info_text += " OCR Needed"
                    # 確保文本位置在畫面內
                    text_pos_y = cargo_bbox_np_for_drawing[1] - 10
                    if text_pos_y < 15: text_pos_y = cargo_bbox_np_for_drawing[3] + 15 # 如果太高就畫在下面
                    cv2.putText(frame_to_capture_np, info_text, (cargo_bbox_np_for_drawing[0], text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)


                    # 創建一個新的 FrameData 包含繪圖後的影像，用於捕獲和上傳
                    frame_data_with_drawing = FrameData(
                        frame_np=frame_to_capture_np,
                        frame_cuda=current_frame_data.frame_cuda,
                        timestamp=current_frame_data.timestamp,
                        detections_raw=current_frame_data.detections_raw
                    )

                    # 捕獲並上傳貨物入庫影像 (使用貨物入庫檔案夾前綴)
                    s3_image_path = self.capture_manager.capture_and_upload_image(
                        self.cargo_processing_event_type,
                        frame_data_with_drawing,
                        self.s3_cargo_checkin_folder, # <-- 傳入貨物入庫檔案夾
                        metadata # metadata 包含 QR/OCR 信息等
                    )

                    # 如果影像成功添加到上傳佇列，則發布事件訊息
                    if s3_image_path:
                        # event_publisher 會自動將 s3_image_path 包含在 payload 中
                        self.event_publisher.publish_event(self.cargo_processing_event_type, s3_image_path=s3_image_path, metadata=metadata)
                        # 記錄事件觸發時間 (用於冷卻)
                        self.event_manager.record_event_triggered(cooldown_key)
                    else:
                        logger.warning(f"未能捕獲或添加到佇列影像用於貨物事件 '{self.cargo_processing_event_type}'。跳過發布事件訊息。")
                else:
                     logger.error("未能獲取當前幀數據用於貨物事件觸發。")

            # else:
            #      logger.debug(f"事件 '{event_type}' 仍在冷卻時間內，跳過觸發。")

        # else:
        #      logger.debug("偵測到允許人物，但未在 ROI 內偵測到貨物。")