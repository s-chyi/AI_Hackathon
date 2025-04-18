# detectors/cargo_detector.py

import logging
from typing import List
import jetson.inference
from events.event_types import EventType
from .base_detector import BaseDetector

logger = logging.getLogger(__name__)

class CargoDetector(BaseDetector):
    """
    專注於偵測貨物和與貨物相關的事件。
    """
    def __init__(self, settings: dict, *args, **kwargs):
        """
        初始化貨物偵測器。
        Args:
            settings (dict): 此偵測器的特定設定 (config.detectors.cargo)。
            *args, **kwargs: 傳遞給 BaseDetector 的其他參數。
        """
        super().__init__(settings, *args, **kwargs)
        self.cargo_class_name = self.settings.get('class_name', 'cargo')
        self.cooldown_seconds = self.settings.get('cooldown_seconds', 30)

        # 可以在這裡初始化更複雜的狀態或模型（例如用於判斷傾斜的分類模型）
        # self.tilt_classifier = kwargs.get('tilt_classifier') # 假設可以傳入分類推論器

    def process(self, frame_cuda: jetson.utils.cudaImage, detections_raw: List[jetson.inference.Detection]):
        """
        處理貨物偵測邏輯。
        Args:
            frame_cuda (jetson.utils.cudaImage): 當前幀的 CUDA 影像數據。
            detections_raw (List[jetson.inference.Detection]): 物件偵測模型輸出的原始偵測結果列表。
        """
        if not self.is_enabled:
            return

        cargo_detections = [
            det for det in detections_raw
            if self.object_detector.class_mapping.get(det.ClassID) == self.cargo_class_name
        ]

        # --------------------------------------------------------------------
        # 實現具體的貨物偵測規則和事件觸發邏輯
        # --------------------------------------------------------------------

        # 規則範例 1: 簡單地偵測到貨物出現
        # (這個事件可能太頻繁，通常需要更具體的規則)
        # if len(cargo_detections) > 0:
        #     metadata = {"count": len(cargo_detections)}
        #     self._trigger_event(
        #         EventType.CARGO_DETECTED.value,
        #         metadata=metadata,
        #         cooldown_override=self.cooldown_seconds
        #     )

        # 規則範例 2: 偵測到貨物傾斜 (需要一個能判斷傾斜的模型或方法)
        # 這裡假設您有另一種方法可以從 detection 判斷傾斜
        # for cargo_det in cargo_detections:
        #     if self._is_cargo_tilted(cargo_det): # 假設有一個內部方法 _is_cargo_tilted
        #         metadata = {"bbox": [cargo_det.Left, cargo_det.Top, cargo_det.Right, cargo_det.Bottom]}
        #         self._trigger_event(
        #              EventType.CARGO_TILTED.value,
        #              metadata=metadata,
        #              cooldown_override=self.cooldown_seconds # 或使用更短/更長的冷卻
        #         )
        #         # 如果只想觸發一次，可以 break

        # 規則範例 3: 偵測到 QR Code (QR Code 讀取通常是獨立於貨物偵測的模塊，但可以放在這裡協調)
        # 假設您有另一個模塊負責讀取 QR Code，並在讀取成功時通知這裡或直接發布事件。
        # 這裡只是一個邏輯上的表示
        # if qr_code_successfully_scanned_in_this_frame: # 從其他地方獲取狀態
        #      qr_code_data = get_qr_code_data() # 從其他地方獲取數據
        #      metadata = {"qr_data": qr_code_data}
        #      self._trigger_event(EventType.QR_CODE_SCANNED.value, metadata=metadata)


    # 輔助方法範例：判斷貨物是否傾斜
    # def _is_cargo_tilted(self, detection: jetson.inference.Detection) -> bool:
    #      """
    #      根據偵測到的貨物 bounding box 或使用分類模型判斷是否傾斜。
    #      這是一個簡化或需要進一步實現的邏輯。
    #      """
    #      # 簡單範例：根據 bounding box 的長寬比或角度估計 (不太準確)
    #      # width = detection.Right - detection.Left
    #      # height = detection.Bottom - detection.Top
    #      # return width / height > 1.5 or height / width > 1.5 # 非常簡陋的判斷

    #      # 更複雜的範例：使用分類模型
    #      # 如果有 tilt_classifier:
    #      #     cargo_image_cuda = crop_cuda_image(frame_cuda, detection.BoundingBox) # 需要實現 CUDA 裁剪
    #      #     tilt_prediction = self.tilt_classifier.infer(cargo_image_cuda)
    #      #     return tilt_prediction == "tilted" # 根據分類結果判斷

    #      return False # 預設不判斷傾斜