# detectors/person_detector.py

import logging
from typing import List
import jetson.inference
from events.event_types import EventType # 引入事件類型
from .base_detector import BaseDetector # 引入基類

logger = logging.getLogger(__name__)

class PersonDetector(BaseDetector):
    """
    專注於偵測人員和與人員相關的事件。
    """
    def __init__(self, settings: dict, *args, **kwargs):
        """
        初始化人員偵測器。
        Args:
            settings (dict): 此偵測器的特定設定 (config.detectors.person)。
            *args, **kwargs: 傳遞給 BaseDetector 的其他參數。
        """
        super().__init__(settings, *args, **kwargs)
        self.person_class_name = self.settings.get('class_name', 'person') # 獲取設定中的人員內部類別名
        self.alert_on_presence = self.settings.get('alert_on_presence', False) # 是否在偵測到人員時觸發事件
        self.cooldown_seconds = self.settings.get('cooldown_seconds', 10) # 人員事件冷卻時間

        # 可以在這裡初始化更複雜的狀態（例如人員跟蹤、計數）
        # self._tracked_persons = {} # 範例：用於跟蹤人員 ID

    def process(self, frame_cuda: jetson.utils.cudaImage, detections_raw: List):
        """
        處理人員偵測邏輯。
        Args:
            frame_cuda (jetson.utils.cudaImage): 當前幀的 CUDA 影像數據。
            detections_raw (List): 物件偵測模型輸出的原始偵測結果列表。
        """
        if not self.is_enabled:
            return

        person_detections = [
            det for det in detections_raw
            if self.object_detector.class_mapping.get(det.ClassID) == self.person_class_name # 根據 class mapping 篩選
        ]

        # --------------------------------------------------------------------
        # 實現具體的人員偵測規則和事件觸發邏輯
        # --------------------------------------------------------------------

        # 規則範例 1: 偵測到人員出現
        if self.alert_on_presence and len(person_detections) > 0:
            # 這裡可以使用所有被偵測到的人員資訊
            # 為了簡化，只使用第一個偵測到的資訊作為事件元數據
            first_person = person_detections[0]
            metadata = {
                "count": len(person_detections),
                "first_person_confidence": first_person.Confidence,
                "first_person_bbox": [first_person.Left, first_person.Top, first_person.Right, first_person.Bottom]
            }
            # 觸發 "PERSON_DETECTED" 事件
            self._trigger_event(
                EventType.PERSON_DETECTED.value,
                metadata=metadata,
                cooldown_override=self.cooldown_seconds # 使用偵測器自己的冷卻時間
            )

        # 規則範例 2: 偵測到人員在特定區域 (需要額外的區域設定和檢查邏輯)
        # if self.settings.get('alert_in_restricted_area', False) and 'restricted_area' in self.settings:
        #     restricted_area_roi = self.settings['restricted_area']
        #     for person_det in person_detections:
        #         if self.is_in_roi(person_det, restricted_area_roi): # 使用基類中的 is_in_roi 方法
        #             metadata = {"bbox": [person_det.Left, person_det.Top, person_det.Right, person_det.Bottom]}
        #             self._trigger_event(EventType.PERSON_IN_RESTRICTED_AREA.value, metadata=metadata)
        #             break # 找到一個在限制區域的人員就觸發一次事件

        # 規則範例 3: 偵測人員跌倒 (需要姿勢估計模型)
        # if self.settings.get('alert_on_fall', False) and self.pose_estimator: # 假設有姿勢估計推論器
        #     # 執行姿勢估計，判斷是否跌倒
        #     # 如果判斷為跌倒，觸發事件
        #     self._trigger_event(EventType.PERSON_FALL_DETECTED.value, metadata={...})

        # 規則範例 4: 偵測人員長時間靜止 (需要目標跟蹤和時間記錄)
        # if self.settings.get('alert_on_idle', False):
        #      # 跟蹤人員目標，記錄每個目標的最後移動時間
        #      # 如果某個目標長時間未移動，觸發事件
        #      pass