# events/event_manager.py

import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EventManager:
    """
    管理邊緣端事件的觸發，處理冷卻時間等邏輯。
    """
    def __init__(self, settings: dict):
        """
        初始化事件管理器。
        Args:
            settings (dict): 事件相關設定。
        """
        self.settings = settings
        self._last_event_time: Dict[str, float] = {} # 記錄上次觸發某個事件類型的時間
        # 可以擴展為記錄更精細的冷卻時間，例如按物體 ID 或區域

    def should_trigger_event(self, event_type: str, cooldown_override: float = None) -> bool:
        """
        判斷某個事件類型是否應該觸發 (基於冷卻時間)。
        Args:
            event_type (str): 事件類型名稱 (string 或 EventType value)。
            cooldown_override (float, optional): 為此特定觸發設置的冷卻時間覆蓋值。
                                                 如果為 None，則使用設定中的預設值。
        Returns:
            bool: 如果事件應該觸發則為 True，否則為 False。
        """
        current_time = time.time()
        last_time = self._last_event_time.get(event_type, 0)

        # 獲取冷卻時間，優先使用覆蓋值，然後是設定中的特定事件類型值，最後是預設值
        cooldown = cooldown_override
        if cooldown is None:
             cooldown = self.settings.get(event_type, {}).get('cooldown_seconds') # 檢查特定事件類型的設定
        if cooldown is None:
             cooldown = self.settings.get('default_cooldown_seconds', 5) # 使用預設值

        if (current_time - last_time) > cooldown:
            return True
        else:
            # logger.debug(f"事件 '{event_type}' 仍在冷卻時間內。") # 如果不需要頻繁輸出，可註釋掉
            return False

    def record_event_triggered(self, event_type: str):
        """
        記錄某個事件類型已觸發的時間。
        Args:
            event_type (str): 事件類型名稱。
        """
        self._last_event_time[event_type] = time.time()
        logger.info(f"事件 '{event_type}' 已記錄觸發時間。")

    # 可以擴展方法來管理更精細的冷卻時間，例如基於 object_id + event_type
    # def should_trigger_per_object(self, event_type: str, object_id: int, cooldown_seconds: float) -> bool:
    #     # 實現按物體 ID 的冷卻邏輯
    #     pass