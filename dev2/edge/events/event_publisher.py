# events/event_publisher.py

import logging
import json
from typing import Dict, Any
from datetime import datetime
from iot_client.aws_iot_client import AWSIoTClient # 引入 IoT 客戶端

logger = logging.getLogger(__name__)

class EventPublisher:
    """
    負責將邊緣事件數據格式化並通過 AWS IoT Core 發布到雲端。
    """
    def __init__(self, iot_client: AWSIoTClient, thing_name: str):
        """
        初始化事件發布器。
        Args:
            iot_client (AWSIoTClient): AWS IoT 客戶端實例。
            thing_name (str): 設備 (Thing) 名稱。
        """
        self.iot_client = iot_client
        self.thing_name = thing_name

    def publish_event(self, event_type: str, s3_image_path: str = None, metadata: Dict[str, Any] = None):
        """
        發布一個邊緣事件到 AWS IoT Core。
        Args:
            event_type (str): 事件類型名稱 (如 EventType.PERSON_DETECTED.value)。
            s3_image_path (str, optional): 事件相關影像在 S3 上的路徑。Defaults to None.
            metadata (Dict[str, Any], optional): 其他與事件相關的元數據。Defaults to None.
        """
        if metadata is None:
            metadata = {}

        # 構建標準事件 Payload
        event_payload = {
            "thing_name": self.thing_name,
            "timestamp": datetime.utcnow().isoformat(), # 使用 UTC 時間戳
            "event_type": event_type,
            "s3_image_path": s3_image_path, # 如果沒有相關影像，則為 None
            "metadata": metadata # 附加額外的元數據 (例如偵測信心度、邊界框、QR Code 內容等)
        }

        # 檢查 IoT 客戶端連接狀態再發布
        if self.iot_client.is_connected():
            # publish_event 方法會返回 Future，這裡選擇不阻塞等待結果
            self.iot_client.publish_event(event_payload)
            logger.info(f"已提交事件 '{event_type}' 到發布佇列。")
        else:
            logger.warning(f"AWS IoT Core 連接斷開，無法發布事件 '{event_type}'。")
            # 可以在這裡添加將事件暫存到本地的邏輯，待連接恢復後發送 (需要更複雜的狀態管理)