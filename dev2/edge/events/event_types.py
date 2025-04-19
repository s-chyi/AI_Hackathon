# events/event_types.py

from enum import Enum

class EventType(Enum):
    """
    定義邊緣端可以觸發的事件類型。
    """
    # 人員相關事件
    UNKNOWN_PERSON_DETECTED = "UNKNOWN_PERSON_DETECTED" # 偵測到人員但無法識別
    KNOWN_PERSON_DETECTED = "KNOWN_PERSON_DETECTED"   # 偵測到已知人員，包含人物 ID

    PERSON_DETECTED = "PERSON_DETECTED" # 原始物件偵測到人物 (可能用於本地顯示或簡易計數)
    PERSON_FOR_IDENTIFICATION = "PERSON_FOR_IDENTIFICATION" # <-- 新增事件，通知雲端進行人臉識別
    PERSON_IN_RESTRICTED_AREA = "PERSON_IN_RESTRICTED_AREA" # 人員進入限制區域 (需要區域設定)
    PERSON_FALL_DETECTED = "PERSON_FALL_DETECTED"     # 偵測到人員跌倒 (需要姿勢估計模型)
    PERSON_IDLE_TOO_LONG = "PERSON_IDLE_TOO_LONG"     # 人員長時間靜止 (需要時間跟蹤)

    # 貨物處理相關事件
    CARGO_INFO_FOR_PROCESSING = "CARGO_INFO_FOR_PROCESSING" # <-- 新增事件，通知雲端處理貨物信息並決定位置

    # 貨物相關事件
    # CARGO_DETECTED = "CARGO_DETECTED"             # 偵測到貨物
    CARGO_TILTED = "CARGO_TILTED"                 # 貨物傾斜 (需要更精確模型)
    CARGO_OUT_OF_BOUNDS = "CARGO_OUT_OF_BOUNDS"   # 貨物超出允許區域 (需要區域設定)
    QR_CODE_SCANNED = "QR_CODE_SCANNED"           # QR Code 掃描成功

    # 動物相關事件
    ANIMAL_DETECTED = "ANIMAL_DETECTED"           # 偵測到動物

    # 工安/安全相關事件
    SAFETY_VIOLATION_PPE = "SAFETY_VIOLATION_PPE" # 個人防護裝備 (PPE) 違規 (如未戴安全帽)

    # 其他事件
    # CAMERA_OFFLINE = "CAMERA_OFFLINE"           # 攝影機離線 (可在 main loop 檢測)
    # EDGE_DEVICE_ERROR = "EDGE_DEVICE_ERROR"     # 邊緣設備自身錯誤
    # ... 根據需求添加更多事件類型