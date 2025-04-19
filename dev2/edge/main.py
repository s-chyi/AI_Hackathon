# main.py

import cv2
import jetson.utils
import jetson.inference
import numpy as np
import time
import threading
import queue
import yaml
import logging
import signal
import json # 引入 json

# 引入我們自己設計的模組
from utils.cuda_utils import numpy_to_cuda
from utils.image_utils import resize_for_display, draw_detections
from utils.s3_uploader import S3Uploader
from iot_client.aws_iot_client import AWSIoTClient
from inference.model_manager import ModelManager
from inference.inferencer import ObjectDetector # 引入具體的推論器
# 移除人臉相關模組導入
# from inference.face_models import FACE_DETECTION_MODEL, FACE_EMBEDDING_MODEL
# from inference.face_inferencers import FaceDetector as FaceDetectorInferencer
# from inference.face_db import KnownFacesDB
# from inference.face_recognizer import FaceRecognizer

from events.event_types import EventType # 引入事件類型
from events.event_manager import EventManager
from events.event_publisher import EventPublisher
# 引入 CaptureManager 和 FrameData
from data_capture.capture_manager import CaptureManager, FrameData

# 引入具體的偵測器
from detectors.person_detector import PersonDetector
from detectors.cargo_detector import CargoDetector # 引入 CargoDetector

# 新增：引入 QR 掃描工具
from utils import qr_scanner

# 配置 logging (這部分可以在載入設定之前完成基礎配置)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全域停止標誌，用於安全退出主循環
stop_requested = threading.Event()

# 全域共享變數：儲存最新的雲端識別結果
# 使用鎖保護，因為可能被不同執行緒訪問
latest_recognition_result = {"person_id": "no_person", "timestamp": 0, "original_event_timestamp": 0, "match_confidence": None, "summary": None} # 增加更多字段
recognition_result_lock = threading.Lock()

def signal_handler(signum, frame):
    """
    處理終止信號 (如 Ctrl+C)。
    """
    logger.info(f"收到信號 {signum}，請求停止應用程式。")
    stop_requested.set()

# 新增：處理雲端識別結果的回調函數
def handle_recognition_result(topic, payload_str):
    logger.info(f"收到雲端識別結果 Topic: {topic}, Payload: {payload_str}")
    try:
        result_data = json.loads(payload_str)
        person_id = result_data.get("person_id", "no_person") # 如果 Payload 中沒有 person_id，設為 no_person
        original_timestamp = result_data.get("original_timestamp", 0) # 邊緣發布事件時的時間戳

        logger.info(f"解析識別結果: Person ID: {person_id}, Original Timestamp: {original_timestamp}")

        # 更新全局共享的最新識別結果狀態
        with recognition_result_lock:
            latest_recognition_result["person_id"] = person_id
            latest_recognition_result["timestamp"] = time.time() # 記錄收到結果的時間 (Unix)
            latest_recognition_result["original_event_timestamp"] = original_timestamp # 邊緣事件的時間戳 (保持原始格式)
            latest_recognition_result["match_confidence"] = result_data.get("match_confidence")
            latest_recognition_result["summary"] = result_data.get("summary")
            latest_recognition_result["face_bbox_rekognition"] = result_data.get("face_bbox_rekognition") # 添加雲端識別人臉

        logger.info(f"已更新最新識別結果狀態：{latest_recognition_result}")

    except json.JSONDecodeError:
        logger.error("無法解析收到的識別結果 Payload (非 JSON 格式)。")
    except Exception as e:
        logger.error(f"處理雲端識別結果時發生錯誤: {e}", exc_info=True)

def main():
    logger.info("應用程式啟動...")

    # 1. 載入設定
    settings = None
    try:
        with open("config/settings.yaml", 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)
        logger.info("設定檔案載入成功。")
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"載入設定檔案時發生錯誤: {e}。應用程式終止。")
        return

    if settings.get('debug', False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("已啟用 DEBUG 級別日誌。")

    # 驗證關鍵設定是否存在 (現在只需要 aws, camera, models, capture)
    # 檢查 models 中至少有 object_detection 設定
    if not all(k in settings for k in ['aws', 'camera', 'models', 'capture']) or \
       'object_detection' not in settings.get('models', {}):
        logger.error("設定檔案中缺少必要的區塊或 'models.object_detection' 設定。應用程式終止。")
        return


    # 註冊信號處理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 2. 初始化模組

    # S3 上傳佇列和執行緒 (必須在 KnownFacesDB 之前初始化，以便 S3 客戶端可用，或者確保 KnownFacesDB 能獨立創建 S3 客戶端)
    # 現在 KnownFacesDB 在內部獨立創建了 S3 客戶端，所以順序不是絕對必需，但先初始化 S3Uploader 是個好習慣
    s3_settings = settings['aws'].get('s3', {})
    s3_upload_queue = queue.Queue(maxsize=s3_settings.get('upload_queue_maxsize', 10))
    s3_uploader = S3Uploader(settings['aws'], s3_upload_queue) # S3Uploader 內部創建 S3 客戶端
    s3_uploader.start()

    # AWS IoT 客戶端
    iot_settings = settings['aws'].get('iot', {})
    if not all(iot_settings.get(k) for k in ['endpoint', 'thing_name', 'cert_path', 'pri_key_path', 'root_ca_path', 'result_topic']):
         logger.error("AWS IoT 設定不完整 (缺少 endpoint, thing_name, 證書路徑或 result_topic)。應用程式終止。")
         s3_uploader.stop()
         s3_uploader.join()
         return

    def handle_cloud_command(topic, payload):
        logger.info(f"收到雲端命令 Topic: {topic}, Payload: {payload}")
        try:
             command_data = json.loads(payload)
             command_type = command_data.get("type")
             logger.info(f"處理命令: {command_type}")
             # 移除 update_known_faces_db 命令處理邏輯
             # if command_type == "update_known_faces_db":
             #     ...
             # ... 處理其他命令邏輯 (例如：重啟程式、修改物件偵測閾值等) ...
             if command_type == "restart_app":
                 logger.info("收到重啟應用程式命令。")
                 # 這裡可以設置一個標誌或使用 os.execv 重新啟動
                 stop_requested.set() # 設置停止標誌，讓主循環結束，然後外部腳本可以重啟
             # ...
        except json.JSONDecodeError:
             logger.error("無法解析收到的命令 Payload (非 JSON 格式)。")
        except Exception as e:
             logger.error(f"處理雲端命令時發生錯誤: {e}", exc_info=True)


    iot_client = AWSIoTClient(iot_settings, command_callback=handle_cloud_command, result_callback=handle_recognition_result)
    if not iot_client.is_connected():
         logger.error("無法連接到 AWS IoT Core。應用程式終止。")
         s3_uploader.stop()
         s3_uploader.join()
         return

    # 模型管理器和推論器 (現在只用於物件偵測)
    model_settings = settings.get('models', {})
    model_manager = ModelManager(model_settings)

    # 載入物件偵測模型 (必需)
    object_detection_model = model_manager.get_model("object_detection")
    if object_detection_model is None:
        logger.error("無法載入物件偵測模型，應用程式終止。")
        iot_client.disconnect()
        s3_uploader.stop()
        s3_uploader.join()
        return
    object_detector_inferencer = ObjectDetector(
        model=object_detection_model,
        class_mapping=model_settings.get('object_detection', {}).get('class_mapping', {})
    )

    # 事件管理器和發布器
    event_settings = settings.get('events', {})
    event_manager = EventManager(event_settings)
    event_publisher = EventPublisher(iot_client, settings['aws']['iot']['thing_name'])

    # 捕獲管理器
    capture_settings = settings.get('capture', {})
    capture_manager = CaptureManager(s3_uploader, settings['aws']['s3'], capture_settings)


    # 偵測器 (根據設定啟用)
    detectors = []
    detector_settings = settings.get('detectors', {})

    # 修改：PersonDetector 的初始化參數
    if detector_settings.get('person', {}).get('enabled', False):
        logger.info("初始化人員偵測器...")
        if object_detector_inferencer:
             person_detector = PersonDetector(
                 settings=detector_settings['person'],
                 object_detector=object_detector_inferencer,
                 event_manager=event_manager,
                 event_publisher=event_publisher,
                 capture_manager=capture_manager
             )
             detectors.append(person_detector)
        else:
             logger.warning("物件偵測器未成功初始化，無法初始化 PersonDetector。")


    # CargoDetector 的初始化 (傳入共享狀態和鎖)
    if detector_settings.get('cargo', {}).get('enabled', False):
        logger.info("初始化貨物偵測器...")
        if object_detector_inferencer: # CargoDetector 也依賴 object_detector
             # 檢查 CargoDetector 必要的設定
             cargo_settings = detector_settings['cargo']
             # allowed_person_ids 是列表，recognition_result_validity_sec 是數字
             if 'allowed_person_ids' not in cargo_settings or 'recognition_result_validity_sec' not in cargo_settings:
                  logger.warning("CargoDetector 設定不完整 (缺少 allowed_person_ids 或 recognition_result_validity_sec)。")
                  # 可以選擇不初始化 CargoDetector 或在內部處理缺失設定
                  pass # 繼續初始化，在 CargoDetector 內部處理缺失設定

             cargo_detector = CargoDetector(
                 settings=detector_settings['cargo'],
                 object_detector=object_detector_inferencer,
                 event_manager=event_manager,
                 event_publisher=event_publisher,
                 capture_manager=capture_manager,
                 # 新增：傳入共享的識別結果狀態和鎖
                 recognition_result_state=latest_recognition_result,
                 recognition_result_lock=recognition_result_lock
             )
             detectors.append(cargo_detector)
        else:
             logger.warning("物件偵測器未成功初始化，無法初始化 CargoDetector。")

    # TODO: 初始化其他偵測器

    # 3. 初始化攝影機
    # ... 攝影機初始化邏輯 (保持不變) ...
    camera_settings = settings.get('camera', {})
    camera_source = camera_settings.get('source', 0)
    camera_width = camera_settings.get('width', 1280)
    camera_height = camera_settings.get('height', 720)
    camera_codec = camera_settings.get('codec', 'MJPG')

    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    fourcc_code = cv2.VideoWriter_fourcc(*camera_codec)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

    if not cap.isOpened():
        logger.error(f"無法開啟攝影機設備 {camera_source}。應用程式終止。")
        iot_client.disconnect()
        s3_uploader.stop()
        s3_uploader.join()
        return

    logger.info(f"攝影機開啟成功，分辨率 {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}，編碼 {camera_codec}。")


    # 4. 主處理迴圈
    logger.info("進入主處理迴圈...")
    processing_frame_count = 0
    start_time = time.time()

    display_settings = settings.get('display', {})
    display_enabled = display_settings.get('enabled', False)
    display_width = display_settings.get('max_width', 800)
    display_height = display_settings.get('max_height', 600)

    while not stop_requested.is_set():
        ret, frame_np = cap.read()
        if not ret:
            logger.warning("無法從攝影機讀取幀。")
            if not stop_requested.is_set():
                time.sleep(0.1)
            continue

        processing_frame_count += 1
        current_time = time.time()

        frame_cuda = None
        try:
             rgb_frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
             frame_cuda = jetson.utils.cudaFromNumpy(rgb_frame_np)
        except Exception as e:
             logger.error(f"NumPy 到 CUDA 轉換失敗: {e}", exc_info=True)
             continue

        # 執行邊緣模型推論 (物件偵測)
        detections_raw = []
        try:
            if object_detector_inferencer:
                detections_raw = object_detector_inferencer.infer(frame_cuda)
        except Exception as e:
            logger.error(f"物件偵測推論失敗: {e}", exc_info=True)
            detections_raw = []

        # 將包含偵測結果的當前幀添加到捕獲管理器的緩衝區
        # 注意：這裡緩衝的是最新一幀，PersonDetector 將直接使用當前幀的 detections_raw 判斷是否有人物
        # 緩衝區主要用於如果未來需要回溯或捕獲延後幀，但目前只捕獲當前幀
        capture_manager.add_frame_to_buffer(frame_np, frame_cuda, detections_raw)


        # 將原始偵測結果和 CUDA 幀傳遞給所有偵測器進行處理
        for detector in detectors:
            try:
                # detectors 現在會從 capture_manager 獲取 buffer，但 process 方法仍然接收當前幀的結果以便觸發判斷
                detector.process(frame_cuda, detections_raw)
            except Exception as e:
                logger.error(f"偵測器 '{detector.__class__.__name__}' 處理失敗: {e}", exc_info=True)


        # 可選：在本地顯示處理後的影像
        # ... 顯示邏輯 (保持不變) ...
        if display_enabled:
            # 在 NumPy 影像上繪製物件偵測框
            frame_to_display = draw_detections(
                frame_np.copy(), # 在拷貝上繪製
                detections_raw,
                object_detector_inferencer.class_mapping
            )

            # 新增：如果 CargoDetector 啟用了並且有配置 ROI，則在顯示的幀上繪製 ROI
            if cargo_detector and cargo_detector.is_enabled and cargo_detector.cargo_roi:
                try:
                    roi = cargo_detector.cargo_roi
                    # 確保 ROI 是有效的 [x1, y1, x2, y2] 格式
                    if isinstance(roi, list) and len(roi) == 4:
                        # 使用 OpenCV 在畫面上繪製矩形
                        # 座標需要是整數
                        p1 = (int(roi[0]), int(roi[1]))
                        p2 = (int(roi[2]), int(roi[3]))
                        color = (0, 0, 255) # 紅色 (BGR 格式)
                        thickness = 2
                        cv2.rectangle(frame_to_display, p1, p2, color, thickness)
                        # 可選：在框附近添加文本標籤
                        # cv2.putText(frame_to_display, "Cargo ROI", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        logger.warning(f"CargoDetector 配置的 ROI 格式無效: {roi}")
                except Exception as e:
                    logger.error(f"在顯示影像上繪製 Cargo ROI 時發生錯誤: {e}", exc_info=True)


            display_frame = resize_for_display(frame_to_display, display_width, display_height)
            cv2.imshow("Edge Detection", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                stop_requested.set()


    # 5. 清理資源
    logger.info("應用程式停止中，開始清理資源...")
    # ... 清理邏輯 (保持不變) ...

    if cap.isOpened():
        cap.release()
        logger.info("攝影機已釋放。")

    if display_enabled:
        cv2.destroyAllWindows()
        logger.info("顯示視窗已關閉。")

    s3_uploader.stop()
    s3_uploader.wait_for_completion()
    s3_uploader.join()
    logger.info("S3 上傳執行緒已停止。")

    iot_client.disconnect()
    logger.info("AWS IoT 連接已斷開。")

    model_manager.unload_all_models()

    logger.info("所有資源已清理，應用程式終止。")

if __name__ == "__main__":
    main()