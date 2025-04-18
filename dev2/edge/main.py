# main.py

import cv2
import jetson_utils
import jetson_inference

import numpy as np
import time
import threading
import queue
import yaml
import logging
import signal # 用於處理停止信號
import json   # 引入 json 用於解析命令 payload

# 引入我們自己設計的模組
from utils.cuda_utils import numpy_to_cuda #, cuda_to_numpy # 如果需要轉回 numpy
from utils.image_utils import resize_for_display, draw_detections #, draw_roi
from utils.s3_uploader import S3Uploader
from iot_client.aws_iot_client import AWSIoTClient
from inference.model_manager import ModelManager
from inference.inferencer import ObjectDetector # 引入具體的推論器
from events.event_types import EventType # 引入事件類型
from events.event_manager import EventManager
from events.event_publisher import EventPublisher
from data_capture.capture_manager import CaptureManager
# 引入具體的偵測器
from detectors.person_detector import PersonDetector
from detectors.cargo_detector import CargoDetector
# 根據需要在 settings.yaml 中啟用或禁用其他偵測器，並在這裡引入和初始化

# 引入人臉識別相關的模組
from inference.face_models import FACE_DETECTION_MODEL, FACE_EMBEDDING_MODEL
from inference.face_inferencers import FaceDetector, FaceEmbedder # 引入具體的人臉推論器
from inference.face_db import KnownFacesDB
from inference.face_recognizer import FaceRecognizer # 引入人臉識別器

# 配置 logging (這部分可以在載入設定之前完成基礎配置)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全域停止標誌，用於安全退出主循環
stop_requested = threading.Event()

def signal_handler(signum, frame):
    """
    處理終止信號 (如 Ctrl+C)。
    """
    logger.info(f"收到信號 {signum}，請求停止應用程式。")
    stop_requested.set()

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

    # 驗證關鍵設定是否存在
    if not all(k in settings for k in ['aws', 'camera', 'models', 'known_faces_db', 'capture']):
         logger.error("設定檔案中缺少必要的頂層區塊 (aws, camera, models, known_faces_db, capture)。應用程式終止。")
         return
    # 這裡可以添加更多對 settings 內容的檢查


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
    if not all(iot_settings.get(k) for k in ['endpoint', 'thing_name', 'cert_path', 'pri_key_path', 'root_ca_path']):
         logger.error("AWS IoT 設定不完整。應用程式終止。")
         s3_uploader.stop()
         s3_uploader.join()
         return

    # 引入 known_faces_db 變數到 handle_cloud_command 函數的作用域，以便可以調用其方法
    known_faces_db = None # 先初始化為 None

    def handle_cloud_command(topic, payload):
        logger.info(f"收到雲端命令 Topic: {topic}, Payload: {payload}")
        try:
             command_data = json.loads(payload)
             command_type = command_data.get("type")
             logger.info(f"處理命令: {command_type}")
             if command_type == "update_known_faces_db":
                 logger.info("收到更新已知人臉數據庫命令...")
                 # 調用 known_faces_db 實例的重新載入方法
                 if known_faces_db: # 確保 known_faces_db 已成功初始化
                      known_faces_db.reload_embeddings_from_s3()
                 else:
                      logger.warning("KnownFacesDB 未初始化，無法執行更新命令。")
             # ... 處理其他命令邏輯 ...
             # elif command_type == "set_threshold":
             # ...
        except json.JSONDecodeError:
             logger.error("無法解析收到的命令 Payload (非 JSON 格式)。")
        except Exception as e:
             logger.error(f"處理雲端命令時發生錯誤: {e}", exc_info=True)


    iot_client = AWSIoTClient(iot_settings, command_callback=handle_cloud_command)
    if not iot_client.is_connected():
         logger.error("無法連接到 AWS IoT Core。應用程式終止。")
         s3_uploader.stop()
         s3_uploader.join()
         return


    # 模型管理器和推論器
    model_settings = settings.get('models', {})
    model_manager = ModelManager(model_settings)

    # ... 載入 object_detection, face_detection, face_embedding 模型的邏輯 (保持不變) ...
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

    face_detection_model = model_manager.get_model(FACE_DETECTION_MODEL)
    face_detector_inferencer = None
    if face_detection_model:
         face_detector_inferencer = FaceDetector(model=face_detection_model)

    face_embedding_model = model_manager.get_model(FACE_EMBEDDING_MODEL)
    face_embedder_inferencer = None
    if face_embedding_model:
         face_embedder_inferencer = FaceEmbedder(model=face_embedding_model)


    # 新增：已知人臉數據庫 (現在需要 AWS 設定)
    known_faces_db_settings = settings.get('known_faces_db', {})
    known_faces_db = KnownFacesDB(known_faces_db_settings, settings['aws']) # <-- 傳入 settings 和 aws 設定
    # KnownFacesDB 在初始化時會自動嘗試從 S3 載入


    # 人臉識別器 (只有當所有依賴模型和數據庫都成功初始化時才初始化)
    face_recognizer = None
    if face_detector_inferencer and face_embedder_inferencer and known_faces_db.get_all_embeddings(): # 檢查數據庫是否有內容
         face_recognizer = FaceRecognizer(settings, face_detector_inferencer, face_embedder_inferencer, known_faces_db)
         logger.info("人臉識別器初始化成功。")
    else:
         logger.warning("人臉識別器初始化失敗或數據庫為空。人員偵測器將不執行人臉識別。")


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

    # PersonDetector 的初始化現在需要 FaceRecognizer 實例
    if detector_settings.get('person', {}).get('enabled', False):
        logger.info("初始化人員偵測器...")
        # 只有當 FaceRecognizer 成功初始化時，才初始化 PersonDetector
        if face_recognizer:
             person_detector = PersonDetector(
                 settings=detector_settings['person'],
                 object_detector=object_detector_inferencer,
                 face_recognizer=face_recognizer, # 傳入 FaceRecognizer 實例
                 event_manager=event_manager,
                 event_publisher=event_publisher,
                 capture_manager=capture_manager # 傳入 CaptureManager 實例
             )
             detectors.append(person_detector)
        else:
             logger.warning("人臉識別器初始化失敗或數據庫為空，無法初始化 PersonDetector。")


    if detector_settings.get('cargo', {}).get('enabled', False):
        logger.info("初始化貨物偵測器...")
        cargo_detector = CargoDetector(
            settings=detector_settings['cargo'],
            object_detector=object_detector_inferencer, # 貨物偵測器需要物件偵測結果
            event_manager=event_manager,
            event_publisher=event_publisher,
            capture_manager=capture_manager
            # 可傳入其他推論器實例，如 tilt_classifier=classification_inferencer
        )
        detectors.append(cargo_detector)

    # TODO: 初始化其他偵測器 (animal, safety, etc.)

    # 3. 初始化攝影機
    camera_settings = settings.get('camera', {}) # 使用 .get() 提供預設值
    camera_source = camera_settings.get('source', 0) # 提供預設值
    camera_width = camera_settings.get('width', 1280)
    camera_height = camera_settings.get('height', 720)
    camera_codec = camera_settings.get('codec', 'MJPG') # 預設使用 MJPG

    # 使用 OpenCV 打開攝影機
    # 根據攝影機型號和驅動，source 可以是 int (設備 ID), str (檔案路徑), 或串流 URL
    cap = cv2.VideoCapture(camera_source)

    # 設定攝影機參數 (如果需要且攝影機支持)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    # 嘗試設定編碼格式以提高讀取效率
    fourcc_code = cv2.VideoWriter_fourcc(*camera_codec)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
    # 設置幀率 (如果需要)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        logger.error(f"無法開啟攝影機設備 {camera_source}。請檢查設備 ID 或權限。應用程式終止。")
        # 在退出前嘗試清理資源
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
             frame_cuda = jetson_utils.cudaFromNumpy(rgb_frame_np)
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
        capture_manager.add_frame_to_buffer(frame_np, frame_cuda, detections_raw)

        # 將原始偵測結果和 CUDA 幀傳遞給所有偵測器進行處理
        for detector in detectors:
            try:
                detector.process(frame_cuda, detections_raw)
            except Exception as e:
                logger.error(f"偵測器 '{detector.__class__.__name__}' 處理失敗: {e}", exc_info=True)


        # 可選：在本地顯示處理後的影像 (用於調試)
        if display_enabled:
            frame_with_detections = draw_detections(
                 frame_np.copy(),
                 detections_raw,
                 object_detector_inferencer.class_mapping
            )
            display_frame = resize_for_display(frame_with_detections, display_width, display_height)
            cv2.imshow("Edge Detection", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                stop_requested.set()
            # 可添加其他按鍵功能，例如手動觸發拍照並上傳
            # elif key == ord('s'):
            #     logger.info("手動觸發拍照上傳...")
            #     # 確保 capture_manager 變數在作用域內
            #     if 'capture_manager' in locals():
            #         s3_path = capture_manager.capture_and_upload_image("manual_trigger")
            #         logger.info(f"手動拍照 S3 路徑: {s3_path}")
            #     else:
            #         logger.warning("Capture manager 未初始化，無法執行手動捕獲。")


        # 控制迴圈速度 (如果需要)
        # time.sleep(0.01) # 避免 CPU/GPU 佔用過高

    # 5. 清理資源
    logger.info("應用程式停止中，開始清理資源...")

    # 停止攝影機
    if cap.isOpened():
        cap.release()
        logger.info("攝影機已釋放。")

    # 關閉顯示視窗
    if display_enabled:
        cv2.destroyAllWindows()
        logger.info("顯示視窗已關閉。")

    # 停止 S3 上傳執行緒並等待其完成佇列任務
    s3_uploader.stop()
    s3_uploader.wait_for_completion() # 等待所有上傳完成
    s3_uploader.join()
    logger.info("S3 上傳執行緒已停止。")

    # 斷開 AWS IoT 連接
    iot_client.disconnect()
    logger.info("AWS IoT 連接已斷開。")

    # 卸載模型
    model_manager.unload_all_models() # 雖然 jetson_inference 可能無實際卸載，但清空內部狀態是個好習慣

    logger.info("所有資源已清理，應用程式終止。")

if __name__ == "__main__":
    main()