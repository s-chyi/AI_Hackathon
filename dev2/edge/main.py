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
    settings = None # 初始化 settings 變數
    try:
        with open("config/settings.yaml", 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f) # <-- 設定檔內容載入到 settings 變數
        logger.info("設定檔案載入成功。")
    except FileNotFoundError:
        logger.error("設定檔案 config/settings.yaml 未找到！應用程式終止。")
        return
    except yaml.YAMLError as e:
        logger.error(f"解析設定檔案時發生錯誤: {e}。應用程式終止。")
        return

    # 設定偵錯級別的日誌輸出（如果設定中有開啟）
    if settings.get('debug', False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("已啟用 DEBUG 級別日誌。")

    # 驗證關鍵設定是否存在
    if 'aws' not in settings or 'camera' not in settings or 'models' not in settings:
        logger.error("設定檔案中缺少必要的頂層區塊 (aws, camera, models)。應用程式終止。")
        return
    # 這裡可以添加更多對 settings 內容的檢查

    # 註冊信號處理器，以便在 Ctrl+C 時能優雅退出
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 2. 初始化模組

    # S3 上傳佇列和執行緒
    s3_settings = settings['aws'].get('s3', {}) # 使用 .get() 提供預設值，避免 KeyErorr
    s3_upload_queue = queue.Queue(maxsize=s3_settings.get('upload_queue_maxsize', 10)) # 提供預設值
    s3_uploader = S3Uploader(settings['aws'], s3_upload_queue)
    s3_uploader.start() # 啟動 S3 上傳執行緒

    # AWS IoT 客戶端 (傳入命令處理回調函數)
    iot_settings = settings['aws'].get('iot', {}) # 使用 .get() 提供預設值
    if not all(k in iot_settings for k in ['endpoint', 'thing_name', 'cert_path', 'pri_key_path', 'root_ca_path']):
        logger.error("AWS IoT 設定不完整 (缺少 endpoint, thing_name, 證書路徑等)。應用程式終止。")
        # 在退出前嘗試清理資源
        s3_uploader.stop()
        s3_uploader.join()
        return

    # command_handler 函數需要在 main 範圍內定義或引入
    def handle_cloud_command(topic, payload):
        logger.info(f"收到雲端命令 Topic: {topic}, Payload: {payload}")
        try:
            command_data = json.loads(payload)
            command_type = command_data.get("type")
            logger.info(f"處理命令: {command_type}")
            # 在這裡實現處理雲端命令的邏輯
            # 根據 command_type 調用相應的功能，例如：
            # if command_type == "set_threshold":
            #     new_threshold = command_data.get("threshold")
            #     if new_threshold is not None and object_detector_inferencer: # 確保推論器存在
            #          # 更新模型閾值 (需要在 Inferencer 中提供更新方法)
            #          # 注意：jetson.inference 的 detectNet 沒有直接更新閾值的方法，可能需要重新載入模型或實現更複雜的邏輯
            #          # 如果模型支援，可以在 ObjectDetector 類中添加一個方法來更新
            #          # object_detector_inferencer.set_confidence_threshold(new_threshold)
            #          logger.info(f"已接收更新偵測閾值命令，新值為 {new_threshold} (實際更新取決於模型和 Inferencer 的實現)")
            # elif command_type == "restart_detector":
            #      detector_name = command_data.get("detector")
            #      # 實現重啟特定偵測器的邏輯 (例如通過 settings 更新 enabled 狀態)
            #      logger.info(f"請求重啟偵測器: {detector_name} (此功能尚未完全實現)")
            # elif command_type == "capture_image":
            #      # 手動觸發拍照並上傳
            #      # 確保 capture_manager 變數在作用域內
            #      if 'capture_manager' in locals():
            #          s3_path = capture_manager.capture_and_upload_image("manual_capture")
            #          logger.info(f"手動捕獲影像，S3 Path: {s3_path}")
            #      else:
            #          logger.warning("Capture manager 未初始化，無法執行手動捕獲。")
            # # ... 其他命令 ...
            # else:
                # logger.warning(f"未知命令類型: {command_type}")
        except json.JSONDecodeError:
            logger.error("無法解析收到的命令 Payload (非 JSON 格式)。")
        except Exception as e:
            logger.error(f"處理雲端命令時發生錯誤: {e}", exc_info=True)

    iot_client = AWSIoTClient(iot_settings, command_callback=handle_cloud_command)
    if not iot_client.is_connected():
        logger.error("無法連接到 AWS IoT Core。應用程式終止。")
        # 在退出前嘗試清理資源
        s3_uploader.stop()
        s3_uploader.join()
        return


    # 模型管理器和推論器
    model_settings = settings.get('models', {})
    model_manager = ModelManager(model_settings)
    # 載入物件偵測模型 (這是大多數偵測器需要的基本模型)
    object_detection_model = model_manager.get_model("object_detection")
    if object_detection_model is None:
        logger.error("無法載入物件偵測模型，應用程式終止。")
        # 在退出前嘗試清理資源
        iot_client.disconnect()
        s3_uploader.stop()
        s3_uploader.join() # 等待上傳執行緒結束
        return

    object_detector_inferencer = ObjectDetector(
        model=object_detection_model,
        class_mapping=model_settings.get('object_detection', {}).get('class_mapping', {}) # 提供預設值
    )
    # 可擴展載入和初始化其他推論器（如果需要）
    # classification_model = model_manager.get_model("classification")
    # classification_inferencer = Classifier(classification_model)

    # 事件管理器和發布器
    event_settings = settings.get('events', {})
    event_manager = EventManager(event_settings)
    event_publisher = EventPublisher(iot_client, settings['aws']['iot']['thing_name'])

    # 捕獲管理器
    capture_manager = CaptureManager(s3_uploader, settings['aws']['s3'])

    # 偵測器 (根據設定啟用)
    detectors = []
    detector_settings = settings.get('detectors', {})

    if detector_settings.get('person', {}).get('enabled', False):
        logger.info("初始化人員偵測器...")
        person_detector = PersonDetector(
            settings=detector_settings['person'],
            object_detector=object_detector_inferencer, # 人員偵測器需要物件偵測結果
            event_manager=event_manager,
            event_publisher=event_publisher,
            capture_manager=capture_manager
        )
        detectors.append(person_detector)

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

    # 確保本地顯示設定存在
    display_settings = settings.get('display', {})
    display_enabled = display_settings.get('enabled', False)
    display_width = display_settings.get('max_width', 800)
    display_height = display_settings.get('max_height', 600)


    while not stop_requested.is_set():
        ret, frame_np = cap.read()
        if not ret:
            logger.warning("無法從攝影機讀取幀。")
            # 這裡可以添加處理攝影機離線的邏輯，例如等待一段時間後重試
            if not stop_requested.is_set(): # 只有在未請求停止時才等待
                time.sleep(0.1)
            continue

        processing_frame_count += 1
        current_time = time.time()

        # 更新捕獲管理器中的當前幀，以便偵測器需要時可以獲取最新影像
        capture_manager.update_frame(frame_np)

        # 將 NumPy 影像轉換為 CUDA 影像
        frame_cuda = None # 初始化為 None
        try:
             # 注意：jetson_utils.cudaFromNumpy 期望輸入是 HWC, RGB
             # OpenCV 的 cap.read() 返回的是 HWC, BGR
             # 所以需要先進行顏色空間轉換，這一步可能比較耗時
             rgb_frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
             frame_cuda = jetson_utils.cudaFromNumpy(rgb_frame_np)
        except Exception as e:
             logger.error(f"NumPy 到 CUDA 轉換失敗: {e}", exc_info=True)
             continue # 跳過當前幀

        # 執行邊緣模型推論 (例如：物件偵測)
        detections_raw = []
        try:
            if object_detector_inferencer: # 確保推論器已初始化
                detections_raw = object_detector_inferencer.infer(frame_cuda)
                # logger.debug(f"偵測到 {len(detections_raw)} 個物體。")
        except Exception as e:
            logger.error(f"物件偵測推論失敗: {e}", exc_info=True)
            detections_raw = [] # 出錯時將結果設置為空列表

        # 將原始偵測結果傳遞給所有偵測器進行處理
        for detector in detectors:
            try:
                detector.process(frame_cuda, detections_raw)
            except Exception as e:
                logger.error(f"偵測器 '{detector.__class__.__name__}' 處理失敗: {e}", exc_info=True)


        # 可選：在本地顯示處理後的影像 (用於調試)
        if display_enabled:
            # 在 NumPy 影像上繪製偵測結果
            # 注意：這裡為了簡化，在原始 frame_np 上繪製。如果需要在 CUDA 圖像上繪製並顯示，
            # 且顯示模塊接受 CUDA 輸入，則需要不同的邏輯。
            # OpenCV 繪圖較慢，可能影響整體幀率
            frame_with_detections = draw_detections(
                 frame_np.copy(), # 在拷貝上繪製，避免影響原始幀
                 detections_raw,
                 object_detector_inferencer.class_mapping # 傳入 class mapping
            )
            # 如果需要繪製 ROI 等，可以在這裡調用 image_utils.draw_roi 等方法

            display_frame = resize_for_display(frame_with_detections, display_width, display_height)
            cv2.imshow("Edge Detection", display_frame)

            # 處理鍵盤輸入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 按 'q' 或 ESC 退出
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