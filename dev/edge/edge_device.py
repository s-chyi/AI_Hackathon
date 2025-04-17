#!/usr/bin/env python3
"""
智能倉儲邊緣設備控制器 - Jetson 版本 (無 IoT Core 連接)
整合物體偵測、異常識別和AWS S3 上傳功能
"""

import cv2
import jetson.inference
import jetson.utils
import numpy as np
import time
import boto3
import json
import os
import threading
import queue
import logging
from datetime import datetime
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError

# --- .env 文件說明 ---
# 請確保您的 .env 文件包含以下變數：
# AWS_ACCESS_KEY=YOUR_AWS_ACCESS_KEY
# AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_KEY
# S3_BUCKET=your-s3-bucket-name
# S3_FOLDER=your-s3-folder-path/  (例如: warehouse-images/)
# S3_REGION=your-aws-region (例如: us-east-1)
# DEVICE_ID=your-unique-device-identifier (例如: jetson-cam-01)
# --- ------------- ---
load_dotenv(verbose=True)

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("edge_device.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EdgeDevice")

# AWS S3 配置
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('S3_BUCKET')
S3_FOLDER = os.getenv('S3_FOLDER', 'images/') # 提供默認值
S3_REGION = os.getenv('S3_REGION')
DEVICE_ID = os.getenv('DEVICE_ID', f'jetson-{int(time.time())}') # 提供默認值

# 檢測設置
CONFIDENCE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 5
MAX_QUEUE_SIZE = 100
BATCH_UPLOAD_SIZE = 5
BATCH_UPLOAD_INTERVAL = 60  # 秒

# 圖像上傳緩存隊列
image_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# 設備狀態 (本地記錄)
device_status = {
    "online": True, # 指應用程式是否正在運行
    "last_upload_attempt": None,
    "last_successful_upload": None,
    "pending_images": 0,
    "device_id": DEVICE_ID
}

class EdgeDevice:
    def __init__(self):
        """初始化邊緣設備控制器"""
        self.net = None
        self.cap = None
        self.s3_client = None
        self.last_detection_time = 0 # 用於冷卻
        self.device_id = DEVICE_ID
        self.camera_id = 10  # /dev/video10 (根據您的設備調整)
        self.is_running = False
        self.upload_thread = None

    def initialize(self):
        """初始化設備和連接"""
        try:
            # 初始化物體偵測網絡
            logger.info("初始化 SSD-MobileNet-v2 偵測網絡...")
            self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=CONFIDENCE_THRESHOLD)

            # 初始化攝像頭
            logger.info(f"連接攝像頭 /dev/video{self.camera_id}...")
            # 嘗試 GStreamer Pipline (如果 /dev/videoX 不穩定)
            # gst_str = f"v4l2src device=/dev/video{self.camera_id} ! video/x-raw, width=1280, height=720 ! videoconvert ! video/x-raw, format=BGR ! appsink"
            # self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                logger.info(f"攝像頭 Width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")


            if not self.cap or not self.cap.isOpened():
                logger.error(f"無法打開攝像頭設備 /dev/video{self.camera_id}")
                return False

            # 初始化 AWS S3 客戶端
            logger.info("連接 AWS S3...")
            if not all([AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, S3_BUCKET, S3_REGION]):
                 logger.error("缺少必要的 AWS S3 配置信息 (ACCESS_KEY, SECRET_KEY, BUCKET, REGION)")
                 return False

            self.s3_client = boto3.client(
                's3',
                region_name=S3_REGION,
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
            logger.info(f"S3 客戶端初始化完成，目標 Bucket: {S3_BUCKET}, 區域: {S3_REGION}")

            # 啟動上傳線程
            self.upload_thread = threading.Thread(target=self.batch_upload_worker)
            self.upload_thread.daemon = True # 主線程退出時，此線程也退出
            self.upload_thread.start()

            logger.info("邊緣設備初始化完成")
            return True

        except Exception as e:
            logger.error(f"初始化失敗: {str(e)}", exc_info=True) # 添加詳細錯誤信息
            return False

    def log_device_status(self):
        """記錄設備狀態到日誌"""
        global device_status
        device_status["timestamp"] = datetime.now().isoformat()
        device_status["pending_images"] = image_queue.qsize()
        logger.info(f"設備狀態: {json.dumps(device_status)}")


    def start_detection(self):
        """開始物體偵測循環"""
        if not self.cap or not self.net:
            logger.error("設備未完全初始化，無法開始偵測")
            return False

        self.is_running = True
        logger.info("開始物體偵測循環")
        last_status_log_time = time.time()

        try:
            while self.is_running and self.cap.isOpened():
                # 捕獲影像
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("無法獲取影像，嘗試重新連接攝像頭...")
                    self.cap.release()
                    time.sleep(2) # 等待一下
                    self.cap = cv2.VideoCapture(self.camera_id)
                    if self.cap is not None:
                       self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                       self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    if not self.cap or not self.cap.isOpened():
                       logger.error("重新連接攝像頭失敗，退出偵測循環。")
                       self.is_running = False # 標記為停止
                       break
                    continue

                # 轉換為 CUDA 格式
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cuda_img = jetson.utils.cudaFromNumpy(frame_rgb)
                except Exception as e:
                    logger.error(f"轉換影像到 CUDA 格式時出錯: {e}")
                    time.sleep(0.1) # 避免快速連續錯誤
                    continue

                # 執行偵測
                detections = self.net.Detect(cuda_img)

                # 處理偵測結果
                frame_with_detections = self.process_detections(frame, detections)

                # 顯示處理後的影像 (可選)
                display_frame = self.resize_for_display(frame_with_detections)
                cv2.imshow("智能倉儲邊緣監控", display_frame)

                # 檢查鍵盤輸入
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    logger.info("收到 ESC 鍵，停止偵測...")
                    break
                elif key == ord('s'):  # 手動捕獲
                    logger.info("收到 's' 鍵，手動觸發捕獲...")
                    self.capture_and_queue(frame=frame, manual=True)

                # 定期記錄設備狀態
                current_time = time.time()
                if current_time - last_status_log_time >= 60:  # 每 60 秒記錄一次
                    self.log_device_status()
                    last_status_log_time = current_time

        except Exception as e:
            logger.error(f"偵測循環中出錯: {str(e)}", exc_info=True)
        finally:
            self.stop()

        return True

    def process_detections(self, frame, detections):
        """處理偵測結果並決定是否捕獲事件"""
        person_detected = False
        current_time = time.time()

        # 複製影像以便繪製
        output_frame = frame.copy()

        for detection in detections:
            class_id = detection.ClassID
            confidence = detection.Confidence
            left = int(detection.Left)
            top = int(detection.Top)
            right = int(detection.Right)
            bottom = int(detection.Bottom)

            # 繪製邊界框
            cv2.rectangle(output_frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # 添加類別標籤和置信度
            class_name = self.net.GetClassDesc(class_id)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(output_frame, label, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 檢查是否是人 (COCO class ID 1)，且置信度足夠
            if class_id == 1 and confidence >= CONFIDENCE_THRESHOLD:
                person_detected = True

                # 顯示"人員已偵測"提示
                cv2.putText(output_frame, "Person Detected!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 如果偵測到人，且冷卻時間已過，則捕獲圖像
        if person_detected and (current_time - self.last_detection_time) > COOLDOWN_SECONDS:
            logger.info(f"偵測到人員，觸發圖像捕獲 (冷卻時間: {COOLDOWN_SECONDS}秒)")
            self.capture_and_queue(frame, detection_type="person")
            self.last_detection_time = current_time # 更新上次偵測時間

        return output_frame

    def capture_and_queue(self, frame=None, detection_type="person", manual=False):
        """捕獲圖像並將其加入上傳隊列"""
        try:
            if frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("捕獲圖像失敗: 無法從攝像頭讀取幀")
                    return False

            # 生成時間戳和文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # 添加毫秒以增加唯一性
            prefix = "manual" if manual else detection_type
            # 確保 S3_FOLDER 以 '/' 結尾
            s3_folder_path = S3_FOLDER if S3_FOLDER.endswith('/') else S3_FOLDER + '/'
            # 建立本地臨時目錄 (如果不存在)
            local_temp_dir = "temp_images"
            os.makedirs(local_temp_dir, exist_ok=True)

            local_filename = os.path.join(local_temp_dir, f"{prefix}_{self.device_id}_{timestamp}.jpg")
            s3_key = f"{s3_folder_path}{prefix}_{self.device_id}_{timestamp}.jpg"

            # 保存圖像到臨時文件
            save_success = cv2.imwrite(local_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90]) # 控制質量以減小文件大小
            if not save_success:
                logger.error(f"無法將圖像保存到本地文件: {local_filename}")
                return False
            logger.info(f"圖像已暫存到本地: {local_filename}")

            # 將圖像加入上傳隊列
            if not image_queue.full():
                image_queue.put((local_filename, s3_key))
                logger.info(f"圖像已加入 S3 上傳隊列: {s3_key} (隊列大小: {image_queue.qsize()})")
            else:
                logger.warning(f"S3 上傳隊列已滿 (大小: {image_queue.maxsize})，無法加入新圖像: {local_filename}。考慮丟棄或增加隊列大小。")
                # 選項：可以考慮刪除本地文件以防佔滿空間
                try:
                   os.remove(local_filename)
                   logger.warning(f"已刪除無法入隊的本地圖像: {local_filename}")
                except OSError as e:
                   logger.error(f"刪除無法入隊的本地圖像失敗: {e}")


            # 記錄事件信息 (替代 IoT 發布)
            event = {
                "type": "detection_event",
                "detection_type": prefix,
                "timestamp": datetime.now().isoformat(), # 使用 ISO 格式時間戳
                "device_id": self.device_id,
                "confidence_threshold": CONFIDENCE_THRESHOLD, # 可以記錄當時的閾值
                "s3_image_key": s3_key,
                "manual_trigger": manual
            }
            logger.info(f"事件記錄: {json.dumps(event)}")

            return True
        except Exception as e:
            logger.error(f"捕獲和隊列處理失敗: {str(e)}", exc_info=True)
            return False

    def batch_upload_worker(self):
        """批量上傳圖像到 S3 的工作線程"""
        global device_status
        logger.info("S3 批量上傳線程已啟動。")
        while True:
            try:
                items_to_upload = []
                start_time = time.time()

                # 等待直到達到批量大小或超時
                while len(items_to_upload) < BATCH_UPLOAD_SIZE and (time.time() - start_time) < BATCH_UPLOAD_INTERVAL:
                    try:
                        # 使用超時 get，避免線程永久阻塞
                        item = image_queue.get(timeout=1.0)
                        items_to_upload.append(item)
                        image_queue.task_done() # 標記任務完成
                    except queue.Empty:
                        # 如果隊列為空，檢查是否該上傳了 (基於時間)
                        if len(items_to_upload) > 0 and (time.time() - start_time) >= BATCH_UPLOAD_INTERVAL:
                           break # 時間到了，即使未滿 batch size 也上傳
                        # 隊列空且未到上傳時間，繼續等待
                        continue

                # 如果收集到需要上傳的圖像
                if items_to_upload:
                    logger.info(f"準備批量上傳 {len(items_to_upload)} 張圖像到 S3...")
                    device_status["last_upload_attempt"] = datetime.now().isoformat()
                    successful_uploads = 0
                    for local_file, s3_key in items_to_upload:
                        try:
                            if os.path.exists(local_file):
                                self.s3_client.upload_file(local_file, S3_BUCKET, s3_key)
                                logger.info(f"成功上傳: s3://{S3_BUCKET}/{s3_key}")
                                successful_uploads += 1
                                # 上傳成功後刪除本地臨時文件
                                try:
                                    os.remove(local_file)
                                    logger.debug(f"已刪除本地臨時文件: {local_file}")
                                except OSError as e:
                                    logger.error(f"刪除本地文件失敗 {local_file}: {e}")
                            else:
                                logger.warning(f"嘗試上傳時，本地文件已不存在: {local_file} (S3 key: {s3_key})")
                        except (NoCredentialsError, ClientError) as e:
                            logger.error(f"S3 上傳權限或客戶端錯誤 {s3_key}: {str(e)}")
                            # 權限錯誤通常無法通過重試解決，可能需要重新放回隊列或放棄
                            # 暫時放棄，避免無限循環
                            logger.warning(f"放棄上傳 (權限/客戶端錯誤): {local_file}")
                            # 考慮是否刪除本地文件
                        except Exception as e:
                            logger.error(f"上傳失敗 {s3_key}: {str(e)}", exc_info=True)
                            # 其他錯誤，可以考慮重新放回隊列 (但要小心無限重試)
                            logger.warning(f"將上傳失敗的任務放回隊列: {local_file}")
                            if os.path.exists(local_file): # 確保文件還在
                                image_queue.put((local_file, s3_key)) # 簡單重試機制

                    if successful_uploads > 0:
                        device_status["last_successful_upload"] = datetime.now().isoformat()
                        logger.info(f"本次批量上傳完成，成功 {successful_uploads} / {len(items_to_upload)} 張。")
                    else:
                         logger.warning(f"本次批量上傳未能成功上傳任何圖像 ({len(items_to_upload)} 個任務)。")
                    # 更新狀態並記錄日誌
                    self.log_device_status()

                # 如果隊列為空，稍微休息一下，避免CPU空轉
                if image_queue.empty() and not items_to_upload:
                   time.sleep(1) # 短暫休眠

            except Exception as e:
                logger.error(f"S3 批量上傳工作線程發生嚴重錯誤: {str(e)}", exc_info=True)
                time.sleep(10)  # 發生錯誤時等待較長時間再重試

    def resize_for_display(self, image, max_width=800, max_height=600):
        """調整圖像大小以適合顯示，保持縱橫比"""
        try:
            h, w = image.shape[:2]
            if h == 0 or w == 0: return image # 無效圖像

            # 計算調整因子
            scale = min(max_width / w, max_height / h)

            # 只有在圖像較大時才調整大小
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA) # 使用 INTER_AREA 縮小
                return resized
            return image
        except Exception as e:
            logger.warning(f"調整圖像大小失敗: {e}")
            return image # 返回原始圖像

    def stop(self):
        """停止設備並釋放資源"""
        if self.is_running:
            logger.info("正在停止邊緣設備...")
            self.is_running = False # 通知偵測循環停止

            # 等待上傳線程處理完畢 (可選，設置超時)
            # 注意：如果隊列很大，這可能會阻塞很久
            # image_queue.join() # 等待隊列為空

            # 釋放攝像頭
            if self.cap and self.cap.isOpened():
                self.cap.release()
                logger.info("攝像頭已釋放")

            # 關閉顯示窗口
            cv2.destroyAllWindows()
            logger.info("顯示窗口已關閉")

            # 不需要斷開 IoT 連接了

            logger.info("邊緣設備已停止。")


def main():
    """主函數"""
    logger.info("啟動智能倉儲邊緣設備控制器 (無 IoT Core 版本)...")
    # 創建臨時圖像目錄
    os.makedirs("temp_images", exist_ok=True)
    # 檢查必要的環境變數
    required_vars = ['AWS_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY', 'S3_BUCKET', 'S3_REGION', 'DEVICE_ID']
    if not all(os.getenv(var) for var in required_vars):
        logger.error(f"啟動失敗：缺少必要的環境變數。請檢查 .env 文件是否包含: {', '.join(required_vars)}")
        return

    # 初始化設備
    device = EdgeDevice()
    if device.initialize():
        try:
            # 開始偵測循環
            device.start_detection()
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C，正在停止程序...")
        finally:
            # 確保在退出前調用 stop
            device.stop()
    else:
        logger.error("設備初始化失敗，無法啟動偵測。")

    logger.info("程序執行完畢。")

if __name__ == "__main__":
    main()