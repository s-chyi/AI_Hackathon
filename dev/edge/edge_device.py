#!/usr/bin/env python3
"""
智能倉儲邊緣設備控制器 - Jetson 版本
整合物體偵測、異常識別和AWS雲端服務連接功能
"""

import cv2
import jetson_inference
import jetson_utils
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
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

load_dotenv(verbose=True)

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/logs/edge_device.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EdgeDevice")

# AWS 配置
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('S3_BUCKET')
S3_FOLDER = os.getenv('S3_FOLDER')
S3_REGION = os.getenv('S3_REGION')
IOT_ENDPOINT = os.getenv('IOT_ENDPOINT')  # 請替換為您的 IoT Core 端點
IOT_TOPIC = os.getenv('IOT_TOPIC')
IOT_CLIENT_ID = os.getenv('IOT_CLIENT_ID')
IOT_CERT_PATH = 'certificates/'  # 存放 IoT 證書的路徑

# 檢測設置
CONFIDENCE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 5
MAX_QUEUE_SIZE = 100
BATCH_UPLOAD_SIZE = 5
BATCH_UPLOAD_INTERVAL = 60  # 秒

# 緩存隊列
image_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
event_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# 設備狀態
device_status = {
    "online": True,
    "last_upload": None,
    "pending_events": 0,
    "battery": 100,  # 假設設備是電池供電的
    "device_id": IOT_CLIENT_ID
}

class EdgeDevice:
    def __init__(self):
        """初始化邊緣設備控制器"""
        self.net = None
        self.cap = None
        self.s3_client = None
        self.iot_client = None
        self.last_upload_time = 0
        self.device_id = IOT_CLIENT_ID
        self.camera_id = 10  # /dev/video10
        self.is_running = False
        self.upload_thread = None
        self.iot_thread = None
        
    def initialize(self):
        """初始化設備和連接"""
        try:
            # 初始化物體偵測網絡
            logger.info("初始化 SSD-MobileNet-v2 偵測網絡...")
            self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=CONFIDENCE_THRESHOLD)
            
            # 初始化攝像頭
            logger.info(f"連接攝像頭 /dev/video{self.camera_id}...")
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            if not self.cap.isOpened():
                logger.error(f"無法打開攝像頭設備 /dev/video{self.camera_id}")
                return False
                
            # 初始化 AWS 客戶端
            logger.info("連接 AWS S3...")
            self.s3_client = boto3.client(
                's3',
                region_name=S3_REGION,
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )

            
            # 初始化 IoT 客戶端
            self.setup_iot_client()
            
            # 啟動上傳線程
            self.upload_thread = threading.Thread(target=self.batch_upload_worker)
            self.upload_thread.daemon = True
            self.upload_thread.start()
            
            # 啟動 IoT 發布線程
            self.iot_thread = threading.Thread(target=self.iot_publish_worker)
            self.iot_thread.daemon = True
            self.iot_thread.start()
            
            logger.info("邊緣設備初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化失敗: {str(e)}")
            return False
    
    def setup_iot_client(self):
        """設置 AWS IoT MQTT 客戶端"""
        try:
            # 創建 MQTT 客戶端
            self.iot_client = AWSIoTMQTTClient(IOT_CLIENT_ID)
            self.iot_client.configureEndpoint(IOT_ENDPOINT, 8883)
            
            # 配置證書
            self.iot_client.configureCredentials(
                f"{IOT_CERT_PATH}AmazonRootCA1.pem",
                f"{IOT_CERT_PATH}private.pem.key",
                f"{IOT_CERT_PATH}certificate.pem.crt"
            )
            
            # 配置連接參數
            self.iot_client.configureAutoReconnectBackoffTime(1, 32, 20)
            self.iot_client.configureOfflinePublishQueueing(-1)  # 無限隊列
            self.iot_client.configureDrainingFrequency(2)  # 每 2 秒嘗試一次
            self.iot_client.configureConnectDisconnectTimeout(10)
            self.iot_client.configureMQTTOperationTimeout(5)
            logger.info("已設置連接參數至 AWS IoT Core")
            # 連接到 IoT Core
            self.iot_client.connect()
            logger.info("已連接到 AWS IoT Core")
            
            # 訂閱命令主題
            self.iot_client.subscribe(f"warehouse/commands/{self.device_id}", 1, self.command_callback)
            logger.info(f"已訂閱命令主題: warehouse/commands/{self.device_id}")
            
            # 發布設備狀態
            self.publish_device_status()
            
            return True
        except Exception as e:
            logger.error(f"IoT 客戶端設置失敗: {str(e)}")
            return False
    
    def command_callback(self, client, userdata, message):
        """處理從雲端接收的命令"""
        try:
            payload = json.loads(message.payload)
            command = payload.get("command")
            
            logger.info(f"收到命令: {command}")
            
            if command == "capture":
                # 手動捕獲一張圖像
                self.capture_and_queue(manual=True)
            elif command == "change_settings":
                # 更改設備設置
                new_settings = payload.get("settings", {})
                if "confidence_threshold" in new_settings:
                    global CONFIDENCE_THRESHOLD
                    CONFIDENCE_THRESHOLD = float(new_settings["confidence_threshold"])
                    self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=CONFIDENCE_THRESHOLD)
                    logger.info(f"已更新置信度閾值為: {CONFIDENCE_THRESHOLD}")
            elif command == "restart":
                # 重啟設備
                logger.info("收到重啟命令，準備重啟設備...")
                self.stop()
                time.sleep(1)
                self.initialize()
                self.start_detection()
        except Exception as e:
            logger.error(f"處理命令時出錯: {str(e)}")
    
    def publish_device_status(self):
        """發布設備狀態到 IoT Core"""
        if not self.iot_client:
            logger.warning("IoT 客戶端未初始化，無法發布狀態")
            return False
            
        try:
            # 更新設備狀態
            global device_status
            device_status["timestamp"] = datetime.now().isoformat()
            device_status["pending_events"] = event_queue.qsize()
            device_status["pending_images"] = image_queue.qsize()
            
            # 發布狀態
            self.iot_client.publish(
                f"warehouse/status/{self.device_id}",
                json.dumps(device_status),
                1
            )
            logger.debug("已發布設備狀態")
            return True
        except Exception as e:
            logger.error(f"發布設備狀態失敗: {str(e)}")
            return False
    
    def start_detection(self):
        """開始物體偵測循環"""
        if not self.cap or not self.net:
            logger.error("設備未完全初始化，無法開始偵測")
            return False
            
        self.is_running = True
        logger.info("開始物體偵測循環")
        
        try:
            while self.is_running and self.cap.isOpened():
                # 捕獲影像
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("無法獲取影像")
                    time.sleep(0.5)
                    continue
                
                # 轉換為 CUDA 格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cuda_img = jetson.utils.cudaFromNumpy(frame_rgb)
                
                # 執行偵測
                detections = self.net.Detect(cuda_img)
                
                # 處理偵測結果
                frame_with_detections = self.process_detections(frame, detections)
                
                # 顯示處理後的影像
                display_frame = self.resize_for_display(frame_with_detections)
                cv2.imshow("智能倉儲邊緣監控", display_frame)
                
                # 檢查鍵盤輸入
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # 手動捕獲
                    self.capture_and_queue(frame=frame, manual=True)
                    
                # 定期發布設備狀態
                if time.time() % 30 < 1:  # 大約每 30 秒
                    self.publish_device_status()
                    
        except Exception as e:
            logger.error(f"偵測循環中出錯: {str(e)}")
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
            
            # 檢查是否是人，且置信度足夠
            if class_id == 1 and confidence >= CONFIDENCE_THRESHOLD:  # COCO 數據集中人的ID是1
                person_detected = True
                
                # 顯示"人員已偵測"提示
                cv2.putText(output_frame, "Person Detected!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        # 如果偵測到人，且冷卻時間已過，則捕獲圖像
        if person_detected and (current_time - self.last_upload_time) > COOLDOWN_SECONDS:
            self.capture_and_queue(frame, detection_type="person")
            self.last_upload_time = current_time
            
        return output_frame
    
    def capture_and_queue(self, frame=None, detection_type="person", manual=False):
        """捕獲圖像並將其加入上傳隊列"""
        try:
            if frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("無法捕獲圖像")
                    return False
            
            # 生成時間戳和文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "manual" if manual else detection_type
            local_filename = f"temp_{prefix}_{timestamp}.jpg"
            s3_key = f"{S3_FOLDER}{prefix}_{self.device_id}_{timestamp}.jpg"
            
            # 保存圖像到臨時文件
            cv2.imwrite(local_filename, frame)
            logger.info(f"已保存圖像: {local_filename}")
            
            # 將圖像加入上傳隊列
            if not image_queue.full():
                image_queue.put((local_filename, s3_key))
                logger.info(f"圖像已加入上傳隊列: {s3_key}")
            else:
                logger.warning("上傳隊列已滿，無法加入新圖像")
                
            # 創建事件並加入事件隊列
            event = {
                "type": "detection",
                "detection_type": prefix,
                "timestamp": timestamp,
                "device_id": self.device_id,
                "confidence": CONFIDENCE_THRESHOLD,
                "image_key": s3_key,
                "manual": manual
            }
            
            if not event_queue.full():
                event_queue.put(event)
            else:
                logger.warning("事件隊列已滿，無法加入新事件")
                
            return True
        except Exception as e:
            logger.error(f"捕獲和隊列處理失敗: {str(e)}")
            return False
    
    def batch_upload_worker(self):
        """批量上傳工作線程"""
        while True:
            try:
                batch = []
                # 收集批量上傳的圖像
                while len(batch) < BATCH_UPLOAD_SIZE and not image_queue.empty():
                    batch.append(image_queue.get())
                    
                # 如果有圖像需要上傳
                if batch:
                    logger.info(f"開始批量上傳 {len(batch)} 張圖像")
                    for local_file, s3_key in batch:
                        try:
                            if os.path.exists(local_file):
                                self.s3_client.upload_file(local_file, S3_BUCKET, s3_key)
                                logger.info(f"已上傳: s3://{S3_BUCKET}/{s3_key}")
                                
                                # 刪除臨時文件
                                os.remove(local_file)
                                logger.debug(f"已刪除臨時文件: {local_file}")
                            else:
                                logger.warning(f"臨時文件不存在: {local_file}")
                        except Exception as e:
                            logger.error(f"上傳失敗 {s3_key}: {str(e)}")
                            # 重新放回隊列，稍後再試
                            if os.path.exists(local_file):
                                image_queue.put((local_file, s3_key))
                    
                    # 更新設備狀態
                    device_status["last_upload"] = datetime.now().isoformat()
                    self.publish_device_status()
                    
                # 等待下一批或直到隊列有足夠的項目
                if image_queue.empty() or len(batch) < BATCH_UPLOAD_SIZE:
                    time.sleep(BATCH_UPLOAD_INTERVAL)
                
            except Exception as e:
                logger.error(f"批量上傳工作線程錯誤: {str(e)}")
                time.sleep(5)  # 發生錯誤時短暫暫停
    
    def iot_publish_worker(self):
        """IoT 事件發布工作線程"""
        while True:
            try:
                batch = []
                # 收集批量發布的事件
                while len(batch) < BATCH_UPLOAD_SIZE and not event_queue.empty():
                    batch.append(event_queue.get())
                    
                # 如果有事件需要發布
                if batch:
                    logger.info(f"開始批量發布 {len(batch)} 個事件")
                    for event in batch:
                        try:
                            if self.iot_client and self.iot_client.connect:
                                self.iot_client.publish(
                                    IOT_TOPIC,
                                    json.dumps(event),
                                    1  # QoS 1
                                )
                                logger.info(f"已發布事件: {event['type']} - {event['timestamp']}")
                            else:
                                logger.warning("IoT 客戶端未連接，將事件放回隊列")
                                event_queue.put(event)
                        except Exception as e:
                            logger.error(f"發布事件失敗: {str(e)}")
                            # 重新放回隊列
                            event_queue.put(event)
                
                # 等待下一批或直到隊列有足夠的項目
                if event_queue.empty() or len(batch) < BATCH_UPLOAD_SIZE:
                    time.sleep(10)  # 事件發布間隔較短
                    
            except Exception as e:
                logger.error(f"IoT 發布工作線程錯誤: {str(e)}")
                time.sleep(5)
    
    def resize_for_display(self, image, max_width=800, max_height=600):
        """調整圖像大小以適合顯示，保持縱橫比"""
        h, w = image.shape[:2]
        
        # 計算調整因子
        scale = min(max_width / w, max_height / h)
        
        # 只有在圖像較大時才調整大小
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            return resized
        return image
    
    def stop(self):
        """停止設備並釋放資源"""
        self.is_running = False
        
        # 釋放攝像頭
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        # 關閉顯示窗口
        cv2.destroyAllWindows()
        
        # 斷開 IoT 連接
        if self.iot_client:
            self.iot_client.disconnect()
            
        logger.info("設備已停止")


def main():
    """主函數"""
    # 創建輸出目錄
    os.makedirs("logs", exist_ok=True)
    
    # 初始化設備
    device = EdgeDevice()
    if device.initialize():
        # 開始偵測循環
        device.start_detection()
    else:
        logger.error("設備初始化失敗，無法啟動")

if __name__ == "__main__":
    main()