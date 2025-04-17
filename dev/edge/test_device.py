#!/usr/bin/env python3
"""
智能倉儲邊緣設備測試套件
用於分步驟測試各個功能模組
"""

import sys
import os
import cv2
import time
import argparse
import unittest
import logging
import json
import boto3
import threading
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("edge_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EdgeTests")

# 導入邊緣設備模組 (假設邊緣設備代碼保存為 edge_device.py)
sys.path.append('.')
try:
    from edge_device import EdgeDevice
except ImportError:
    logger.error("無法導入 EdgeDevice 模組，請確保 edge_device.py 在當前目錄")
    sys.exit(1)

class TestCameraModule(unittest.TestCase):
    """測試攝像頭模組功能"""
    
    def setUp(self):
        """測試前的準備工作"""
        # 使用模擬的AWS服務客戶端
        self.mock_s3 = MagicMock()
        self.mock_iot = MagicMock()
        
        # 創建 EdgeDevice 實例但不初始化任何連接
        self.device = EdgeDevice()
        self.device.s3_client = self.mock_s3
        self.device.iot_client = self.mock_iot
    
    def test_camera_connection(self):
        """測試攝像頭連接"""
        # 測試不同的攝像頭ID
        for camera_id in [0, 10]:  # 嘗試默認攝像頭和/dev/video10
            logger.info(f"嘗試連接攝像頭 ID: {camera_id}")
            cap = cv2.VideoCapture(camera_id)
            is_opened = cap.isOpened()
            
            if is_opened:
                logger.info(f"成功連接到攝像頭 {camera_id}")
                # 讀取一幀
                ret, frame = cap.read()
                self.assertTrue(ret, f"無法從攝像頭 {camera_id} 讀取影像")
                
                # 檢查影像尺寸
                height, width = frame.shape[:2]
                logger.info(f"攝像頭 {camera_id} 影像尺寸: {width}x{height}")
                
                # 保存測試圖像
                test_file = f"test_camera_{camera_id}.jpg"
                cv2.imwrite(test_file, frame)
                logger.info(f"測試圖像已保存至 {test_file}")
                
                # 釋放攝像頭
                cap.release()
            else:
                logger.warning(f"無法連接到攝像頭 {camera_id}")
        
        # 至少一個攝像頭應該可用
        # 這裡我們只是記錄結果，不強制斷言，因為測試環境可能沒有攝像頭
        
    def test_resize_for_display(self):
        """測試圖像縮放功能"""
        # 創建測試圖像
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        test_image[:] = (0, 0, 255)  # 紅色測試圖像
        
        # 測試縮放功能
        resized = self.device.resize_for_display(test_image, 800, 600)
        
        # 檢查縮放後的尺寸
        height, width = resized.shape[:2]
        self.assertLessEqual(width, 800, "寬度應該小於等於800")
        self.assertLessEqual(height, 600, "高度應該小於等於600")
        
        # 檢查縱橫比是否維持
        original_ratio = 1920 / 1080
        resized_ratio = width / height
        self.assertAlmostEqual(original_ratio, resized_ratio, delta=0.1, 
                              msg="縮放後的縱橫比應該與原圖相近")

class TestObjectDetection(unittest.TestCase):
    """測試物體偵測功能"""
    
    def setUp(self):
        """測試前的準備工作"""
        # 跳過如果沒有GPU或jetson.inference
        try:
            import jetson.inference
            import jetson.utils
            self.skip_tests = False
        except ImportError:
            logger.warning("無法導入 jetson.inference，跳過物體偵測測試")
            self.skip_tests = True
    
    def test_load_detection_model(self):
        """測試載入偵測模型"""
        if self.skip_tests:
            self.skipTest("缺少 jetson.inference 模組")
            
        try:
            import jetson.inference
            # 嘗試載入模型
            net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
            self.assertIsNotNone(net, "應該成功載入偵測模型")
            logger.info("成功載入 SSD-MobileNet-v2 模型")
            
            # 檢查模型屬性
            class_count = net.GetNumClasses()
            logger.info(f"模型類別數量: {class_count}")
            self.assertGreater(class_count, 0, "模型應該支持至少一個類別")
            
            # 測試獲取類別名稱
            class_desc = net.GetClassDesc(1)  # 人的類別ID是1
            logger.info(f"類別 ID 1 描述: {class_desc}")
            self.assertEqual(class_desc.lower(), "person", "類別1應該是person")
            
        except Exception as e:
            self.fail(f"載入偵測模型失敗: {str(e)}")
    
    def test_image_detection(self):
        """測試靜態圖像的物體偵測"""
        if self.skip_tests:
            self.skipTest("缺少 jetson.inference 模組")
            
        try:
            import jetson.inference
            import jetson.utils
            
            # 檢查測試圖像是否存在
            test_image_path = "test_camera_0.jpg"
            if not os.path.exists(test_image_path):
                logger.warning(f"測試圖像 {test_image_path} 不存在，嘗試創建...")
                # 嘗試捕獲一幀
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(test_image_path, frame)
                        logger.info(f"創建了測試圖像 {test_image_path}")
                    cap.release()
            
            if not os.path.exists(test_image_path):
                self.skipTest(f"無法找到或創建測試圖像 {test_image_path}")
                
            # 載入模型和圖像
            net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
            cv_img = cv2.imread(test_image_path)
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            cuda_img = jetson.utils.cudaFromNumpy(rgb_img)
            
            # 執行偵測
            detections = net.Detect(cuda_img)
            
            # 處理並記錄結果
            logger.info(f"在圖像中偵測到 {len(detections)} 個物體")
            
            # 繪製偵測結果
            output_img = cv_img.copy()
            for detection in detections:
                class_id = detection.ClassID
                confidence = detection.Confidence
                left = int(detection.Left)
                top = int(detection.Top)
                right = int(detection.Right)
                bottom = int(detection.Bottom)
                
                # 繪製邊界框
                cv2.rectangle(output_img, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # 添加類別標籤和置信度
                class_name = net.GetClassDesc(class_id)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(output_img, label, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                logger.info(f"偵測到: {class_name}, 置信度: {confidence:.2f}")
            
            # 保存結果圖像
            detection_output = "test_detection_output.jpg"
            cv2.imwrite(detection_output, output_img)
            logger.info(f"偵測結果已保存至 {detection_output}")
            
        except Exception as e:
            self.fail(f"靜態圖像物體偵測測試失敗: {str(e)}")


class TestAWSConnectivity(unittest.TestCase):
    """測試 AWS 服務連接"""
    
    def setUp(self):
        """測試前的準備工作"""
        # 設置測試變量
        self.region = 'us-west-2'  # 使用您的區域
        self.bucket = 'test-bucket'  # 使用測試桶名
        self.test_image = "test_aws_upload.jpg"
        
        # 創建測試圖像
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (0, 255, 0)  # 綠色測試圖像
        cv2.imwrite(self.test_image, img)
    
    def tearDown(self):
        """測試後清理"""
        # 刪除測試圖像
        if os.path.exists(self.test_image):
            os.remove(self.test_image)
    
    def test_s3_connectivity(self):
        """測試 S3 連接和上傳功能"""
        try:
            # 創建 S3 客戶端
            s3_client = boto3.client('s3', region_name=self.region)
            
            # 列出存儲桶
            response = s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            logger.info(f"可用的 S3 存儲桶: {buckets}")
            
            # 檢查測試桶是否存在
            if self.bucket in buckets:
                # 嘗試上傳測試圖像
                test_key = f"test_images/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                s3_client.upload_file(self.test_image, self.bucket, test_key)
                logger.info(f"成功上傳測試圖像到 s3://{self.bucket}/{test_key}")
                
                # 驗證文件已上傳
                response = s3_client.list_objects_v2(Bucket=self.bucket, Prefix=test_key)
                self.assertIn('Contents', response, "上傳的文件應該可以被列出")
                
                # 刪除測試文件
                s3_client.delete_object(Bucket=self.bucket, Key=test_key)
                logger.info(f"已刪除測試文件 s3://{self.bucket}/{test_key}")
            else:
                logger.warning(f"測試桶 {self.bucket} 不存在，跳過上傳測試")
        
        except Exception as e:
            logger.error(f"S3 連接測試失敗: {str(e)}")
            self.fail(f"S3 連接測試失敗: {str(e)}")
    
    @patch('AWSIoTPythonSDK.MQTTLib.AWSIoTMQTTClient')
    def test_iot_connectivity(self, mock_mqtt_client):
        """測試 AWS IoT 連接功能 (使用模擬)"""
        # 設置模擬的返回值
        mock_instance = mock_mqtt_client.return_value
        mock_instance.connect.return_value = True
        
        try:
            # 創建 EdgeDevice 實例
            device = EdgeDevice()
            device.iot_client = mock_instance
            
            # 測試發布設備狀態
            result = device.publish_device_status()
            self.assertTrue(result, "發布設備狀態應該成功")
            
            # 驗證 connect 和 publish 方法被調用
            mock_instance.connect.assert_called_once()
            mock_instance.publish.assert_called_once()
            
            logger.info("IoT 連接和發布測試成功")
            
        except Exception as e:
            logger.error(f"IoT 連接測試失敗: {str(e)}")
            self.fail(f"IoT 連接測試失敗: {str(e)}")


class TestQueueingSystem(unittest.TestCase):
    """測試隊列系統功能"""
    
    def setUp(self):
        """測試前的準備工作"""
        self.device = EdgeDevice()
        # 模擬必要的組件
        self.device.s3_client = MagicMock()
        self.device.iot_client = MagicMock()
        self.device.net = MagicMock()
        
        # 創建測試圖像
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.test_frame, "Test Frame", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def test_capture_and_queue(self):
        """測試捕獲和隊列功能"""
        # 設置模擬的攝像頭
        self.device.cap = MagicMock()
        self.device.cap.read.return_value = (True, self.test_frame)
        
        # 測試手動捕獲
        result = self.device.capture_and_queue(frame=self.test_frame, manual=True)
        self.assertTrue(result, "手動捕獲應該成功")
        
        # 測試偵測捕獲
        result = self.device.capture_and_queue(frame=self.test_frame, detection_type="person")
        self.assertTrue(result, "偵測捕獲應該成功")
        
        # 驗證圖像和事件是否已加入隊列
        from edge_device import image_queue, event_queue
        self.assertGreater(image_queue.qsize(), 0, "圖像隊列應該有項目")
        self.assertGreater(event_queue.qsize(), 0, "事件隊列應該有項目")
        
        # 檢查生成的臨時文件
        temp_files = [f for f in os.listdir('.') if f.startswith('temp_')]
        self.assertGreater(len(temp_files), 0, "應該生成臨時文件")
        
        # 清理臨時文件
        for file in temp_files:
            os.remove(file)
    
    def test_batch_upload_worker(self):
        """測試批量上傳工作線程"""
        # 修改批量上傳設置以加速測試
        import edge_device
        original_batch_size = edge_device.BATCH_UPLOAD_SIZE
        original_interval = edge_device.BATCH_UPLOAD_INTERVAL
        edge_device.BATCH_UPLOAD_SIZE = 2
        edge_device.BATCH_UPLOAD_INTERVAL = 1
        
        try:
            # 準備測試數據
            test_local_file = "test_batch_upload.jpg"
            cv2.imwrite(test_local_file, self.test_frame)
            
            # 添加測試項目到隊列
            edge_device.image_queue.put((test_local_file, "test/key1.jpg"))
            edge_device.image_queue.put((test_local_file, "test/key2.jpg"))
            
            # 啟動批量上傳線程
            upload_thread = threading.Thread(target=self.device.batch_upload_worker)
            upload_thread.daemon = True
            upload_thread.start()
            
            # 等待上傳完成
            time.sleep(3)
            
            # 驗證上傳調用
            self.assertEqual(self.device.s3_client.upload_file.call_count, 2, 
                           "S3 上傳應該被調用兩次")
            
            # 檢查臨時文件是否被刪除
            self.assertFalse(os.path.exists(test_local_file), 
                           "上傳後臨時文件應該被刪除")
            
        finally:
            # 恢復原始設置
            edge_device.BATCH_UPLOAD_SIZE = original_batch_size
            edge_device.BATCH_UPLOAD_INTERVAL = original_interval


class TestEndToEnd(unittest.TestCase):
    """端到端集成測試"""
    
    def setUp(self):
        """測試前的準備工作"""
        # 這個測試需要完整的設置
        self.device = None
    
    def test_initialization(self):
        """測試設備初始化"""
        self.device = EdgeDevice()
        try:
            result = self.device.initialize()
            if result:
                logger.info("設備初始化成功")
                self.assertTrue(result)
                
                # 檢查組件是否正確初始化
                self.assertIsNotNone(self.device.net, "偵測網絡應該被初始化")
                self.assertIsNotNone(self.device.cap, "攝像頭應該被初始化")
                self.assertIsNotNone(self.device.s3_client, "S3 客戶端應該被初始化")
                
                # 停止設備
                self.device.stop()
            else:
                logger.warning("設備初始化失敗，跳過此測試")
                self.skipTest("設備初始化失敗")
        except Exception as e:
            logger.error(f"設備初始化測試失敗: {str(e)}")
            if self.device:
                self.device.stop()
            self.fail(f"設備初始化測試失敗: {str(e)}")
    
    def test_short_detection_run(self):
        """測試短時間的偵測運行"""
        self.device = EdgeDevice()
        try:
            if self.device.initialize():
                # 創建一個線程來運行偵測，這樣我們可以在短時間後停止它
                detection_thread = threading.Thread(target=self.device.start_detection)
                detection_thread.daemon = True
                detection_thread.start()
                
                # 讓偵測運行幾秒鐘
                logger.info("偵測正在運行，將在 5 秒後停止...")
                time.sleep(5)
                
                # 停止偵測
                self.device.is_running = False
                time.sleep(1)  # 給線程一些時間來停止
                
                logger.info("偵測已停止")
                self.assertFalse(self.device.is_running, "設備應該停止運行")
            else:
                logger.warning("設備初始化失敗，跳過偵測測試")
                self.skipTest("設備初始化失敗")
        except Exception as e:
            logger.error(f"偵測運行測試失敗: {str(e)}")
            if self.device:
                self.device.stop()
            self.fail(f"偵測運行測試失敗: {str(e)}")
        finally:
            # 確保設備正確停止
            if self.device:
                self.device.stop()


def main():
    """執行測試套件"""
    # 創建輸出目錄
    os.makedirs("test_results", exist_ok=True)
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='邊緣設備功能測試')
    parser.add_argument('--module', choices=['camera', 'detection', 'aws', 'queue', 'e2e', 'all'], 
                       default='all', help='要測試的模組')
    args = parser.parse_args()
    
    # 組織測試套件
    suite = unittest.TestSuite()
    
    if args.module == 'camera' or args.module == 'all':
        suite.addTest(unittest.makeSuite(TestCameraModule))
    
    if args.module == 'detection' or args.module == 'all':
        suite.addTest(unittest.makeSuite(TestObjectDetection))
    
    if args.module == 'aws' or args.module == 'all':
        suite.addTest(unittest.makeSuite(TestAWSConnectivity))
    
    if args.module == 'queue' or args.module == 'all':
        suite.addTest(unittest.makeSuite(TestQueueingSystem))
    
    if args.module == 'e2e' or args.module == 'all':
        suite.addTest(unittest.makeSuite(TestEndToEnd))
    
    # 執行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 報告結果
    print(f"\n測試結果摘要:")
    print(f"運行次數: {result.testsRun}")
    print(f"失敗次數: {len(result.failures)}")
    print(f"錯誤次數: {len(result.errors)}")
    
    # 返回退出碼
    sys.exit(len(result.failures) + len(result.errors))

if __name__ == "__main__":
    main()