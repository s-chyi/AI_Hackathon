# utils/s3_uploader.py

import threading
import queue
import boto3
import logging
import os
from botocore.exceptions import NoCredentialsError, ClientError

# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class S3Uploader(threading.Thread):
    """
    使用獨立執行緒處理 S3 上傳的類別。
    接收佇列中的上傳任務，並異步執行。
    """
    def __init__(self, aws_settings: dict, upload_queue: queue.Queue):
        """
        初始化 S3 上傳器。
        Args:
            aws_settings (dict): AWS 相關設定，包含 region, s3 config 等。
            upload_queue (queue.Queue): 儲存待上傳任務的佇列 (tuple: (bytes, s3_key))。
        """
        super().__init__(daemon=True) # 設定為 daemon 執行緒，主程式結束時會自動終止
        self.aws_settings = aws_settings
        self.upload_queue = upload_queue
        self.s3_client = self._create_s3_client()
        self._stop_event = threading.Event() # 用於安全停止執行緒

    def _create_s3_client(self):
        """
        建立 Boto3 S3 客戶端實例。
        可以從設定檔、環境變數或 IAM Role 獲取憑證。
        """
        try:
            # 優先使用設定檔中的 Access Key/Secret Key (如果提供)
            if 'access_key_id' in self.aws_settings and 'secret_access_key' in self.aws_settings:
                return boto3.client('s3',
                    region_name=self.aws_settings.get('region'),
                    aws_access_key_id=self.aws_settings['access_key_id'],
                    aws_secret_access_key=self.aws_settings['secret_access_key']
                )
            # 其次使用 profile (如果提供)
            elif 'profile_name' in self.aws_settings:
                 return boto3.client('s3',
                    region_name=self.aws_settings.get('region'),
                    profile_name=self.aws_settings['profile_name']
                )
            # 否則依賴環境變數或 EC2/ECS 的 IAM Role (Jetson 上較可能使用環境變數或 profile)
            else:
                 return boto3.client('s3',
                    region_name=self.aws_settings.get('region')
                )
        except NoCredentialsError:
            logger.error("AWS 憑證找不到，無法建立 S3 客戶端。請檢查設定檔或環境變數。")
            return None
        except Exception as e:
            logger.error(f"建立 S3 客戶端時發生錯誤: {e}")
            return None

    def run(self):
        """
        執行緒的主體，不斷從佇列中獲取任務並上傳。
        """
        if not self.s3_client:
            logger.error("S3 客戶端初始化失敗，上傳執行緒終止。")
            return

        logger.info("S3 上傳執行緒啟動...")
        while not self._stop_event.is_set():
            try:
                # 設置 timeout 避免在佇列為空時永久阻塞，以便檢查停止事件
                task = self.upload_queue.get(timeout=1.0)
            except queue.Empty:
                continue # 佇列為空，繼續循環檢查停止事件

            if task is None:
                logger.info("收到停止任務，S3 上傳執行緒準備結束。")
                break # 收到 None 任務，表示停止

            try:
                image_data, s3_key = task
                bucket_name = self.aws_settings['s3']['bucket_name']
                logger.info(f"開始上傳: s3://{bucket_name}/{s3_key} ({len(image_data)} bytes)")

                # 使用 put_object 進行上傳
                self.s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=image_data)
                logger.info(f"成功上傳: s3://{bucket_name}/{s3_key}")

            except ClientError as e:
                 logger.error(f"S3 上傳失敗 (ClientError): {e}. Key: {s3_key}", exc_info=True)
                 # 可以在這裡添加重試邏輯或將任務放回佇列的機制 (根據需求複雜度)
            except Exception as e:
                logger.error(f"S3 上傳失敗 (其他錯誤): {e}. Key: {s3_key}", exc_info=True)
            finally:
                self.upload_queue.task_done() # 通知佇列任務已完成

        logger.info("S3 上傳執行緒已終止。")

    def stop(self):
        """
        請求停止上傳執行緒。
        """
        logger.info("請求停止 S3 上傳執行緒...")
        self._stop_event.set() # 設定停止事件
        # 將 None 任務放入佇列，喚醒正在等待的 get()
        try:
             self.upload_queue.put_nowait(None)
        except queue.Full:
             pass # 如果佇列滿了，就無法放入 None，等待 timeout 結束

    def put_upload_task(self, image_data: bytes, s3_key: str):
        """
        將一個上傳任務添加到佇列。
        Args:
            image_data (bytes): 圖片的二進位數據。
            s3_key (str): 上傳到 S3 的目標 Key (檔案路徑)。
        """
        try:
            self.upload_queue.put_nowait((image_data, s3_key)) # 非阻塞地放入佇列
            logger.debug(f"已將任務添加到 S3 上傳佇列: {s3_key}")
        except queue.Full:
            logger.warning(f"S3 上傳佇列已滿，丟棄任務: {s3_key}")
            # 佇列滿了可以選擇丟棄任務或阻塞等待，這裡選擇丟棄以保持主迴圈響應

    def wait_for_completion(self):
        """
        等待佇列中的所有任務完成。
        """
        logger.info("等待 S3 上傳佇列清空...")
        self.upload_queue.join()
        logger.info("S3 上傳佇列已清空。")