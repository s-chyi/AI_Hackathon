# inference/face_db.py

import json
import logging
import numpy as np
import os
import boto3 # 引入 boto3
from botocore.exceptions import NoCredentialsError, ClientError, ParamValidationError # 引入 S3 相關異常
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class KnownFacesDB:
    """
    管理已知人臉的特徵向量數據庫，從 S3 載入並支持本地緩存。
    """
    def __init__(self, settings: dict, aws_settings: dict):
        """
        初始化已知人臉數據庫。
        Args:
            settings (dict): 已知人臉數據庫相關設定 (config.known_faces_db)。
            aws_settings (dict): AWS 相關設定，包含 region, credentials 等 (用於 S3 訪問)。
        """
        self.settings = settings
        self.aws_settings = aws_settings
        self.s3_embeddings_url = self.settings.get('s3_embeddings_url')
        self.local_cache_path = self.settings.get('local_cache_path')
        self._known_embeddings: Dict[str, np.ndarray] = {} # {person_id: embedding_np}
        self._s3_client = self._create_s3_client() # 創建 S3 客戶端

        if not self.s3_embeddings_url:
             logger.error("settings.yaml 中未設定 known_faces_db.s3_embeddings_url。")
             # 這裡不終止程式，但數據庫將為空

        # 嘗試從 S3 載入數據庫
        self.reload_embeddings_from_s3()

        # 如果 S3 載入失敗且存在本地緩存，則從本地緩存載入
        if not self._known_embeddings and self.local_cache_path and os.path.exists(self.local_cache_path):
             logger.warning("從 S3 載入失敗，嘗試從本地緩存載入已知人臉數據庫。")
             self._load_embeddings_from_local(self.local_cache_path)


        logger.info(f"已知人臉數據庫初始化完成，載入 {len(self._known_embeddings)} 個 embedding。")


    def _create_s3_client(self):
        """
        建立 Boto3 S3 客戶端實例。使用與 S3Uploader 類似的邏輯獲取憑證。
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
            logger.error(f"建立 S3 客戶端時發生錯誤: {e}", exc_info=True)
            return None


    def _parse_s3_url(self, s3_url: str) -> Optional[Tuple[str, str]]:
        """
        解析 S3 URL (e.g., s3://bucket-name/key/to/file.json) 為 bucket 名稱和 key。
        Returns:
            Optional[Tuple[str, str]]: (bucket_name, s3_key) 元組，如果格式無效則為 None。
        """
        if not s3_url or not s3_url.startswith("s3://"):
            return None
        parts = s3_url[5:].split('/', 1)
        if len(parts) != 2:
            # 如果只有 bucket 名稱而沒有 key，也是無效的檔案路徑
            return None
        return parts[0], parts[1]


    def reload_embeddings_from_s3(self):
        """
        從 S3 下載已知人臉數據庫檔案並載入。
        如果成功，也將下載的檔案保存到本地作為緩存。
        """
        if not self._s3_client:
            logger.warning("S3 客戶端未初始化，無法從 S3 載入已知人臉數據庫。")
            return

        if not self.s3_embeddings_url:
             logger.warning("未設定 known_faces_db.s3_embeddings_url，無法從 S3 載入。")
             return

        s3_info = self._parse_s3_url(self.s3_embeddings_url)
        if not s3_info:
             logger.error(f"無效的 S3 URL 格式: {self.s3_embeddings_url}")
             return

        bucket_name, s3_key = s3_info
        logger.info(f"嘗試從 S3 載入已知人臉數據庫: s3://{bucket_name}/{s3_key}")

        try:
            # 下載檔案到一個 BytesIO 對象或臨時檔案
            # 使用臨時檔案更簡單，特別是對於可能比較大的檔案
            temp_file_path = self.local_cache_path + ".tmp" if self.local_cache_path else f"/tmp/known_faces_db_{int(time.time())}.tmp" # 確保路徑可用寫入
            
            # 確保本地緩存目錄存在 (如果設定了 local_cache_path)
            if self.local_cache_path:
                local_dir = os.path.dirname(self.local_cache_path)
                if local_dir and not os.path.exists(local_dir):
                    os.makedirs(local_dir, exist_ok=True)
                    logger.info(f"創建本地緩存目錄: {local_dir}")


            self._s3_client.download_file(bucket_name, s3_key, temp_file_path)
            logger.info(f"成功從 S3 下載檔案到臨時路徑: {temp_file_path}")

            # 從下載的臨時檔案載入數據庫
            self._load_embeddings_from_local(temp_file_path)

            # 如果載入成功且設定了本地緩存路徑，將臨時檔案移動到緩存路徑
            if self._known_embeddings and self.local_cache_path and temp_file_path != self.local_cache_path:
                 try:
                     os.replace(temp_file_path, self.local_cache_path) # 安全地替換舊的緩存檔案
                     logger.info(f"已將新的數據庫保存到本地緩存: {self.local_cache_path}")
                 except Exception as e:
                     logger.warning(f"保存本地緩存檔案失敗: {e}", exc_info=True)

        except FileNotFoundError:
            logger.error(f"S3 中的檔案不存在: s3://{bucket_name}/{s3_key}")
            self._known_embeddings = {} # 檔案不存在，數據庫為空
        except NoCredentialsError:
            logger.error("AWS 憑證找不到，無法從 S3 下載已知人臉數據庫。")
            self._known_embeddings = {}
        except ClientError as e:
            logger.error(f"從 S3 下載檔案時發生 ClientError: {e}. Bucket: {bucket_name}, Key: {s3_key}", exc_info=True)
            self._known_embeddings = {}
        except ParamValidationError as e:
             logger.error(f"S3 客戶端參數驗證失敗: {e}", exc_info=True)
             self._known_embeddings = {}
        except Exception as e:
            logger.error(f"從 S3 下載或處理已知人臉數據庫時發生意外錯誤: {e}", exc_info=True)
            self._known_embeddings = {}
        finally:
             # 清理臨時檔案 (如果存在)
             if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                 try:
                     os.remove(temp_file_path)
                     logger.debug(f"已清理臨時檔案: {temp_file_path}")
                 except Exception as e:
                      logger.warning(f"清理臨時檔案失敗: {e}", exc_info=True)


    def _load_embeddings_from_local(self, file_path: str):
        """
        從本地檔案載入已知人臉特徵向量。
        Args:
            file_path (str): 本地檔案路徑。
        """
        if not os.path.exists(file_path):
            logger.warning(f"本地檔案 '{file_path}' 未找到，無法載入本地緩存。")
            self._known_embeddings = {} # 確保狀態正確
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                new_embeddings: Dict[str, np.ndarray] = {}
                for person_id, embedding_list in data.items():
                    # 將列表轉換回 NumPy 陣列
                    new_embeddings[str(person_id)] = np.asarray(embedding_list, dtype=np.float32) # 確保數據類型
                self._known_embeddings = new_embeddings # 更新數據庫
                logger.info(f"從本地檔案 '{file_path}' 載入 {len(self._known_embeddings)} 個已知人臉 embedding。")
        except Exception as e:
            logger.error(f"載入本地檔案 '{file_path}' 時發生錯誤: {e}", exc_info=True)
            self._known_embeddings = {} # 載入失敗則清空數據庫

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        獲取所有已知人臉的 embedding。
        Returns:
            Dict[str, np.ndarray]: 已知人臉 embedding 字典。
        """
        # 返回數據庫的拷貝，避免外部修改內部狀態
        return { pid: emb.copy() for pid, emb in self._known_embeddings.items() }

    # 不需要 add_embedding 和 save_embeddings 方法，因為更新應該由雲端進行並推送到 S3