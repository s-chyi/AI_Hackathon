# inference/model_manager.py

import jetson.inference
import logging
import os # 引入 os 模組用於路徑檢查

logger = logging.getLogger(__name__)

class ModelManager:
    """
    載入和管理邊緣端 AI 模型。
    """
    def __init__(self, model_settings: dict):
        """
        初始化模型管理器。
        Args:
            model_settings (dict): 模型相關設定，包含模型路徑、閾值等。
        """
        self.model_settings = model_settings
        self.models = {} # 字典存放載入的模型實例

    def load_model(self, model_type: str):
        """
        載入指定類型的模型。
        Args:
            model_type (str): 模型類型名稱 (如 "object_detection")。
        Returns:
            object: 載入的模型實例，如果設定中沒有該類型或載入失敗則為 None。
        """
        if model_type not in self.model_settings or not self.model_settings[model_type]:
            logger.warning(f"設定中沒有找到模型類型 '{model_type}' 的配置。")
            return None

        model_config = self.model_settings[model_type]
        threshold = model_config.get('threshold', 0.5) # 預設閾值 0.5

        net = None # 初始化模型實例為 None

        if model_type == "object_detection":
            # 修正：優先檢查是否指定了內建模型名稱
            built_in_name = model_config.get('built_in_model_name')
            if built_in_name:
                 try:
                     logger.info(f"載入內建物件偵測模型: '{built_in_name}', 閾值: {threshold}")
                     # 使用內建模型名稱作為第一個參數
                     net = jetson.inference.detectNet(built_in_name, threshold=threshold)
                     logger.info(f"內建物件偵測模型 '{built_in_name}' 載入成功。")
                 except Exception as e:
                    logger.error(f"載入內建物件偵測模型 '{built_in_name}' 時發生錯誤: {e}", exc_info=True)
                    return None # 載入失敗，返回 None

            else: # 如果沒有指定內建模型名稱，則嘗試載入自定義檔案
                 model_file_path = model_config.get('model_file')
                 labels_file_path = model_config.get('labels_file')
                 input_blob = model_config.get('input_blob')
                 output_cvg = model_config.get('output_cvg')
                 output_bbox = model_config.get('output_bbox')

                 if not model_file_path or not labels_file_path:
                      logger.error(f"模型類型 '{model_type}' 需要設定 'built_in_model_name' 或 'model_file' 和 'labels_file'。")
                      return None

                 # 可選：檢查檔案是否存在
                 if not os.path.exists(model_file_path):
                      logger.error(f"模型檔案 '{model_file_path}' 不存在。請檢查路徑。")
                      return None
                 if not os.path.exists(labels_file_path):
                      logger.error(f"標籤檔案 '{labels_file_path}' 不存在。請檢查路徑。")
                      return None

                 try:
                     logger.info(f"載入物件偵測模型檔案: {model_file_path}, 標籤檔案: {labels_file_path}, 閾值: {threshold}")

                     # 使用關鍵字參數 model= 和 labels= 載入自定義檔案
                     net = jetson.inference.detectNet(
                         model=model_file_path,
                         labels=labels_file_path,
                         threshold=threshold,
                         input_blob=input_blob if input_blob else None,
                         output_cvg=output_cvg if output_cvg else None,
                         output_bbox=output_bbox if output_bbox else None
                     )

                     logger.info(f"物件偵測模型檔案 '{model_file_path}' 載入成功。")
                 except Exception as e:
                     logger.error(f"載入物件偵測模型檔案 '{model_file_path}' 時發生錯誤: {e}", exc_info=True)
                     return None # 載入失敗，返回 None

            # 如果 net 成功載入 (無論是內建還是自定義)
            if net:
                self.models[model_type] = net
                # 確保 class_mapping 與模型標籤對應 (對於內建模型，GetClassDesc 可能有用)
                # 對於自定義模型，class_mapping 必須與 labels_file 內容對應
                # 如果使用內建模型，可能需要 GetClassDesc 來驗證 class_mapping 是否正確對應
                # 例如：logger.debug(f"Model has {net.Get ); # 可選，用於檢查模型輸出的類別數量
                return net

        # ... 可擴展載入其他類型的模型 ...
        else:
            logger.warning(f"未知模型類型 '{model_type}'。")
            return None

    def get_model(self, model_type: str):
        """
        獲取已載入的模型實例。如果模型尚未載入，則嘗試載入。
        Args:
            model_type (str): 模型類型名稱。
        Returns:
            object: 模型實例，如果找不到或載入失敗則為 None。
        """
        if model_type not in self.models:
            logger.info(f"模型類型 '{model_type}' 尚未載入，嘗試載入...")
            return self.load_model(model_type)
        return self.models[model_type]

    def unload_all_models(self):
        """
        卸載所有已載入的模型 (如果底層庫支持)。
        """
        logger.info("卸載所有模型...")
        # jetson.inference 目前沒有明確的 unload 方法，這裡留空或根據需要處理
        self.models = {}
        logger.info("所有模型已卸載。")