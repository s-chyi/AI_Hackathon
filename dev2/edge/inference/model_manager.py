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
        # 確保模型類型是我們期望的字符串
        model_type_str = model_type if isinstance(model_type, str) else str(model_type)

        if model_type_str not in self.model_settings or not self.model_settings[model_type_str]:
            logger.warning(f"設定中沒有找到模型類型 '{model_type_str}' 的配置。")
            return None

        model_config = self.model_settings[model_type_str]
        threshold = model_config.get('threshold')

        net = None

        # 載入物件偵測模型 (只保留物件偵測)
        if model_type_str == "object_detection":
             built_in_name = model_config.get('built_in_model_name')
             if built_in_name:
                 try:
                     logger.info(f"載入內建物件偵測模型: '{built_in_name}', 閾值: {threshold}")
                     net = jetson.inference.detectNet(built_in_name, threshold=threshold)
                     logger.info(f"內建物件偵測模型 '{built_in_name}' 載入成功。")
                     self.models[model_type_str] = net
                     return net
                 except Exception as e:
                    logger.error(f"載入內建物件偵測模型 '{built_in_name}' 時發生錯誤: {e}", exc_info=True)
                    return None
             else:
                 model_file_path = model_config.get('model_file')
                 labels_file_path = model_config.get('labels_file')
                 input_blob = model_config.get('input_blob')
                 output_cvg = model_config.get('output_cvg')
                 output_bbox = model_config.get('output_bbox')

                 if not model_file_path or not labels_file_path:
                      logger.error(f"模型類型 '{model_type_str}' 需要設定 'built_in_model_name' 或 'model_file' 和 'labels_file'。")
                      return None

                 if not os.path.exists(model_file_path) or not os.path.exists(labels_file_path):
                     logger.error(f"模型或標籤檔案不存在 ('{model_file_path}', '{labels_file_path}')。")
                     return None

                 try:
                     logger.info(f"載入物件偵測模型檔案: {model_file_path}, 標籤檔案: {labels_file_path}, 閾值: {threshold}")
                     net = jetson_inference.detectNet(
                         model=model_file_path, labels=labels_file_path, threshold=threshold,
                         input_blob=input_blob, output_cvg=output_cvg, output_bbox=output_bbox
                     )
                     logger.info(f"物件偵測模型檔案 '{model_file_path}' 載入成功。")
                     self.models[model_type_str] = net
                     return net
                 except Exception as e:
                     logger.error(f"載入物件偵測模型檔案 '{model_file_path}' 時發生錯誤: {e}", exc_info=True)
                     return None


        # 如果模型類型不是 object_detection，則報錯或警告
        else:
            logger.warning(f"嘗試載入未知或不受支持的模型類型 '{model_type_str}'。")
            return None

    def get_model(self, model_type: str):
        """
        獲取已載入的模型實例。如果模型尚未載入，則嘗試載入。
        Args:
            model_type (str): 模型類型名稱 (如 "object_detection")。
        Returns:
            object: 模型實例，如果找不到或載入失敗則為 None。
        """
        model_type_str = model_type if isinstance(model_type, str) else str(model_type)

        if model_type_str not in self.models:
            logger.info(f"模型類型 '{model_type_str}' 尚未載入，嘗試載入...")
            # 確保只對 object_detection 進行載入
            if model_type_str == "object_detection":
                return self.load_model(model_type_str)
            else:
                 logger.warning(f"嘗試獲取非物件偵測模型 '{model_type_str}'，但不再通過 ModelManager 載入。")
                 return None


        return self.models.get(model_type_str)


    def unload_all_models(self):
        """
        卸載所有已載入的模型 (現在只用於物件偵測)。
        """
        logger.info("卸載所有模型。")
        self.models = {}
        logger.info("所有模型已卸載 (內部狀態已清除)。")