# inference/model_manager.py

import jetson.inference
import logging

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
        model_path = model_config.get('model_path')
        threshold = model_config.get('threshold', 0.5) # 預設閾值 0.5

        if not model_path:
            logger.error(f"模型類型 '{model_type}' 的 model_path 未設定。")
            return None

        if model_type == "object_detection":
            try:
                # jetson.inference.detectNet 可以直接載入一些預訓練模型名稱，或指定檔案路徑
                logger.info(f"載入物件偵測模型: {model_path}, 閾值: {threshold}")
                net = jetson.inference.detectNet(model_path, threshold=threshold)
                self.models[model_type] = net
                logger.info(f"物件偵測模型 '{model_path}' 載入成功。")
                return net
            except Exception as e:
                logger.error(f"載入物件偵測模型 '{model_path}' 時發生錯誤: {e}", exc_info=True)
                return None
        # 可擴展載入其他類型的模型，如分類模型、姿勢估計模型等
        # elif model_type == "classification":
        #     pass # 載入分類模型的邏輯
        # elif model_type == "pose_estimation":
        #     pass # 載入姿勢估計模型的邏輯
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