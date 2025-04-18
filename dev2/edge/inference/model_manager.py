# inference/model_manager.py

import jetson.inference
import logging
import os # 引入 os 模組用於路徑檢查
from .face_models import FACE_DETECTION_MODEL, FACE_EMBEDDING_MODEL

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
                    net = jetson.inference.detectNet(built_in_name, threshold=threshold)
                    logger.info(f"內建物件偵測模型 '{built_in_name}' 載入成功。")
                except Exception as e:
                    logger.error(f"載入內建物件偵測模型 '{built_in_name}' 時發生錯誤: {e}", exc_info=True)
                return None
            else:
                # 載入自定義物件偵測模型的邏輯
                model_file_path = model_config.get('model_file')
                labels_file_path = model_config.get('labels_file')
                input_blob = model_config.get('input_blob')
                output_cvg = model_config.get('output_cvg')
                output_bbox = model_config.get('output_bbox')

                if not model_file_path or not labels_file_path:
                    logger.error(f"模型類型 '{model_type}' 需要設定 'built_in_model_name' 或 'model_file' 和 'labels_file'。")
                    return None

                if not os.path.exists(model_file_path) or not os.path.exists(labels_file_path):
                    logger.error(f"模型或標籤檔案不存在 ('{model_file_path}', '{labels_file_path}')。")
                    return None

                try:
                    logger.info(f"載入物件偵測模型檔案: {model_file_path}, 標籤檔案: {labels_file_path}, 閾值: {threshold}")
                    net = jetson.inference.detectNet(
                        model=model_file_path, labels=labels_file_path, threshold=threshold,
                        input_blob=input_blob, output_cvg=output_cvg, output_bbox=output_bbox
                    )
                    logger.info(f"物件偵測模型檔案 '{model_file_path}' 載入成功。")
                except Exception as e:
                    logger.error(f"載入物件偵測模型檔案 '{model_file_path}' 時發生錯誤: {e}", exc_info=True)
                    return None
        # 載入人臉偵測模型
        elif model_type == FACE_DETECTION_MODEL:
            built_in_name = model_config.get('built_in_model_name')
            if not built_in_name:
                logger.error(f"模型類型 '{model_type}' 需要設定 'built_in_model_name'。")
                return None
            try:
                logger.info(f"載入人臉偵測模型: '{built_in_name}', 閾值: {threshold}")
                net = jetson.inference.detectNet(built_in_name, threshold=threshold) # 人臉偵測也使用 detectNet
                logger.info(f"人臉偵測模型 '{built_in_name}' 載入成功。")
            except Exception as e:
                    logger.error(f"載入人臉偵測模型 '{built_in_name}' 時發生錯誤: {e}", exc_info=True)
                    return None


        # 載入人臉特徵提取模型
        elif model_type == FACE_EMBEDDING_MODEL:
            built_in_name = model_config.get('built_in_model_name')
            if not built_in_name:
                    logger.error(f"模型類型 '{model_type}' 需要設定 'built_in_model_name'。")
                    return None
            try:
                logger.info(f"載入人臉特徵提取模型: '{built_in_name}'")
                # Jetson-inference 的 resnet18-facenet 模型通常載入到 poseNet 中
                # 如果您的模型載入方式不同，請調整這裡
                net = jetson.inference.poseNet(built_in_name) # 人臉 embedding 使用 poseNet
                logger.info(f"人臉特徵提取模型 '{built_in_name}' 載入成功。")
            except Exception as e:
                    logger.error(f"載入人臉特徵提取模型 '{built_in_name}' 時發生錯誤: {e}", exc_info=True)
                    return None

        else:
            logger.warning(f"未知模型類型 '{model_type}'。")
            return None

        # 如果 net 成功載入 (無論是內建還是自定義)
        if net:
            self.models[model_type] = net
            return net
        else:
            return None # 模型載入失敗

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