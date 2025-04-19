# utils/qr_scanner.py

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

# 嘗試導入 pyzbar，這是常見的 QR Code 掃描庫
try:
    from pyzbar import pyzbar
except ImportError:
    pyzbar = None
    logging.warning("pyzbar 庫未安裝。QR Code 掃描功能將不可用。請執行 'pip install pyzbar opencv-python'。")
    # 注意：pyzbar 可能需要安裝額外的依賴，例如 libzbar0

logger = logging.getLogger(__name__)


def scan_qr_code(image_np: np.ndarray) -> Optional[str]:
    """
    在影像中掃描並讀取 QR Code。
    Args:
        image_np (np.ndarray): OpenCV 格式的影像。
    Returns:
        Optional[str]: 讀取到的 QR Code 數據字符串，如果未偵測到或讀取失敗則為 None。
    """
    if pyzbar is None:
        logger.error("pyzbar 庫未安裝，無法執行 QR Code 掃描。")
        return None

    if image_np is None or image_np.size == 0:
         logger.warning("輸入影像無效，無法掃描 QR Code。")
         return None

    # 將影像轉換為灰階，有利於 QR Code 偵測
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    try:
        # 偵測並解碼影像中的 QR Code
        barcodes = pyzbar.decode(gray_image)

        # 這裡假設影像中只有一個與我們相關的 QR Code，或者我們只處理第一個
        if barcodes:
            # 解碼成功，返回數據
            qr_data = barcodes[0].data.decode('utf-8')
            logger.info(f"成功掃描到 QR Code: {qr_data}")
            # 可選：繪製偵測到的 QR 框 (用於調試或保存影像)
            # for barcode in barcodes:
            #      (x, y, w, h) = barcode.rect
            #      cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #      logger.debug(f"QR Code 框: ({x}, {y}, {w}, {h})")
            # 返回讀取到的數據
            return qr_data
        else:
            # 未偵測到 QR Code
            # logger.debug("未偵測到 QR Code。") # 避免頻繁日誌
            return None

    except Exception as e:
        logger.error(f"掃描 QR Code 時發生錯誤: {e}", exc_info=True)
        return None

# 可選：添加一個函數來掃描特定 ROI 中的 QR Code
# def scan_qr_code_in_roi(image_np: np.ndarray, roi: List[int]) -> Optional[str]:
#     """
#     在指定的 ROI 區域中掃描 QR Code。
#     Args:
#         image_np (np.ndarray): 原始 OpenCV 影像。
#         roi (List[int]): 感興趣區域 [x1, y1, x2, y2]。
#     Returns:
#         Optional[str]: 讀取到的 QR Code 數據，或 None。
#     """
#     if image_np is None or image_np.size == 0 or not roi:
#          return None
#     try:
#          # 裁剪 ROI 區域
#          roi_image = image_np[roi[1]:roi[3], roi[0]:roi[2]]
#          return scan_qr_code(roi_image)
#     except Exception as e:
#          logger.error(f"在 ROI 中掃描 QR Code 時發生錯誤: {e}", exc_info=True)
#          return None