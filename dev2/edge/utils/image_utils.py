# utils/image_utils.py

import cv2
import numpy as np

def resize_for_display(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """
    根據最大寬高限制調整影像大小，用於本地顯示。
    Args:
        image (np.ndarray): 原始 OpenCV 影像。
        max_width (int): 最大顯示寬度。
        max_height (int): 最大顯示高度。
    Returns:
        np.ndarray: 調整大小後的影像。
    """
    h, w = image.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def draw_detections(image: np.ndarray, detections: list, class_mapping: dict) -> np.ndarray:
    """
    在影像上繪製偵測結果的邊框和標籤。
    Args:
        image (np.ndarray): 原始 OpenCV 影像。
        detections (list): jetson.inference.detectNet 輸出的偵測結果列表。
        class_mapping (dict): 模型類別 ID 到內部類別名稱的映射。
    Returns:
        np.ndarray: 繪製後的影像。
    """
    output_image = image.copy() # 避免修改原始影像
    for det in detections:
        class_id = det.ClassID
        confidence = det.Confidence
        left, top, right, bottom = int(det.Left), int(det.Top), int(det.Right), int(det.Bottom)

        # 根據信心度和類別繪製不同的顏色或標籤
        color = (0, 255, 0) # 綠色
        label = class_mapping.get(class_id, f"Class {class_id}") # 使用內部類別名或原始 ID
        label += f": {confidence:.2f}"

        cv2.rectangle(output_image, (left, top), (right, bottom), color, 2)
        cv2.putText(output_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output_image

# 可擴展添加其他繪圖函數，如繪製 ROI
# def draw_roi(image: np.ndarray, roi: list, color=(255, 0, 0), thickness=2) -> np.ndarray:
#     """
#     在影像上繪製感興趣區域 (ROI)。
#     Args:
#         image (np.ndarray): 原始 OpenCV 影像。
#         roi (list): ROI 座標 [x1, y1, x2, y2]。
#         color (tuple): 繪製顏色 (BGR)。
#         thickness (int): 線條粗細。
#     Returns:
#         np.ndarray: 繪製後的影像。
#     """
#     output_image = image.copy()
#     cv2.rectangle(output_image, (roi[0], roi[1]), (roi[2], roi[3]), color, thickness)
#     return output_image