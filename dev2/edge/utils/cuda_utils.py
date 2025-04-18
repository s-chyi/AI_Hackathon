# utils/cuda_utils.py

import jetson.utils
import numpy as np

def numpy_to_cuda(image_np: np.ndarray) -> jetson.utils.cudaImage:
    """
    將 OpenCV (NumPy) 影像轉換為 Jetson.utils CUDA 影像格式。
    Args:
        image_np (np.ndarray): OpenCV 格式的影像 (HWC, BGR)。
    Returns:
        jetson.utils.cudaImage: CUDA 影像格式。
    """
    # OpenCV 預設是 BGR 格式，jetson.utils.cudaFromNumpy 需要 RGB
    # 注意：使用 cv2.COLOR_BGR2RGB 轉換會建立一份拷貝
    # 如果性能極限，可以考慮在原始 BGR CUDA 表面操作，但較複雜
    rgb_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return jetson.utils.cudaFromNumpy(rgb_image_np)

# 如果需要將 CUDA 影像轉回 NumPy (例如本地顯示前)，可以使用：
# def cuda_to_numpy(cuda_img: jetson.utils.cudaImage) -> np.ndarray:
#     """
#     將 Jetson.utils CUDA 影像轉換回 OpenCV (NumPy) 影像格式。
#     Args:
#         cuda_img (jetson.utils.cudaImage): CUDA 影像格式。
#     Returns:
#         np.ndarray: OpenCV 格式的影像 (HWC, BGR)。
#     """
#     numpy_img = jetson.utils.cudaToNumpy(cuda_img)
#     # 從 RGB 轉回 BGR
#     return cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)