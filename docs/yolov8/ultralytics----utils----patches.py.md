# `.\yolov8\ultralytics\utils\patches.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Monkey patches to update/extend functionality of existing functions."""

import time
from pathlib import Path

import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # 将cv2.imshow赋值给_imshow变量，避免递归错误


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    """
    Read an image from a file.

    Args:
        filename (str): Path to the file to read.
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Defaults to cv2.IMREAD_COLOR.

    Returns:
        (np.ndarray): The read image.
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)  # 使用cv2.imdecode函数读取文件并返回图像数据


def imwrite(filename: str, img: np.ndarray, params=None):
    """
    Write an image to a file.

    Args:
        filename (str): Path to the file to write.
        img (np.ndarray): Image to write.
        params (list of ints, optional): Additional parameters. See OpenCV documentation.

    Returns:
        (bool): True if the file was written, False otherwise.
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)  # 使用cv2.imencode将图像编码并写入文件
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray):
    """
    Displays an image in the specified window.

    Args:
        winname (str): Name of the window.
        mat (np.ndarray): Image to be shown.
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)  # 使用_imshow显示指定名称的窗口中的图像


# PyTorch functions ----------------------------------------------------------------------------------------------------
_torch_load = torch.load  # 将torch.load赋值给_torch_load变量，避免递归错误
_torch_save = torch.save


def torch_load(*args, **kwargs):
    """
    Load a PyTorch model with updated arguments to avoid warnings.

    This function wraps torch.load and adds the 'weights_only' argument for PyTorch 1.13.0+ to prevent warnings.

    Args:
        *args (Any): Variable length argument list to pass to torch.load.
        **kwargs (Any): Arbitrary keyword arguments to pass to torch.load.

    Returns:
        (Any): The loaded PyTorch object.

    Note:
        For PyTorch versions 2.0 and above, this function automatically sets 'weights_only=False'
        if the argument is not provided, to avoid deprecation warnings.
    """
    from ultralytics.utils.torch_utils import TORCH_1_13  # 导入TORCH_1_13变量，用于检测PyTorch版本

    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False  # 如果使用的是PyTorch 1.13及以上版本且没有指定'weights_only'参数，则设置为False

    return _torch_load(*args, **kwargs)  # 调用torch.load加载模型


def torch_save(*args, use_dill=True, **kwargs):
    """
    Optionally use dill to serialize lambda functions where pickle does not, adding robustness with 3 retries and
    exponential standoff in case of save failure.

    ```py
    # 此处代码块是省略部分，不需要注释
    ```
    """
    pass  # torch_save函数暂时没有实现内容，直接返回
    """
    Args:
        *args (tuple): Positional arguments to pass to torch.save.
        use_dill (bool): Whether to try using dill for serialization if available. Defaults to True.
        **kwargs (Any): Keyword arguments to pass to torch.save.
    """
    # 尝试使用 dill 序列化库（如果可用），否则使用 pickle
    try:
        assert use_dill
        import dill as pickle
    except (AssertionError, ImportError):
        import pickle

    # 如果 kwargs 中没有指定 pickle_module，则默认使用 pickle 库
    if "pickle_module" not in kwargs:
        kwargs["pickle_module"] = pickle

    # 最多尝试保存 4 次（包括初始尝试），以处理可能的运行时错误
    for i in range(4):  # 3 retries
        try:
            # 调用 _torch_save 函数尝试保存数据
            return _torch_save(*args, **kwargs)
        except RuntimeError as e:  # unable to save, possibly waiting for device to flush or antivirus scan
            # 如果是最后一次尝试保存，则抛出原始的 RuntimeError
            if i == 3:
                raise e
            # 等待指数增长的时间，用于避免设备刷新或者反病毒扫描等问题
            time.sleep((2**i) / 2)  # exponential standoff: 0.5s, 1.0s, 2.0s
```