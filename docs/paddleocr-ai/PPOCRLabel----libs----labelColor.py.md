# `.\PaddleOCR\PPOCRLabel\libs\labelColor.py`

```
import PIL.Image
import numpy as np


def rgb2hsv(rgb):
    # type: (np.ndarray) -> np.ndarray
    """Convert rgb to hsv.

    Parameters
    ----------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Input rgb image.

    Returns
    -------
    hsv: numpy.ndarray, (H, W, 3), np.uint8
        Output hsv image.

    """
    # 将 RGB 图像转换为 PIL.Image 对象
    hsv = PIL.Image.fromarray(rgb, mode="RGB")
    # 将 RGB 图像转换为 HSV 模式
    hsv = hsv.convert("HSV")
    # 将 HSV 图像转换为 numpy 数组
    hsv = np.array(hsv)
    return hsv


def hsv2rgb(hsv):
    # type: (np.ndarray) -> np.ndarray
    """Convert hsv to rgb.

    Parameters
    ----------
    hsv: numpy.ndarray, (H, W, 3), np.uint8
        Input hsv image.

    Returns
    -------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Output rgb image.

    """
    # 将 HSV 图像转换为 PIL.Image 对象
    rgb = PIL.Image.fromarray(hsv, mode="HSV")
    # 将 HSV 图像转换为 RGB 模式
    rgb = rgb.convert("RGB")
    # 将 RGB 图像转换为 numpy 数组
    rgb = np.array(rgb)
    return rgb


def label_colormap(n_label=256, value=None):
    """Label colormap.

    Parameters
    ----------
    n_label: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.

    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    # 创建一个形状为 (n_label, 3) 的零数组，用于存储颜色映射
    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            # 从 id 中获取每个颜色通道的值
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        # 将计算得到的 RGB 值存储到颜色映射数组中
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    # 如果值不为空
    if value is not None:
        # 将颜色映射数组重塑为一行，转换为 HSV 颜色空间
        hsv = rgb2hsv(cmap.reshape(1, -1, 3))
        # 如果值是浮点数
        if isinstance(value, float):
            # 将 HSV 颜色空间中饱和度和亮度的值乘以给定的浮点数
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            # 如果值是整数，断言值为整数类型
            assert isinstance(value, int)
            # 将 HSV 颜色空间中饱和度和亮度的值设置为给定的整数值
            hsv[:, 1:, 2] = value
        # 将 HSV 颜色空间转换回 RGB 颜色空间，并重塑为原始形状
        cmap = hsv2rgb(hsv).reshape(-1, 3)
    # 返回处理后的颜色映射数组
    return cmap
```