# `.\diffusers\utils\pil_utils.py`

```py
# 从类型提示模块导入 List
from typing import List

# 导入 PIL.Image 模块用于图像处理
import PIL.Image
# 导入 PIL.ImageOps 模块提供图像操作功能
import PIL.ImageOps
# 从 packaging 模块导入 version 用于版本比较
from packaging import version
# 从 PIL 导入 Image 类用于图像处理
from PIL import Image


# 检查 PIL 的版本是否大于或等于 9.1.0
if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    # 定义图像插值方式的字典，使用新版的 Resampling 属性
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    # 定义图像插值方式的字典，使用旧版的插值常量
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


# 将 torch 图像转换为 PIL 图像的函数
def pt_to_pil(images):
    """
    Convert a torch image to a PIL image.
    """
    # 将图像标准化到 [0, 1] 范围内
    images = (images / 2 + 0.5).clamp(0, 1)
    # 将图像转移到 CPU，并调整维度为 (batch, height, width, channels)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    # 调用 numpy_to_pil 函数将 numpy 图像转换为 PIL 图像
    images = numpy_to_pil(images)
    # 返回 PIL 图像
    return images


# 将 numpy 图像或图像批量转换为 PIL 图像的函数
def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    # 如果图像是三维的，增加一个维度以表示批量
    if images.ndim == 3:
        images = images[None, ...]
    # 将图像值转换为 0-255 的整数，并取整
    images = (images * 255).round().astype("uint8")
    # 检查是否为单通道图像（灰度图像）
    if images.shape[-1] == 1:
        # 对于灰度图像，使用模式 "L" 创建 PIL 图像
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        # 对于彩色图像，直接从数组创建 PIL 图像
        pil_images = [Image.fromarray(image) for image in images]

    # 返回 PIL 图像列表
    return pil_images


# 创建图像网格的函数，便于可视化
def make_image_grid(images: List[PIL.Image.Image], rows: int, cols: int, resize: int = None) -> PIL.Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    # 断言图像数量与网格大小一致
    assert len(images) == rows * cols

    # 如果指定了调整大小，则对每个图像进行调整
    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    # 获取每个图像的宽度和高度
    w, h = images[0].size
    # 创建一个新的 RGB 图像，用于放置网格
    grid = Image.new("RGB", size=(cols * w, rows * h))

    # 将每个图像放置到网格中
    for i, img in enumerate(images):
        # 计算每个图像的放置位置
        grid.paste(img, box=(i % cols * w, i // cols * h))
    # 返回生成的图像网格
    return grid
```