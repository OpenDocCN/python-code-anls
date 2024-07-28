# `.\comic-translate\modules\utils\inpainting.py`

```py
# https://github.com/Sanster/lama-cleaner/blob/main/lama_cleaner/helper.py

# 导入所需的模块
import io
import os
import sys
from typing import List, Optional

# 从 urllib.parse 模块导入 urlparse 函数
from urllib.parse import urlparse

# 导入 OpenCV 库
import cv2
# 导入 Pillow 库中的 Image、ImageOps 和 PngImagePlugin 模块
from PIL import Image, ImageOps, PngImagePlugin
# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 loguru 库中的 logger 对象
from loguru import logger
# 从 torch.hub 模块中导入 download_url_to_file 和 get_dir 函数
from torch.hub import download_url_to_file, get_dir
# 导入 hashlib 库中的 md5 函数
import hashlib

# 计算文件的 MD5 值
def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


# 根据 URL 获取缓存文件路径
def get_cache_path_by_url(url):
    parts = urlparse(url)
    # 获取当前工作目录下的 models 文件夹路径
    dir = os.path.join(os.getcwd(), "models")
    # 在 models 文件夹下创建 inpainting 文件夹路径
    model_dir = os.path.join(dir, "inpainting")

    # 如果 inpainting 文件夹不存在，则创建该文件夹
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # 获取 URL 中的文件名，并与 model_dir 拼接成缓存文件路径
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


# 下载模型文件
def download_model(url, model_md5: str = None):
    # 获取模型文件的缓存路径
    cached_file = get_cache_path_by_url(url)

    # 如果缓存文件不存在
    if not os.path.exists(cached_file):
        # 输出提示信息到标准错误流，显示下载进度
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        # 使用 download_url_to_file 函数下载模型文件
        download_url_to_file(url, cached_file, hash_prefix, progress=True)

        # 如果指定了模型的 MD5 值
        if model_md5:
            # 计算下载文件的 MD5 值
            _md5 = md5sum(cached_file)
            # 如果 MD5 值匹配
            if model_md5 == _md5:
                logger.info(f"Download model success, md5: {_md5}")
            else:
                try:
                    # 删除下载错误的模型文件
                    os.remove(cached_file)
                    logger.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart comic-translate."
                        f"If you still have errors, please try download model manually first https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
                    )
                except:
                    logger.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, please delete {cached_file} and restart comic-translate."
                    )
                # 终止程序执行
                exit(-1)

    # 返回模型文件的缓存路径
    return cached_file


# 将 x 对 mod 取上整除余数
def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


# 处理错误情况，打印模型路径、模型 MD5 值和异常信息
def handle_error(model_path, model_md5, e):
    _md5 = md5sum(model_path)
    # 如果文件的 MD5 校验值与期望的模型 MD5 校验值不一致
    if _md5 != model_md5:
        # 尝试删除已下载的错误模型文件
        try:
            os.remove(model_path)
            # 记录错误日志，指出错误的模型 MD5 和期望的 MD5，并提示手动下载模型的步骤
            logger.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart lama-cleaner."
                f"If you still have errors, please try download model manually first https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
            )
        # 如果删除失败，则记录错误日志，提示手动删除文件并重启 lama-cleaner
        except:
            logger.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, please delete {model_path} and restart lama-cleaner."
            )
    # 如果文件的 MD5 校验值与期望的模型 MD5 校验值一致
    else:
        # 记录错误日志，指出加载模型失败，并提示用户提交错误到 GitHub 并附上错误截图
        logger.error(
            f"Failed to load model {model_path},"
            f"please submit an issue at https://github.com/Sanster/lama-cleaner/issues and include a screenshot of the error:\n{e}"
        )
    # 退出程序，返回错误码 -1
    exit(-1)
# 根据给定的 URL 或路径加载 JIT 模型，并将其加载到指定的设备上
def load_jit_model(url_or_path, device, model_md5: str):
    # 如果给定的路径存在，则直接使用该路径
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        # 否则，通过下载模型函数获取模型路径
        model_path = download_model(url_or_path, model_md5)

    # 记录日志，显示正在从哪个路径加载模型
    logger.info(f"Loading model from: {model_path}")
    try:
        # 加载 JIT 模型，并将其移到指定的设备上
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except Exception as e:
        # 处理加载模型时可能发生的异常
        handle_error(model_path, model_md5, e)
    
    # 设置模型为评估模式
    model.eval()
    return model


# 加载普通 PyTorch 模型，并将其加载到指定的设备上
def load_model(model: torch.nn.Module, url_or_path, device, model_md5):
    # 如果给定的路径存在，则直接使用该路径
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        # 否则，通过下载模型函数获取模型路径
        model_path = download_model(url_or_path, model_md5)

    try:
        # 记录日志，显示正在从哪个路径加载模型
        logger.info(f"Loading model from: {model_path}")
        # 加载模型的状态字典，并且进行严格模式的加载
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        # 将模型移到指定的设备上
        model.to(device)
    except Exception as e:
        # 处理加载模型时可能发生的异常
        handle_error(model_path, model_md5, e)
    
    # 设置模型为评估模式
    model.eval()
    return model


# 将 numpy 数组转换为字节流（bytes）
def numpy_to_bytes(image_numpy: np.ndarray, ext: str) -> bytes:
    # 使用 OpenCV 将图像数组编码为指定格式的图像数据
    data = cv2.imencode(
        f".{ext}",
        image_numpy,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )[1]
    # 将编码后的图像数据转换为字节流
    image_bytes = data.tobytes()
    return image_bytes


# 将 PIL 图像对象转换为字节流（bytes）
def pil_to_bytes(pil_img, ext: str, quality: int = 95, exif_infos={}) -> bytes:
    with io.BytesIO() as output:
        kwargs = {k: v for k, v in exif_infos.items() if v is not None}
        if ext == "png" and "parameters" in kwargs:
            # 如果是 PNG 格式，并且包含特定参数，则添加到 PNG 图像信息中
            pnginfo_data = PngImagePlugin.PngInfo()
            pnginfo_data.add_text("parameters", kwargs["parameters"])
            kwargs["pnginfo"] = pnginfo_data

        # 将 PIL 图像对象保存为指定格式的图像数据，并存储在输出流中
        pil_img.save(
            output,
            format=ext,
            quality=quality,
            **kwargs,
        )
        # 获取输出流中的字节数据
        image_bytes = output.getvalue()
    return image_bytes


# 加载图像字节数据并返回 numpy 数组及其 alpha 通道（如果有的话）
def load_img(img_bytes, gray: bool = False, return_exif: bool = False):
    alpha_channel = None
    # 从图像字节数据中创建 PIL 图像对象
    image = Image.open(io.BytesIO(img_bytes))

    if return_exif:
        # 如果需要返回 exif 信息，则获取图像的元数据信息
        info = image.info or {}
        exif_infos = {"exif": image.getexif(), "parameters": info.get("parameters")}

    try:
        # 尝试对图像进行方向调整，以使其正常显示
        image = ImageOps.exif_transpose(image)
    except:
        # 如果出错，不做任何处理
        pass

    if gray:
        # 如果需要返回灰度图像，则将图像转换为灰度模式
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            # 如果图像模式为 RGBA，则获取 alpha 通道并转换为 RGB 模式
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            # 否则，将图像转换为 RGB 模式
            image = image.convert("RGB")
            np_img = np.array(image)

    if return_exif:
        # 如果需要返回 exif 信息，则同时返回 numpy 数组、alpha 通道和 exif 信息
        return np_img, alpha_channel, exif_infos
    # 否则，只返回 numpy 数组及其 alpha 通道（如果有的话）
    return np_img, alpha_channel


# 对输入的 numpy 数组进行归一化处理，返回归一化后的数组
def norm_img(np_img):
    if len(np_img.shape) == 2:
        # 如果输入数组是灰度图像，则扩展其维度
        np_img = np_img[:, :, np.newaxis]
    # 将数组维度重新排列为 (通道数, 宽度, 高度)
    np_img = np.transpose(np_img, (2, 0, 1))
    # 将数组类型转换为 float32，并进行归一化处理（0-1 范围）
    np_img = np_img.astype("float32") / 255
    return np_img


# 调整输入图像的尺寸，使其最大尺寸不超过指定的限制
def resize_max_size(
    np_img, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    # 获取图像的高度和宽度
    h, w = np_img.shape[:2]
    
    # 检查图像的长边是否大于指定的尺寸限制
    if max(h, w) > size_limit:
        # 计算缩放比例，使图像长边缩放到指定的尺寸限制
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)  # 计算新的宽度
        new_h = int(h * ratio + 0.5)  # 计算新的高度
        # 使用指定的插值方法对图像进行缩放，并返回缩放后的图像
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)
    else:
        # 如果图像的长边没有超过指定的尺寸限制，则返回原始图像
        return np_img
# 根据指定的模数对图像进行填充，使其高度和宽度都是模数的倍数
def pad_img_to_modulo(
    img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None
):
    """
    Args:
        img: 输入的图像数组，形状为 [H, W, C]
        mod: 指定的模数，用于确定填充后的高度和宽度
        square: 是否将输出调整为正方形
        min_size: 如果指定，将确保输出图像的高度和宽度不小于该值

    Returns:
        np.ndarray: 填充后的图像数组
    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    # 计算调整后的输出高度和宽度，使其都是模数的倍数
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        # 如果指定了最小尺寸，确保输出的高度和宽度不小于最小尺寸
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        # 如果需要输出为正方形，调整高度和宽度为最大值
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    # 使用对称模式对图像进行填充，保持边缘的对称性
    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def boxes_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: 掩码图像数组，形状为 (h, w, 1)，像素值范围在 0~255

    Returns:
        List[np.ndarray]: 包含边界框坐标的列表
    """
    height, width = mask.shape[:2]
    # 使用阈值处理得到二值图像
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    # 提取轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        # 获取轮廓的边界框坐标
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h]).astype(int)

        # 确保边界框坐标在图像范围内
        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)

    return boxes


def only_keep_largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    Args:
        mask: 掩码图像数组，形状为 (h, w)，像素值范围在 0~255

    Returns:
        np.ndarray: 只保留最大轮廓的掩码图像数组
    """
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    # 提取轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_index = -1
    for i, cnt in enumerate(contours):
        # 计算轮廓的面积
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_index = i

    if max_index != -1:
        # 如果找到最大面积的轮廓，创建一个新的掩码图像并将其绘制上最大轮廓
        new_mask = np.zeros_like(mask)
        return cv2.drawContours(new_mask, contours, max_index, 255, -1)
    else:
        # 如果没有找到有效的轮廓，返回原始掩码图像
        return mask
```