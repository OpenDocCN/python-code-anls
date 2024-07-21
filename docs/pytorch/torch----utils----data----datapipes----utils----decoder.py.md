# `.\pytorch\torch\utils\data\datapipes\utils\decoder.py`

```py
# mypy: allow-untyped-defs
# 声明允许有未注释的类型定义

# 从NVIDIA的webdataset库中部分借鉴的实现
# 可在以下链接找到原始代码：
# https://github.com/tmbdev/webdataset/blob/master/webdataset/autodecode.py

# 导入所需的模块
import io
import json
import os.path
import pickle
import tempfile

import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper

# 定义模块的公开接口
__all__ = [
    "Decoder",
    "ImageHandler",
    "MatHandler",
    "audiohandler",
    "basichandlers",
    "extension_extract_fn",
    "handle_extension",
    "imagehandler",
    "mathandler",
    "videohandler",
]

################################################################
# 处理基本数据类型
################################################################
def basichandlers(extension: str, data):
    """将原始数据（字节流）转换为Python对象。

    根据文件扩展名加载数据到支持相应扩展的Python对象。

    Args:
        extension (str): 文件扩展名
        data (byte stream): 要加载到Python对象中的数据

    Returns:
        object: 数据加载到支持扩展的Python对象中

    示例:
        >>> import pickle
        >>> data = pickle.dumps('some data')
        >>> new_data = basichandlers('pickle', data)
        >>> new_data
        some data

    扩展名与数据的转换如下:
        - txt, text, transcript: 以utf-8解码的str格式数据
        - cls, cls2, class, count, index, inx, id: int类型数据
        - json, jsn: 加载的json数据
        - pickle, pyd: 加载的pickle数据
        - pt: 加载的torch数据
    """

    if extension in "txt text transcript":
        return data.decode("utf-8")

    if extension in "cls cls2 class count index inx id".split():
        try:
            return int(data)
        except ValueError:
            return None

    if extension in "json jsn":
        return json.loads(data)

    if extension in "pyd pickle".split():
        return pickle.loads(data)

    if extension in "pt".split():
        stream = io.BytesIO(data)
        return torch.load(stream)

    return None

################################################################
# 处理图片
################################################################
imagespecs = {
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),
    "l": ("numpy", "float", "l"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),
    "torchl8": ("torch", "uint8", "l"),
    "torchrgb8": ("torch", "uint8", "rgb"),
    "torchrgba8": ("torch", "uint8", "rgba"),
    "torchl": ("torch", "float", "l"),
    "torchrgb": ("torch", "float", "rgb"),
    # 定义一个字典，包含不同的键值对，每个键值对代表一个图片处理库及其相关参数
    "torch": ("torch", "float", "rgb"),
    # 使用 torch 库处理图片，数据类型为 float，颜色模式为 RGB
    "torchrgba": ("torch", "float", "rgba"),
    # 使用 torch 库处理图片，数据类型为 float，颜色模式为 RGBA
    "pill": ("pil", None, "l"),
    # 使用 PIL 库处理图片，不指定数据类型，颜色模式为灰度（L）
    "pil": ("pil", None, "rgb"),
    # 使用 PIL 库处理图片，不指定数据类型，颜色模式为 RGB
    "pilrgb": ("pil", None, "rgb"),
    # 使用 PIL 库处理图片，不指定数据类型，颜色模式为 RGB
    "pilrgba": ("pil", None, "rgba"),
    # 使用 PIL 库处理图片，不指定数据类型，颜色模式为 RGBA
}

# 定义一个处理文件扩展名的函数，返回一个处理器函数，用于给定扩展名列表
def handle_extension(extensions, f):
    """
    根据扩展名列表返回一个解码处理器函数。

    扩展名可以是一个以空格分隔的列表。
    扩展名可以包含点号，在这种情况下，给定给 f 的键必须包含相应数量的扩展名组件。
    比较是不区分大小写的。
    示例:
    handle_extension("jpg jpeg", my_decode_jpg)  # 对于任何 file.jpg 都会调用
    handle_extension("seg.jpg", special_case_jpg)  # 仅对 file.seg.jpg 调用
    """
    extensions = extensions.lower().split()

    def g(key, data):
        extension = key.lower().split(".")

        for target in extensions:
            target = target.split(".")
            if len(target) > len(extension):
                continue

            if extension[-len(target) :] == target:
                return f(data)
            return None

    return g


class ImageHandler:
    """
    使用给定的 `imagespec` 解码图像数据。

    `imagespec` 指定了图像如何解码：
    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchl: torch float l
    - torchrgb: torch float rgb
    - torch: torch float rgb
    - torchrgba: torch float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba
    """

    def __init__(self, imagespec):
        assert imagespec in list(
            imagespecs.keys()
        ), f"unknown image specification: {imagespec}"
        self.imagespec = imagespec.lower()
    # 定义一个特殊方法，使实例对象可以像函数一样被调用，处理图像文件的解码
    def __call__(self, extension, data):
        # 检查文件扩展名是否在支持的图像格式列表中，如果不在则返回空
        if extension.lower() not in "jpg jpeg png ppm pgm pbm pnm".split():
            return None

        try:
            import numpy as np  # 尝试导入 numpy 库
        except ModuleNotFoundError as e:
            # 如果导入失败，抛出异常提示用户安装 numpy 库
            raise ModuleNotFoundError(
                "Package `numpy` is required to be installed for default image decoder."
                "Please use `pip install numpy` to install the package"
            ) from e

        try:
            import PIL.Image  # 尝试导入 PIL 库中的 Image 模块
        except ModuleNotFoundError as e:
            # 如果导入失败，抛出异常提示用户安装 Pillow 库
            raise ModuleNotFoundError(
                "Package `PIL` is required to be installed for default image decoder."
                "Please use `pip install Pillow` to install the package"
            ) from e

        imagespec = self.imagespec  # 获取实例对象的图像规范
        atype, etype, mode = imagespecs[imagespec]  # 获取图像规范中的数据

        # 使用数据创建一个 BytesIO 流对象
        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)  # 用 PIL 库打开流中的图像数据
            img.load()  # 加载图像数据
            img = img.convert(mode.upper())  # 将图像转换为指定模式
            if atype == "pil":
                return img  # 如果图像类型为 PIL，则直接返回 PIL.Image 对象
            elif atype == "numpy":
                result = np.asarray(img)  # 将 PIL.Image 对象转换为 numpy 数组
                assert (
                    result.dtype == np.uint8
                ), f"numpy image array should be type uint8, but got {result.dtype}"  # 断言数组类型为 uint8
                if etype == "uint8":
                    return result  # 如果目标类型为 uint8，则直接返回 numpy 数组
                else:
                    return result.astype("f") / 255.0  # 否则将数组转换为浮点数并归一化后返回
            elif atype == "torch":
                result = np.asarray(img)  # 将 PIL.Image 对象转换为 numpy 数组
                assert (
                    result.dtype == np.uint8
                ), f"numpy image array should be type uint8, but got {result.dtype}"  # 断言数组类型为 uint8

                if etype == "uint8":
                    result = np.array(result.transpose(2, 0, 1))  # 转置数组维度顺序
                    return torch.tensor(result)  # 将数组转换为 Torch 张量并返回
                else:
                    result = np.array(result.transpose(2, 0, 1))  # 转置数组维度顺序
                    return torch.tensor(result) / 255.0  # 将数组转换为浮点数并归一化后返回
            return None  # 如果未匹配到任何图像类型，返回空
################################################################
# sample decoder
################################################################
# 从文件路径中提取文件扩展名
def extension_extract_fn(pathname):
    # 使用 os 模块的 splitext 函数提取文件扩展名
    ext = os.path.splitext(pathname)[1]
    # 如果存在扩展名，则去除开头的点号
    if ext:
        ext = ext[1:]
    # 返回变量 ext 的值作为函数的返回结果
    return ext
class Decoder:
    """
    Decode key/data sets using a list of handlers.

    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """

    def __init__(self, *handler, key_fn=extension_extract_fn):
        # 初始化 Decoder 对象，接受一系列处理函数作为参数，并设置默认的键提取函数
        self.handlers = list(handler) if handler else []
        self.key_fn = key_fn

    # 将新的处理函数插入到处理函数列表的开头，确保新处理函数具有最高优先级
    def add_handler(self, *handler):
        if not handler:
            return
        self.handlers = list(handler) + self.handlers

    @staticmethod
    def _is_stream_handle(data):
        # 静态方法：检查数据是否是流处理对象
        obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
        return isinstance(obj_to_check, (io.BufferedIOBase, io.RawIOBase))

    def decode1(self, key, data):
        # 解码单个数据项
        if not data:
            return data

        # 如果数据是流处理对象，则需要在解码之前读取所有内容
        if Decoder._is_stream_handle(data):
            ds = data
            # 由于流对象的读取行为可能不同，因此使用此方法确保读取所有内容
            data = b"".join(data)
            ds.close()

        # 通过每个处理函数依次处理数据，直到某个处理函数返回非空结果
        for f in self.handlers:
            result = f(key, data)
            if result is not None:
                return result
        return data

    def decode(self, data):
        # 解码数据集合
        result = {}
        # 如果数据是单个元组 (路径名, 数据流)
        if isinstance(data, tuple):
            data = [data]

        if data is not None:
            for k, v in data:
                # 如果键以 '_' 开头，将字节流解码为 UTF-8 字符串
                if k[0] == "_":
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                        result[k] = v
                        continue
                # 使用键提取函数获取键，然后调用 decode1 方法处理数据
                result[k] = self.decode1(self.key_fn(k), v)
        return result

    def __call__(self, data):
        # 可调用对象的实现，直接调用 decode 方法
        return self.decode(data)
```