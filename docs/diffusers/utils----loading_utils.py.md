# `.\diffusers\utils\loading_utils.py`

```py
# 导入操作系统模块
import os
# 导入临时文件模块
import tempfile
# 导入类型相关的类型提示
from typing import Callable, List, Optional, Union

# 导入 PIL 库中的图像模块
import PIL.Image
# 导入 PIL 库中的图像处理模块
import PIL.ImageOps
# 导入请求库
import requests

# 从本地模块中导入一些实用工具
from .import_utils import BACKENDS_MAPPING, is_imageio_available


# 定义加载图像的函数
def load_image(
    # 接受字符串或 PIL 图像作为输入
    image: Union[str, PIL.Image.Image], 
    # 可选的转换方法，用于加载后处理图像
    convert_method: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]] = None
) -> PIL.Image.Image:
    """
    加载给定的 `image` 为 PIL 图像。

    参数:
        image (`str` 或 `PIL.Image.Image`):
            要转换为 PIL 图像格式的图像。
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], *optional*):
            加载后应用于图像的转换方法。如果为 `None`，则图像将被转换为 "RGB"。

    返回:
        `PIL.Image.Image`:
            一个 PIL 图像。
    """
    # 检查 image 是否为字符串类型
    if isinstance(image, str):
        # 如果字符串以 http 或 https 开头，则认为是 URL
        if image.startswith("http://") or image.startswith("https://"):
            # 通过请求获取图像并打开为 PIL 图像
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        # 检查给定路径是否为有效文件
        elif os.path.isfile(image):
            # 打开本地文件为 PIL 图像
            image = PIL.Image.open(image)
        else:
            # 如果路径无效，抛出错误
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    # 检查 image 是否为 PIL 图像对象
    elif isinstance(image, PIL.Image.Image):
        # 如果是 PIL 图像，保持不变
        image = image
    else:
        # 如果格式不正确，抛出错误
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    # 应用 EXIF 转换，调整图像方向
    image = PIL.ImageOps.exif_transpose(image)

    # 如果提供了转换方法，则应用该方法
    if convert_method is not None:
        image = convert_method(image)
    else:
        # 否则将图像转换为 RGB 格式
        image = image.convert("RGB")

    # 返回处理后的图像
    return image


# 定义加载视频的函数
def load_video(
    # 接受视频的字符串路径或 URL
    video: str,
    # 可选的转换方法，用于加载后处理图像列表
    convert_method: Optional[Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]]] = None,
) -> List[PIL.Image.Image]:
    """
    加载给定的 `video` 为 PIL 图像列表。

    参数:
        video (`str`):
            视频的 URL 或路径，转换为 PIL 图像列表。
        convert_method (Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]], *optional*):
            加载后应用于视频的转换方法。如果为 `None`，则图像将被转换为 "RGB"。

    返回:
        `List[PIL.Image.Image]`:
            视频作为 PIL 图像列表。
    """
    # 检查视频是否为 URL
    is_url = video.startswith("http://") or video.startswith("https://")
    # 检查视频路径是否为有效文件
    is_file = os.path.isfile(video)
    # 标记是否创建了临时文件
    was_tempfile_created = False

    # 如果既不是 URL 也不是文件，抛出错误
    if not (is_url or is_file):
        raise ValueError(
            f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {video} is not a valid path."
        )

    # 如果是 URL，获取视频数据
    if is_url:
        video_data = requests.get(video, stream=True).raw
        # 获取视频的文件后缀，如果没有则默认使用 .mp4
        suffix = os.path.splitext(video)[1] or ".mp4"
        # 创建一个带后缀的临时文件
        video_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
        # 标记已创建临时文件
        was_tempfile_created = True
        # 将视频数据写入临时文件
        with open(video_path, "wb") as f:
            f.write(video_data.read())

        # 更新视频变量为临时文件路径
        video = video_path
    # 初始化一个空列表，用于存储 PIL 图像
        pil_images = []
        # 检查视频文件是否为 GIF 格式
        if video.endswith(".gif"):
            # 打开 GIF 文件
            gif = PIL.Image.open(video)
            try:
                # 持续读取 GIF 的每一帧
                while True:
                    # 将当前帧的副本添加到列表中
                    pil_images.append(gif.copy())
                    # 移动到下一帧
                    gif.seek(gif.tell() + 1)
            # 捕捉到 EOFError 时停止循环
            except EOFError:
                pass
    
        else:
            # 检查 imageio 库是否可用
            if is_imageio_available():
                # 导入 imageio 库
                import imageio
            else:
                # 如果不可用，抛出导入错误
                raise ImportError(BACKENDS_MAPPING["imageio"][1].format("load_video"))
    
            try:
                # 尝试获取 ffmpeg 执行文件
                imageio.plugins.ffmpeg.get_exe()
            except AttributeError:
                # 如果未找到 ffmpeg，抛出属性错误
                raise AttributeError(
                    "`Unable to find an ffmpeg installation on your machine. Please install via `pip install imageio-ffmpeg"
                )
    
            # 使用 imageio 创建视频读取器
            with imageio.get_reader(video) as reader:
                # 读取所有帧
                for frame in reader:
                    # 将每一帧转换为 PIL 图像并添加到列表中
                    pil_images.append(PIL.Image.fromarray(frame))
    
        # 如果创建了临时文件，删除它
        if was_tempfile_created:
            os.remove(video_path)
    
        # 如果提供了转换方法，应用该方法到 PIL 图像列表
        if convert_method is not None:
            pil_images = convert_method(pil_images)
    
        # 返回 PIL 图像列表
        return pil_images
```