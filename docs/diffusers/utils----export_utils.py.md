# `.\diffusers\utils\export_utils.py`

```py
# 导入所需的模块
import io  # 用于处理输入输出
import random  # 用于生成随机数
import struct  # 用于处理C语言风格的二进制数据
import tempfile  # 用于创建临时文件
from contextlib import contextmanager  # 用于创建上下文管理器
from typing import List, Union  # 用于类型注解

import numpy as np  # 用于数值计算
import PIL.Image  # 用于处理图像
import PIL.ImageOps  # 用于图像操作

from .import_utils import BACKENDS_MAPPING, is_imageio_available, is_opencv_available  # 导入工具函数和映射
from .logging import get_logger  # 导入日志记录器

# 创建全局随机数生成器
global_rng = random.Random()

# 获取当前模块的日志记录器
logger = get_logger(__name__)

# 定义上下文管理器以便于缓冲写入
@contextmanager
def buffered_writer(raw_f):
    # 创建缓冲写入对象
    f = io.BufferedWriter(raw_f)
    # 生成缓冲写入对象
    yield f
    # 刷新缓冲区，确保所有数据写入
    f.flush()

# 导出图像列表为 GIF 文件
def export_to_gif(image: List[PIL.Image.Image], output_gif_path: str = None, fps: int = 10) -> str:
    # 如果没有提供输出路径，则创建一个临时文件
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name

    # 保存图像为 GIF 文件
    image[0].save(
        output_gif_path,
        save_all=True,  # 保存所有帧
        append_images=image[1:],  # 附加后续图像
        optimize=False,  # 不优化图像
        duration=1000 // fps,  # 设置帧间隔
        loop=0,  # 循环次数
    )
    # 返回生成的 GIF 文件路径
    return output_gif_path

# 导出网格数据为 PLY 文件
def export_to_ply(mesh, output_ply_path: str = None):
    """
    写入网格的 PLY 文件。
    """
    # 如果没有提供输出路径，则创建一个临时文件
    if output_ply_path is None:
        output_ply_path = tempfile.NamedTemporaryFile(suffix=".ply").name

    # 获取网格顶点坐标并转换为 NumPy 数组
    coords = mesh.verts.detach().cpu().numpy()
    # 获取网格面信息并转换为 NumPy 数组
    faces = mesh.faces.cpu().numpy()
    # 获取网格顶点的 RGB 颜色信息并堆叠为数组
    rgb = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)

    # 使用缓冲写入器打开输出文件
    with buffered_writer(open(output_ply_path, "wb")) as f:
        f.write(b"ply\n")  # 写入 PLY 文件头
        f.write(b"format binary_little_endian 1.0\n")  # 指定文件格式
        f.write(bytes(f"element vertex {len(coords)}\n", "ascii"))  # 写入顶点数量
        f.write(b"property float x\n")  # 写入 x 坐标属性
        f.write(b"property float y\n")  # 写入 y 坐标属性
        f.write(b"property float z\n")  # 写入 z 坐标属性
        if rgb is not None:  # 如果有 RGB 颜色信息
            f.write(b"property uchar red\n")  # 写入红色属性
            f.write(b"property uchar green\n")  # 写入绿色属性
            f.write(b"property uchar blue\n")  # 写入蓝色属性
        if faces is not None:  # 如果有面信息
            f.write(bytes(f"element face {len(faces)}\n", "ascii"))  # 写入面数量
            f.write(b"property list uchar int vertex_index\n")  # 写入顶点索引属性
        f.write(b"end_header\n")  # 写入文件头结束标记

        if rgb is not None:  # 如果有 RGB 颜色信息
            rgb = (rgb * 255.499).round().astype(int)  # 将 RGB 值转换为整数
            vertices = [
                (*coord, *rgb)  # 合并坐标和颜色信息
                for coord, rgb in zip(
                    coords.tolist(),
                    rgb.tolist(),
                )
            ]
            format = struct.Struct("<3f3B")  # 定义数据格式
            for item in vertices:  # 写入每个顶点数据
                f.write(format.pack(*item))  # 使用打包格式写入数据
        else:  # 如果没有 RGB 信息
            format = struct.Struct("<3f")  # 定义仅包含坐标的数据格式
            for vertex in coords.tolist():  # 写入每个顶点坐标
                f.write(format.pack(*vertex))  # 使用打包格式写入数据

        if faces is not None:  # 如果有面信息
            format = struct.Struct("<B3I")  # 定义面数据格式
            for tri in faces.tolist():  # 写入每个面数据
                f.write(format.pack(len(tri), *tri))  # 使用打包格式写入数据

    # 返回生成的 PLY 文件路径
    return output_ply_path

# 导出网格数据为 OBJ 文件
def export_to_obj(mesh, output_obj_path: str = None):
    # 如果没有提供输出路径，则创建一个临时文件
    if output_obj_path is None:
        output_obj_path = tempfile.NamedTemporaryFile(suffix=".obj").name

    # 获取网格顶点坐标并转换为 NumPy 数组
    verts = mesh.verts.detach().cpu().numpy()
    # 获取网格面信息并转换为 NumPy 数组
    faces = mesh.faces.cpu().numpy()
    # 将网格顶点的颜色通道合并成一个多维数组
        vertex_colors = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)
        # 将顶点坐标和颜色组合成格式化字符串，形成顶点列表
        vertices = [
            "{} {} {} {} {} {}".format(*coord, *color) for coord, color in zip(verts.tolist(), vertex_colors.tolist())
        ]
    
        # 将每个三角形的索引格式化为面定义字符串，索引加1以符合 OBJ 格式
        faces = ["f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1)) for tri in faces.tolist()]
    
        # 将顶点和面数据合并为最终输出列表
        combined_data = ["v " + vertex for vertex in vertices] + faces
    
        # 打开指定路径的文件以写入数据
        with open(output_obj_path, "w") as f:
            # 将合并的数据写入文件，每个元素占一行
            f.writelines("\n".join(combined_data))
# 导出视频的私有函数，接受视频帧列表、输出视频路径和帧率作为参数
def _legacy_export_to_video(
    # 视频帧，可以是 NumPy 数组或 PIL 图像的列表，输出视频的路径，帧率
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 10
):
    # 检查 OpenCV 是否可用
    if is_opencv_available():
        # 导入 OpenCV 库
        import cv2
    else:
        # 如果不可用，抛出导入错误
        raise ImportError(BACKENDS_MAPPING["opencv"][1].format("export_to_video"))
    
    # 如果没有提供输出视频路径，则创建一个临时文件
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    # 如果视频帧是 NumPy 数组，则将其值从 [0, 1] 乘以 255 并转换为 uint8 类型
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    # 如果视频帧是 PIL 图像，则将其转换为 NumPy 数组
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    # 获取视频编码器，指定使用 mp4v 编码
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # 获取视频帧的高度、宽度和通道数
    h, w, c = video_frames[0].shape
    # 创建视频写入对象，设置输出路径、编码器、帧率和帧尺寸
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    
    # 遍历每一帧，转换颜色并写入视频
    for i in range(len(video_frames)):
        # 将帧从 RGB 转换为 BGR 格式
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        # 将转换后的帧写入视频文件
        video_writer.write(img)

    # 返回输出视频的路径
    return output_video_path


# 导出视频的公共函数，接受视频帧列表、输出视频路径和帧率作为参数
def export_to_video(
    # 视频帧，可以是 NumPy 数组或 PIL 图像的列表，输出视频的路径，帧率
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 10
) -> str:
    # TODO: Dhruv. 在 Diffusers 版本 0.33.0 发布时删除
    # 为了防止现有代码中断而添加的
    # 检查 imageio 是否可用
    if not is_imageio_available():
        # 记录警告信息，提示用户建议使用 imageio 和 imageio-ffmpeg 作为后端
        logger.warning(
            (
                "It is recommended to use `export_to_video` with `imageio` and `imageio-ffmpeg` as a backend. \n"
                "These libraries are not present in your environment. Attempting to use legacy OpenCV backend to export video. \n"
                "Support for the OpenCV backend will be deprecated in a future Diffusers version"
            )
        )
        # 如果不使用 imageio，则调用旧版导出函数
        return _legacy_export_to_video(video_frames, output_video_path, fps)

    # 如果 imageio 可用，则导入它
    if is_imageio_available():
        import imageio
    else:
        # 如果不可用，抛出导入错误
        raise ImportError(BACKENDS_MAPPING["imageio"][1].format("export_to_video"))

    # 尝试获取 imageio ffmpeg 插件的执行文件
    try:
        imageio.plugins.ffmpeg.get_exe()
    except AttributeError:
        # 如果未找到兼容的 ffmpeg 安装，抛出属性错误
        raise AttributeError(
            (
                "Found an existing imageio backend in your environment. Attempting to export video with imageio. \n"
                "Unable to find a compatible ffmpeg installation in your environment to use with imageio. Please install via `pip install imageio-ffmpeg"
            )
        )

    # 如果没有提供输出视频路径，则创建一个临时文件
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    # 如果视频帧是 NumPy 数组，则将其值从 [0, 1] 乘以 255 并转换为 uint8 类型
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    # 如果视频帧是 PIL 图像，则将其转换为 NumPy 数组
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    # 使用 imageio 创建视频写入器，指定输出路径和帧率
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        # 遍历每一帧，将其附加到视频写入器中
        for frame in video_frames:
            writer.append_data(frame)

    # 返回输出视频的路径
    return output_video_path
```