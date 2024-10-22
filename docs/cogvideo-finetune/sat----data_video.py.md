# `.\cogvideo-finetune\sat\data_video.py`

```py
# 导入io模块，处理输入输出操作
import io
# 导入os模块，进行操作系统相关的功能
import os
# 导入sys模块，访问与Python解释器相关的变量和函数
import sys
# 从functools模块导入partial，用于创建部分函数应用
from functools import partial
# 导入math模块，进行数学计算
import math
# 导入torchvision.transforms模块，进行图像变换
import torchvision.transforms as TT
# 从sgm.webds模块导入MetaDistributedWebDataset，进行分布式数据集处理
from sgm.webds import MetaDistributedWebDataset
# 导入random模块，进行随机数生成
import random
# 从fractions模块导入Fraction，处理有理数
from fractions import Fraction
# 从typing模块导入Union、Optional、Dict、Any和Tuple，进行类型注解
from typing import Union, Optional, Dict, Any, Tuple
# 从torchvision.io.video导入av，处理视频输入输出
from torchvision.io.video import av
# 导入numpy库，进行数组和矩阵操作
import numpy as np
# 导入torch库，进行深度学习操作
import torch
# 从torchvision.io导入_video_opt，处理视频选项
from torchvision.io import _video_opt
# 从torchvision.io.video导入多个函数，用于视频处理
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
# 从torchvision.transforms.functional导入center_crop和resize，进行图像裁剪和调整大小
from torchvision.transforms.functional import center_crop, resize
# 从torchvision.transforms导入InterpolationMode，处理插值模式
from torchvision.transforms import InterpolationMode
# 导入decord库，进行视频读取
import decord
# 从decord模块导入VideoReader，读取视频
from decord import VideoReader
# 从torch.utils.data导入Dataset，构建数据集类
from torch.utils.data import Dataset


# 定义读取视频的函数，返回视频帧和音频帧
def read_video(
    filename: str,  # 视频文件的路径
    start_pts: Union[float, Fraction] = 0,  # 视频开始的展示时间
    end_pts: Optional[Union[float, Fraction]] = None,  # 视频结束的展示时间
    pts_unit: str = "pts",  # 展示时间的单位，默认为'pts'
    output_format: str = "THWC",  # 输出视频张量的格式
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    从文件中读取视频，返回视频帧和音频帧

    参数:
        filename (str): 视频文件的路径
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            视频的开始展示时间
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            结束展示时间
        pts_unit (str, optional): start_pts和end_pts值的单位,
            可以是'pts'或'sec'。默认为'pts'。
        output_format (str, optional): 输出视频张量的格式，可以是'THWC'（默认）或'TCHW'。

    返回:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): 视频帧
        aframes (Tensor[K, L]): 音频帧，其中K为通道数，L为点数
        info (Dict): 视频和音频的元数据。可以包含video_fps（float）和audio_fps（int）字段
    """

    # 将输出格式转换为大写
    output_format = output_format.upper()
    # 检查输出格式是否有效
    if output_format not in ("THWC", "TCHW"):
        # 如果无效，抛出错误
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    # 检查AV（音频视频）是否可用
    _check_av_available()

    # 如果结束时间点为空，则设置为无穷大
    if end_pts is None:
        end_pts = float("inf")

    # 检查结束时间点是否大于开始时间点
    if end_pts < start_pts:
        # 如果不满足条件，抛出错误
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    # 初始化信息字典，用于存储视频和音频的元数据
    info = {}
    # 初始化音频帧列表，用于存储音频数据
    audio_frames = []
    # 设置音频时间基准
    audio_timebase = _video_opt.default_timebase
    # 使用指定的文件名打开音频/视频容器，并忽略元数据错误
    with av.open(filename, metadata_errors="ignore") as container:
        # 检查容器中是否有音频流
        if container.streams.audio:
            # 获取音频流的时间基准
            audio_timebase = container.streams.audio[0].time_base
        # 检查容器中是否有视频流
        if container.streams.video:
            # 从视频流中读取指定时间范围内的帧
            video_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],  # 指定视频流
                {"video": 0},  # 额外参数，指示视频流索引
            )
            # 获取视频流的平均帧率
            video_fps = container.streams.video[0].average_rate
            # 防止潜在的损坏文件导致错误
            if video_fps is not None:
                # 将视频帧率保存到信息字典中
                info["video_fps"] = float(video_fps)

        # 再次检查音频流
        if container.streams.audio:
            # 从音频流中读取指定时间范围内的帧
            audio_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.audio[0],  # 指定音频流
                {"audio": 0},  # 额外参数，指示音频流索引
            )
            # 将音频帧率保存到信息字典中
            info["audio_fps"] = container.streams.audio[0].rate

    # 将音频帧转换为 NumPy 数组格式
    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    # 创建一个空的张量，用于存放视频帧
    vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    # 如果有音频帧
    if aframes_list:
        # 将音频帧列表沿着第一个维度拼接
        aframes = np.concatenate(aframes_list, 1)
        # 将 NumPy 数组转换为 PyTorch 张量
        aframes = torch.as_tensor(aframes)
        # 如果时间单位是秒，将开始和结束时间点转换为帧数
        if pts_unit == "sec":
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            # 如果结束时间点不是无穷大，进行转换
            if end_pts != float("inf"):
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        # 对齐音频帧
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        # 如果没有音频帧，创建一个空的音频张量
        aframes = torch.empty((1, 0), dtype=torch.float32)

    # 如果输出格式为 TCHW
    if output_format == "TCHW":
        # 将张量维度从 [T,H,W,C] 转换为 [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    # 返回视频帧、音频帧和信息字典
    return vframes, aframes, info
# 根据给定的图像尺寸调整数组，以便进行矩形裁剪
def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    # 判断数组的宽高比是否大于目标图像的宽高比
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        # 调整数组大小，使其适应目标图像的宽度，保持高度比例
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        # 调整数组大小，使其适应目标图像的高度，保持宽度比例
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    # 获取调整后数组的高度和宽度
    h, w = arr.shape[2], arr.shape[3]
    # 移除数组的第一个维度
    arr = arr.squeeze(0)

    # 计算高度和宽度的差值
    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    # 根据重塑模式确定裁剪的起始位置
    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        # 计算中心裁剪的起始位置
        top, left = delta_h // 2, delta_w // 2
    else:
        # 如果重塑模式不支持，则抛出异常
        raise NotImplementedError
    # 从数组中裁剪出指定区域
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    # 返回裁剪后的数组
    return arr


# 填充视频的最后一帧，以确保帧数达到指定数量
def pad_last_frame(tensor, num_frames):
    # 检查当前帧数是否少于指定帧数
    if len(tensor) < num_frames:
        # 计算需要填充的帧数
        pad_length = num_frames - len(tensor)
        # 使用最后一帧进行填充而不是使用零帧
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        # 将原始帧和填充帧拼接在一起
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        # 如果帧数足够，返回前指定数量的帧
        return tensor[:num_frames]


# 加载视频并根据参数进行采样
def load_video(
    video_data,
    sampling="uniform",
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    # 设置视频读取的桥接方式为torch
    decord.bridge.set_bridge("torch")
    # 创建视频读取器对象
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    # 确定要读取的原始视频长度
    if nb_read_frames is not None:
        ori_vlen = nb_read_frames
    else:
        ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    # 计算最大寻址位置
    max_seek = int(ori_vlen - skip_frms_num - num_frames / wanted_fps * actual_fps)
    # 随机选择起始帧
    start = random.randint(skip_frms_num, max_seek + 1)
    # 计算结束帧
    end = int(start + num_frames / wanted_fps * actual_fps)
    n_frms = num_frames

    # 如果采样模式为均匀，生成采样索引
    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / n_frms).astype(int)
    else:
        # 如果采样模式不支持，则抛出异常
        raise NotImplementedError

    # 从视频读取器中获取一批帧
    temp_frms = vr.get_batch(np.arange(start, end))
    # 确保获取的帧不为空
    assert temp_frms is not None
    # 将获取的帧转换为张量
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    # 根据索引提取需要的帧
    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]

    # 返回填充后的帧
    return pad_last_frame(tensor_frms, num_frames)


import threading


# 在子线程中加载视频并设置超时
def load_video_with_timeout(*args, **kwargs):
    video_container = {}

    # 定义目标函数，用于在子线程中执行视频加载
    def target_function():
        video = load_video(*args, **kwargs)
        video_container["video"] = video

    # 创建并启动新线程
    thread = threading.Thread(target=target_function)
    thread.start()
    # 设置超时为20秒
    timeout = 20
    # 等待线程执行完成，最多等待指定时间
    thread.join(timeout)
    # 检查线程是否仍在运行
        if thread.is_alive():
            # 如果线程还在运行，打印超时信息
            print("Loading video timed out")
            # 抛出超时异常
            raise TimeoutError
        # 从视频容器中获取视频数据，如果不存在则返回 None，并确保数据是连续的
        return video_container.get("video", None).contiguous()
# 定义处理视频的函数，接收多个参数以配置视频处理
def process_video(
    video_path,  # 视频文件路径或字节流
    image_size=None,  # 可选参数，处理后的图像大小
    duration=None,  # 可选参数，已知持续时间以加快处理
    num_frames=4,  # 希望处理的帧数，默认为4
    wanted_fps=None,  # 可选参数，期望的帧率
    actual_fps=None,  # 可选参数，实际的帧率
    skip_frms_num=0.0,  # 忽略的帧数，避免过渡帧
    nb_read_frames=None,  # 可选参数，读取的帧数
):
    """
    video_path: str or io.BytesIO  # 视频路径或字节流类型
    image_size: .  # 图像大小的描述
    duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.  # 预先知道持续时间以加快处理
    num_frames: wanted num_frames.  # 希望的帧数
    wanted_fps: .  # 期望的帧率描述
    skip_frms_num: ignore the first and the last xx frames, avoiding transitions.  # 忽略首尾帧以避免过渡
    """

    # 调用函数加载视频，设置超时处理
    video = load_video_with_timeout(
        video_path,  # 视频路径
        duration=duration,  # 视频持续时间
        num_frames=num_frames,  # 希望的帧数
        wanted_fps=wanted_fps,  # 期望的帧率
        actual_fps=actual_fps,  # 实际的帧率
        skip_frms_num=skip_frms_num,  # 忽略的帧数
        nb_read_frames=nb_read_frames,  # 读取的帧数
    )

    # --- 复制并修改图像处理 ---
    video = video.permute(0, 3, 1, 2)  # 将视频的维度顺序调整为 [时间, 通道, 高, 宽]

    # 如果指定了图像大小，则进行调整
    if image_size is not None:
        video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")  # 调整视频尺寸

    return video  # 返回处理后的视频


# 定义处理视频数据的函数
def process_fn_video(src, image_size, fps, num_frames, skip_frms_num=0.0, txt_key="caption"):
    while True:  # 无限循环以处理数据源中的每个项
        r = next(src)  # 获取下一个视频数据项
        if "mp4" in r:  # 如果数据项包含 mp4 格式
            video_data = r["mp4"]  # 提取 mp4 视频数据
        elif "avi" in r:  # 如果数据项包含 avi 格式
            video_data = r["avi"]  # 提取 avi 视频数据
        else:  # 如果没有视频数据
            print("No video data found")  # 输出提示信息
            continue  # 继续下一个循环

        # 检查文本键是否存在
        if txt_key not in r:
            txt = ""  # 如果不存在，设置为空字符串
        else:
            txt = r[txt_key]  # 否则提取文本信息

        # 如果文本是字节类型，则解码为字符串
        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")  # 解码字节为 UTF-8 字符串
        else:
            txt = str(txt)  # 转换为字符串类型

        # 尝试获取视频的持续时间
        duration = r.get("duration", None)  # 获取持续时间，默认为 None
        if duration is not None:  # 如果持续时间存在
            duration = float(duration)  # 转换为浮点数
        else:
            continue  # 如果不存在，则跳过当前循环

        # 尝试获取视频的实际帧率
        actual_fps = r.get("fps", None)  # 获取实际帧率，默认为 None
        if actual_fps is not None:  # 如果实际帧率存在
            actual_fps = float(actual_fps)  # 转换为浮点数
        else:
            continue  # 如果不存在，则跳过当前循环

        # 计算所需帧数和持续时间
        required_frames = num_frames / fps * actual_fps + 2 * skip_frms_num  # 计算所需帧数
        required_duration = num_frames / fps + 2 * skip_frms_num / actual_fps  # 计算所需持续时间

        # 如果视频持续时间小于所需持续时间，则跳过
        if duration is not None and duration < required_duration:
            continue  # 跳过当前循环

        try:
            # 调用处理视频函数
            frames = process_video(
                io.BytesIO(video_data),  # 将视频数据转换为字节流
                num_frames=num_frames,  # 设置希望处理的帧数
                wanted_fps=fps,  # 设置期望的帧率
                image_size=image_size,  # 设置图像大小
                duration=duration,  # 设置视频持续时间
                actual_fps=actual_fps,  # 设置实际帧率
                skip_frms_num=skip_frms_num,  # 设置忽略的帧数
            )
            frames = (frames - 127.5) / 127.5  # 进行帧归一化处理
        except Exception as e:  # 捕获处理过程中的任何异常
            print(e)  # 输出异常信息
            continue  # 继续下一个循环

        # 创建包含处理结果的字典
        item = {
            "mp4": frames,  # 存储处理后的视频帧
            "txt": txt,  # 存储相关文本
            "num_frames": num_frames,  # 存储希望的帧数
            "fps": fps,  # 存储帧率
        }

        yield item  # 生成处理后的项目


# 定义视频数据集类，继承自 MetaDistributedWebDataset
class VideoDataset(MetaDistributedWebDataset):
    # 初始化方法，构造类的实例
        def __init__(
            self,
            path,  # 数据路径
            image_size,  # 图像尺寸
            num_frames,  # 帧数
            fps,  # 每秒帧数
            skip_frms_num=0.0,  # 跳过的帧数，默认为0.0
            nshards=sys.maxsize,  # 分片数，默认为系统最大值
            seed=1,  # 随机种子，默认为1
            meta_names=None,  # 元数据名称，默认为None
            shuffle_buffer=1000,  # 打乱缓冲区大小，默认为1000
            include_dirs=None,  # 包含的目录，默认为None
            txt_key="caption",  # 文本键，默认为"caption"
            **kwargs,  # 其他额外参数
        ):
            # 如果种子为-1，则生成一个随机种子
            if seed == -1:
                seed = random.randint(0, 1000000)
            # 如果元数据名称为空，则设置为空列表
            if meta_names is None:
                meta_names = []
    
            # 如果路径以";"开头，则将其分割为路径和包含目录
            if path.startswith(";"):
                path, include_dirs = path.split(";", 1)
            # 调用父类的初始化方法，传递必要的参数
            super().__init__(
                path,  # 传递的数据路径
                partial(  # 使用偏函数包装处理视频的函数
                    process_fn_video, num_frames=num_frames, image_size=image_size, fps=fps, skip_frms_num=skip_frms_num
                ),
                seed,  # 随机种子
                meta_names=meta_names,  # 元数据名称
                shuffle_buffer=shuffle_buffer,  # 打乱缓冲区大小
                nshards=nshards,  # 分片数
                include_dirs=include_dirs,  # 包含的目录
            )
    
        # 类方法，用于创建数据集的实例
        @classmethod
        def create_dataset_function(cls, path, args, **kwargs):
            # 返回类的实例，使用给定的路径和其他参数
            return cls(path, **kwargs)
# 定义一个 SFTDataset 类，继承自 Dataset
class SFTDataset(Dataset):
    # 初始化方法，接收数据目录、视频尺寸、帧率、最大帧数和跳过的帧数
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3):
        """
        skip_frms_num: 忽略前面和后面的 xx 帧，避免过渡效果。
        """
        # 调用父类的初始化方法
        super(SFTDataset, self).__init__()

        # 设置视频尺寸
        self.video_size = video_size
        # 设置帧率
        self.fps = fps
        # 设置最大帧数
        self.max_num_frames = max_num_frames
        # 设置跳过的帧数
        self.skip_frms_num = skip_frms_num

        # 初始化视频路径列表
        self.video_paths = []
        # 初始化字幕列表
        self.captions = []

        # 遍历数据目录，获取所有视频文件
        for root, dirnames, filenames in os.walk(data_dir):
            # 遍历当前目录下的所有文件
            for filename in filenames:
                # 检查文件是否以 ".mp4" 结尾
                if filename.endswith(".mp4"):
                    # 获取视频文件的完整路径
                    video_path = os.path.join(root, filename)
                    # 将视频路径添加到列表中
                    self.video_paths.append(video_path)

                    # 构造对应的字幕文件路径
                    caption_path = video_path.replace(".mp4", ".txt").replace("videos", "labels")
                    # 检查字幕文件是否存在
                    if os.path.exists(caption_path):
                        # 如果存在，读取第一行作为字幕
                        caption = open(caption_path, "r").read().splitlines()[0]
                    else:
                        # 如果不存在，字幕为空字符串
                        caption = ""
                    # 将字幕添加到列表中
                    self.captions.append(caption)
    # 定义获取视频帧的方法，支持通过索引访问
    def __getitem__(self, index):
        # 设置桥接库为 Torch
        decord.bridge.set_bridge("torch")

        # 根据索引获取视频文件路径
        video_path = self.video_paths[index]
        # 创建视频读取器，设置高度和宽度为 -1 表示自动
        vr = VideoReader(uri=video_path, height=-1, width=-1)
        # 获取视频的实际帧率
        actual_fps = vr.get_avg_fps()
        # 获取视频的原始帧数
        ori_vlen = len(vr)

        # 如果视频帧数与目标帧率计算后超过最大帧数限制
        if ori_vlen / actual_fps * self.fps > self.max_num_frames:
            # 将帧数限制为最大帧数
            num_frames = self.max_num_frames
            # 计算起始帧
            start = int(self.skip_frms_num)
            # 计算结束帧
            end = int(start + num_frames / self.fps * actual_fps)
            # 确保结束帧不超过原始帧数
            end_safty = min(int(start + num_frames / self.fps * actual_fps), int(ori_vlen))
            # 生成采样索引
            indices = np.arange(start, end, (end - start) // num_frames).astype(int)
            # 从视频读取器中获取帧数据
            temp_frms = vr.get_batch(np.arange(start, end_safty))
            # 确保获取到的帧数据不为空
            assert temp_frms is not None
            # 将帧数据转换为张量，如果已经是张量则保持不变
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            # 根据采样索引选择相应的帧数据
            tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
        else:
            # 如果原始帧数大于最大帧数限制
            if ori_vlen > self.max_num_frames:
                # 将帧数限制为最大帧数
                num_frames = self.max_num_frames
                # 计算起始帧
                start = int(self.skip_frms_num)
                # 计算结束帧
                end = int(ori_vlen - self.skip_frms_num)
                # 生成采样索引
                indices = np.arange(start, end, max((end - start) // num_frames, 1)).astype(int)
                # 从视频读取器中获取全部帧数据
                temp_frms = vr.get_batch(np.arange(start, end))
                # 确保获取到的帧数据不为空
                assert temp_frms is not None
                # 将帧数据转换为张量，如果已经是张量则保持不变
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                # 根据采样索引选择相应的帧数据
                tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
            else:
                # 定义一个函数用于计算小于等于 n 的 4k+1 的最近值
                def nearest_smaller_4k_plus_1(n):
                    remainder = n % 4
                    if remainder == 0:
                        return n - 3  # 返回 n 减去 3
                    else:
                        return n - remainder + 1  # 返回 n 减去余数再加 1

                # 计算起始帧
                start = int(self.skip_frms_num)
                # 计算结束帧
                end = int(ori_vlen - self.skip_frms_num)
                # 根据函数获取符合条件的帧数
                num_frames = nearest_smaller_4k_plus_1(end - start)  # 3D VAE 需要帧数为 4k+1
                # 重新计算结束帧
                end = int(start + num_frames)
                # 从视频读取器中获取帧数据
                temp_frms = vr.get_batch(np.arange(start, end))
                # 确保获取到的帧数据不为空
                assert temp_frms is not None
                # 将帧数据转换为张量，如果已经是张量则保持不变
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms

        # 填充最后一帧以满足最大帧数要求
        tensor_frms = pad_last_frame(
            tensor_frms, self.max_num_frames
        )  # 因为索引的长度可能小于帧数，需处理四舍五入误差
        # 调整张量维度，从 [T, H, W, C] 转为 [T, C, H, W]
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        # 对张量进行中心矩形裁剪处理
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")
        # 归一化处理，将像素值转换到 [-1, 1]
        tensor_frms = (tensor_frms - 127.5) / 127.5

        # 构建返回的字典，包含处理后的帧数据、对应的文本和帧数及帧率
        item = {
            "mp4": tensor_frms,
            "txt": self.captions[index],
            "num_frames": num_frames,
            "fps": self.fps,
        }
        # 返回处理后的项
        return item

    # 定义获取视频路径数量的方法
    def __len__(self):
        # 返回视频路径的数量
        return len(self.video_paths)

    # 定义一个类方法
    @classmethod
    # 创建数据集的类方法，接收路径和其他参数
    def create_dataset_function(cls, path, args, **kwargs):
        # 实例化类，传入数据目录路径和额外参数
        return cls(data_dir=path, **kwargs)
```