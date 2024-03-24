# `.\lucidrains\magvit2-pytorch\magvit2_pytorch\data.py`

```py
# 导入必要的库
from pathlib import Path
from functools import partial

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

import cv2
from PIL import Image
from torchvision import transforms as T, utils

from beartype import beartype
from beartype.typing import Tuple, List
from beartype.door import is_bearable

import numpy as np

from einops import rearrange

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 返回输入值
def identity(t, *args, **kwargs):
    return t

# 将输入值转换为元组
def pair(val):
    return val if isinstance(val, tuple) else (val, val)

# 在指定维度上填充张量
def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# 调整张量的帧数
def cast_num_frames(t, *, frames):
    f = t.shape[-3]

    if f == frames:
        return t

    if f > frames:
        return t[..., :frames, :, :]

    return pad_at_dim(t, (0, frames - f), dim = -3)

# 将图像转换为指定格式
def convert_image_to_fn(img_type, image):
    if not exists(img_type) or image.mode == img_type:
        return image

    return image.convert(img_type)

# 如果路径没有后缀，则添加后缀
def append_if_no_suffix(path: str, suffix: str):
    path = Path(path)

    if path.suffix == '':
        path = path.parent / (path.name + suffix)

    assert path.suffix == suffix, f'{str(path)} needs to have suffix {suffix}'

    return str(path)

# 通道到图像模式的映射
CHANNEL_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}

# 图像相关的辅助函数和数据集

# 图像数据集类
class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        convert_image_to = None,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        folder = Path(folder)
        assert folder.is_dir(), f'{str(folder)} must be a folder containing images'
        self.folder = folder

        self.image_size = image_size

        exts = exts + [ext.upper() for ext in exts]
        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        if exists(channels) and not exists(convert_image_to):
            convert_image_to = CHANNEL_TO_MODE.get(channels)

        self.transform = T.Compose([
            T.Lambda(partial(convert_image_to_fn, convert_image_to)),
            T.Resize(image_size, antialias = True),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# 张量的形状 (channels, frames, height, width) -> gif

# 处理读取和写入 gif

# 逐帧读取图像
def seek_all_images(img: Tensor, channels = 3):
    mode = CHANNEL_TO_MODE.get(channels)

    assert exists(mode), f'channels {channels} invalid'

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# 张量的形状 (channels, frames, height, width) -> gif

# 将视频张量转换为 gif
@beartype
def video_tensor_to_gif(
    tensor: Tensor,
    path: str,
    duration = 120,
    loop = 0,
    optimize = True
):
    path = append_if_no_suffix(path, '.gif')
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(str(path), save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> 张量 (channels, frame, height, width)

# 将 gif 转换为张量
def gif_to_tensor(
    path: str,
    channels = 3,
    transform = T.ToTensor()
):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

# 处理读取和写入 mp4

# 将视频转换为张量
def video_to_tensor(
    path: str,              # 视频文件的路径，需要导入的视频
    num_frames = -1,        # 要存储在输出张量中的帧数，默认为-1表示存储所有帧
    crop_size = None        # 裁剪尺寸，默认为None表示不进行裁剪
# 定义一个函数，将视频文件转换为张量
def video_to_tensor(path: str) -> Tensor:  # 返回形状为 (1, 通道数, 帧数, 高度, 宽度) 的张量

    # 使用 OpenCV 打开视频文件
    video = cv2.VideoCapture(path)

    frames = []  # 存储视频帧的列表
    check = True

    # 循环读取视频帧
    while check:
        check, frame = video.read()

        if not check:
            continue

        # 如果存在裁剪尺寸，则对帧进行中心裁剪
        if exists(crop_size):
            frame = crop_center(frame, *pair(crop_size))

        # 将帧重新排列为 (1, ...) 的形状并添加到 frames 列表中
        frames.append(rearrange(frame, '... -> 1 ...'))

    # 将帧列表转换为 numpy 数组，然后合并帧并转换为 numpy 数组
    frames = np.array(np.concatenate(frames[:-1], axis=0))
    frames = rearrange(frames, 'f h w c -> c f h w')

    # 将 numpy 数组转换为 PyTorch 张量并转换为浮点数类型
    frames_torch = torch.tensor(frames).float()

    # 将张量值归一化到 [0, 1] 范围
    frames_torch /= 255.
    # 将张量沿着第一个维度翻转，从 BGR 格式转换为 RGB 格式
    frames_torch = frames_torch.flip(dims=(0,))

    # 返回指定数量的帧数
    return frames_torch[:, :num_frames, :, :]

# 定义一个函数，将张量转换为视频文件
@beartype
def tensor_to_video(
    tensor: Tensor,        # PyTorch 视频张量
    path: str,             # 要保存的视频路径
    fps=25,                # 保存视频的帧率
    video_format='MP4V'    # 视频格式，默认为 MP4
):
    # 如果路径没有后缀，则添加 .mp4 后缀
    path = append_if_no_suffix(path, '.mp4')

    # 将张量移动到 CPU
    tensor = tensor.cpu()

    # 获取张量的帧数、高度和宽度
    num_frames, height, width = tensor.shape[-3:]

    # 使用指定的视频格式创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    video = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    frames = []  # 存储视频帧的列表

    # 遍历每一帧，将张量转换为 numpy 数组并写入视频
    for idx in range(num_frames):
        numpy_frame = tensor[:, idx, :, :].numpy()
        numpy_frame = np.uint8(rearrange(numpy_frame, 'c h w -> h w c'))
        video.write(numpy_frame)

    # 释放 VideoWriter 对象
    video.release()

    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()

    return video

# 定义一个函数，对图像进行中心裁剪
def crop_center(
    img: Tensor,  # 输入图像张���
    cropx: int,   # 最终图像在 x 方向上的长度
    cropy: int    # 最终图像在 y 方向上的长度
) -> Tensor:      # 返回裁剪后的图像张量
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:(starty + cropy), startx:(startx + cropx), :]

# 视频数据集类
class VideoDataset(Dataset):
    def __init__(
        self,
        folder,              # 视频文件夹路径
        image_size,          # 图像尺寸
        channels=3,          # 通道数，默认为 3
        num_frames=17,       # 帧数，默认为 17
        force_num_frames=True,  # 是否强制指定帧数，默认为 True
        exts=['gif', 'mp4']  # 视频文件扩展名列表，默认为 ['gif', 'mp4']
    ):
        super().__init__()
        folder = Path(folder)
        assert folder.is_dir(), f'{str(folder)} must be a folder containing videos'
        self.folder = folder

        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        # 定义图像转换操作
        self.transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size)
        ])

        # 定义将视频路径转换为张量的函数
        self.gif_to_tensor = partial(gif_to_tensor, channels=self.channels, transform=self.transform)
        self.mp4_to_tensor = partial(video_to_tensor, crop_size=self.image_size)

        # 定义将帧数转换为指定数量的函数
        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        ext = path.suffix
        path_str = str(path)

        if ext == '.gif':
            tensor = self.gif_to_tensor(path_str)
        elif ext == '.mp4':
            tensor = self.mp4_to_tensor(path_str)
            frames = tensor.unbind(dim=1)
            tensor = torch.stack([*map(self.transform, frames)], dim=1)
        else:
            raise ValueError(f'unknown extension {ext}')

        return self.cast_num_frames_fn(tensor)

# 重写数据加载器以能够整理张量和字符串
def collate_tensors_and_strings(data):
    if is_bearable(data, List[Tensor]):
        return (torch.stack(data),)

    data = zip(*data)
    output = []
    # 遍历数据列表中的每个元素
    for datum in data:
        # 检查数据是否为可接受的类型（元组中包含张量）
        if is_bearable(datum, Tuple[Tensor, ...]):
            # 如果是，则将张量堆叠成一个张量
            datum = torch.stack(datum)
        # 检查数据是否为可接受的类型（元组中包含字符串）
        elif is_bearable(datum, Tuple[str, ...]):
            # 如果是，则将元组转换为列表
            datum = list(datum)
        else:
            # 如果数据类型不符合要求，则引发值错误异常
            raise ValueError('detected invalid type being passed from dataset')

        # 将处理后的数据添加到输出列表中
        output.append(datum)

    # 将输出列表转换为元组并返回
    return tuple(output)
# 定义一个函数DataLoader，接受任意数量的位置参数和关键字参数
def DataLoader(*args, **kwargs):
    # 返回PytorchDataLoader对象，使用指定的参数和自定义的collate函数
    return PytorchDataLoader(*args, collate_fn = collate_tensors_and_strings, **kwargs)
```