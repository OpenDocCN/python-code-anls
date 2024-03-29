# `.\lucidrains\phenaki-pytorch\phenaki_pytorch\data.py`

```py
# 导入所需的库
from pathlib import Path
import cv2
from PIL import Image
from functools import partial
from typing import Tuple, List
from beartype.door import is_bearable
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader
from torchvision import transforms as T, utils
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

# 调整帧数
def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f == frames:
        return t
    if f > frames:
        return t[:, :frames]
    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

# 将图像转换为指定格式
def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# 图像相关的辅助函数和数据集

# 图像数据集类
class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
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

# 处理读取和写入 GIF

# 通道数对应的图像模式
CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

# 读取 GIF 中的所有图像
def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# 将视频张量转换为 GIF
def video_tensor_to_gif(
    tensor,
    path,
    duration = 120,
    loop = 0,
    optimize = True
):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# GIF 转换为张量
def gif_to_tensor(
    path,
    channels = 3,
    transform = T.ToTensor()
):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

# 处理读取和写入 MP4

# 将视频转换为张量
def video_to_tensor(
    path: str,              # 要导入的视频路径
    num_frames = -1,        # 要存储在输出张量中的帧数
    crop_size = None
) -> torch.Tensor:          # 形状为 (1, 通道数, 帧数, 高度, 宽度)

    video = cv2.VideoCapture(path)

    frames = []
    check = True

    while check:
        check, frame = video.read()

        if not check:
            continue

        if exists(crop_size):
            frame = crop_center(frame, *pair(crop_size))

        frames.append(rearrange(frame, '... -> 1 ...'))

    frames = np.array(np.concatenate(frames[:-1], axis = 0))  # 将帧列表转换为 numpy 数组
    frames = rearrange(frames, 'f h w c -> c f h w')

    frames_torch = torch.tensor(frames).float()

    return frames_torch[:, :num_frames, :, :]

# 将张量转换为视频
def tensor_to_video(
    tensor,                # Pytorch 视频张量
    path: str,             # 要保存的视频路径
    fps = 25,              # 保存视频的帧率
    # 定义视频格式为 MP4V
    video_format = 'MP4V'
# Import the video and cut it into frames.
def read_zip(fname):
    # 将张量移回 CPU
    tensor = tensor.cpu()

    # 获取张量的帧数、高度和宽度
    num_frames, height, width = tensor.shape[-3:]

    # 使用指定的视频格式创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*video_format) # Changes in this line can allow for different video formats.
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))

    frames = []

    # 遍历每一帧，将张量转换为 numpy 数组并写入视频
    for idx in range(num_frames):
        numpy_frame = tensor[:, idx, :, :].numpy()
        numpy_frame = np.uint8(rearrange(numpy_frame, 'c h w -> h w c'))
        video.write(numpy_frame)

    # 释放视频对象
    video.release()

    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()

    # 返回视频对象
    return video

# 将图像中心裁剪为指定大小
def crop_center(
    img,        # tensor
    cropx,      # Length of the final image in the x direction.
    cropy       # Length of the final image in the y direction.
) -> torch.Tensor:
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:(starty + cropy), startx:(startx + cropx), :]

# 视频数据集类
class VideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 17,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif', 'mp4']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # 定义数据转换流程
        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

        # 定义将视频路径转换为张量的函数
        self.gif_to_tensor = partial(gif_to_tensor, channels = self.channels, transform = self.transform)
        self.mp4_to_tensor = partial(video_to_tensor, crop_size = self.image_size)

        # 定义将帧数转换为指定数量的函数
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        ext = path.suffix

        # 根据文���扩展名选择相应的处理方式
        if ext == '.gif':
            tensor = self.gif_to_tensor(path)
        elif ext == '.mp4':
            tensor = self.mp4_to_tensor(str(path))
        else:
            raise ValueError(f'unknown extension {ext}')

        # 转换帧数并返回张量
        return self.cast_num_frames_fn(tensor)

# 重写数据加载器以能够整理字符串
def collate_tensors_and_strings(data):
    if is_bearable(data, List[torch.Tensor]):
        return (torch.stack(data, dim = 0),)

    data = zip(*data)
    output = []

    for datum in data:
        if is_bearable(datum, Tuple[torch.Tensor, ...]):
            datum = torch.stack(datum, dim = 0)
        elif is_bearable(datum, Tuple[str, ...]):
            datum = list(datum)
        else:
            raise ValueError('detected invalid type being passed from dataset')

        output.append(datum)

    return tuple(output)

# 创建数据加载器
def DataLoader(*args, **kwargs):
    return PytorchDataLoader(*args, collate_fn = collate_tensors_and_strings, **kwargs)
```