# `.\lucidrains\nuwa-pytorch\nuwa_pytorch\image_utils.py`

```
# 导入 torch 库
import torch
# 导入 torchvision.transforms 库并重命名为 T
import torchvision.transforms as T
# 从 PIL 库中导入 Image 类
from PIL import Image

# 定义常量

# 通道数到模式的映射关系
CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

# 遍历所有图像帧
def seek_all_images(img, channels = 3):
    # 检查通道数是否在 CHANNELS_TO_MODE 中
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    # 获取对应通道数的图像模式
    mode = CHANNELS_TO_MODE[channels]

    # 初始化帧数为 0
    i = 0
    # 循环直到遇到异常
    while True:
        try:
            # 尝试定位到第 i 帧
            img.seek(i)
            # 将图像转换为指定模式
            yield img.convert(mode)
        except EOFError:
            # 遇到文件结尾异常时退出循环
            break
        # 帧数加一
        i += 1

# 将张量转换为 GIF 图像
def video_tensor_to_gif(tensor, path, duration = 80, loop = 0, optimize = True):
    # 将张量中的每一帧转换为 PIL 图像
    images = map(T.ToPILImage(), tensor.unbind(0))
    # 获取第一帧图像和剩余图像
    first_img, *rest_imgs = images
    # 保存 GIF 图像到指定路径
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    # 返回图像列表
    return images

# 将 GIF 图像转换为张量 (帧数, 通道数, 高度, 宽度)
def gif_to_tensor(path, channels = 3):
    # 打开 GIF 图像
    img = Image.open(path)
    # 获取图像中的每一帧并转换为张量
    tensors = tuple(map(T.ToTensor(), seek_all_images(img, channels = channels)))
    # 沿着第 0 维度堆叠张量
    return torch.stack(tensors, dim = 0)
```