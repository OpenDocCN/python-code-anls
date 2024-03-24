# `.\lucidrains\stylegan2-pytorch\stylegan2_pytorch\diff_augment.py`

```
# 导入必要的库
from functools import partial
import random
import torch
import torch.nn.functional as F

# 定义一个函数，用于对输入进行不同类型的数据增强
def DiffAugment(x, types=[]):
    # 遍历每种数据增强类型
    for p in types:
        # 遍历每种数据增强函数
        for f in AUGMENT_FNS[p]:
            # 对输入数据进行数据增强操作
            x = f(x)
    # 返回处理后的数据
    return x.contiguous()

# 定义不同的数据增强函数

# 亮度随机增强函数
def rand_brightness(x, scale):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * scale
    return x

# 饱和度随机增强函数
def rand_saturation(x, scale):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (((torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * 2.0 * scale) + 1.0) + x_mean
    return x

# 对比度随机增强函数
def rand_contrast(x, scale):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (((torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * 2.0 * scale) + 1.0) + x_mean
    return x

# 随机平移增强函数
def rand_translation(x, ratio=0.125):
    # 计算平移的像素数
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    # 生成随机的平移量
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    # 创建平移后的图像
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

# 随机偏移增强函数
def rand_offset(x, ratio=1, ratio_h=1, ratio_v=1):
    w, h = x.size(2), x.size(3)

    imgs = []
    for img in x.unbind(dim = 0):
        max_h = int(w * ratio * ratio_h)
        max_v = int(h * ratio * ratio_v)

        value_h = random.randint(0, max_h) * 2 - max_h
        value_v = random.randint(0, max_v) * 2 - max_v

        if abs(value_h) > 0:
            img = torch.roll(img, value_h, 2)

        if abs(value_v) > 0:
            img = torch.roll(img, value_v, 1)

        imgs.append(img)

    return torch.stack(imgs)

# 水平偏移增强函数
def rand_offset_h(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=ratio, ratio_v=0)

# 垂直偏移增强函数
def rand_offset_v(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=0, ratio_v=ratio)

# 随机遮挡增强函数
def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

# 定义不同数据增强类型对应的数据增强函数
AUGMENT_FNS = {
    'brightness': [partial(rand_brightness, scale=1.)],
    'lightbrightness': [partial(rand_brightness, scale=.65)],
    'contrast':  [partial(rand_contrast, scale=.5)],
    'lightcontrast':  [partial(rand_contrast, scale=.25)],
    'saturation': [partial(rand_saturation, scale=1.)],
    'lightsaturation': [partial(rand_saturation, scale=.5)],
    'color': [partial(rand_brightness, scale=1.), partial(rand_saturation, scale=1.), partial(rand_contrast, scale=0.5)],
}
    # 'lightcolor'键对应的值是一个包含三个函数的列表，分别用于随机调整亮度、饱和度和对比度
    'lightcolor': [partial(rand_brightness, scale=0.65), partial(rand_saturation, scale=.5), partial(rand_contrast, scale=0.5)],
    # 'offset'键对应的值是一个包含一个函数的列表，用于生成随机偏移量
    'offset': [rand_offset],
    # 'offset_h'键对应的值是一个包含一个函数的列表，用于生成水平方向的随机偏移量
    'offset_h': [rand_offset_h],
    # 'offset_v'键对应的值是一个包含一个函数的列表，用于生成垂直方向的随机偏移量
    'offset_v': [rand_offset_v],
    # 'translation'键对应的值是一个包含一个函数的列表，用于生成随机平移
    'translation': [rand_translation],
    # 'cutout'键对应的值是一个包含一个函数的列表，用于生成随机遮挡
    'cutout': [rand_cutout],
# 闭合大括号，表示代码块的结束
```