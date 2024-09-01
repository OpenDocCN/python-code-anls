# `.\flux\src\flux\sampling.py`

```py
# 导入数学库
import math
# 导入 Callable 类型
from typing import Callable

# 导入 PyTorch 库
import torch
# 从 einops 导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 torch 导入 Tensor 类型
from torch import Tensor

# 从 model 模块导入 Flux 类
from .model import Flux
# 从 modules.conditioner 模块导入 HFEmbedder 类
from .modules.conditioner import HFEmbedder


# 生成噪声的函数
def get_noise(
    num_samples: int,  # 生成的样本数量
    height: int,  # 高度
    width: int,  # 宽度
    device: torch.device,  # 计算设备
    dtype: torch.dtype,  # 数据类型
    seed: int,  # 随机种子
):
    return torch.randn(
        num_samples,  # 样本数量
        16,  # 通道数
        # 允许打包的高度和宽度
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,  # 指定设备
        dtype=dtype,  # 指定数据类型
        generator=torch.Generator(device=device).manual_seed(seed),  # 使用指定种子初始化随机生成器
    )


# 准备数据的函数
def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape  # 获取批量大小、通道数、高度和宽度
    if bs == 1 and not isinstance(prompt, str):  # 如果批量大小为1且提示不是字符串
        bs = len(prompt)  # 设置批量大小为提示列表的长度

    # 调整图像形状以适应后续处理
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:  # 如果批量大小为1且实际批量大于1
        img = repeat(img, "1 ... -> bs ...", bs=bs)  # 复制图像以适应批量大小

    img_ids = torch.zeros(h // 2, w // 2, 3)  # 创建图像ID的零张量
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]  # 设置行ID
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]  # 设置列ID
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)  # 将ID张量重复以适应批量大小

    if isinstance(prompt, str):  # 如果提示是字符串
        prompt = [prompt]  # 将提示转换为列表
    txt = t5(prompt)  # 使用 t5 模型处理文本提示
    if txt.shape[0] == 1 and bs > 1:  # 如果文本的批量大小为1且实际批量大于1
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)  # 复制文本以适应批量大小
    txt_ids = torch.zeros(bs, txt.shape[1], 3)  # 创建文本ID的零张量

    vec = clip(prompt)  # 使用 clip 模型处理文本提示
    if vec.shape[0] == 1 and bs > 1:  # 如果向量的批量大小为1且实际批量大于1
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)  # 复制向量以适应批量大小

    return {
        "img": img,  # 返回处理后的图像
        "img_ids": img_ids.to(img.device),  # 返回图像ID，转移到图像所在设备
        "txt": txt.to(img.device),  # 返回处理后的文本，转移到图像所在设备
        "txt_ids": txt_ids.to(img.device),  # 返回文本ID，转移到图像所在设备
        "vec": vec.to(img.device),  # 返回处理后的向量，转移到图像所在设备
    }


# 计算时间移位的函数
def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)  # 计算时间移位值


# 获取线性函数的函数
def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15  # 默认参数值
) -> Callable[[float], float]:  # 返回一个接受浮点数并返回浮点数的函数
    m = (y2 - y1) / (x2 - x1)  # 计算线性函数的斜率
    b = y1 - m * x1  # 计算线性函数的截距
    return lambda x: m * x + b  # 返回线性函数


# 获取调度时间的函数
def get_schedule(
    num_steps: int,  # 步骤数量
    image_seq_len: int,  # 图像序列长度
    base_shift: float = 0.5,  # 基础偏移量
    max_shift: float = 1.15,  # 最大偏移量
    shift: bool = True,  # 是否应用偏移
) -> list[float]:  # 返回浮点数列表
    # 生成从1到0的时间步长
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # 如果启用了偏移
    if shift:
        # 基于线性估算估计 mu
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)  # 应用时间移位

    return timesteps.tolist()  # 返回时间步长的列表


# 去噪函数
def denoise(
    model: Flux,  # 模型
    # 模型输入
    img: Tensor,  # 输入图像
    img_ids: Tensor,  # 图像ID
    txt: Tensor,  # 处理后的文本
    txt_ids: Tensor,  # 文本ID
    vec: Tensor,  # 处理后的向量
    # 采样参数
    timesteps: list[float],  # 时间步长
    guidance: float = 4.0,  # 引导强度
):
    # 为每个图像创建引导向量
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    # 遍历当前时间步和前一个时间步的配对
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        # 创建一个张量 t_vec，其形状与 img 的第一个维度相同，值为 t_curr，数据类型和设备与 img 相同
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        # 使用当前时间步 t_vec 及其他参数调用模型，获得预测结果 pred
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        # 更新 img，增加预测结果 pred 和时间步差 (t_prev - t_curr) 的乘积
        img = img + (t_prev - t_curr) * pred

    # 返回更新后的 img
    return img
# 定义一个函数，用于对 Tensor 进行重排列，调整维度
def unpack(x: Tensor, height: int, width: int) -> Tensor:
    # 使用 rearrange 函数重排列 Tensor 的维度
    return rearrange(
        x,
        # 指定输入维度和输出维度的转换规则
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        # 根据输入的 height 和 width 计算重排列后的维度
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
```