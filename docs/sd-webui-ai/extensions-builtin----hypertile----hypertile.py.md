# `stable-diffusion-webui\extensions-builtin\hypertile\hypertile.py`

```py
"""
Hypertile module for splitting attention layers in SD-1.5 U-Net and SD-1.5 VAE
Warn: The patch works well only if the input image has a width and height that are multiples of 128
Original author: @tfernd Github: https://github.com/tfernd/HyperTile
"""

# 导入必要的库
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from functools import wraps, cache
import math
import torch.nn as nn
import random
from einops import rearrange

# 定义数据类 HypertileParams
@dataclass
class HypertileParams:
    depth = 0
    layer_name = ""
    tile_size: int = 0
    swap_size: int = 0
    aspect_ratio: float = 1.0
    forward = None
    enabled = False

# TODO add SD-XL layers
# 定义深度层次的字典
DEPTH_LAYERS = {
    0: [
        # SD 1.5 U-Net (diffusers)
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "input_blocks.1.1.transformer_blocks.0.attn1",
        "input_blocks.2.1.transformer_blocks.0.attn1",
        "output_blocks.9.1.transformer_blocks.0.attn1",
        "output_blocks.10.1.transformer_blocks.0.attn1",
        "output_blocks.11.1.transformer_blocks.0.attn1",
        # SD 1.5 VAE
        "decoder.mid_block.attentions.0",
        "decoder.mid.attn_1",
    ],
    1: [
        # SD 1.5 U-Net (diffusers) 第一组注意力机制
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1",  # 下采样块1的第一个注意力机制
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1",  # 下采样块1的第二个注意力机制
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1",    # 上采样块2的第一个注意力机制
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1",    # 上采样块2的第二个注意力机制
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1",    # 上采样块2的第三个注意力机制
        # SD 1.5 U-Net (ldm) 第一组注意力机制
        "input_blocks.4.1.transformer_blocks.0.attn1",            # 输入块4.1的第一个注意力机制
        "input_blocks.5.1.transformer_blocks.0.attn1",            # 输入块5.1的第一个注意力机制
        "output_blocks.6.1.transformer_blocks.0.attn1",           # 输出块6.1的第一个注意力机制
        "output_blocks.7.1.transformer_blocks.0.attn1",           # 输出块7.1的第一个注意力机制
        "output_blocks.8.1.transformer_blocks.0.attn1",           # 输出块8.1的第一个注意力机制
    ],
    2: [
        # SD 1.5 U-Net (diffusers) 第二组注意力机制
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1",  # 下采样块2的第一个注意力机制
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1",  # 下采样块2的第二个注意力机制
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1",    # 上采样块1的第一个注意力机制
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1",    # 上采样块1的第二个注意力机制
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1",    # 上采样块1的第三个注意力机制
        # SD 1.5 U-Net (ldm) 第二组注意力机制
        "input_blocks.7.1.transformer_blocks.0.attn1",            # 输入块7.1的第一个注意力机制
        "input_blocks.8.1.transformer_blocks.0.attn1",            # 输入块8.1的第一个注意力机制
        "output_blocks.3.1.transformer_blocks.0.attn1",           # 输出块3.1的第一个注意力机制
        "output_blocks.4.1.transformer_blocks.0.attn1",           # 输出块4.1的第一个注意力机制
        "output_blocks.5.1.transformer_blocks.0.attn1",           # 输出块5.1的第一个注意力机制
    ],
    3: [
        # SD 1.5 U-Net (diffusers) 第三组注意力机制
        "mid_block.attentions.0.transformer_blocks.0.attn1",      # 中间块的第一个注意力机制
        # SD 1.5 U-Net (ldm) 第三组注意力机制
        "middle_block.1.transformer_blocks.0.attn1",              # 中间块1的第一个注意力机制
    ],
# XL layers, thanks for GitHub@gel-crabs for the help
# 定义了深度为XL的层次结构，感谢GitHub上的gel-crabs提供帮助
DEPTH_LAYERS_XL = {
    0: [
        # SD 1.5 U-Net (diffusers)
        # SD 1.5 U-Net (扩散器)
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        # SD 1.5 U-Net (ldm)
        "input_blocks.4.1.transformer_blocks.0.attn1",
        "input_blocks.5.1.transformer_blocks.0.attn1",
        "output_blocks.3.1.transformer_blocks.0.attn1",
        "output_blocks.4.1.transformer_blocks.0.attn1",
        "output_blocks.5.1.transformer_blocks.0.attn1",
        # SD 1.5 VAE
        # SD 1.5 VAE
        "decoder.mid_block.attentions.0",
        "decoder.mid.attn_1",
    ],
    ],
    2: [
        # SD 1.5 U-Net (diffusers)
        # SD 1.5 U-Net (扩散器)
        "mid_block.attentions.0.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        # SD 1.5 U-Net (ldm)
        "middle_block.1.transformer_blocks.0.attn1",
        "middle_block.1.transformer_blocks.1.attn1",
        "middle_block.1.transformer_blocks.2.attn1",
        "middle_block.1.transformer_blocks.3.attn1",
        "middle_block.1.transformer_blocks.4.attn1",
        "middle_block.1.transformer_blocks.5.attn1",
        "middle_block.1.transformer_blocks.6.attn1",
        "middle_block.1.transformer_blocks.7.attn1",
        "middle_block.1.transformer_blocks.8.attn1",
        "middle_block.1.transformer_blocks.9.attn1",
    ],
    3 : [] # TODO - separate layers for SD-XL
    # 3: [] # 待办事项 - 为SD-XL分离层次
}

# 创建一个随机数生成器实例
RNG_INSTANCE = random.Random()

# 缓存装饰器，用于缓存函数的返回值
@cache
def get_divisors(value: int, min_value: int, /, max_options: int = 1) -> list[int]:
    """
    Returns divisors of value that
        x * min_value <= value
    in big -> small order, amount of divisors is limited by max_options
    """
    # 确保最大选项数至少为1
    max_options = max(1, max_options) # at least 1 option should be returned
    # 确保最小值不超过给定值
    min_value = min(min_value, value)
    # 生成一个列表，包含从最小值到给定值范围内所有能整除给定值的数，按照从小到大的顺序排列
    divisors = [i for i in range(min_value, value + 1) if value % i == 0] # divisors in small -> big order
    # 生成一个列表，包含给定值除以divisors列表中的每个元素的商，取前max_options个元素，按照从大到小的顺序排列
    ns = [value // i for i in divisors[:max_options]]  # has at least 1 element # big -> small order
    # 返回ns列表
    return ns
# 返回值 value 的随机因子，满足 x * min_value <= value
def random_divisor(value: int, min_value: int, /, max_options: int = 1) -> int:
    # 获取 value 的因子列表，最小值为 min_value，最大选项数为 max_options
    ns = get_divisors(value, min_value, max_options=max_options) # get cached divisors
    # 生成一个随机索引
    idx = RNG_INSTANCE.randint(0, len(ns) - 1)

    return ns[idx]


# 设置随机数生成器的种子
def set_hypertile_seed(seed: int) -> None:
    RNG_INSTANCE.seed(seed)


# 计算给定宽度和高度的最大瓦片大小
@cache
def largest_tile_size_available(width: int, height: int) -> int:
    # 计算宽度和高度的最大公约数
    gcd = math.gcd(width, height)
    largest_tile_size_available = 1
    # 计算最大瓦片大小，始终为 2 的幂
    while gcd % (largest_tile_size_available * 2) == 0:
        largest_tile_size_available *= 2
    return largest_tile_size_available


# 找到满足 h*w = hw 和 h/w = aspect_ratio 的 h 和 w
def iterative_closest_divisors(hw:int, aspect_ratio:float) -> tuple[int, int]:
    # 获取 hw 的所有因子
    divisors = [i for i in range(2, hw + 1) if hw % i == 0] # all divisors of hw
    # 获取所有因子对
    pairs = [(i, hw // i) for i in divisors] # all pairs of divisors of hw
    # 计算所有因子对的比率
    ratios = [w/h for h, w in pairs] # all ratios of pairs of divisors of hw
    # 找到最接近 aspect_ratio 的比率
    closest_ratio = min(ratios, key=lambda x: abs(x - aspect_ratio)) # closest ratio to aspect_ratio
    # 找到最接近 aspect_ratio 的因子对
    closest_pair = pairs[ratios.index(closest_ratio)] # closest pair of divisors to aspect_ratio
    return closest_pair


# 找到满足 h*w = hw 和 h/w = aspect_ratio 的 h 和 w
@cache
def find_hw_candidates(hw:int, aspect_ratio:float) -> tuple[int, int]:
    # 计算满足条件 h*w = hw 和 h/w = aspect_ratio 的 h 和 w
    h, w = round(math.sqrt(hw * aspect_ratio)), round(math.sqrt(hw / aspect_ratio))
    # find h and w such that h*w = hw and h/w = aspect_ratio
    # 如果给定的高度和宽度乘积不等于目标面积
    if h * w != hw:
        # 计算宽度的候选值
        w_candidate = hw / h
        # 检查宽度是否为整数
        if not w_candidate.is_integer():
            # 计算高度的候选值
            h_candidate = hw / w
            # 检查高度是否为整数
            if not h_candidate.is_integer():
                # 如果高度和宽度都不是整数，则调用递归函数 iterative_closest_divisors
                return iterative_closest_divisors(hw, aspect_ratio)
            else:
                # 如果高度是整数，则更新高度值
                h = int(h_candidate)
        else:
            # 如果宽度是整数，则更新宽度值
            w = int(w_candidate)
    # 返回更新后的高度和宽度
    return h, w
# 定义一个装饰器函数，用于实现自注意力机制的前向传播
def self_attn_forward(params: HypertileParams, scale_depth=True) -> Callable:

    @wraps(params.forward)
    def wrapper(*args, **kwargs):
        # 如果自注意力机制未启用，则直接调用原始的前向传播函数
        if not params.enabled:
            return params.forward(*args, **kwargs)

        # 计算潜在瓦片大小，取最大值为128或params.tile_size的八分之一
        latent_tile_size = max(128, params.tile_size) // 8
        x = args[0]

        # 如果输入数据维度为4，表示为VAE模型
        if x.ndim == 4:
            b, c, h, w = x.shape

            # 随机选择h和w的因子，使其能够被latent_tile_size整除
            nh = random_divisor(h, latent_tile_size, params.swap_size)
            nw = random_divisor(w, latent_tile_size, params.swap_size)

            # 如果nh * nw大于1，则将输入数据x重新排列成(b nh nw) c h w的形式，分割成nh * nw个瓦片
            if nh * nw > 1:
                x = rearrange(x, "b c (nh h) (nw w) -> (b nh nw) c h w", nh=nh, nw=nw)  # split into nh * nw tiles

            # 调用params.forward函数进行前向传播
            out = params.forward(x, *args[1:], **kwargs)

            # 如果nh * nw大于1，则将输出数据out重新排列成(b nh nw) c h w的形式
            if nh * nw > 1:
                out = rearrange(out, "(b nh nw) c h w -> b c (nh h) (nw w)", nh=nh, nw=nw)

        # 如果输入数据维度不为4，表示为U-Net模型
        else:
            hw: int = x.size(1)
            h, w = find_hw_candidates(hw, params.aspect_ratio)
            assert h * w == hw, f"Invalid aspect ratio {params.aspect_ratio} for input of shape {x.shape}, hw={hw}, h={h}, w={w}"

            # 根据深度缩放因子计算nh和nw，使其能够被latent_tile_size * factor整除
            factor = 2 ** params.depth if scale_depth else 1
            nh = random_divisor(h, latent_tile_size * factor, params.swap_size)
            nw = random_divisor(w, latent_tile_size * factor, params.swap_size)

            # 如果nh * nw大于1，则将输入数据x重新排列成(b nh nw) (h w) c的形式
            if nh * nw > 1:
                x = rearrange(x, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)

            # 调用params.forward函数进行前向传播
            out = params.forward(x, *args[1:], **kwargs)

            # 如果nh * nw大于1，则将输出数据out重新排列成b nh nw hw c的形式，再将其转换为b (nh h nw w) c的形式
            if nh * nw > 1:
                out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)

        return out

    return wrapper

# 定义一个函数，用于对模型进行超瓦片处理
def hypertile_hook_model(model: nn.Module, width, height, *, enable=False, tile_size_max=128, swap_size=1, max_depth=3, is_sdxl=False):
    # 获取模型中的超瓦片层信息
    hypertile_layers = getattr(model, "__webui_hypertile_layers", None)
    # 如果未提供 hypertile_layers 参数，则进行以下操作
    if hypertile_layers is None:
        # 如果未启用超级瓦片，则直接返回
        if not enable:
            return

        # 初始化 hypertile_layers 字典
        hypertile_layers = {}
        # 根据是否为 SDXL 模型选择不同的深度层次
        layers = DEPTH_LAYERS_XL if is_sdxl else DEPTH_LAYERS

        # 遍历深度范围为 0 到 3
        for depth in range(4):
            # 遍历模型中的每个层和模块
            for layer_name, module in model.named_modules():
                # 如果层名以指定深度的任一名称结尾，则执行以下操作
                if any(layer_name.endswith(try_name) for try_name in layers[depth]):
                    # 创建 HypertileParams 对象
                    params = HypertileParams()
                    # 将 params 附加到模块的 __webui_hypertile_params 属性上
                    module.__webui_hypertile_params = params
                    # 保存模块的原始 forward 方法，并将新的 self_attn_forward 方法赋给模块的 forward 方法
                    params.forward = module.forward
                    params.depth = depth
                    params.layer_name = layer_name
                    module.forward = self_attn_forward(params)

                    # 将层名添加到 hypertile_layers 字典中
                    hypertile_layers[layer_name] = 1

        # 将 hypertile_layers 字典赋给模型的 __webui_hypertile_layers 属性
        model.__webui_hypertile_layers = hypertile_layers

    # 计算宽高比
    aspect_ratio = width / height
    # 计算瓦片大小，取最小值
    tile_size = min(largest_tile_size_available(width, height), tile_size_max)

    # 遍历模型中的每个层和模块
    for layer_name, module in model.named_modules():
        # 如果层名在 hypertile_layers 字典中
        if layer_name in hypertile_layers:
            # 获取模块的 HypertileParams 对象
            params = module.__webui_hypertile_params

            # 设置瓦片大小、交换大小、宽高比和启用状态
            params.tile_size = tile_size
            params.swap_size = swap_size
            params.aspect_ratio = aspect_ratio
            params.enabled = enable and params.depth <= max_depth
```