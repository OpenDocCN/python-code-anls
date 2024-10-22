# `.\cogvideo-finetune\inference\gradio_composite_demo\utils.py`

```py
# 导入数学库
import math
# 从 typing 模块导入 Union 和 List 类型注解
from typing import Union, List

# 导入 PyTorch 库
import torch
# 导入操作系统相关功能
import os
# 导入日期和时间处理功能
from datetime import datetime
# 导入 NumPy 库
import numpy as np
# 导入 itertools 库
import itertools
# 导入图像处理库 PIL
import PIL.Image
# 导入 safetensors 库用于处理张量
import safetensors.torch
# 导入进度条显示库 tqdm
import tqdm
# 导入日志记录库
import logging
# 从 diffusers.utils 导入视频导出功能
from diffusers.utils import export_to_video
# 导入模型加载器
from spandrel import ModelLoader

# 创建一个记录器，命名为当前文件名
logger = logging.getLogger(__file__)


# 定义加载 PyTorch 文件的函数
def load_torch_file(ckpt, device=None, dtype=torch.float16):
    # 如果未指定设备，则默认使用 CPU
    if device is None:
        device = torch.device("cpu")
    # 检查文件扩展名，判断是否为 safetensors 格式
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        # 加载 safetensors 文件
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        # 检查当前 PyTorch 版本是否支持 weights_only 参数
        if not "weights_only" in torch.load.__code__.co_varnames:
            logger.warning(
                "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
            )

        # 加载普通 PyTorch 文件
        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        # 如果包含 global_step，记录调试信息
        if "global_step" in pl_sd:
            logger.debug(f"Global Step: {pl_sd['global_step']}")
        # 根据不同的键获取模型状态字典
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        elif "params_ema" in pl_sd:
            sd = pl_sd["params_ema"]
        else:
            sd = pl_sd

    # 将加载的张量转换为指定数据类型
    sd = {k: v.to(dtype) for k, v in sd.items()}
    # 返回状态字典
    return sd


# 定义替换状态字典前缀的函数
def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    # 如果需要过滤键，则初始化输出字典
    if filter_keys:
        out = {}
    else:
        out = state_dict
    # 遍历所有要替换的前缀
    for rp in replace_prefix:
        # 找到以指定前缀开头的所有键，并生成新键
        replace = list(
            map(
                lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])),
                filter(lambda a: a.startswith(rp), state_dict.keys()),
            )
        )
        # 遍历需要替换的键值对
        for x in replace:
            # 从状态字典中移除旧键，添加新键
            w = state_dict.pop(x[0])
            out[x[1]] = w
    # 返回更新后的字典
    return out


# 定义计算模块大小的函数
def module_size(module):
    module_mem = 0
    # 获取模块的状态字典
    sd = module.state_dict()
    # 遍历状态字典中的每个键
    for k in sd:
        t = sd[k]
        # 计算模块内所有张量的元素总数乘以元素大小，累加到模块内存大小
        module_mem += t.nelement() * t.element_size()
    # 返回模块的内存大小
    return module_mem


# 定义计算平铺缩放步骤的函数
def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    # 计算平铺所需的步骤数
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


# 使用无梯度模式定义平铺缩放多维函数
@torch.inference_mode()
def tiled_scale_multidim(
    samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", pbar=None
):
    # 获取平铺的维度数量
    dims = len(tile)
    # 打印样本的数据类型
    print(f"samples dtype:{samples.dtype}")
    # 初始化输出张量，形状为样本数量和调整后的通道数
    output = torch.empty(
        [samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:])),
        device=output_device,
    )
    # 遍历样本的每个元素
        for b in range(samples.shape[0]):
            # 获取当前样本的切片
            s = samples[b : b + 1]
            # 初始化输出张量，大小根据上采样比例计算
            out = torch.zeros(
                [s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])),
                device=output_device,
            )
            # 初始化输出分母张量，用于后续归一化
            out_div = torch.zeros(
                [s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])),
                device=output_device,
            )
    
            # 生成所有可能的切片位置
            for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(s.shape[2:], tile))):
                # 设置输入为当前样本
                s_in = s
                # 用于存储上采样的位置
                upscaled = []
    
                # 遍历每个维度
                for d in range(dims):
                    # 计算当前切片的位置，确保不越界
                    pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                    # 确定当前切片的长度
                    l = min(tile[d], s.shape[d + 2] - pos)
                    # 从样本中提取相应的切片
                    s_in = s_in.narrow(d + 2, pos, l)
                    # 记录上采样位置
                    upscaled.append(round(pos * upscale_amount))
    
                # 对输入进行处理，得到上采样的结果
                ps = function(s_in).to(output_device)
                # 创建与 ps 相同形状的全一掩码
                mask = torch.ones_like(ps)
                # 计算羽化的大小
                feather = round(overlap * upscale_amount)
                # 为每个维度应用羽化处理
                for t in range(feather):
                    for d in range(2, dims + 2):
                        # 处理掩码的前端羽化
                        m = mask.narrow(d, t, 1)
                        m *= (1.0 / feather) * (t + 1)
                        # 处理掩码的后端羽化
                        m = mask.narrow(d, mask.shape[d] - 1 - t, 1)
                        m *= (1.0 / feather) * (t + 1)
    
                # 定义输出张量
                o = out
                o_d = out_div
                # 将上采样结果添加到输出张量中
                for d in range(dims):
                    o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                    o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])
    
                # 更新输出张量和分母张量
                o += ps * mask
                o_d += mask
    
                # 如果有进度条，更新进度
                if pbar is not None:
                    pbar.update(1)
    
            # 将结果存回输出张量，进行归一化处理
            output[b : b + 1] = out / out_div
        # 返回最终输出
        return output
# 定义一个函数，用于对样本进行分块缩放
def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    pbar=None,
):
    # 调用 tiled_scale_multidim 函数，传递参数以执行缩放
    return tiled_scale_multidim(
        samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device, pbar
    )


# 定义一个函数，从检查点加载上采样模型
def load_sd_upscale(ckpt, inf_device):
    # 从指定设备加载模型权重文件
    sd = load_torch_file(ckpt, device=inf_device)
    # 检查权重字典中是否存在特定键
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        # 替换字典中键的前缀
        sd = state_dict_prefix_replace(sd, {"module.": ""})
    # 加载模型并将其转换为半精度
    out = ModelLoader().load_from_state_dict(sd).half()
    # 返回加载的模型
    return out


# 定义一个函数，用于对给定的张量进行上采样
def upscale(upscale_model, tensor: torch.Tensor, inf_device, output_device="cpu") -> torch.Tensor:
    # 计算上采样模型所需的内存
    memory_required = module_size(upscale_model.model)
    memory_required += (
        (512 * 512 * 3) * tensor.element_size() * max(upscale_model.scale, 1.0) * 384.0
    )  # 384.0 是模型内存占用的估算值，TODO: 需要更准确
    memory_required += tensor.nelement() * tensor.element_size()
    # 打印所需内存的大小
    print(f"UPScaleMemory required: {memory_required / 1024 / 1024 / 1024} GB")

    # 将上采样模型移至指定的设备
    upscale_model.to(inf_device)
    # 定义分块的大小和重叠量
    tile = 512
    overlap = 32

    # 计算总的处理步骤
    steps = tensor.shape[0] * get_tiled_scale_steps(
        tensor.shape[3], tensor.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
    )

    # 初始化进度条
    pbar = ProgressBar(steps, desc="Tiling and Upscaling")

    # 调用 tiled_scale 进行上采样
    s = tiled_scale(
        samples=tensor.to(torch.float16),
        function=lambda a: upscale_model(a),
        tile_x=tile,
        tile_y=tile,
        overlap=overlap,
        upscale_amount=upscale_model.scale,
        pbar=pbar,
    )

    # 将模型移回输出设备
    upscale_model.to(output_device)
    # 返回上采样后的结果
    return s


# 定义一个函数，用于对批量的潜变量进行上采样并拼接
def upscale_batch_and_concatenate(upscale_model, latents, inf_device, output_device="cpu") -> torch.Tensor:
    # 初始化一个空列表以存储上采样的潜变量
    upscaled_latents = []
    # 遍历每个潜变量
    for i in range(latents.size(0)):
        latent = latents[i]
        # 对当前潜变量进行上采样
        upscaled_latent = upscale(upscale_model, latent, inf_device, output_device)
        # 将上采样结果添加到列表中
        upscaled_latents.append(upscaled_latent)
    # 返回拼接后的张量
    return torch.stack(upscaled_latents)


# 定义一个函数，用于保存视频文件
def save_video(tensor: Union[List[np.ndarray], List[PIL.Image.Image]], fps: int = 8):
    # 获取当前时间戳并格式化为字符串
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 定义视频文件的保存路径
    video_path = f"./output/{timestamp}.mp4"
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # 将张量导出为视频文件
    export_to_video(tensor, video_path, fps=fps)
    # 返回视频文件路径
    return video_path


# 定义一个进度条类
class ProgressBar:
    def __init__(self, total, desc=None):
        # 初始化总步骤数和当前步骤数
        self.total = total
        self.current = 0
        # 创建进度条对象
        self.b_unit = tqdm.tqdm(total=total, desc="ProgressBar context index: 0" if desc is None else desc)

    # 更新进度条的方法
    def update(self, value):
        # 如果传入值超过总数，则设置为总数
        if value > self.total:
            value = self.total
        # 更新当前进度
        self.current = value
        # 刷新进度条显示
        if self.b_unit is not None:
            self.b_unit.set_description("ProgressBar context index: {}".format(self.current))
            self.b_unit.refresh()

            # 更新进度
            self.b_unit.update(self.current)
```