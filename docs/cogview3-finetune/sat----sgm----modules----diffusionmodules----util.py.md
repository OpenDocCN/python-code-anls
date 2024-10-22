# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\util.py`

```
"""
引用自
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
和
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
和
https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py

感谢！
"""

# 导入数学库以执行数学运算
import math
# 从 typing 模块导入可选类型
from typing import Optional

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从 einops 库导入 rearrange 和 repeat 函数，用于张量操作
from einops import rearrange, repeat


# 创建一个 beta 调度函数
def make_beta_schedule(
    schedule,  # 调度类型
    n_timestep,  # 时间步数
    linear_start=1e-4,  # 线性调度的起始值
    linear_end=2e-2,  # 线性调度的结束值
):
    # 如果调度类型为线性
    if schedule == "linear":
        # 生成从起始值到结束值的线性空间并平方
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )
    # 返回生成的 beta 值作为 NumPy 数组
    return betas.numpy()


# 将张量中的值提取到指定形状的输出张量中
def extract_into_tensor(a, t, x_shape):
    # 解构 t 的形状，获取批大小
    b, *_ = t.shape
    # 从 a 中根据 t 的索引提取值
    out = a.gather(-1, t)
    # 将输出调整为指定的形状
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# 混合检查点功能的定义
def mixed_checkpoint(func, inputs: dict, params, flag):
    """
    评估函数而不缓存中间激活，从而减少内存使用，但在反向传播中增加计算量。
    与原始的检查点函数不同，它也可以处理非张量输入。
    :param func: 要评估的函数。
    :param inputs: 传递给 `func` 的参数字典。
    :param params: `func` 依赖但不明确作为参数传递的参数序列。
    :param flag: 如果为 False，则禁用梯度检查点。
    """
    # 如果标志为真
    if flag:
        # 获取所有张量键
        tensor_keys = [key for key in inputs if isinstance(inputs[key], torch.Tensor)]
        # 获取所有张量输入
        tensor_inputs = [
            inputs[key] for key in inputs if isinstance(inputs[key], torch.Tensor)
        ]
        # 获取所有非张量键
        non_tensor_keys = [
            key for key in inputs if not isinstance(inputs[key], torch.Tensor)
        ]
        # 获取所有非张量输入
        non_tensor_inputs = [
            inputs[key] for key in inputs if not isinstance(inputs[key], torch.Tensor)
        ]
        # 将所有输入和参数组合成元组
        args = tuple(tensor_inputs) + tuple(non_tensor_inputs) + tuple(params)
        # 调用混合检查点功能并返回结果
        return MixedCheckpointFunction.apply(
            func,
            len(tensor_inputs),
            len(non_tensor_inputs),
            tensor_keys,
            non_tensor_keys,
            *args,
        )
    else:
        # 直接调用函数并返回结果
        return func(**inputs)


# 定义混合检查点功能的类
class MixedCheckpointFunction(torch.autograd.Function):
    @staticmethod
    # 定义前向传播方法
    def forward(
        ctx,  # 上下文对象
        run_function,  # 要运行的函数
        length_tensors,  # 张量的长度
        length_non_tensors,  # 非张量的长度
        tensor_keys,  # 张量的键
        non_tensor_keys,  # 非张量的键
        *args,  # 其他参数
    ):
        # 设置结束张量的数量
        ctx.end_tensors = length_tensors
        # 设置结束非张量的数量
        ctx.end_non_tensors = length_tensors + length_non_tensors
        # 初始化 GPU 自动混合精度参数
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),  # 检查自动混合精度是否启用
            "dtype": torch.get_autocast_gpu_dtype(),  # 获取当前自动混合精度数据类型
            "cache_enabled": torch.is_autocast_cache_enabled(),  # 检查缓存是否启用
        }
        # 确保张量键和非张量键的数量与传入长度一致
        assert (
            len(tensor_keys) == length_tensors
            and len(non_tensor_keys) == length_non_tensors
        )

        # 将输入张量映射到字典，键为 tensor_keys，值为相应的 args
        ctx.input_tensors = {
            key: val for (key, val) in zip(tensor_keys, list(args[: ctx.end_tensors]))
        }
        # 将输入非张量映射到字典，键为 non_tensor_keys，值为相应的 args
        ctx.input_non_tensors = {
            key: val
            for (key, val) in zip(
                non_tensor_keys, list(args[ctx.end_tensors : ctx.end_non_tensors])
            )
        }
        # 保存运行函数
        ctx.run_function = run_function
        # 获取剩余输入参数
        ctx.input_params = list(args[ctx.end_non_tensors :])

        # 在不计算梯度的上下文中运行
        with torch.no_grad():
            # 调用运行函数并传入输入张量和非张量
            output_tensors = ctx.run_function(
                **ctx.input_tensors, **ctx.input_non_tensors
            )
        # 返回输出张量
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # 将输入张量中的所有张量设为需要梯度
        ctx.input_tensors = {
            key: ctx.input_tensors[key].detach().requires_grad_(True)
            for key in ctx.input_tensors
        }

        # 启用梯度计算并设置自动混合精度上下文
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # 创建输入张量的浅拷贝以避免原地修改
            shallow_copies = {
                key: ctx.input_tensors[key].view_as(ctx.input_tensors[key])
                for key in ctx.input_tensors
            }
            # shallow_copies.update(additional_args)
            # 调用运行函数并传入浅拷贝和非张量输入
            output_tensors = ctx.run_function(**shallow_copies, **ctx.input_non_tensors)
        # 计算输出张量相对于输入张量和参数的梯度
        input_grads = torch.autograd.grad(
            output_tensors,
            list(ctx.input_tensors.values()) + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        # 删除上下文中的输入张量和参数
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        # 返回梯度和填充的 None 值
        return (
            (None, None, None, None, None)
            + input_grads[: ctx.end_tensors]
            + (None,) * (ctx.end_non_tensors - ctx.end_tensors)
            + input_grads[ctx.end_tensors :]
        )
# 定义一个检查点函数，用于评估给定的函数，降低内存使用，同时增加计算开销
def checkpoint(func, inputs, params, flag):
    """
    评估一个函数，避免缓存中间激活，减少内存使用，但在反向传播时增加计算量。
    :param func: 要评估的函数。
    :param inputs: 传递给 `func` 的参数序列。
    :param params: `func` 依赖的参数序列，但并不显式作为参数接受。
    :param flag: 如果为 False，禁用梯度检查点。
    """
    # 如果 flag 为 True，启用梯度检查点
    if flag:
        # 将输入参数和依赖参数组合成一个元组
        args = tuple(inputs) + tuple(params)
        # 应用检查点函数，返回计算结果
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        # 如果 flag 为 False，直接调用 func 函数
        return func(*inputs)


# 定义检查点函数的类，继承自 torch.autograd.Function
class CheckpointFunction(torch.autograd.Function):
    # 定义前向传播的静态方法
    @staticmethod
    def forward(ctx, run_function, length, *args):
        # 将运行的函数保存到上下文中
        ctx.run_function = run_function
        # 保存输入张量（前 length 个参数）
        ctx.input_tensors = list(args[:length])
        # 保存输入参数（后面的参数）
        ctx.input_params = list(args[length:])
        # 获取当前的 GPU 自动混合精度设置
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        # 在不计算梯度的情况下运行函数
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        # 返回输出张量
        return output_tensors

    # 定义反向传播的静态方法
    @staticmethod
    def backward(ctx, *output_grads):
        # 将输入张量分离，并设置为需要梯度
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        # 启用梯度计算并设置自动混合精度
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # 解决一个 bug，确保运行函数的第一个操作不会修改分离的张量存储
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            # 使用浅拷贝运行函数以获取输出张量
            output_tensors = ctx.run_function(*shallow_copies)
        # 计算输入梯度
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        # 删除上下文中的输入张量
        del ctx.input_tensors
        # 删除上下文中的输入参数
        del ctx.input_params
        # 删除输出张量
        del output_tensors
        # 返回 None 和输入梯度
        return (None, None) + input_grads


# 定义时间步嵌入的函数
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False, dtype=torch.float32):
    """
    创建正弦时间步嵌入。
    :param timesteps: 一个 1-D 张量，包含每个批次元素的 N 个索引。
                      这些索引可以是小数。
    :param dim: 输出的维度。
    :param max_period: 控制嵌入的最小频率。
    :return: 一个形状为 [N x dim] 的位置嵌入张量。
    """
    # 如果不只是重复时序
        if not repeat_only:
            # 计算半个维度，用于频率计算
            half = dim // 2
            # 生成频率数组，基于最大周期和半个维度
            freqs = torch.exp(
                -math.log(max_period)  # 计算最大周期的对数
                * torch.arange(start=0, end=half, dtype=torch.float32)  # 生成从0到half的浮点数数组
                / half  # 归一化，使频率在0到1之间
            ).to(device=timesteps.device)  # 将频率数组移动到与timesteps相同的设备
            # 计算时序与频率的乘积，准备嵌入向量
            args = timesteps[:, None].float() * freqs[None]
            # 通过计算余弦和正弦生成嵌入向量
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            # 如果维度是奇数，添加额外的零向量
            if dim % 2:
                embedding = torch.cat(
                    [embedding, torch.zeros_like(embedding[:, :1])], dim=-1  # 在最后一维追加零向量
                )
        # 如果是重复时序，生成与时序相同的嵌入
        else:
            embedding = repeat(timesteps, "b -> b d", d=dim)  # 根据时序生成重复的嵌入向量
        # 返回嵌入向量并设置数据类型
        return embedding.to(dtype)  # 将嵌入向量转换为指定的数据类型
# 将模块的参数归零并返回该模块
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    # 遍历模块的所有参数
    for p in module.parameters():
        # 分离参数并将其值归零
        p.detach().zero_()
    # 返回修改后的模块
    return module


# 对模块的参数进行缩放并返回该模块
def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    # 遍历模块的所有参数
    for p in module.parameters():
        # 分离参数并按比例缩放
        p.detach().mul_(scale)
    # 返回修改后的模块
    return module


# 计算张量中所有非批次维度的平均值
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    # 计算并返回张量在非批次维度上的平均值
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# 创建标准化层
def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    # 返回一个具有指定输入通道数的GroupNorm32标准化层
    return GroupNorm32(32, channels)


# SiLU激活函数类，兼容PyTorch 1.5
class SiLU(nn.Module):
    # 定义前向传播方法
    def forward(self, x):
        # 返回输入与其Sigmoid值的乘积
        return x * torch.sigmoid(x)


# 自定义GroupNorm类，继承自nn.GroupNorm
class GroupNorm32(nn.GroupNorm):
    # 定义前向传播方法
    def forward(self, x):
        # 调用父类的前向方法并返回与输入相同的数据类型
        return super().forward(x).type(x.dtype)


# 创建1D、2D或3D卷积模块
def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    # 根据维度选择对应的卷积层
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    # 如果维度不支持，抛出错误
    raise ValueError(f"unsupported dimensions: {dims}")


# 创建线性模块
def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    # 返回一个线性层
    return nn.Linear(*args, **kwargs)


# 创建1D、2D或3D平均池化模块
def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    # 根据维度选择对应的平均池化层
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    # 如果维度不支持，抛出错误
    raise ValueError(f"unsupported dimensions: {dims}")


# AlphaBlender类，用于合并不同策略
class AlphaBlender(nn.Module):
    # 支持的合并策略
    strategies = ["learned", "fixed", "learned_with_images"]

    # 初始化方法
    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()  # 调用父类初始化
        # 保存合并策略和重排模式
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        # 确保合并策略是支持的选项之一
        assert (
            merge_strategy in self.strategies
        ), f"merge_strategy needs to be in {self.strategies}"

        # 根据合并策略注册混合因子
        if self.merge_strategy == "fixed":
            # 注册固定混合因子
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif (
            self.merge_strategy == "learned"
            or self.merge_strategy == "learned_with_images"
        ):
            # 注册可学习的混合因子
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            # 抛出不支持的合并策略错误
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")
    # 定义获取 alpha 值的函数，接受一个图像指示器作为输入，返回一个张量
    def get_alpha(self, image_only_indicator: torch.Tensor) -> torch.Tensor:
        # 根据合并策略选择 alpha 值
        if self.merge_strategy == "fixed":
            # 如果策略是固定，则 alpha 等于混合因子
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            # 如果策略是学习的，则通过 sigmoid 函数计算 alpha
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            # 如果策略是基于图像学习的，确保提供了图像指示器
            assert image_only_indicator is not None, "need image_only_indicator ..."
            # 根据图像指示器的布尔值，决定 alpha 的值
            alpha = torch.where(
                image_only_indicator.bool(),
                # 如果为真，则 alpha 为全 1 的张量
                torch.ones(1, 1, device=image_only_indicator.device),
                # 否则通过 sigmoid 函数处理混合因子，并调整维度
                rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),
            )
            # 根据给定的重排模式调整 alpha 的形状
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            # 如果没有匹配的合并策略，抛出未实现的错误
            raise NotImplementedError
        # 返回计算得到的 alpha 值
        return alpha
    
    # 定义前向传播的函数，接受空间和时间的输入张量，返回一个张量
    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 调用 get_alpha 函数获取 alpha 值
        alpha = self.get_alpha(image_only_indicator)
        # 根据 alpha 值和输入张量进行加权求和
        x = (
            # x_spatial 乘以 alpha，转换为相同的数据类型
            alpha.to(x_spatial.dtype) * x_spatial
            + 
            # (1.0 - alpha) 乘以 x_temporal，转换为相同的数据类型
            (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        )
        # 返回加权后的结果张量
        return x
```