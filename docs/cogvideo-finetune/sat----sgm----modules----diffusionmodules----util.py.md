# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\util.py`

```py
# 该模块源自多个开源项目，感谢它们的贡献
"""
adopted from
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
and
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
and
https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py

thanks!
"""

# 导入数学库
import math
# 从 typing 导入可选类型
from typing import Optional

# 导入 PyTorch 库
import torch
import torch.nn as nn
# 从 einops 导入重排和重复函数
from einops import rearrange, repeat


# 创建 beta 调度的函数
def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
):
    # 如果调度方式为线性，生成 beta 值的序列
    if schedule == "linear":
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    # 返回 numpy 格式的 beta 值
    return betas.numpy()


# 从张量中提取特定的值并调整形状
def extract_into_tensor(a, t, x_shape):
    # 获取 t 的第一个维度的大小
    b, *_ = t.shape
    # 根据索引 t 从 a 中提取数据
    out = a.gather(-1, t)
    # 返回调整形状后的输出
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# 混合检查点函数，用于在不缓存中间激活的情况下评估函数
def mixed_checkpoint(func, inputs: dict, params, flag):
    """
    在不缓存中间激活的情况下评估函数，减少内存消耗，但会增加反向传播的计算量。
    该实现允许非张量输入。
    :param func: 要评估的函数。
    :param inputs: 传递给 `func` 的参数字典。
    :param params: func 依赖但不作为参数的参数序列。
    :param flag: 如果为 False，禁用梯度检查点。
    """
    # 如果启用标志，处理张量输入
    if flag:
        # 获取所有张量类型的输入键
        tensor_keys = [key for key in inputs if isinstance(inputs[key], torch.Tensor)]
        # 获取所有张量类型的输入值
        tensor_inputs = [inputs[key] for key in inputs if isinstance(inputs[key], torch.Tensor)]
        # 获取所有非张量类型的输入键
        non_tensor_keys = [key for key in inputs if not isinstance(inputs[key], torch.Tensor)]
        # 获取所有非张量类型的输入值
        non_tensor_inputs = [inputs[key] for key in inputs if not isinstance(inputs[key], torch.Tensor)]
        # 构建参数元组
        args = tuple(tensor_inputs) + tuple(non_tensor_inputs) + tuple(params)
        # 应用混合检查点函数
        return MixedCheckpointFunction.apply(
            func,
            len(tensor_inputs),
            len(non_tensor_inputs),
            tensor_keys,
            non_tensor_keys,
            *args,
        )
    else:
        # 如果禁用标志，直接调用函数
        return func(**inputs)


# 定义混合检查点函数的类
class MixedCheckpointFunction(torch.autograd.Function):
    @staticmethod
    # 定义前向传播方法
    def forward(
        ctx,
        run_function,
        length_tensors,
        length_non_tensors,
        tensor_keys,
        non_tensor_keys,
        *args,
    ):
        # 将长度张量赋值给上下文对象的属性
        ctx.end_tensors = length_tensors
        # 将长度非张量赋值给上下文对象的属性
        ctx.end_non_tensors = length_tensors + length_non_tensors
        # 创建包含 GPU 自动混合精度参数的字典
        ctx.gpu_autocast_kwargs = {
            # 检查自动混合精度是否启用
            "enabled": torch.is_autocast_enabled(),
            # 获取自动混合精度的 GPU 数据类型
            "dtype": torch.get_autocast_gpu_dtype(),
            # 检查自动混合精度缓存是否启用
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        # 断言张量键和非张量键的长度与预期相符
        assert len(tensor_keys) == length_tensors and len(non_tensor_keys) == length_non_tensors

        # 创建输入张量字典，将键与对应的值配对
        ctx.input_tensors = {key: val for (key, val) in zip(tensor_keys, list(args[: ctx.end_tensors]))}
        # 创建输入非张量字典，将键与对应的值配对
        ctx.input_non_tensors = {
            key: val for (key, val) in zip(non_tensor_keys, list(args[ctx.end_tensors : ctx.end_non_tensors]))
        }
        # 将运行函数赋值给上下文对象的属性
        ctx.run_function = run_function
        # 将输入参数赋值为剩余的参数
        ctx.input_params = list(args[ctx.end_non_tensors :])

        # 在无梯度计算上下文中执行
        with torch.no_grad():
            # 调用运行函数，并传入输入张量和非张量
            output_tensors = ctx.run_function(**ctx.input_tensors, **ctx.input_non_tensors)
        # 返回输出张量
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # 创建额外参数的字典（已注释掉）
        # additional_args = {key: ctx.input_tensors[key] for key in ctx.input_tensors if not isinstance(ctx.input_tensors[key],torch.Tensor)}
        # 将输入张量设为不跟踪梯度并要求梯度
        ctx.input_tensors = {key: ctx.input_tensors[key].detach().requires_grad_(True) for key in ctx.input_tensors}

        # 在启用梯度计算和混合精度上下文中执行
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # 创建输入张量的浅拷贝以避免修改原张量
            shallow_copies = {key: ctx.input_tensors[key].view_as(ctx.input_tensors[key]) for key in ctx.input_tensors}
            # shallow_copies.update(additional_args)  # 更新额外参数（已注释掉）
            # 调用运行函数，并传入浅拷贝和非张量
            output_tensors = ctx.run_function(**shallow_copies, **ctx.input_non_tensors)
        # 计算输出张量相对于输入张量和输入参数的梯度
        input_grads = torch.autograd.grad(
            output_tensors,
            list(ctx.input_tensors.values()) + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        # 删除输入张量
        del ctx.input_tensors
        # 删除输入参数
        del ctx.input_params
        # 删除输出张量
        del output_tensors
        # 返回梯度和占位符
        return (
            (None, None, None, None, None)
            + input_grads[: ctx.end_tensors]
            + (None,) * (ctx.end_non_tensors - ctx.end_tensors)
            + input_grads[ctx.end_tensors :]
        )
# 定义一个检查点函数，允许不缓存中间激活以降低内存使用
def checkpoint(func, inputs, params, flag):
    # 文档字符串，解释函数的用途和参数
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    # 如果标志为真，执行检查点功能
    if flag:
        # 将输入和参数组合为一个元组
        args = tuple(inputs) + tuple(params)
        # 应用检查点功能并返回结果
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        # 直接调用函数并返回结果
        return func(*inputs)

# 定义检查点功能类，继承自torch的自动梯度功能
class CheckpointFunction(torch.autograd.Function):
    # 静态方法，用于前向传播
    @staticmethod
    def forward(ctx, run_function, length, *args):
        # 将运行的函数和输入张量保存到上下文中
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        # 保存GPU自动混合精度的相关设置
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        # 在无梯度计算模式下执行函数
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        # 返回输出张量
        return output_tensors

    # 静态方法，用于反向传播
    @staticmethod
    def backward(ctx, *output_grads):
        # 将输入张量标记为需要梯度计算
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        # 启用梯度计算和混合精度模式
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # 创建输入张量的浅拷贝，以避免存储修改问题
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            # 执行反向传播，获取输出张量
            output_tensors = ctx.run_function(*shallow_copies)
        # 计算输入梯度
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        # 清理上下文中的临时变量
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        # 返回None和输入梯度
        return (None, None) + input_grads

# 定义时间步嵌入函数，创建正弦时间步嵌入
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False, dtype=torch.float32):
    # 文档字符串，解释函数的用途和参数
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    # 检查是否只重复使用嵌入
        if not repeat_only:
            # 计算维度的一半
            half = dim // 2
            # 计算频率值，使用指数衰减函数生成频率序列
            freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
                device=timesteps.device
            )
            # 计算时间步长与频率的乘积，形成角度参数
            args = timesteps[:, None].float() * freqs[None]
            # 计算余弦和正弦值并在最后一维合并
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            # 如果维度为奇数，添加一列全零以保持维度一致
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            # 如果只重复，直接生成与时间步长相同的嵌入
            embedding = repeat(timesteps, "b -> b d", d=dim)
        # 将嵌入转换为指定的数据类型并返回
        return embedding.to(dtype)
# 定义一个将模块参数归零的函数，并返回该模块
def zero_module(module):
    """
    将模块的参数置为零，并返回该模块。
    """
    # 遍历模块的所有参数
    for p in module.parameters():
        # 将参数分离并置为零
        p.detach().zero_()
    # 返回处理后的模块
    return module


# 定义一个对模块参数进行缩放的函数，并返回该模块
def scale_module(module, scale):
    """
    将模块的参数进行缩放，并返回该模块。
    """
    # 遍历模块的所有参数
    for p in module.parameters():
        # 将参数分离并进行缩放
        p.detach().mul_(scale)
    # 返回处理后的模块
    return module


# 定义一个计算非批次维度均值的函数
def mean_flat(tensor):
    """
    对所有非批次维度进行求均值。
    """
    # 计算并返回张量的均值，忽略第一维（批次维度）
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# 定义一个创建标准归一化层的函数
def normalization(channels):
    """
    创建一个标准归一化层。
    :param channels: 输入通道的数量。
    :return: 一个用于归一化的 nn.Module。
    """
    # 返回一个具有指定输入通道数的 GroupNorm32 对象
    return GroupNorm32(32, channels)


# 定义一个 SiLU 激活函数类，支持 PyTorch 1.5
class SiLU(nn.Module):
    # 定义前向传播方法
    def forward(self, x):
        # 返回 x 与其 sigmoid 值的乘积
        return x * torch.sigmoid(x)


# 定义一个扩展自 nn.GroupNorm 的 GroupNorm32 类
class GroupNorm32(nn.GroupNorm):
    # 定义前向传播方法
    def forward(self, x):
        # 调用父类的 forward 方法，并返回与输入相同数据类型的结果
        return super().forward(x).type(x.dtype)


# 定义一个创建卷积模块的函数，支持 1D、2D 和 3D 卷积
def conv_nd(dims, *args, **kwargs):
    """
    创建一个 1D、2D 或 3D 卷积模块。
    """
    # 判断维度是否为 1
    if dims == 1:
        # 返回 1D 卷积模块
        return nn.Conv1d(*args, **kwargs)
    # 判断维度是否为 2
    elif dims == 2:
        # 返回 2D 卷积模块
        return nn.Conv2d(*args, **kwargs)
    # 判断维度是否为 3
    elif dims == 3:
        # 返回 3D 卷积模块
        return nn.Conv3d(*args, **kwargs)
    # 抛出不支持的维度错误
    raise ValueError(f"unsupported dimensions: {dims}")


# 定义一个创建线性模块的函数
def linear(*args, **kwargs):
    """
    创建一个线性模块。
    """
    # 返回一个线性模块
    return nn.Linear(*args, **kwargs)


# 定义一个创建平均池化模块的函数，支持 1D、2D 和 3D 池化
def avg_pool_nd(dims, *args, **kwargs):
    """
    创建一个 1D、2D 或 3D 平均池化模块。
    """
    # 判断维度是否为 1
    if dims == 1:
        # 返回 1D 平均池化模块
        return nn.AvgPool1d(*args, **kwargs)
    # 判断维度是否为 2
    elif dims == 2:
        # 返回 2D 平均池化模块
        return nn.AvgPool2d(*args, **kwargs)
    # 判断维度是否为 3
    elif dims == 3:
        # 返回 3D 平均池化模块
        return nn.AvgPool3d(*args, **kwargs)
    # 抛出不支持的维度错误
    raise ValueError(f"unsupported dimensions: {dims}")


# 定义一个 AlphaBlender 类，用于实现不同的混合策略
class AlphaBlender(nn.Module):
    # 定义支持的混合策略
    strategies = ["learned", "fixed", "learned_with_images"]

    # 初始化方法，设置混合因子和重排模式
    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        # 保存混合策略和重排模式
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        # 确保混合策略在支持的策略中
        assert merge_strategy in self.strategies, f"merge_strategy needs to be in {self.strategies}"

        # 根据混合策略注册混合因子
        if self.merge_strategy == "fixed":
            # 注册一个固定的混合因子
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            # 注册一个可学习的混合因子
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            # 抛出未知混合策略错误
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")
    # 定义获取 alpha 值的函数，输入为图像指示器，返回一个张量
        def get_alpha(self, image_only_indicator: torch.Tensor) -> torch.Tensor:
            # 根据合并策略选择 alpha 值的计算方式
            if self.merge_strategy == "fixed":
                # 如果合并策略为固定值，则直接使用 mix_factor 作为 alpha
                alpha = self.mix_factor
            elif self.merge_strategy == "learned":
                # 如果合并策略为学习得到的值，则对 mix_factor 应用 sigmoid 函数
                alpha = torch.sigmoid(self.mix_factor)
            elif self.merge_strategy == "learned_with_images":
                # 如果合并策略为图像学习，需要确保提供图像指示器
                assert image_only_indicator is not None, "need image_only_indicator ..."
                # 根据图像指示器选择 alpha 值，真值对应 1，假值则使用 mix_factor 的 sigmoid 结果
                alpha = torch.where(
                    image_only_indicator.bool(),  # 将图像指示器转换为布尔值
                    torch.ones(1, 1, device=image_only_indicator.device),  # 为真值创建全 1 的张量
                    rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),  # 为假值应用 sigmoid 并调整维度
                )
                # 根据 rearrange_pattern 重新排列 alpha 的维度
                alpha = rearrange(alpha, self.rearrange_pattern)
            else:
                # 如果合并策略不在已知范围内，抛出未实现错误
                raise NotImplementedError
            # 返回计算得到的 alpha 值
            return alpha
    
    # 定义前向传播函数，输入为空间和时间张量，以及可选的图像指示器
        def forward(
            self,
            x_spatial: torch.Tensor,  # 空间输入张量
            x_temporal: torch.Tensor,  # 时间输入张量
            image_only_indicator: Optional[torch.Tensor] = None,  # 可选的图像指示器
        ) -> torch.Tensor:
            # 调用 get_alpha 函数获取 alpha 值
            alpha = self.get_alpha(image_only_indicator)
            # 计算最终输出张量，结合空间和时间输入，使用 alpha 进行加权
            x = alpha.to(x_spatial.dtype) * x_spatial + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
            # 返回计算得到的输出张量
            return x
```