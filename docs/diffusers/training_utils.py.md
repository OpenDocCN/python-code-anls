# `.\diffusers\training_utils.py`

```py
# 导入标准库和第三方库
import contextlib  # 上下文管理器相关功能
import copy  # 复制对象的功能
import math  # 数学相关功能
import random  # 随机数生成器
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union  # 类型提示

# 导入 NumPy 库
import numpy as np  # 数组处理
# 导入 PyTorch 库
import torch  # 深度学习框架

# 从当前目录下的模型模块导入 UNet2DConditionModel 类
from .models import UNet2DConditionModel
# 从调度器模块导入 SchedulerMixin 类
from .schedulers import SchedulerMixin
# 从工具模块导入多个实用函数
from .utils import (
    convert_state_dict_to_diffusers,  # 转换状态字典为 Diffusers 格式
    convert_state_dict_to_peft,  # 转换状态字典为 PEFT 格式
    deprecate,  # 标记过时的功能
    is_peft_available,  # 检查 PEFT 是否可用
    is_torch_npu_available,  # 检查是否可用 NPU
    is_torchvision_available,  # 检查 torchvision 是否可用
    is_transformers_available,  # 检查 Transformers 是否可用
)

# 如果 Transformers 可用，导入相关库
if is_transformers_available():
    import transformers  # 导入 Transformers 库

# 如果 PEFT 可用，导入相关功能
if is_peft_available():
    from peft import set_peft_model_state_dict  # 导入设置 PEFT 模型状态字典的功能

# 如果 torchvision 可用，导入相关功能
if is_torchvision_available():
    from torchvision import transforms  # 导入图像变换功能

# 如果 NPU 可用，导入相关库，但不使用警告
if is_torch_npu_available():
    import torch_npu  # noqa: F401，表示此导入未使用

# 定义设置随机种子的函数
def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    # 设置 Python 的随机种子
    random.seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    # 如果 NPU 可用，设置所有 NPU 的随机种子
    if is_torch_npu_available():
        torch.npu.manual_seed_all(seed)
    else:
        # 设置 CUDA 的随机种子，即使 CUDA 不可用也安全调用
        torch.cuda.manual_seed_all(seed)

# 定义计算信噪比（SNR）的函数
def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    # 获取噪声调度器的累积 alpha 值
    alphas_cumprod = noise_scheduler.alphas_cumprod
    # 计算 alpha 的平方根
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    # 计算 (1 - alpha) 的平方根
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # 扩展张量的维度
    # 参考链接，调整张量维度
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    # 直到维度匹配为止，增加最后一个维度
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    # 扩展 alpha 到与 timesteps 相同的形状
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    # 同样处理 (1 - alpha) 的平方根
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    # 扩展 sigma 到与 timesteps 相同的形状
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # 计算信噪比
    snr = (alpha / sigma) ** 2
    # 返回计算得到的 SNR
    return snr

# 定义解析插值模式的函数
def resolve_interpolation_mode(interpolation_type: str):
    """
    Maps a string describing an interpolation function to the corresponding torchvision `InterpolationMode` enum. The
    full list of supported enums is documented at
    https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.functional.InterpolationMode.
    # 定义参数说明，描述插值方法的类型
    Args:
        interpolation_type (`str`):
            # 字符串，描述插值方法。目前支持 `bilinear`, `bicubic`, `box`, `nearest`,
            # `nearest_exact`, `hamming`, 和 `lanczos`，与 torchvision 中的插值模式相对应。

    # 定义返回值说明，返回 torchvision 的插值模式枚举
    Returns:
        `torchvision.transforms.InterpolationMode`: 一个 `InterpolationMode` 枚举，用于 torchvision 的 `resize`
        # 变换。
    """
    # 检查 torchvision 是否可用
    if not is_torchvision_available():
        # 如果不可用，抛出导入错误并提示用户安装 torchvision
        raise ImportError(
            "Please make sure to install `torchvision` to be able to use the `resolve_interpolation_mode()` function."
        )

    # 判断插值类型是否为 bilinear
    if interpolation_type == "bilinear":
        # 设置插值模式为 BILINEAR
        interpolation_mode = transforms.InterpolationMode.BILINEAR
    # 判断插值类型是否为 bicubic
    elif interpolation_type == "bicubic":
        # 设置插值模式为 BICUBIC
        interpolation_mode = transforms.InterpolationMode.BICUBIC
    # 判断插值类型是否为 box
    elif interpolation_type == "box":
        # 设置插值模式为 BOX
        interpolation_mode = transforms.InterpolationMode.BOX
    # 判断插值类型是否为 nearest
    elif interpolation_type == "nearest":
        # 设置插值模式为 NEAREST
        interpolation_mode = transforms.InterpolationMode.NEAREST
    # 判断插值类型是否为 nearest_exact
    elif interpolation_type == "nearest_exact":
        # 设置插值模式为 NEAREST_EXACT
        interpolation_mode = transforms.InterpolationMode.NEAREST_EXACT
    # 判断插值类型是否为 hamming
    elif interpolation_type == "hamming":
        # 设置插值模式为 HAMMING
        interpolation_mode = transforms.InterpolationMode.HAMMING
    # 判断插值类型是否为 lanczos
    elif interpolation_type == "lanczos":
        # 设置插值模式为 LANCZOS
        interpolation_mode = transforms.InterpolationMode.LANCZOS
    # 如果插值类型不支持，抛出值错误
    else:
        raise ValueError(
            # 提示用户给定的插值模式不被支持，并列出当前支持的插值模式
            f"The given interpolation mode {interpolation_type} is not supported. Currently supported interpolation"
            f" modes are `bilinear`, `bicubic`, `box`, `nearest`, `nearest_exact`, `hamming`, and `lanczos`."
        )

    # 返回最终的插值模式
    return interpolation_mode
# 定义函数，计算梦境并更新潜在变量
def compute_dream_and_update_latents(
    # UNet模型，用于生成预测
    unet: UNet2DConditionModel,
    # 噪声调度器，用于在给定时间步添加噪声
    noise_scheduler: SchedulerMixin,
    # 噪声调度器使用的时间步
    timesteps: torch.Tensor,
    # 噪声张量，形状与noisy_latents相同
    noise: torch.Tensor,
    # 先前的噪声潜在变量，来自训练循环
    noisy_latents: torch.Tensor,
    # 目标张量，用于在移除eps后进行预测
    target: torch.Tensor,
    # 编码器隐藏状态，来自文本模型的文本嵌入
    encoder_hidden_states: torch.Tensor,
    # 梦境细节保留水平的浮点值
    dream_detail_preservation: float = 1.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    实现"DREAM (Diffusion Rectification and Estimation-Adaptive Models)"，见http://arxiv.org/abs/2312.00210。
    DREAM有助于对齐训练与采样，使训练更加高效和准确，代价是多一步无梯度的前向计算。

    参数:
        `unet`: 用于生成预测的状态unet。
        `noise_scheduler`: 用于给定时间步添加噪声的噪声调度器。
        `timesteps`: 噪声调度器使用的时间步。
        `noise`: 形状与noisy_latents相同的噪声张量。
        `noisy_latents`: 来自训练循环的先前噪声潜在变量。
        `target`: 移除eps后要预测的真实目标张量。
        `encoder_hidden_states`: 来自文本模型的文本嵌入。
        `dream_detail_preservation`: 表示细节保留水平的浮点值。
          参考文献。

    返回:
        `tuple[torch.Tensor, torch.Tensor]`: 调整后的noisy_latents和target。
    """
    # 获取时间步对应的累积alpha值，并将其移动到时间步所在设备
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
    # 计算1减去累积alpha的平方根
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # 计算梦境系数，使用论文中lambda = sqrt(1 - alpha) ** p，p取1
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation

    pred = None
    # 禁用梯度计算
    with torch.no_grad():
        # 使用UNet生成预测结果
        pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    _noisy_latents, _target = (None, None)
    # 如果预测类型为"epsilon"
    if noise_scheduler.config.prediction_type == "epsilon":
        # 将预测的噪声存储在predicted_noise中
        predicted_noise = pred
        # 计算实际噪声与预测噪声的差值，并从计算图中分离
        delta_noise = (noise - predicted_noise).detach()
        # 按梦境系数缩放差值噪声
        delta_noise.mul_(dream_lambda)
        # 更新噪声潜在变量
        _noisy_latents = noisy_latents.add(sqrt_one_minus_alphas_cumprod * delta_noise)
        # 更新目标张量
        _target = target.add(delta_noise)
    # 如果预测类型为"v_prediction"，抛出未实现错误
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    # 否则抛出值错误，说明未知的预测类型
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # 返回更新后的噪声潜在变量和目标张量
    return _noisy_latents, _target


# 定义函数，获取UNet的LoRA状态字典
def unet_lora_state_dict(unet: UNet2DConditionModel) -> Dict[str, torch.Tensor]:
    r"""
    返回:
        仅包含LoRA参数的状态字典。
    """
    # 初始化一个空字典，用于存储LoRA状态
    lora_state_dict = {}
    # 遍历 UNet 模型的所有命名模块
        for name, module in unet.named_modules():
            # 检查模块是否具有设置 Lora 层的属性
            if hasattr(module, "set_lora_layer"):
                # 获取模块的 Lora 层
                lora_layer = getattr(module, "lora_layer")
                # 确保 Lora 层不为空
                if lora_layer is not None:
                    # 获取当前 Lora 层的状态字典
                    current_lora_layer_sd = lora_layer.state_dict()
                    # 遍历 Lora 层状态字典中的所有矩阵名称和参数
                    for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items():
                        # 矩阵名称可以是 "down" 或 "up"，将参数保存到字典中
                        lora_state_dict[f"{name}.lora.{lora_layer_matrix_name}"] = lora_param
    
        # 返回包含 Lora 层参数的状态字典
        return lora_state_dict
# 定义一个函数，将模型参数转换为指定的数据类型
def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    # 检查输入的模型是否为列表，如果不是则将其转换为列表
    if not isinstance(model, list):
        model = [model]
    # 遍历模型列表中的每个模型
    for m in model:
        # 遍历模型的所有参数
        for param in m.parameters():
            # 仅将可训练的参数转换为指定的数据类型
            if param.requires_grad:
                param.data = param.to(dtype)


# 定义一个函数，将 LoRA 状态字典设置到文本编码器中
def _set_state_dict_into_text_encoder(
    lora_state_dict: Dict[str, torch.Tensor], prefix: str, text_encoder: torch.nn.Module
):
    """
    将来自 `transformers` 的 `lora_state_dict` 设置到 `text_encoder` 中。

    Args:
        lora_state_dict: 要设置的状态字典。
        prefix: 字符串标识符，用于检索属于 `text_encoder` 的状态字典部分。
        text_encoder: 要设置 `lora_state_dict` 的地方。
    """

    # 创建一个新的状态字典，只包含以 prefix 开头的键值对
    text_encoder_state_dict = {
        f'{k.replace(prefix, "")}': v for k, v in lora_state_dict.items() if k.startswith(prefix)
    }
    # 将状态字典转换为 PEFT 格式
    text_encoder_state_dict = convert_state_dict_to_peft(convert_state_dict_to_diffusers(text_encoder_state_dict))
    # 将转换后的状态字典设置到文本编码器中
    set_peft_model_state_dict(text_encoder, text_encoder_state_dict, adapter_name="default")


# 定义一个函数，计算用于时序采样的密度
def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """计算在进行 SD3 训练时采样时序的密度。

    参考：这是由 Rafie Walker 提供的 https://github.com/huggingface/diffusers/pull/8528。

    SD3 论文参考： https://arxiv.org/abs/2403.03206v1。
    """
    # 根据加权方案选择计算方式
    if weighting_scheme == "logit_normal":
        # 参见 SD3 论文中的公式
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        # 应用 sigmoid 函数
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        # 生成均匀分布的随机数
        u = torch.rand(size=(batch_size,), device="cpu")
        # 根据模式调整 u 的值
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        # 如果没有匹配的方案，则生成随机数
        u = torch.rand(size=(batch_size,), device="cpu")
    # 返回计算的 u 值
    return u


# 定义一个函数，为 SD3 训练计算损失加权方案
def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """计算 SD3 训练的损失加权方案。

    参考：这是由 Rafie Walker 提供的 https://github.com/huggingface/diffusers/pull/8528。

    SD3 论文参考： https://arxiv.org/abs/2403.03206v1。
    """
    # 根据加权方案选择计算方式
    if weighting_scheme == "sigma_sqrt":
        # 计算加权值为 sigma 的平方的倒数
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        # 根据 sigma 计算底部值，并计算加权
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        # 如果没有匹配的方案，则返回与 sigmas 相同大小的全1张量
        weighting = torch.ones_like(sigmas)
    # 返回计算的加权值
    return weighting


# 定义一个类，用于实现模型权重的指数移动平均
class EMAModel:
    """
    模型权重的指数移动平均
    """
    # 初始化方法，设置 EMA 模型的参数
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],  # 模型参数的可迭代对象
        decay: float = 0.9999,  # 衰减因子的默认值
        min_decay: float = 0.0,  # 最小衰减值
        update_after_step: int = 0,  # 更新后开始计算的步骤
        use_ema_warmup: bool = False,  # 是否使用 EMA 暖启动
        inv_gamma: Union[float, int] = 1.0,  # 反伽马值，用于计算衰减
        power: Union[float, int] = 2 / 3,  # 衰减的幂
        foreach: bool = False,  # 是否使用逐项更新
        model_cls: Optional[Any] = None,  # 模型类的可选参数
        model_config: Dict[str, Any] = None,  # 模型配置的字典
        **kwargs,  # 其他关键字参数
    # 类方法，用于从预训练模型加载
    @classmethod
    def from_pretrained(cls, path, model_cls, foreach=False) -> "EMAModel":
        # 从给定路径加载模型配置，返回未使用的参数
        _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
        # 从预训练路径加载模型
        model = model_cls.from_pretrained(path)

        # 创建 EMA 模型，传入模型参数及其他配置
        ema_model = cls(model.parameters(), model_cls=model_cls, model_config=model.config, foreach=foreach)

        # 加载 EMA 模型的状态字典
        ema_model.load_state_dict(ema_kwargs)
        # 返回创建的 EMA 模型
        return ema_model

    # 保存预训练模型的方法
    def save_pretrained(self, path):
        # 检查是否定义了模型类
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        # 检查是否定义了模型配置
        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        # 根据模型配置创建模型
        model = self.model_cls.from_config(self.model_config)
        # 获取当前模型的状态字典
        state_dict = self.state_dict()
        # 从状态字典中删除 "shadow_params" 项
        state_dict.pop("shadow_params", None)

        # 将状态字典注册到模型配置中
        model.register_to_config(**state_dict)
        # 将当前模型的参数复制到新模型中
        self.copy_to(model.parameters())
        # 保存模型到指定路径
        model.save_pretrained(path)

    # 计算衰减因子的方法
    def get_decay(self, optimization_step: int) -> float:
        """
        计算指数移动平均的衰减因子。
        """
        # 计算当前步骤，确保不为负
        step = max(0, optimization_step - self.update_after_step - 1)

        # 如果当前步骤小于等于0，返回0.0
        if step <= 0:
            return 0.0

        # 根据是否使用 EMA 暖启动计算当前衰减值
        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        # 确保当前衰减值不大于最大衰减值
        cur_decay_value = min(cur_decay_value, self.decay)
        # 确保衰减值不小于最小衰减值
        cur_decay_value = max(cur_decay_value, self.min_decay)
        # 返回计算出的衰减值
        return cur_decay_value

    # 在无梯度计算的上下文中执行的装饰器
    @torch.no_grad()
    # 定义一个方法，接收参数列表，参数可以是可迭代的 PyTorch 参数
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        # 检查传入的参数是否是 PyTorch 模块
        if isinstance(parameters, torch.nn.Module):
            # 定义一个弃用警告消息，提示用户不应传递模块
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage.step` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            # 调用弃用函数，显示警告信息
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage.step`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            # 获取模块的参数列表
            parameters = parameters.parameters()

        # 将参数转换为列表形式
        parameters = list(parameters)

        # 增加优化步骤计数
        self.optimization_step += 1

        # 计算指数移动平均的衰减因子
        decay = self.get_decay(self.optimization_step)
        # 保存当前的衰减值
        self.cur_decay_value = decay
        # 计算 1 - 衰减值
        one_minus_decay = 1 - decay

        # 初始化上下文管理器为无操作上下文
        context_manager = contextlib.nullcontext
        # 检查是否可用 transformers 库以及 DeepSpeed 的 Zero3 功能
        if is_transformers_available() and transformers.deepspeed.is_deepspeed_zero3_enabled():
            # 导入 DeepSpeed 库
            import deepspeed

        # 如果使用 foreach 模式
        if self.foreach:
            # 如果可用，使用 DeepSpeed 的 GatheredParameters 上下文管理器
            if is_transformers_available() and transformers.deepspeed.is_deepspeed_zero3_enabled():
                context_manager = deepspeed.zero.GatheredParameters(parameters, modifier_rank=None)

            # 使用上下文管理器进行操作
            with context_manager():
                # 筛选出需要梯度计算的参数
                params_grad = [param for param in parameters if param.requires_grad]
                # 与阴影参数配对，筛选出需要梯度的阴影参数
                s_params_grad = [
                    s_param for s_param, param in zip(self.shadow_params, parameters) if param.requires_grad
                ]

                # 如果需要梯度的参数数量少于总参数数量
                if len(params_grad) < len(parameters):
                    # 复制不需要梯度的参数值到阴影参数
                    torch._foreach_copy_(
                        [s_param for s_param, param in zip(self.shadow_params, parameters) if not param.requires_grad],
                        [param for param in parameters if not param.requires_grad],
                        non_blocking=True,
                    )

                # 更新阴影参数的值，使用指数移动平均更新
                torch._foreach_sub_(
                    s_params_grad, torch._foreach_sub(s_params_grad, params_grad), alpha=one_minus_decay
                )

        # 如果不使用 foreach 模式
        else:
            # 遍历阴影参数和输入参数
            for s_param, param in zip(self.shadow_params, parameters):
                # 如果可用，使用 DeepSpeed 的 GatheredParameters 上下文管理器
                if is_transformers_available() and transformers.deepspeed.is_deepspeed_zero3_enabled():
                    context_manager = deepspeed.zero.GatheredParameters(param, modifier_rank=None)

                # 使用上下文管理器进行操作
                with context_manager():
                    # 如果参数需要梯度
                    if param.requires_grad:
                        # 使用指数移动平均更新阴影参数
                        s_param.sub_(one_minus_decay * (s_param - param))
                    else:
                        # 直接复制参数到阴影参数
                        s_param.copy_(param)
    # 定义一个将当前平均参数复制到给定参数集合的方法
    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        # 方法文档，描述参数的作用
        """
        Copy current averaged parameters into given collection of parameters.
    
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        # 将输入参数转换为列表
        parameters = list(parameters)
        # 如果设置了 foreach，则使用批量操作复制数据
        if self.foreach:
            torch._foreach_copy_(
                # 提取参数的数据
                [param.data for param in parameters],
                # 将影子参数的数据复制到目标参数的设备
                [s_param.to(param.device).data for s_param, param in zip(self.shadow_params, parameters)],
            )
        else:
            # 否则逐一复制影子参数的数据到目标参数
            for s_param, param in zip(self.shadow_params, parameters):
                param.data.copy_(s_param.to(param.device).data)
    
    # 定义一个将内部缓冲区移动到固定内存的方法
    def pin_memory(self) -> None:
        r"""
        Move internal buffers of the ExponentialMovingAverage to pinned memory. Useful for non-blocking transfers for
        offloading EMA params to the host.
        """
        # 将影子参数移动到固定内存
        self.shadow_params = [p.pin_memory() for p in self.shadow_params]
    
    # 定义一个将内部缓冲区移动到指定设备的方法
    def to(self, device=None, dtype=None, non_blocking=False) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
    
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() 方法处理 None 的情况
        self.shadow_params = [
            # 如果参数是浮点型，则同时移动设备和数据类型
            p.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if p.is_floating_point()
            # 否则只移动设备
            else p.to(device=device, non_blocking=non_blocking)
            for p in self.shadow_params
        ]
    
    # 定义一个返回 ExponentialMovingAverage 状态字典的方法
    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict. This method is used by accelerate during
        checkpointing to save the ema state dict.
        """
        # 返回状态字典，遵循 PyTorch 约定，返回张量的引用
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,
        }
    
    # 定义一个保存当前参数以便后续恢复的方法
    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Save the current parameters for restoring later.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        # 将当前参数暂时存储在 CPU 上的克隆
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]
    # 定义一个恢复方法，用于恢复存储的参数
    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        恢复通过 `store` 方法存储的参数。此方法对于使用 EMA 参数验证模型非常有用，而不会影响
        原始优化过程。在调用 `copy_to()` 方法之前存储参数。验证（或保存模型）后，使用此方法
        恢复以前的参数。
            parameters: `torch.nn.Parameter` 的可迭代对象；要更新为存储参数的参数。如果为 `None`，
            则将使用此 `ExponentialMovingAverage` 初始化时的参数。
        """
        # 检查是否有存储的参数，如果没有则引发运行时错误
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights " "to `restore()`")
        # 如果使用并行操作，则使用 foreach 复制参数
        if self.foreach:
            torch._foreach_copy_(
                # 获取每个参数的数据进行复制
                [param.data for param in parameters], [c_param.data for c_param in self.temp_stored_params]
            )
        # 否则，逐一复制存储的参数数据
        else:
            for c_param, param in zip(self.temp_stored_params, parameters):
                # 将存储的参数数据复制到当前参数
                param.data.copy_(c_param.data)
    
        # 更好地节省内存，将临时存储的参数置为 None
        self.temp_stored_params = None
    # 定义一个加载状态字典的方法，用于加载指数移动平均的状态
    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Args:
        加载指数移动平均状态。此方法在检查点时由 accelerate 使用，以保存 ema 状态字典。
            state_dict (dict): EMA 状态。应为从 :meth:`state_dict` 调用返回的对象。
        """
        # 深拷贝状态字典，以与模块 API 保持一致
        state_dict = copy.deepcopy(state_dict)

        # 获取衰减值，如果未提供则使用当前值
        self.decay = state_dict.get("decay", self.decay)
        # 检查衰减值是否在有效范围内
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        # 获取最小衰减值，如果未提供则使用当前值
        self.min_decay = state_dict.get("min_decay", self.min_decay)
        # 检查最小衰减值是否为浮点数
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        # 获取优化步骤，如果未提供则使用当前值
        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        # 检查优化步骤是否为整数
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        # 获取更新后步骤，如果未提供则使用当前值
        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        # 检查更新后步骤是否为整数
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        # 获取是否使用 EMA 预热，如果未提供则使用当前值
        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        # 检查是否使用 EMA 预热的值是否为布尔值
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        # 获取反伽马值，如果未提供则使用当前值
        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        # 检查反伽马值是否为浮点数或整数
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        # 获取幂值，如果未提供则使用当前值
        self.power = state_dict.get("power", self.power)
        # 检查幂值是否为浮点数或整数
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        # 获取影子参数，如果未提供则为 None
        shadow_params = state_dict.get("shadow_params", None)
        # 如果影子参数不为 None，则进行检查
        if shadow_params is not None:
            self.shadow_params = shadow_params
            # 检查影子参数是否为列表
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            # 检查所有影子参数是否都是张量
            if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")
```