# `.\diffusers\schedulers\scheduling_consistency_decoder.py`

```py
# 导入数学库
import math
# 从数据类模块导入数据类装饰器
from dataclasses import dataclass
# 导入可选类型、元组和联合类型
from typing import Optional, Tuple, Union

# 导入 PyTorch 库
import torch

# 从配置工具中导入混合类和注册配置的装饰器
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入基本输出类
from ..utils import BaseOutput
# 从 PyTorch 工具中导入随机张量生成函数
from ..utils.torch_utils import randn_tensor
# 从调度工具中导入调度混合类
from .scheduling_utils import SchedulerMixin


# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 传入的扩散时间步数
    max_beta=0.999,  # 最大的 beta 值，默认为 0.999
    alpha_transform_type="cosine",  # alpha 变换类型，默认为 "cosine"
):
    """
    创建一个 beta 调度程序，该调度程序离散化给定的 alpha_t_bar 函数，该函数定义了随时间变化的 (1-beta) 的累积乘积，从 t = [0,1]。

    包含一个函数 alpha_bar，该函数接受参数 t 并将其转换为扩散过程中到该部分的 (1-beta) 的累积乘积。

    参数:
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用小于 1 的值以防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认为 `cosine`): alpha_bar 的噪声调度类型。
                     选择 `cosine` 或 `exp`

    返回:
        betas (`np.ndarray`): 调度程序用于步骤模型输出的 betas
    """
    # 如果 alpha_transform_type 为 "cosine"
    if alpha_transform_type == "cosine":
        # 定义 alpha_bar 函数，计算基于余弦的 alpha 值
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 如果 alpha_transform_type 为 "exp"
    elif alpha_transform_type == "exp":
        # 定义 alpha_bar 函数，计算基于指数的 alpha 值
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    # 如果传入的 alpha_transform_type 不支持
    else:
        # 抛出错误，提示不支持的 alpha_transform_type
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化一个空列表，用于存储 beta 值
    betas = []
    # 遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        # 计算当前时间步 t1
        t1 = i / num_diffusion_timesteps
        # 计算下一个时间步 t2
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算 beta 值，并确保不超过 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 返回作为 PyTorch 张量的 beta 值
    return torch.tensor(betas, dtype=torch.float32)


# 定义一个数据类，用于调度器的输出
@dataclass
class ConsistencyDecoderSchedulerOutput(BaseOutput):
    """
    调度器 `step` 函数的输出类。

    参数:
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            先前时间步的计算样本 `(x_{t-1})`。`prev_sample` 应作为去噪循环中的下一个模型输入。
    """

    # 定义前一个样本的张量
    prev_sample: torch.Tensor


# 定义一致性解码调度器类，继承自调度混合类和配置混合类
class ConsistencyDecoderScheduler(SchedulerMixin, ConfigMixin):
    # 定义调度器的顺序
    order = 1

    # 使用装饰器注册到配置
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1024,  # 训练时间步数，默认为 1024
        sigma_data: float = 0.5,  # 数据的 sigma 值，默认为 0.5
    ):
        # 计算与 alpha_bar 相关的 beta 值
        betas = betas_for_alpha_bar(num_train_timesteps)

        # 计算 alpha 值，alpha = 1 - beta
        alphas = 1.0 - betas
        # 计算 alpha 的累积乘积
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 计算累积 alpha 的平方根
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        # 计算 1 - 累积 alpha 的平方根
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # 计算 sigma 值，sigma = sqrt(1 / alpha_cumprod - 1)
        sigmas = torch.sqrt(1.0 / alphas_cumprod - 1)

        # 计算累积 alpha 的倒数的平方根
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)

        # 计算 c_skip，用于后续的跳跃连接
        self.c_skip = sqrt_recip_alphas_cumprod * sigma_data**2 / (sigmas**2 + sigma_data**2)
        # 计算 c_out，输出通道的缩放因子
        self.c_out = sigmas * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5
        # 计算 c_in，输入通道的缩放因子
        self.c_in = sqrt_recip_alphas_cumprod / (sigmas**2 + sigma_data**2) ** 0.5

    def set_timesteps(
        # 定义设置时间步的函数，接收可选的推理步骤数和设备类型
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
    ):
        # 如果推理步骤数不是 2，抛出错误
        if num_inference_steps != 2:
            raise ValueError("Currently more than 2 inference steps are not supported.")

        # 设置时间步为指定的张量
        self.timesteps = torch.tensor([1008, 512], dtype=torch.long, device=device)
        # 将 sqrt_alphas_cumprod 移动到指定设备
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        # 将 sqrt_one_minus_alphas_cumprod 移动到指定设备
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        # 将 c_skip 移动到指定设备
        self.c_skip = self.c_skip.to(device)
        # 将 c_out 移动到指定设备
        self.c_out = self.c_out.to(device)
        # 将 c_in 移动到指定设备
        self.c_in = self.c_in.to(device)

    @property
    # 初始化噪声的标准差，使用 timesteps[0] 的值
    def init_noise_sigma(self):
        return self.sqrt_one_minus_alphas_cumprod[self.timesteps[0]]

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器的互换性。

        Args:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *optional*):
                扩散链中的当前时间步。

        Returns:
            `torch.Tensor`:
                一个缩放后的输入样本。
        """
        # 返回缩放后的样本，使用当前时间步对应的 c_in
        return sample * self.c_in[timestep]

    def step(
        # 定义模型推理的步骤函数，接收模型输出、时间步、样本等参数
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    # 定义函数返回类型为一致性解码器调度器输出或元组
    ) -> Union[ConsistencyDecoderSchedulerOutput, Tuple]:
        """
        通过逆向 SDE 预测前一个时间步的样本。此函数从学习到的模型输出（通常是预测的噪声）传播扩散过程。
        
        参数：
            model_output (`torch.Tensor`):
                来自学习扩散模型的直接输出。
            timestep (`float`):
                扩散链中的当前时间步。
            sample (`torch.Tensor`):
                由扩散过程创建的当前样本实例。
            generator (`torch.Generator`, *可选*):
                随机数生成器。
            return_dict (`bool`, *可选*, 默认为 `True`):
                是否返回一致性解码器调度器输出或元组。
    
        返回：
            [`~schedulers.scheduling_consistency_models.ConsistencyDecoderSchedulerOutput`] 或 `tuple`:
                如果 return_dict 为 `True`，返回一致性解码器调度器输出；否则返回一个元组，元组的第一个元素是样本张量。
        """
        # 计算当前时间步的输出，结合模型输出和样本
        x_0 = self.c_out[timestep] * model_output + self.c_skip[timestep] * sample
    
        # 获取当前时间步在时间步列表中的索引
        timestep_idx = torch.where(self.timesteps == timestep)[0]
    
        # 检查当前时间步是否为最后一个时间步
        if timestep_idx == len(self.timesteps) - 1:
            # 如果是最后一个时间步，前一个样本为当前输出
            prev_sample = x_0
        else:
            # 否则生成噪声，并计算前一个样本
            noise = randn_tensor(x_0.shape, generator=generator, dtype=x_0.dtype, device=x_0.device)
            prev_sample = (
                self.sqrt_alphas_cumprod[self.timesteps[timestep_idx + 1]].to(x_0.dtype) * x_0
                + self.sqrt_one_minus_alphas_cumprod[self.timesteps[timestep_idx + 1]].to(x_0.dtype) * noise
            )
    
        # 如果不返回字典，则返回前一个样本的元组
        if not return_dict:
            return (prev_sample,)
    
        # 否则返回一致性解码器调度器输出
        return ConsistencyDecoderSchedulerOutput(prev_sample=prev_sample)
```