# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\denoiser_scaling.py`

```py
# 从 abc 模块导入 ABC 类和 abstractmethod 装饰器
from abc import ABC, abstractmethod
# 从 typing 模块导入 Any 和 Tuple 类型
from typing import Any, Tuple

# 导入 torch 库
import torch


# 定义一个抽象基类 DenoiserScaling，继承自 ABC
class DenoiserScaling(ABC):
    # 定义一个抽象方法 __call__，接受一个 torch.Tensor 参数并返回一个四元组
    @abstractmethod
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


# 定义 EDMScaling 类
class EDMScaling:
    # 初始化方法，接受一个 sigma_data 参数，默认值为 0.5
    def __init__(self, sigma_data: float = 0.5):
        # 设置实例变量 sigma_data
        self.sigma_data = sigma_data

    # 定义 __call__ 方法，接受一个 torch.Tensor 参数并返回一个四元组
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 计算 c_skip，使用 sigma 和 sigma_data
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        # 计算 c_out，结合 sigma 和 sigma_data
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        # 计算 c_in，涉及 sigma 和 sigma_data
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        # 计算 c_noise，基于 sigma 的对数
        c_noise = 0.25 * sigma.log()
        # 返回计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 EpsScaling 类
class EpsScaling:
    # 定义 __call__ 方法，接受一个 torch.Tensor 参数并返回一个四元组
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 创建与 sigma 形状相同的全 1 张量，设备与 sigma 相同
        c_skip = torch.ones_like(sigma, device=sigma.device)
        # c_out 为 sigma 的负值
        c_out = -sigma
        # 计算 c_in，涉及 sigma 的平方
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        # 复制 sigma 作为 c_noise
        c_noise = sigma.clone()
        # 返回计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 VScaling 类
class VScaling:
    # 定义 __call__ 方法，接受一个 torch.Tensor 参数并返回一个四元组
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 计算 c_skip，涉及 sigma 的平方
        c_skip = 1.0 / (sigma**2 + 1.0)
        # 计算 c_out，结合 sigma 的负值
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        # 计算 c_in，涉及 sigma 的平方
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        # 复制 sigma 作为 c_noise
        c_noise = sigma.clone()
        # 返回计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 VScalingWithEDMcNoise 类，继承自 DenoiserScaling
class VScalingWithEDMcNoise(DenoiserScaling):
    # 定义 __call__ 方法，接受一个 torch.Tensor 参数并返回一个四元组
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 计算 c_skip，涉及 sigma 的平方
        c_skip = 1.0 / (sigma**2 + 1.0)
        # 计算 c_out，结合 sigma 的负值
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        # 计算 c_in，涉及 sigma 的平方
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        # 计算 c_noise，基于 sigma 的对数
        c_noise = 0.25 * sigma.log()
        # 返回计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 VideoScaling 类，类似于 VScaling
class VideoScaling:  # similar to VScaling
    # 定义 __call__ 方法，接受一个 torch.Tensor 和可选的其他模型输入
    def __call__(
        self, alphas_cumprod_sqrt: torch.Tensor, **additional_model_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 将 alphas_cumprod_sqrt 赋值给 c_skip
        c_skip = alphas_cumprod_sqrt
        # 计算 c_out，涉及 alphas_cumprod_sqrt
        c_out = -((1 - alphas_cumprod_sqrt**2) ** 0.5)
        # 创建与 alphas_cumprod_sqrt 形状相同的全 1 张量，设备相同
        c_in = torch.ones_like(alphas_cumprod_sqrt, device=alphas_cumprod_sqrt.device)
        # 复制 additional_model_inputs 中的 "idx" 作为 c_noise
        c_noise = additional_model_inputs["idx"].clone()
        # 返回计算结果
        return c_skip, c_out, c_in, c_noise
```