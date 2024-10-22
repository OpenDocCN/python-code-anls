# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\denoiser_scaling.py`

```
# 从 abc 模块导入抽象基类和抽象方法
from abc import ABC, abstractmethod
# 从 typing 模块导入任意类型和元组
from typing import Any, Tuple

# 导入 PyTorch 库
import torch


# 定义一个抽象基类 DenoiserScaling
class DenoiserScaling(ABC):
    # 定义一个抽象方法 __call__
    @abstractmethod
    def __call__(
        self, sigma: torch.Tensor, **additional_model_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 抽象方法没有具体实现
        pass


# 定义 EDMScaling 类
class EDMScaling:
    # 初始化方法，接受一个 sigma 数据，默认值为 0.5
    def __init__(self, sigma_data: float = 0.5):
        # 将 sigma_data 保存为实例变量
        self.sigma_data = sigma_data

    # 定义一个可调用方法 __call__
    def __call__(
        self, sigma: torch.Tensor, **additional_model_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 计算 c_skip 的值
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        # 计算 c_out 的值
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        # 计算 c_in 的值
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        # 计算 c_noise 的值
        c_noise = 0.25 * sigma.log()
        # 返回四个计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 EpsScaling 类
class EpsScaling:
    # 定义一个可调用方法 __call__
    def __call__(
        self, sigma: torch.Tensor, **additional_model_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 创建与 sigma 相同形状的全 1 张量作为 c_skip
        c_skip = torch.ones_like(sigma, device=sigma.device)
        # 计算 c_out 的值为 -sigma
        c_out = -sigma
        # 计算 c_in 的值
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        # 复制 sigma 作为 c_noise
        c_noise = sigma.clone()
        # 返回四个计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 VScaling 类
class VScaling:
    # 定义一个可调用方法 __call__
    def __call__(
        self, sigma: torch.Tensor, **additional_model_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 计算 c_skip 的值
        c_skip = 1.0 / (sigma**2 + 1.0)
        # 计算 c_out 的值
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        # 计算 c_in 的值
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        # 复制 sigma 作为 c_noise
        c_noise = sigma.clone()
        # 返回四个计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 VScalingWithEDMcNoise 类，继承自 DenoiserScaling
class VScalingWithEDMcNoise(DenoiserScaling):
    # 定义一个可调用方法 __call__
    def __call__(
        self, sigma: torch.Tensor, **additional_model_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 计算 c_skip 的值
        c_skip = 1.0 / (sigma**2 + 1.0)
        # 计算 c_out 的值
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        # 计算 c_in 的值
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        # 计算 c_noise 的值
        c_noise = 0.25 * sigma.log()
        # 返回四个计算结果
        return c_skip, c_out, c_in, c_noise

# 定义 ZeroSNRScaling 类，类似于 VScaling
class ZeroSNRScaling: # similar to VScaling
    # 定义一个可调用方法 __call__
    def __call__(
        self, alphas_cumprod_sqrt: torch.Tensor, **additional_model_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 将 alphas_cumprod_sqrt 作为 c_skip
        c_skip = alphas_cumprod_sqrt
        # 计算 c_out 的值
        c_out = - (1 - alphas_cumprod_sqrt**2) ** 0.5
        # 创建与 alphas_cumprod_sqrt 相同形状的全 1 张量作为 c_in
        c_in = torch.ones_like(alphas_cumprod_sqrt, device=alphas_cumprod_sqrt.device)
        # 复制额外输入的 'idx' 作为 c_noise
        c_noise = additional_model_inputs['idx'].clone()
        # 返回四个计算结果
        return c_skip, c_out, c_in, c_noise
```