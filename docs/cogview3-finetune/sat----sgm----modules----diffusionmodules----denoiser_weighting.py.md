# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\denoiser_weighting.py`

```
# 导入 PyTorch 库
import torch


# 定义一个单位加权类
class UnitWeighting:
    # 定义可调用方法，接收参数 sigma
    def __call__(self, sigma):
        # 返回与 sigma 形状相同的全 1 张量，设备与 sigma 相同
        return torch.ones_like(sigma, device=sigma.device)


# 定义 EDM 加权类
class EDMWeighting:
    # 初始化方法，设置 sigma_data 的默认值为 0.5
    def __init__(self, sigma_data=0.5):
        # 将传入的 sigma_data 存储为实例变量
        self.sigma_data = sigma_data

    # 定义可调用方法，接收参数 sigma
    def __call__(self, sigma):
        # 根据公式计算加权值并返回
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


# 定义 V 加权类，继承自 EDMWeighting
class VWeighting(EDMWeighting):
    # 初始化方法
    def __init__(self):
        # 调用父类初始化方法，设置 sigma_data 为 1.0
        super().__init__(sigma_data=1.0)


# 定义 Eps 加权类
class EpsWeighting:
    # 定义可调用方法，接收参数 sigma
    def __call__(self, sigma):
        # 返回 sigma 的 -2 次幂
        return sigma**-2.0
```