# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\discretizer.py`

```py
# 从 abc 模块导入抽象方法，用于定义抽象基类
from abc import abstractmethod
# 从 functools 模块导入 partial，用于创建偏函数
from functools import partial

# 导入 numpy 库，主要用于数值计算
import numpy as np
# 导入 torch 库，主要用于深度学习和张量操作
import torch

# 从自定义模块中导入 make_beta_schedule 函数，用于生成 beta 时间表
from ...modules.diffusionmodules.util import make_beta_schedule
# 从自定义模块中导入 append_zero 函数，用于处理数组
from ...util import append_zero


# 定义函数 generate_roughly_equally_spaced_steps，用于生成大致均匀间隔的步骤
def generate_roughly_equally_spaced_steps(
    num_substeps: int, max_step: int
) -> np.ndarray:
    # 生成从 max_step-1 到 0 的均匀间隔数组，包含 num_substeps 个元素，并将结果反转
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]

# 定义函数 sub_generate_roughly_equally_spaced_steps，用于生成两个子步骤的均匀间隔
def sub_generate_roughly_equally_spaced_steps(
    num_substeps_1: int, num_substeps_2: int, max_step: int
) -> np.ndarray:
    # 生成第二组子步骤的均匀间隔
    substeps_2 = np.linspace(max_step - 1, 0, num_substeps_2, endpoint=False).astype(int)[::-1]
    # 生成第一组子步骤的均匀间隔
    substeps_1 = np.linspace(num_substeps_2 - 1, 0, num_substeps_1, endpoint=False).astype(int)[::-1]
    # 返回根据第一组子步骤索引获取第二组子步骤的数组
    return substeps_2[substeps_1]

# 定义离散化的抽象基类 Discretization
class Discretization:
    # 定义可调用方法，接受 n、do_append_zero、device 和 flip 参数
    def __call__(self, n, do_append_zero=True, device="cpu", flip=False):
        # 调用 get_sigmas 方法获取 sigma 值
        sigmas = self.get_sigmas(n, device=device)
        # 如果 do_append_zero 为真，向 sigmas 添加零
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        # 根据 flip 参数决定是否反转 sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))

    # 定义抽象方法 get_sigmas，必须在子类中实现
    @abstractmethod
    def get_sigmas(self, n, device):
        pass


# 定义 EDMDiscretization 类，继承自 Discretization
class EDMDiscretization(Discretization):
    # 初始化方法，设置 sigma 的最小值、最大值和 rho 值
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    # 实现抽象方法 get_sigmas，计算 sigma 值
    def get_sigmas(self, n, device="cpu"):
        # 生成从 0 到 1 的等间隔张量
        ramp = torch.linspace(0, 1, n, device=device)
        # 计算最小和最大 rho 的倒数
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        # 根据公式计算 sigmas
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        # 返回计算得到的 sigmas
        return sigmas


# 定义 LegacyDDPMDiscretization 类，继承自 Discretization
class LegacyDDPMDiscretization(Discretization):
    # 初始化方法，设置线性开始、结束和时间步数
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
    ):
        # 调用父类初始化方法
        super().__init__()
        self.num_timesteps = num_timesteps
        # 使用 make_beta_schedule 生成 beta 时间表
        betas = make_beta_schedule(
            "linear", num_timesteps, linear_start=linear_start, linear_end=linear_end
        )
        # 计算 alphas
        alphas = 1.0 - betas
        # 计算累积的 alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # 创建将 numpy 数组转换为 torch 张量的偏函数
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

    # 实现抽象方法 get_sigmas，根据 n 计算 sigma 值
    def get_sigmas(self, n, device="cpu"):
        # 如果 n 小于总时间步数
        if n < self.num_timesteps:
            # 生成大致均匀间隔的时间步
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            # 获取对应的累积 alphas
            alphas_cumprod = self.alphas_cumprod[timesteps]
        # 如果 n 等于总时间步数
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        # 如果 n 大于总时间步数，抛出异常
        else:
            raise ValueError

        # 创建将 numpy 数组转换为 torch 张量的偏函数，指定设备
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        # 计算 sigmas 值
        sigmas = to_torch((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        # 返回反转的 sigmas
        return torch.flip(sigmas, (0,)) # sigma_t: 14.4 -> 0.029

# 定义 ZeroSNRDDPMDiscretization 类，继承自 Discretization
class ZeroSNRDDPMDiscretization(Discretization):
    # 初始化方法，设置线性开始、结束、时间步数和 shift_scale
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.,
    # 初始化父类
        ):
            super().__init__()
            # 设置时间步数
            self.num_timesteps = num_timesteps
            # 生成线性调度的 beta 值
            betas = make_beta_schedule(
                "linear", num_timesteps, linear_start=linear_start, linear_end=linear_end
            )
            # 计算 alpha 值
            alphas = 1.0 - betas
            # 计算累积的 alpha 值
            self.alphas_cumprod = np.cumprod(alphas, axis=0)
            # 将数据转换为 torch 张量的函数
            self.to_torch = partial(torch.tensor, dtype=torch.float32)
    
            # 对累积的 alpha 值进行缩放
            self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1-shift_scale) * self.alphas_cumprod)
    
        # 获取 sigma 值的方法
        def get_sigmas(self, n, device="cpu", return_idx=False):
            # 判断请求的时间步数是否小于总时间步数
            if n < self.num_timesteps:
                # 生成等间距的时间步
                timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
                # 获取对应的累积 alpha 值
                alphas_cumprod = self.alphas_cumprod[timesteps]
            # 判断请求的时间步数是否等于总时间步数
            elif n == self.num_timesteps:
                alphas_cumprod = self.alphas_cumprod
            # 如果超出范围，抛出异常
            else:
                raise ValueError
    
            # 将 alpha 值转换为 torch 张量
            to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
            alphas_cumprod = to_torch(alphas_cumprod)
            # 计算累积 alpha 值的平方根
            alphas_cumprod_sqrt = alphas_cumprod.sqrt()
            # 克隆初始和最终的 alpha 值
            alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
            alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
    
            # 对平方根进行归一化处理
            alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
            alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)
    
            # 根据返回标志返回结果
            if return_idx:
                return torch.flip(alphas_cumprod_sqrt, (0,)), timesteps
            else:
                # 返回反转的平方根 alpha 值
                return torch.flip(alphas_cumprod_sqrt, (0,)) # sqrt(alpha_t): 0 -> 0.99
            
        # 使对象可调用的方法
        def __call__(self, n, do_append_zero=True, device="cpu", flip=False, return_idx=False):
            # 根据返回标志调用获取 sigma 值的方法
            if return_idx:
                sigmas, idx = self.get_sigmas(n, device=device, return_idx=True)
                sigmas = append_zero(sigmas) if do_append_zero else sigmas
                # 根据 flip 标志返回结果
                return (sigmas, idx) if not flip else (torch.flip(sigmas, (0,)), torch.flip(idx, (0,)))
            else:
                # 获取 sigma 值并处理
                sigmas = self.get_sigmas(n, device=device)
                sigmas = append_zero(sigmas) if do_append_zero else sigmas
                # 根据 flip 标志返回结果
                return sigmas if not flip else torch.flip(sigmas, (0,))
```