# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\discretizer.py`

```py
# 从 abc 模块导入抽象方法，用于定义抽象基类
from abc import abstractmethod
# 从 functools 模块导入 partial，用于创建部分应用的函数
from functools import partial

# 导入 numpy 库，通常用于数值计算
import numpy as np
# 导入 PyTorch 库，通常用于深度学习
import torch

# 从自定义模块中导入 make_beta_schedule 函数，用于生成 beta 调度
from ...modules.diffusionmodules.util import make_beta_schedule
# 从自定义模块中导入 append_zero 函数，用于处理 sigma 数组
from ...util import append_zero


# 定义一个函数，用于生成大致均匀间隔的步数
def generate_roughly_equally_spaced_steps(num_substeps: int, max_step: int) -> np.ndarray:
    # 使用 linspace 生成从 max_step-1 到 0 的均匀间隔的数组，反转并转换为整数
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


# 定义一个抽象基类 Discretization
class Discretization:
    # 定义一个可调用方法，用于处理输入参数
    def __call__(self, n, do_append_zero=True, device="cpu", flip=False, return_idx=False):
        # 根据 return_idx 的值获取 sigma 和索引
        if return_idx:
            sigmas, idx = self.get_sigmas(n, device=device, return_idx=return_idx)
        else:
            # 获取 sigma，不返回索引
            sigmas = self.get_sigmas(n, device=device, return_idx=return_idx)
        # 如果 do_append_zero 为真，则在 sigmas 末尾添加零
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        # 根据 flip 的值决定返回的 sigmas 和索引
        if return_idx:
            return sigmas if not flip else torch.flip(sigmas, (0,)), idx
        else:
            return sigmas if not flip else torch.flip(sigmas, (0,))

    # 定义一个抽象方法 get_sigmas，必须在子类中实现
    @abstractmethod
    def get_sigmas(self, n, device):
        pass


# 定义 EDMDiscretization 类，继承自 Discretization
class EDMDiscretization(Discretization):
    # 初始化方法，设置 sigma_min、sigma_max 和 rho 的默认值
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min  # 设置最小 sigma 值
        self.sigma_max = sigma_max  # 设置最大 sigma 值
        self.rho = rho  # 设置 rho 值

    # 实现 get_sigmas 方法
    def get_sigmas(self, n, device="cpu"):
        # 在指定设备上生成从 0 到 1 的均匀分布数组
        ramp = torch.linspace(0, 1, n, device=device)
        # 计算 sigma_min 和 sigma_max 的倒数
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        # 根据公式计算 sigmas
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas  # 返回计算出的 sigmas


# 定义 LegacyDDPMDiscretization 类，继承自 Discretization
class LegacyDDPMDiscretization(Discretization):
    # 初始化方法，设置线性开始、结束和时间步数的默认值
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
    ):
        super().__init__()  # 调用父类的初始化方法
        self.num_timesteps = num_timesteps  # 设置时间步数
        # 生成 beta 调度并计算对应的 alpha 值
        betas = make_beta_schedule("linear", num_timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1.0 - betas  # 计算 alpha 值
        # 计算 alpha 值的累积乘积
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # 创建一个部分应用的 torch.tensor 函数，指定数据类型
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

    # 实现 get_sigmas 方法
    def get_sigmas(self, n, device="cpu"):
        # 如果 n 小于时间步数，生成均匀间隔的时间步
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]  # 根据时间步获取对应的 alpha 值
        # 如果 n 等于时间步数，直接使用累积 alpha 值
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        # 如果 n 超过时间步数，抛出值错误
        else:
            raise ValueError

        # 创建一个部分应用的 torch.tensor 函数，指定数据类型和设备
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        # 计算 sigmas
        sigmas = to_torch((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return torch.flip(sigmas, (0,))  # 反转返回 sigmas


# 定义 ZeroSNRDDPMDiscretization 类，继承自 Discretization
class ZeroSNRDDPMDiscretization(Discretization):
    # 初始化方法，设置线性开始、结束、时间步数和其他参数的默认值
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.0,  # 噪声调度参数
        keep_start=False,  # 是否保持起始状态
        post_shift=False,  # 是否在后续进行偏移
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 如果保留起始值且没有后移，则对线性起始值进行缩放
        if keep_start and not post_shift:
            linear_start = linear_start / (shift_scale + (1 - shift_scale) * linear_start)
        # 设置时间步数
        self.num_timesteps = num_timesteps
        # 创建线性调度的 beta 值
        betas = make_beta_schedule("linear", num_timesteps, linear_start=linear_start, linear_end=linear_end)
        # 计算 alpha 值
        alphas = 1.0 - betas
        # 计算累积的 alpha 值
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # 将部分功能固定为转换为 torch 张量
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

        # SNR 偏移处理
        if not post_shift:
            # 调整累积的 alpha 值
            self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1 - shift_scale) * self.alphas_cumprod)

        # 存储后移状态
        self.post_shift = post_shift
        # 存储偏移缩放因子
        self.shift_scale = shift_scale

    def get_sigmas(self, n, device="cpu", return_idx=False):
        # 如果 n 小于时间步数，则生成等间隔的时间步
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            # 取出对应的累积 alpha 值
            alphas_cumprod = self.alphas_cumprod[timesteps]
        # 如果 n 等于时间步数，直接使用全部累积 alpha 值
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        # 如果 n 超过时间步数，则抛出错误
        else:
            raise ValueError

        # 将累积 alpha 值转换为 torch 张量
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod = to_torch(alphas_cumprod)
        # 计算累积 alpha 值的平方根
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        # 备份初始和最终的累积 alpha 值的平方根
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

        # 调整平方根的累积 alpha 值
        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

        # 如果开启后移，则进一步调整平方根的累积 alpha 值
        if self.post_shift:
            alphas_cumprod_sqrt = (
                alphas_cumprod_sqrt**2 / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)
            ) ** 0.5

        # 根据是否返回索引来决定返回值
        if return_idx:
            return torch.flip(alphas_cumprod_sqrt, (0,)), timesteps
        else:
            # 返回反转的平方根 alpha 值
            return torch.flip(alphas_cumprod_sqrt, (0,))  # sqrt(alpha_t): 0 -> 0.99
```