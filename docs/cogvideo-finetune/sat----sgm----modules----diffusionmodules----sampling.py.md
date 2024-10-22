# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\sampling.py`

```py
# 部分移植自 https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

# 从类型提示模块导入字典和联合类型
from typing import Dict, Union

# 导入 PyTorch 库
import torch
# 从 OmegaConf 导入列表配置和 OmegaConf 类
from omegaconf import ListConfig, OmegaConf
# 从 tqdm 导入进度条功能
from tqdm import tqdm

# 从自定义模块导入必要的函数
from ...modules.diffusionmodules.sampling_utils import (
    get_ancestral_step,  # 获取祖先步骤的函数
    linear_multistep_coeff,  # 线性多步骤系数的函数
    to_d,  # 转换到 d 的函数
    to_neg_log_sigma,  # 转换为负对数 sigma 的函数
    to_sigma,  # 转换为 sigma 的函数
)
# 从 util 模块导入辅助函数
from ...util import append_dims, default, instantiate_from_config
from ...util import SeededNoise  # 导入带种子的噪声生成器

# 从 guiders 模块导入动态 CFG 类
from .guiders import DynamicCFG

# 默认引导器配置
DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}

# 定义基础扩散采样器类
class BaseDiffusionSampler:
    # 初始化方法，设置相关配置
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],  # 离散化配置
        num_steps: Union[int, None] = None,  # 步骤数
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,  # 引导器配置
        verbose: bool = False,  # 是否详细输出
        device: str = "cuda",  # 使用的设备
    ):
        self.num_steps = num_steps  # 保存步骤数
        # 根据配置实例化离散化对象
        self.discretization = instantiate_from_config(discretization_config)
        # 根据配置实例化引导器对象，使用默认值如果未提供
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose  # 设置详细输出标志
        self.device = device  # 设置设备

    # 准备采样循环的方法
    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        # 计算 sigma 值
        sigmas = self.discretization(self.num_steps if num_steps is None else num_steps, device=self.device)
        # 默认情况下使用条件输入
        uc = default(uc, cond)

        # 对输入进行缩放
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)  # 获取 sigma 数量

        # 创建与输入样本数相同的全 1 张量
        s_in = x.new_ones([x.shape[0]]).float()

        # 返回准备好的参数
        return x, s_in, sigmas, num_sigmas, cond, uc

    # 去噪声的方法
    def denoise(self, x, denoiser, sigma, cond, uc):
        # 准备输入并进行去噪声处理
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        # 使用引导器进一步处理去噪声结果
        denoised = self.guider(denoised, sigma)
        # 返回去噪声后的结果
        return denoised

    # 获取 sigma 生成器的方法
    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)  # 创建 sigma 生成器范围
        if self.verbose:  # 如果启用详细输出
            print("#" * 30, " Sampling setting ", "#" * 30)  # 输出分隔符
            print(f"Sampler: {self.__class__.__name__}")  # 输出采样器类名
            print(f"Discretization: {self.discretization.__class__.__name__}")  # 输出离散化类名
            print(f"Guider: {self.guider.__class__.__name__}")  # 输出引导器类名
            # 包装 sigma 生成器为 tqdm 对象，以便显示进度条
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        # 返回 sigma 生成器
        return sigma_generator

# 定义单步扩散采样器类，继承自基础扩散采样器
class SingleStepDiffusionSampler(BaseDiffusionSampler):
    # 定义采样步骤的抽象方法
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError  # 抛出未实现错误

    # 欧拉步骤的方法
    def euler_step(self, x, d, dt):
        # 计算下一个状态
        return x + dt * d

# 定义 EDM 采样器类，继承自单步扩散采样器
class EDMSampler(SingleStepDiffusionSampler):
    # 初始化方法，设置相关参数
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类初始化

        self.s_churn = s_churn  # 设置 churn 参数
        self.s_tmin = s_tmin  # 设置最小时间
        self.s_tmax = s_tmax  # 设置最大时间
        self.s_noise = s_noise  # 设置噪声参数
    # 定义sampler_step函数，用于执行采样步骤
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        # 计算sigma_hat，用于计算噪声
        sigma_hat = sigma * (gamma + 1.0)
        # 如果gamma大于0，生成服从标准正态分布的随机数eps，并将其乘以s_noise，再加到x上
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        # 使用denoiser对x进行去噪，得到denoised
        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        # 计算d，用于后续步骤
        d = to_d(x, sigma_hat, denoised)
        # 计算dt，用于后续步骤
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        # 使用欧拉步骤更新x
        euler_step = self.euler_step(x, d, dt)
        # 使用可能的修正步骤更新x
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, denoiser, cond, uc)
        # 返回更新后的x
        return x

    # 定义__call__函数，用于执行整个采样过程
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        # 准备采样循环所需的变量
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        # 遍历sigmas，执行采样步骤
        for i in self.get_sigma_gen(num_sigmas):
            # 计算gamma，用于控制噪声的大小
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            # 执行采样步骤
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        # 返回最终的x
        return x
# 定义 DDIMSampler 类，继承自 SingleStepDiffusionSampler 类
class DDIMSampler(SingleStepDiffusionSampler):
    # 初始化方法，接受 s_noise 参数，默认值为 0.1
    def __init__(self, s_noise=0.1, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # 设置实例属性 s_noise 为传入的 s_noise 参数值
        self.s_noise = s_noise

    # sampler_step 方法，接受 sigma、next_sigma、denoiser、x、cond、uc 和 s_noise 参数
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, s_noise=0.0):
        # 使用 denoiser 对 x 进行去噪，得到 denoised
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        # 计算 d，使用 to_d 函数
        d = to_d(x, sigma, denoised)
        # 计算 dt
        dt = append_dims(next_sigma * (1 - s_noise**2) ** 0.5 - sigma, x.ndim)
        # 计算 euler_step
        euler_step = x + dt * d + s_noise * append_dims(next_sigma, x.ndim) * torch.randn_like(x)
        # 调用 possible_correction_step 方法，得到 x
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, denoiser, cond, uc)
        # 返回 x
        return x

    # __call__ 方法，接受 denoiser、x、cond、uc 和 num_steps 参数
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        # 调用 prepare_sampling_loop 方法，得到 x、s_in、sigmas、num_sigmas、cond 和 uc
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        # 遍历 sigmas
        for i in self.get_sigma_gen(num_sigmas):
            # 调用 sampler_step 方法，得到 x
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                self.s_noise,
            )

        # 返回 x
        return x


# 定义 AncestralSampler 类，继承自 SingleStepDiffusionSampler 类
class AncestralSampler(SingleStepDiffusionSampler):
    # 初始化方法，接受 eta 和 s_noise 参数，默认值分别为 1.0 和 1.0
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # 设置实例属性 eta 为传入的 eta 参数值
        self.eta = eta
        # 设置实例属性 s_noise 为传入的 s_noise 参数值
        self.s_noise = s_noise
        # 设置实例属性 noise_sampler 为一个 lambda 函数，用于生成噪声
        self.noise_sampler = lambda x: torch.randn_like(x)

    # ancestral_euler_step 方法，接受 x、denoised、sigma 和 sigma_down 参数
    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        # 计算 d，使用 to_d 函数
        d = to_d(x, sigma, denoised)
        # 计算 dt
        dt = append_dims(sigma_down - sigma, x.ndim)
        # 调用 euler_step 方法，得到结果
        return self.euler_step(x, d, dt)

    # ancestral_step 方法，接受 x、sigma、next_sigma 和 sigma_up 参数
    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        # 根据条件进行赋值操作
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        # 返回结果
        return x

    # __call__ 方法，接受 denoiser、x、cond、uc 和 num_steps 参数
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        # 调用 prepare_sampling_loop 方法，得到 x、s_in、sigmas、num_sigmas、cond 和 uc
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        # 遍历 sigmas
        for i in self.get_sigma_gen(num_sigmas):
            # 调用 sampler_step 方法，得到 x
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        # 返回 x
        return x


# 定义 LinearMultistepSampler 类，继承自 BaseDiffusionSampler 类
class LinearMultistepSampler(BaseDiffusionSampler):
    # 初始化方法，接受 order 参数，默认值为 4
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # 设置实例属性 order 为传入的 order 参数值
        self.order = order
    # 定义可调用方法，用于执行去噪操作
        def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
            # 准备采样循环，返回处理后的输入、sigma、以及条件信息
            x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)
    
            # 初始化去噪结果列表
            ds = []
            # 将 sigma 从计算图中分离并转移到 CPU，然后转换为 NumPy 数组
            sigmas_cpu = sigmas.detach().cpu().numpy()
            # 遍历生成的 sigma 值
            for i in self.get_sigma_gen(num_sigmas):
                # 计算当前 sigma
                sigma = s_in * sigmas[i]
                # 使用去噪器处理输入数据，获取去噪结果
                denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs)
                # 进一步处理去噪结果
                denoised = self.guider(denoised, sigma)
                # 将当前输入、sigma 和去噪结果转换为目标格式
                d = to_d(x, sigma, denoised)
                # 将去噪结果添加到结果列表
                ds.append(d)
                # 如果结果列表超过预设顺序，则移除最旧的结果
                if len(ds) > self.order:
                    ds.pop(0)
                # 计算当前顺序
                cur_order = min(i + 1, self.order)
                # 计算线性多步法的系数
                coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
                # 更新输入 x，使用加权和的方式结合去噪结果
                x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    
            # 返回最终的去噪结果
            return x
# 定义一个基于 Euler 方法的 EDM 采样器类，继承自 EDMSampler
class EulerEDMSampler(EDMSampler):
    # 可能的修正步骤，接受多个参数
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        # 直接返回 euler_step，没有进行任何修正
        return euler_step


# 定义一个基于 Heun 方法的 EDM 采样器类，继承自 EDMSampler
class HeunEDMSampler(EDMSampler):
    # 可能的修正步骤，接受多个参数
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        # 检查 next_sigma 的总和是否小于 1e-14
        if torch.sum(next_sigma) < 1e-14:
            # 如果所有噪声水平为 0，则返回 euler_step，避免网络评估
            return euler_step
        else:
            # 使用 denoiser 去噪 euler_step
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            # 将 euler_step 和去噪结果转换为新的 d 值
            d_new = to_d(euler_step, next_sigma, denoised)
            # 计算 d 的新值，取 d 和 d_new 的平均
            d_prime = (d + d_new) / 2.0

            # 如果噪声水平不为 0，则应用修正
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step)
            # 返回修正后的 x
            return x


# 定义一个基于 Euler 的祖先采样器类，继承自 AncestralSampler
class EulerAncestralSampler(AncestralSampler):
    # 进行采样步骤，接受多个参数
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        # 获取祖先步骤的上下界
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        # 去噪 x
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        # 进行 Euler 采样步骤
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        # 进行祖先步骤
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        # 返回最终的 x
        return x


# 定义一个基于 DPM++ 的祖先采样器类，继承自 AncestralSampler
class DPMPP2SAncestralSampler(AncestralSampler):
    # 获取变量，接受 sigma 和 sigma_down
    def get_variables(self, sigma, sigma_down):
        # 将 sigma 和 sigma_down 转换为负对数
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        # 计算时间差 h
        h = t_next - t
        # 计算 s 值
        s = t + 0.5 * h
        # 返回 h, s, t, t_next
        return h, s, t, t_next

    # 计算多重值，接受 h, s, t, t_next
    def get_mult(self, h, s, t, t_next):
        # 计算多个乘数
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        # 返回所有计算出的乘数
        return mult1, mult2, mult3, mult4

    # 进行采样步骤，接受多个参数
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        # 获取祖先步骤的上下界
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        # 去噪 x
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        # 进行 Euler 采样步骤
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        # 检查 sigma_down 的总和是否小于 1e-14
        if torch.sum(sigma_down) < 1e-14:
            # 如果所有噪声水平为 0，则返回 x_euler，避免网络评估
            x = x_euler
        else:
            # 获取变量
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            # 计算多重值并调整维度
            mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)]

            # 计算新的 x2
            x2 = mult[0] * x - mult[1] * denoised
            # 对 x2 进行去噪
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            # 计算最终的 x_dpmpp2s
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # 如果噪声水平不为 0，则应用修正
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        # 进行祖先步骤
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        # 返回最终的 x
        return x


# 定义一个基于 DPM++ 的采样器类，继承自 BaseDiffusionSampler
class DPMPP2MSampler(BaseDiffusionSampler):
    # 定义一个获取变量的函数，接受当前和下一个噪声级别，以及可选的上一个噪声级别
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        # 将 sigma 和 next_sigma 转换为负对数形式
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        # 计算两个时间点之间的差值
        h = t_next - t

        # 如果上一个噪声级别存在
        if previous_sigma is not None:
            # 计算当前时间与上一个时间的差值
            h_last = t - to_neg_log_sigma(previous_sigma)
            # 计算当前和上一个时间差的比值
            r = h_last / h
            # 返回差值 h、比值 r、当前和下一个时间
            return h, r, t, t_next
        else:
            # 返回差值 h 和当前、下一个时间，但不返回比值 r
            return h, None, t, t_next

    # 定义一个获取乘数的函数，接受多个参数
    def get_mult(self, h, r, t, t_next, previous_sigma):
        # 计算当前和下一个时间的 sigma 乘数
        mult1 = to_sigma(t_next) / to_sigma(t)
        # 计算 h 的负值的指数减一
        mult2 = (-h).expm1()

        # 如果上一个噪声级别存在
        if previous_sigma is not None:
            # 计算与 r 相关的乘数
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            # 返回所有乘数
            return mult1, mult2, mult3, mult4
        else:
            # 返回前两个乘数
            return mult1, mult2

    # 定义采样步骤函数，接受多个参数
    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        # 使用去噪器对输入 x 进行去噪
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        # 获取变量 h、r、t、t_next
        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        # 获取乘数，并将维度调整以匹配 x 的维度
        mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, r, t, t_next, previous_sigma)]

        # 计算标准化后的 x
        x_standard = mult[0] * x - mult[1] * denoised
        # 如果没有旧的去噪结果或下一个噪声级别接近零
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # 保存网络评估，如果所有噪声级别为 0 或为第一步
            return x_standard, denoised
        else:
            # 计算去噪后的旧结果
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            # 计算高级 x
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # 如果噪声级别不为 0 且不是第一步，则应用修正
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard)

        # 返回最终的 x 和去噪结果
        return x, denoised

    # 定义调用函数，接受去噪器、输入 x、条件和其他可选参数
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        # 准备采样循环的参数，包括对输入 x、条件、噪声级别等的处理
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        # 初始化旧去噪结果
        old_denoised = None
        # 遍历生成的噪声级别
        for i in self.get_sigma_gen(num_sigmas):
            # 执行采样步骤，更新 x 和旧去噪结果
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        # 返回最终的 x
        return x
# 定义 SDEDPMPP2MSampler 类，继承自 BaseDiffusionSampler
class SDEDPMPP2MSampler(BaseDiffusionSampler):
    # 获取变量 h、r 和时间参数 t、t_next
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        # 将 sigma 和 next_sigma 转换为负对数形式
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        # 计算 h 为 t_next 和 t 的差值
        h = t_next - t

        # 如果 previous_sigma 不为 None
        if previous_sigma is not None:
            # 计算上一个 sigma 的负对数值
            h_last = t - to_neg_log_sigma(previous_sigma)
            # 计算 r 为 h_last 和 h 的比值
            r = h_last / h
            # 返回 h、r、t 和 t_next
            return h, r, t, t_next
        else:
            # 返回 h 和 None（无 r），以及 t 和 t_next
            return h, None, t, t_next

    # 计算乘数值
    def get_mult(self, h, r, t, t_next, previous_sigma):
        # 计算 mult1 为 t_next 和 t 的 sigma 比值乘以 h 的负指数
        mult1 = to_sigma(t_next) / to_sigma(t) * (-h).exp()
        # 计算 mult2 为 (-2*h) 的 expm1 值
        mult2 = (-2 * h).expm1()

        # 如果 previous_sigma 不为 None
        if previous_sigma is not None:
            # 计算 mult3 为 1 + 1/(2*r)
            mult3 = 1 + 1 / (2 * r)
            # 计算 mult4 为 1/(2*r)
            mult4 = 1 / (2 * r)
            # 返回 mult1、mult2、mult3 和 mult4
            return mult1, mult2, mult3, mult4
        else:
            # 返回 mult1 和 mult2
            return mult1, mult2

    # 执行采样步骤
    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        # 使用 denoiser 对 x 进行去噪处理
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        # 获取 h、r、t 和 t_next
        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        # 计算乘数，并调整维度
        mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, r, t, t_next, previous_sigma)]
        # 计算噪声乘数并调整维度
        mult_noise = append_dims(next_sigma * (1 - (-2 * h).exp()) ** 0.5, x.ndim)

        # 计算标准化后的 x
        x_standard = mult[0] * x - mult[1] * denoised + mult_noise * torch.randn_like(x)
        # 如果 old_denoised 为 None 或 next_sigma 的和小于 1e-14
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # 返回标准化后的 x 和去噪后的结果
            return x_standard, denoised
        else:
            # 计算去噪的差异
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            # 计算高级 x
            x_advanced = mult[0] * x - mult[1] * denoised_d + mult_noise * torch.randn_like(x)

            # 如果噪声水平不为 0 且不是第一步，应用修正
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard)

        # 返回最终的 x 和去噪后的结果
        return x, denoised

    # 调用采样器，执行采样循环
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, **kwargs):
        # 准备采样循环，初始化输入和 sigma
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        # 初始化 old_denoised 为 None
        old_denoised = None
        # 遍历 sigma 生成器
        for i in self.get_sigma_gen(num_sigmas):
            # 执行采样步骤并更新 x 和 old_denoised
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        # 返回最终的 x
        return x


# 定义 SdeditEDMSampler 类，继承自 EulerEDMSampler
class SdeditEDMSampler(EulerEDMSampler):
    # 初始化函数，设置编辑比例
    def __init__(self, edit_ratio=0.5, *args, **kwargs):
        # 调用父类的初始化函数
        super().__init__(*args, **kwargs)

        # 设置编辑比例
        self.edit_ratio = edit_ratio
    # 定义一个可调用的方法，接受多个参数进行图像去噪
    def __call__(self, denoiser, image, randn, cond, uc=None, num_steps=None, edit_ratio=None):
        # 克隆 randn，创建 randn_unit 用于后续计算
        randn_unit = randn.clone()
        # 准备采样循环，处理 randn、条件、未条件和步骤数
        randn, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(randn, cond, uc, num_steps)
    
        # 如果未指定 num_steps，则使用对象的默认步骤数
        if num_steps is None:
            num_steps = self.num_steps
        # 如果未指定 edit_ratio，则使用对象的默认编辑比例
        if edit_ratio is None:
            edit_ratio = self.edit_ratio
        # 初始化 x 为 None，用于后续存储结果
        x = None
    
        # 遍历 sigma 生成器，获取每个 sigma 的值
        for i in self.get_sigma_gen(num_sigmas):
            # 如果当前步骤比例小于 edit_ratio，则跳过此次循环
            if i / num_steps < edit_ratio:
                continue
            # 如果 x 为 None，则初始化 x 为图像与噪声的组合
            if x is None:
                x = image + randn_unit * append_dims(s_in * sigmas[i], len(randn_unit.shape))
    
            # 计算 gamma 值，依据条件限制调整
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            # 进行一次采样步骤，更新 x 的值
            x = self.sampler_step(
                s_in * sigmas[i],     # 当前 sigma 的输入
                s_in * sigmas[i + 1], # 下一个 sigma 的输入
                denoiser,             # 去噪器
                x,                    # 当前图像
                cond,                 # 条件信息
                uc,                   # 未条件信息
                gamma,                # gamma 值
            )
    
        # 返回最终处理后的图像
        return x
# 定义一个名为 VideoDDIMSampler 的类，继承自 BaseDiffusionSampler
class VideoDDIMSampler(BaseDiffusionSampler):
    # 初始化函数，接受固定帧数和 sdedit 标志，及其他参数
    def __init__(self, fixed_frames=0, sdedit=False, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 设置固定帧数
        self.fixed_frames = fixed_frames
        # 设置 sdedit 标志
        self.sdedit = sdedit

    # 准备采样循环，接受输入数据和条件
    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        # 进行离散化，计算 alpha 的平方根累积乘积和时间步
        alpha_cumprod_sqrt, timesteps = self.discretization(
            self.num_steps if num_steps is None else num_steps,  # 使用给定的步数或默认步数
            device=self.device,  # 指定设备
            return_idx=True,  # 返回索引
            do_append_zero=False,  # 不追加零
        )
        # 在 alpha_cumprod_sqrt 末尾添加一个值为 1 的新张量
        alpha_cumprod_sqrt = torch.cat([alpha_cumprod_sqrt, alpha_cumprod_sqrt.new_ones([1])])
        # 创建一个新的时间步张量，并在开头添加一个值为 -1 的零张量
        timesteps = torch.cat([torch.tensor(list(timesteps)).new_zeros([1]) - 1, torch.tensor(list(timesteps))])

        # 如果 uc 为空，使用 cond 作为默认值
        uc = default(uc, cond)

        # 计算 alpha_cumprod_sqrt 的元素数量
        num_sigmas = len(alpha_cumprod_sqrt)

        # 创建一个新的张量 s_in，初始值为 1
        s_in = x.new_ones([x.shape[0]])

        # 返回多个变量
        return x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps

    # 去噪函数，接受多个输入参数
    def denoise(self, x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep=None, idx=None, scale=None, scale_emb=None):
        # 初始化额外模型输入的字典
        additional_model_inputs = {}

        # 检查 scale 是否为张量且不为 1
        if isinstance(scale, torch.Tensor) == False and scale == 1:
            # 为额外模型输入添加当前时间步的索引
            additional_model_inputs["idx"] = x.new_ones([x.shape[0]]) * timestep
            # 如果 scale_emb 不为 None，添加到额外输入
            if scale_emb is not None:
                additional_model_inputs["scale_emb"] = scale_emb
            # 调用去噪器进行去噪，并转换为 float32 类型
            denoised = denoiser(x, alpha_cumprod_sqrt, cond, **additional_model_inputs).to(torch.float32)
        else:
            # 创建一个新的索引张量，包含当前时间步的重复值
            additional_model_inputs["idx"] = torch.cat([x.new_ones([x.shape[0]]) * timestep] * 2)
            # 调用去噪器进行去噪，准备输入并转换为 float32 类型
            denoised = denoiser(
                *self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs
            ).to(torch.float32)
            # 如果 guider 是 DynamicCFG 的实例，进行动态调整
            if isinstance(self.guider, DynamicCFG):
                denoised = self.guider(
                    denoised, (1 - alpha_cumprod_sqrt**2) ** 0.5, step_index=self.num_steps - timestep, scale=scale
                )
            else:
                # 否则，进行普通的调整
                denoised = self.guider(denoised, (1 - alpha_cumprod_sqrt**2) ** 0.5, scale=scale)
        # 返回去噪后的结果
        return denoised

    # 采样步骤函数，接受多个输入参数
    def sampler_step(
        self,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
        scale=None,
        scale_emb=None,
    ):
        # 调用 denoise 方法获取去噪结果
        denoised = self.denoise(
            x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx, scale=scale, scale_emb=scale_emb
        ).to(torch.float32)

        # 计算 a_t 和 b_t 值
        a_t = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5
        b_t = next_alpha_cumprod_sqrt - alpha_cumprod_sqrt * a_t

        # 更新 x 的值，通过加权当前值和去噪后的值
        x = append_dims(a_t, x.ndim) * x + append_dims(b_t, x.ndim) * denoised
        # 返回更新后的 x
        return x
    # 定义一个可调用的方法，用于处理去噪过程
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, scale_emb=None):
        # 准备采样循环的输入，返回处理后的数据和相关参数
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
    
        # 遍历生成的 sigma 值
        for i in self.get_sigma_gen(num_sigmas):
            # 执行采样步骤，更新输入数据
            x = self.sampler_step(
                s_in * alpha_cumprod_sqrt[i],  # 当前 sigma 的缩放输入
                s_in * alpha_cumprod_sqrt[i + 1],  # 下一个 sigma 的缩放输入
                denoiser,  # 去噪器对象
                x,  # 当前输入
                cond,  # 条件输入
                uc,  # 可选的额外条件
                idx=self.num_steps - i,  # 当前步骤索引
                timestep=timesteps[-(i + 1)],  # 当前时间步
                scale=scale,  # 缩放因子
                scale_emb=scale_emb,  # 嵌入的缩放因子
            )
    
        # 返回处理后的结果
        return x
# 定义 VPSDEDPMPP2M 采样器类，继承自 VideoDDIMSampler
class VPSDEDPMPP2MSampler(VideoDDIMSampler):
    # 获取变量，计算多个参数
    def get_variables(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt=None):
        # 计算 alpha 的累积乘积
        alpha_cumprod = alpha_cumprod_sqrt**2
        # 计算 lamb 的对数值
        lamb = ((alpha_cumprod / (1 - alpha_cumprod)) ** 0.5).log()
        # 计算下一个 alpha 的累积乘积
        next_alpha_cumprod = next_alpha_cumprod_sqrt**2
        # 计算下一个 lamb 的对数值
        lamb_next = ((next_alpha_cumprod / (1 - next_alpha_cumprod)) ** 0.5).log()
        # 计算 h 值
        h = lamb_next - lamb

        # 如果存在前一个 alpha 的累积乘积
        if previous_alpha_cumprod_sqrt is not None:
            # 计算前一个 alpha 的累积乘积
            previous_alpha_cumprod = previous_alpha_cumprod_sqrt**2
            # 计算前一个 lamb 的对数值
            lamb_previous = ((previous_alpha_cumprod / (1 - previous_alpha_cumprod)) ** 0.5).log()
            # 计算 h_last 值
            h_last = lamb - lamb_previous
            # 计算 r 值
            r = h_last / h
            # 返回 h、r、lamb 和 lamb_next
            return h, r, lamb, lamb_next
        else:
            # 返回 h、None、lamb 和 lamb_next
            return h, None, lamb, lamb_next

    # 计算乘数
    def get_mult(self, h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt):
        # 计算第一个乘数
        mult1 = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5 * (-h).exp()
        # 计算第二个乘数
        mult2 = (-2 * h).expm1() * next_alpha_cumprod_sqrt

        # 如果存在前一个 alpha 的累积乘积
        if previous_alpha_cumprod_sqrt is not None:
            # 计算第三个乘数
            mult3 = 1 + 1 / (2 * r)
            # 计算第四个乘数
            mult4 = 1 / (2 * r)
            # 返回所有乘数
            return mult1, mult2, mult3, mult4
        else:
            # 返回前两个乘数
            return mult1, mult2

    # 执行采样步骤
    def sampler_step(
        self,
        old_denoised,
        previous_alpha_cumprod_sqrt,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
        scale=None,
        scale_emb=None,
    ):
        # 使用去噪器处理输入，得到去噪后的结果
        denoised = self.denoise(
            x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx, scale=scale, scale_emb=scale_emb
        ).to(torch.float32)
        # 如果索引为 1，返回去噪结果
        if idx == 1:
            return denoised, denoised

        # 获取相关变量
        h, r, lamb, lamb_next = self.get_variables(
            alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt
        )
        # 获取乘数
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        ]
        # 计算噪声乘数
        mult_noise = append_dims((1 - next_alpha_cumprod_sqrt**2) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5, x.ndim)

        # 计算标准化 x
        x_standard = mult[0] * x - mult[1] * denoised + mult_noise * torch.randn_like(x)
        # 如果 old_denoised 为 None 或者下一个 alpha 的累积乘积小于阈值
        if old_denoised is None or torch.sum(next_alpha_cumprod_sqrt) < 1e-14:
            # 返回标准化的 x 和去噪后的结果
            return x_standard, denoised
        else:
            # 计算去噪后的差异
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            # 计算高级 x
            x_advanced = mult[0] * x - mult[1] * denoised_d + mult_noise * torch.randn_like(x)

            # 更新 x
            x = x_advanced

        # 返回最终的 x 和去噪结果
        return x, denoised
    # 定义可调用方法，接收去噪器、输入数据、条件、上采样因子及其它参数
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, scale_emb=None):
        # 准备采样循环所需的输入及参数
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
    
        # 如果固定帧数大于0，提取前固定帧数的图像
        if self.fixed_frames > 0:
            prefix_frames = x[:, : self.fixed_frames]
        # 初始化去噪后的图像为 None
        old_denoised = None
        # 遍历生成的 sigma 值
        for i in self.get_sigma_gen(num_sigmas):
            # 如果固定帧数大于0，进行处理
            if self.fixed_frames > 0:
                # 如果启用 SD 编辑模式
                if self.sdedit:
                    # 生成与前缀帧同形状的随机噪声
                    rd = torch.randn_like(prefix_frames)
                    # 计算带噪声的前缀帧
                    noised_prefix_frames = alpha_cumprod_sqrt[i] * prefix_frames + rd * append_dims(
                        s_in * (1 - alpha_cumprod_sqrt[i] ** 2) ** 0.5, len(prefix_frames.shape)
                    )
                    # 将带噪声的前缀帧与剩余帧连接
                    x = torch.cat([noised_prefix_frames, x[:, self.fixed_frames :]], dim=1)
                else:
                    # 直接将前缀帧与剩余帧连接
                    x = torch.cat([prefix_frames, x[:, self.fixed_frames :]], dim=1)
            # 执行去噪步骤
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * alpha_cumprod_sqrt[i - 1],
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i + 1)],
                scale=scale,
                scale_emb=scale_emb,
            )
    
        # 如果固定帧数大于0，重构最终输出
        if self.fixed_frames > 0:
            x = torch.cat([prefix_frames, x[:, self.fixed_frames :]], dim=1)
    
        # 返回最终的去噪结果
        return x
# 定义 VPODEDPMPP2MSampler 类，继承自 VideoDDIMSampler
class VPODEDPMPP2MSampler(VideoDDIMSampler):
    # 获取变量，计算当前和下一个 alpha 的平方根
    def get_variables(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt=None):
        # 计算 alpha 的平方
        alpha_cumprod = alpha_cumprod_sqrt**2
        # 计算 lambda 值并取其对数
        lamb = ((alpha_cumprod / (1 - alpha_cumprod)) ** 0.5).log()
        # 计算下一个 alpha 的平方
        next_alpha_cumprod = next_alpha_cumprod_sqrt**2
        # 计算下一个 lambda 值并取其对数
        lamb_next = ((next_alpha_cumprod / (1 - next_alpha_cumprod)) ** 0.5).log()
        # 计算 h 值
        h = lamb_next - lamb

        # 如果提供了上一个 alpha 的平方根
        if previous_alpha_cumprod_sqrt is not None:
            # 计算上一个 alpha 的平方
            previous_alpha_cumprod = previous_alpha_cumprod_sqrt**2
            # 计算上一个 lambda 值并取其对数
            lamb_previous = ((previous_alpha_cumprod / (1 - previous_alpha_cumprod)) ** 0.5).log()
            # 计算上一个 h 值
            h_last = lamb - lamb_previous
            # 计算 r 值
            r = h_last / h
            # 返回 h, r, lamb, lamb_next
            return h, r, lamb, lamb_next
        else:
            # 如果没有上一个 alpha，返回 h 和其他计算值
            return h, None, lamb, lamb_next

    # 获取乘数
    def get_mult(self, h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt):
        # 计算第一个乘数
        mult1 = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5
        # 计算第二个乘数
        mult2 = (-h).expm1() * next_alpha_cumprod_sqrt

        # 如果提供了上一个 alpha 的平方根
        if previous_alpha_cumprod_sqrt is not None:
            # 计算第三个乘数
            mult3 = 1 + 1 / (2 * r)
            # 计算第四个乘数
            mult4 = 1 / (2 * r)
            # 返回所有乘数
            return mult1, mult2, mult3, mult4
        else:
            # 返回前两个乘数
            return mult1, mult2

    # 采样步骤
    def sampler_step(
        self,
        old_denoised,
        previous_alpha_cumprod_sqrt,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
    ):
        # 使用去噪器对输入 x 进行去噪处理
        denoised = self.denoise(x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx).to(torch.float32)
        # 如果索引为 1，返回去噪结果
        if idx == 1:
            return denoised, denoised

        # 获取变量
        h, r, lamb, lamb_next = self.get_variables(
            alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt
        )
        # 获取乘数并调整维度
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        ]

        # 计算标准化的 x
        x_standard = mult[0] * x - mult[1] * denoised
        # 如果没有旧的去噪结果或下一个 alpha 的平方根总和接近 0
        if old_denoised is None or torch.sum(next_alpha_cumprod_sqrt) < 1e-14:
            # 返回标准化的 x 和去噪结果
            return x_standard, denoised
        else:
            # 计算去噪后的结果
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            # 计算高级的 x
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # 更新 x
            x = x_advanced

        # 返回最终的 x 和去噪结果
        return x, denoised
    # 定义可调用对象，接受去噪器、输入数据、条件及其他参数
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, **kwargs):
        # 准备采样循环所需的输入数据和参数
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
    
        # 初始化旧去噪结果为 None
        old_denoised = None
        # 遍历生成的 sigma 值，进行采样步骤
        for i in self.get_sigma_gen(num_sigmas):
            # 执行单步采样，并更新当前输入和旧去噪结果
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * alpha_cumprod_sqrt[i - 1],  # 第一步不使用旧去噪
                s_in * alpha_cumprod_sqrt[i],  # 当前 sigma 的值
                s_in * alpha_cumprod_sqrt[i + 1],  # 下一个 sigma 的值
                denoiser,  # 去噪器
                x,  # 当前输入数据
                cond,  # 条件输入
                uc=uc,  # 额外条件（可选）
                idx=self.num_steps - i,  # 当前步骤索引
                timestep=timesteps[-(i + 1)],  # 当前时间步
            )
    
        # 返回最终生成的输入数据
        return x
```