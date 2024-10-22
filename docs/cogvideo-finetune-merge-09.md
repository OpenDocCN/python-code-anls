# CogVideo & CogVideoX 微调代码源码解析（十）



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

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\sampling_utils.py`

```py
# 导入 PyTorch 库
import torch
# 从 scipy 导入积分功能
from scipy import integrate

# 从上层模块导入 append_dims 函数
from ...util import append_dims
# 从 einops 导入 rearrange 函数，用于重排张量
from einops import rearrange


# 定义无动态阈值类
class NoDynamicThresholding:
    # 定义调用方法，接受无条件输入、条件输入和缩放因子
    def __call__(self, uncond, cond, scale):
        # 如果 scale 是张量，则调整其维度以匹配条件的维度
        scale = append_dims(scale, cond.ndim) if isinstance(scale, torch.Tensor) else scale
        # 返回无条件输入和缩放后的条件与无条件差值之和
        return uncond + scale * (cond - uncond)


# 定义静态阈值类
class StaticThresholding:
    # 定义调用方法，接受无条件输入、条件输入和缩放因子
    def __call__(self, uncond, cond, scale):
        # 计算无条件输入和缩放后的条件与无条件差值之和
        result = uncond + scale * (cond - uncond)
        # 将结果限制在 -1.0 到 1.0 之间
        result = torch.clamp(result, min=-1.0, max=1.0)
        # 返回处理后的结果
        return result


# 定义动态阈值函数
def dynamic_threshold(x, p=0.95):
    # 获取输入张量的维度
    N, T, C, H, W = x.shape
    # 将张量重排为适合计算的形状
    x = rearrange(x, "n t c h w -> n c (t h w)")
    # 计算给定分位数的左侧和右侧阈值
    l, r = x.quantile(q=torch.tensor([1 - p, p], device=x.device), dim=-1, keepdim=True)
    # 计算阈值的最大值
    s = torch.maximum(-l, r)
    # 创建阈值掩码，用于过滤
    threshold_mask = (s > 1).expand(-1, -1, H * W * T)
    # 如果阈值掩码中有任何 True 值，则应用阈值处理
    if threshold_mask.any():
        x = torch.where(threshold_mask, x.clamp(min=-1 * s, max=s), x)
    # 恢复张量的原始形状
    x = rearrange(x, "n c (t h w) -> n t c h w", t=T, h=H, w=W)
    # 返回处理后的张量
    return x


# 定义第二种动态阈值处理函数
def dynamic_thresholding2(x0):
    p = 0.995  # 参考论文“Imagen”中的超参数
    # 保存输入张量的原始数据类型
    origin_dtype = x0.dtype
    # 将输入张量转换为浮点数
    x0 = x0.to(torch.float32)
    # 计算输入张量绝对值的分位数
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    # 将分位数与 1 进行比较并调整维度
    s = append_dims(torch.maximum(s, torch.ones_like(s).to(s.device)), x0.dim())
    # 限制输入张量的值在 -s 和 s 之间
    x0 = torch.clamp(x0, -s, s)  # / s
    # 返回转换为原始数据类型的张量
    return x0.to(origin_dtype)


# 定义潜在动态阈值处理函数
def latent_dynamic_thresholding(x0):
    p = 0.9995  # 参考论文中的超参数
    # 保存输入张量的原始数据类型
    origin_dtype = x0.dtype
    # 将输入张量转换为浮点数
    x0 = x0.to(torch.float32)
    # 计算输入张量绝对值的分位数
    s = torch.quantile(torch.abs(x0), p, dim=2)
    # 调整分位数的维度以匹配输入张量
    s = append_dims(s, x0.dim())
    # 限制输入张量的值在 -s 和 s 之间，并进行归一化处理
    x0 = torch.clamp(x0, -s, s) / s
    # 返回转换为原始数据类型的张量
    return x0.to(origin_dtype)


# 定义第三种动态阈值处理函数
def dynamic_thresholding3(x0):
    p = 0.995  # 参考论文“Imagen”中的超参数
    # 保存输入张量的原始数据类型
    origin_dtype = x0.dtype
    # 将输入张量转换为浮点数
    x0 = x0.to(torch.float32)
    # 计算输入张量绝对值的分位数
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    # 将分位数与 1 进行比较并调整维度
    s = append_dims(torch.maximum(s, torch.ones_like(s).to(s.device)), x0.dim())
    # 限制输入张量的值在 -s 和 s 之间
    x0 = torch.clamp(x0, -s, s)  # / s
    # 返回转换为原始数据类型的张量
    return x0.to(origin_dtype)


# 定义动态阈值类
class DynamicThresholding:
    # 定义调用方法，接受无条件输入、条件输入和缩放因子
    def __call__(self, uncond, cond, scale):
        # 计算无条件输入的均值和标准差
        mean = uncond.mean()
        std = uncond.std()
        # 计算无条件输入和缩放后的条件与无条件差值之和
        result = uncond + scale * (cond - uncond)
        # 计算结果的均值和标准差
        result_mean, result_std = result.mean(), result.std()
        # 标准化结果，使其具有相同的标准差
        result = (result - result_mean) / result_std * std
        # result = dynamic_thresholding3(result)  # 可选的进一步处理
        # 返回处理后的结果
        return result


# 定义动态阈值版本1类
class DynamicThresholdingV1:
    # 初始化时接收缩放因子
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
    # 定义一个函数，接受三个参数 uncond, cond, scale
    def __call__(self, uncond, cond, scale):
        # 计算结果，根据公式 uncond + scale * (cond - uncond)
        result = uncond + scale * (cond - uncond)
        # 对结果进行反缩放，除以缩放因子
        unscaled_result = result / self.scale_factor
        # 获取结果的形状信息，分别为 Batch size, Time steps, Channels, Height, Width
        B, T, C, H, W = unscaled_result.shape
        # 将结果重新排列成 "b t c h w" 的形式
        flattened = rearrange(unscaled_result, "b t c h w -> b c (t h w)")
        # 计算每个通道的均值，并在第二维度上增加一个维度
        means = flattened.mean(dim=2).unsqueeze(2)
        # 对结果进行重新中心化，减去均值
        recentered = flattened - means
        # 计算每个通道的绝对值的最大值
        magnitudes = recentered.abs().max()
        # 对结果进行归一化，除以最大值
        normalized = recentered / magnitudes
        # 对结果进行动态阈值处理
        thresholded = latent_dynamic_thresholding(normalized)
        # 对结果进行反归一化，乘以最大值
        denormalized = thresholded * magnitudes
        # 对结果进行重新中心化，加上均值
        uncentered = denormalized + means
        # 将结果重新排列成 "b c (t h w)" 的形式
        unflattened = rearrange(uncentered, "b c (t h w) -> b t c h w", t=T, h=H, w=W)
        # 对结果进行缩放，乘以缩放因子
        scaled_result = unflattened * self.scale_factor
        # 返回缩放后的结果
        return scaled_result
# 定义一个动态阈值处理类
class DynamicThresholdingV2:
    # 定义类的调用方法，接受无条件值、条件值和缩放比例作为参数
    def __call__(self, uncond, cond, scale):
        # 获取无条件值的形状信息
        B, T, C, H, W = uncond.shape
        # 计算条件值和无条件值的差值
        diff = cond - uncond
        # 计算最小目标值
        mim_target = uncond + diff * 4.0
        # 计算配置目标值
        cfg_target = uncond + diff * 8.0

        # 将最小目标值展平为二维数组
        mim_flattened = rearrange(mim_target, "b t c h w -> b c (t h w)")
        # 将配置目标值展平为二维数组
        cfg_flattened = rearrange(cfg_target, "b t c h w -> b c (t h w)")
        # 计算最小目标值的均值
        mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
        # 计算配置目标值的均值
        cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
        # 计算最小目标值的中心化值
        mim_centered = mim_flattened - mim_means
        # 计算配置目标值的中心化值
        cfg_centered = cfg_flattened - cfg_means

        # 计算最小目标值的标准差
        mim_scaleref = mim_centered.std(dim=2).unsqueeze(2)
        # 计算配置目标值的标准差
        cfg_scaleref = cfg_centered.std(dim=2).unsqueeze(2)

        # 对配置目标值进行重新归一化
        cfg_renormalized = cfg_centered / cfg_scaleref * mim_scaleref

        # 将结果还原为原始形状
        result = cfg_renormalized + cfg_means
        unflattened = rearrange(result, "b c (t h w) -> b t c h w", t=T, h=H, w=W)

        return unflattened


# 定义一个线性多步系数函数
def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    # 如果阶数大于i，则抛出异常
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    # 定义一个内部函数，接受tau作为参数
    def fn(tau):
        prod = 1.0
        # 遍历阶数
        for k in range(order):
            # 如果j等于k，则跳过
            if j == k:
                continue
            # 计算乘积
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    # 对内部函数进行积分
    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


# 定义一个获取祖先步长的函数
def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    # 如果eta为假，则返回sigma_to和0.0
    if not eta:
        return sigma_to, 0.0
    # 计算上行步长
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    # 计算下行步长
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


# 定义一个转换为d的函数
def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


# 定义一个转换为负对数sigma的函数
def to_neg_log_sigma(sigma):
    return sigma.log().neg()


# 定义一个转换为sigma的函数
def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()
```

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\sigma_sampling.py`

```py
import torch
import torch.distributed

from sat import mpu

from ...util import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        # 初始化函数，设置默认的 p_mean 和 p_std 值
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        # 调用函数，生成 n_samples 个样本
        # 通过随机数生成 log_sigma，其均值为 p_mean，标准差为 p_std
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        # 返回 log_sigma 的指数形式
        return log_sigma.exp()


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, uniform_sampling=False):
        # 初始化函数，设置离散化配置、索引数量、是否追加零、是否翻转和是否均匀采样
        self.num_idx = num_idx
        # 实例化离散化配置，得到 sigmas
        self.sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        # 获取数据并行的世界大小
        world_size = mpu.get_data_parallel_world_size()
        self.uniform_sampling = uniform_sampling
        if self.uniform_sampling:
            # 如果进行均匀采样
            i = 1
            while True:
                if world_size % i != 0 or num_idx % (world_size // i) != 0:
                    i += 1
                else:
                    # 计算组数
                    self.group_num = world_size // i
                    break

            assert self.group_num > 0
            assert world_size % self.group_num == 0
            # 计算每组的宽度，即每组中的排名数量
            self.group_width = world_size // self.group_num  # the number of rank in one group
            # 计算每个组的 sigma 区间
            self.sigma_interval = self.num_idx // self.group_num

    def idx_to_sigma(self, idx):
        # 根据索引获取 sigma
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None, return_idx=False):
        # 调用函数，生成 n_samples 个样本
        if self.uniform_sampling:
            # 如果进行均匀采样
            rank = mpu.get_data_parallel_rank()
            # 计算组索引
            group_index = rank // self.group_width
            # 生成索引，范围为 group_index * sigma_interval 到 (group_index + 1) * sigma_interval
            idx = default(
                rand,
                torch.randint(
                    group_index * self.sigma_interval, (group_index + 1) * self.sigma_interval, (n_samples,)
                ),
            )
        else:
            # 如果不进行均匀采样
            # 生成索引，范围为 0 到 num_idx
            idx = default(
                rand,
                torch.randint(0, self.num_idx, (n_samples,)),
            )
        if return_idx:
            # 如果返回索引
            # 返回索引对应的 sigma 和索引
            return self.idx_to_sigma(idx), idx
        else:
            # 如果不返回索引
            # 返回索引对应的 sigma
            return self.idx_to_sigma(idx)


class PartialDiscreteSampling:
    def __init__(self, discretization_config, total_num_idx, partial_num_idx, do_append_zero=False, flip=True):
        # 初始化函数，设置离散化配置、总索引数量、部分索引数量、是否追加零和是否翻转
        self.total_num_idx = total_num_idx
        self.partial_num_idx = partial_num_idx
        # 实例化离散化配置，得到 sigmas
        self.sigmas = instantiate_from_config(discretization_config)(
            total_num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        # 根据索引获取 sigma
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        # 调用函数，生成 n_samples 个样本
        # 生成索引，范围为 0 到 partial_num_idx
        idx = default(
            rand,
            torch.randint(0, self.partial_num_idx, (n_samples,)),
        )
        # 返回索引对应的 sigma
        return self.idx_to_sigma(idx)
```

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

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\wrappers.py`

```py
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从 packaging 库导入 version 模块以便进行版本比较
from packaging import version

# 定义 OpenAIWrapper 的模块路径
OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


# 定义身份包装器类，继承自 nn.Module
class IdentityWrapper(nn.Module):
    # 初始化函数，接收扩散模型、是否编译模型的标志和数据类型
    def __init__(self, diffusion_model, compile_model: bool = False, dtype: torch.dtype = torch.float32):
        # 调用父类的初始化函数
        super().__init__()
        # 判断 PyTorch 版本，选择是否编译模型
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_model
            else lambda x: x  # 如果不满足条件，返回原始输入
        )
        # 编译扩散模型并赋值给实例变量
        self.diffusion_model = compile(diffusion_model)
        # 设置数据类型
        self.dtype = dtype

    # 前向传播函数，接收任意数量的位置参数和关键字参数
    def forward(self, *args, **kwargs):
        # 调用扩散模型的前向传播，并返回结果
        return self.diffusion_model(*args, **kwargs)


# 定义 OpenAIWrapper 类，继承自 IdentityWrapper
class OpenAIWrapper(IdentityWrapper):
    # 重写前向传播函数，接收输入张量 x、时间步 t、上下文字典 c 以及其他关键字参数
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        # 将上下文字典中的每个张量转换为指定的数据类型
        for key in c:
            c[key] = c[key].to(self.dtype)

        # 如果输入张量是 4 维，按维度 1 拼接上下文中的 "concat" 张量
        if x.dim() == 4:
            x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        # 如果输入张量是 5 维，按维度 2 拼接上下文中的 "concat" 张量
        elif x.dim() == 5:
            x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=2)
        # 如果输入张量的维度不符合要求，抛出值错误
        else:
            raise ValueError("Input tensor must be 4D or 5D")

        # 调用扩散模型的前向传播，传入处理后的张量、时间步和上下文等
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )
```

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\__init__.py`

```py
# 从当前包的 denoiser 模块导入 Denoiser 类
from .denoiser import Denoiser
# 从当前包的 discretizer 模块导入 Discretization 类
from .discretizer import Discretization
# 从当前包的 model 模块导入 Decoder、Encoder 和 Model 类
from .model import Decoder, Encoder, Model
# 从当前包的 openaimodel 模块导入 UNetModel 类
from .openaimodel import UNetModel
# 从当前包的 sampling 模块导入 BaseDiffusionSampler 类
from .sampling import BaseDiffusionSampler
# 从当前包的 wrappers 模块导入 OpenAIWrapper 类
from .wrappers import OpenAIWrapper
```

# `.\cogvideo-finetune\sat\sgm\modules\distributions\distributions.py`

```py
# 导入 numpy 库并简化为 np
import numpy as np
# 导入 PyTorch 库
import torch


# 定义抽象类 AbstractDistribution
class AbstractDistribution:
    # 抽象方法，生成样本
    def sample(self):
        raise NotImplementedError()

    # 抽象方法，返回分布的众数
    def mode(self):
        raise NotImplementedError()


# 定义 DiracDistribution 类，继承自 AbstractDistribution
class DiracDistribution(AbstractDistribution):
    # 初始化 DiracDistribution，接受一个值
    def __init__(self, value):
        self.value = value

    # 重写 sample 方法，返回固定值
    def sample(self):
        return self.value

    # 重写 mode 方法，返回固定值
    def mode(self):
        return self.value


# 定义 DiagonalGaussianDistribution 类
class DiagonalGaussianDistribution(object):
    # 初始化，接受参数和一个确定性标志
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        # 将参数分为均值和对数方差
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # 将对数方差限制在指定范围内
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        # 计算标准差和方差
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        # 如果是确定性模式，将方差和标准差设为零
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    # 生成样本
    def sample(self):
        # 使用均值和标准差生成随机样本
        x = self.mean + self.std * torch.randn_like(self.mean).to(device=self.parameters.device)
        return x

    # 计算 KL 散度
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])  # 确定性时 KL 散度为 0
        else:
            if other is None:
                # 计算与标准正态分布的 KL 散度
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                # 计算两个分布间的 KL 散度
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    # 计算负对数似然
    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])  # 确定性时 NLL 为 0
        logtwopi = np.log(2.0 * np.pi)  # 计算 2π 的对数
        # 计算负对数似然值
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    # 返回均值
    def mode(self):
        return self.mean


# 定义 normal_kl 函数，计算两个高斯分布之间的 KL 散度
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    计算两个高斯分布之间的 KL 散度。
    形状会自动广播，支持批量比较和标量等用例。
    """
    tensor = None
    # 找到第一个是 Tensor 的对象
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    # 确保至少有一个参数是 Tensor
    assert tensor is not None, "at least one argument must be a Tensor"

    # 强制将方差转换为 Tensor
    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]
    # 计算并返回某种损失值，具体公式由多个部分组成
        return 0.5 * (  # 返回整个计算结果的一半
            -1.0 + logvar2 - logvar1 +  # 计算对数方差差值，并减去常数项
            torch.exp(logvar1 - logvar2) +  # 计算指数项，表示方差的相对差异
            ((mean1 - mean2) ** 2) * torch.exp(-logvar2)  # 计算均值差的平方乘以方差的倒数
        )
```

# `.\cogvideo-finetune\sat\sgm\modules\distributions\__init__.py`

```py
请提供需要注释的代码。
```

# `.\cogvideo-finetune\sat\sgm\modules\ema.py`

```py
# 导入 PyTorch 库
import torch
# 从 PyTorch 导入神经网络模块
from torch import nn


# 定义一个名为 LitEma 的类，继承自 nn.Module
class LitEma(nn.Module):
    # 初始化函数，接受模型、衰减因子和更新次数的使用标志
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        # 调用父类构造函数
        super().__init__()
        # 检查衰减因子是否在有效范围内
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        # 初始化模型参数名称到阴影参数名称的映射字典
        self.m_name2s_name = {}
        # 注册衰减因子为一个缓冲区
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        # 根据是否使用更新次数注册相应的缓冲区
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int) if use_num_upates else torch.tensor(-1, dtype=torch.int),
        )

        # 遍历模型的每个命名参数
        for name, p in model.named_parameters():
            # 只处理需要梯度的参数
            if p.requires_grad:
                # 将参数名称中的点替换为字符，以便注册为缓冲区
                s_name = name.replace(".", "")
                # 更新名称映射字典
                self.m_name2s_name.update({name: s_name})
                # 注册参数的副本为缓冲区
                self.register_buffer(s_name, p.clone().detach().data)

        # 初始化存储的参数列表
        self.collected_params = []

    # 重置更新次数的函数
    def reset_num_updates(self):
        # 删除当前的更新次数缓冲区
        del self.num_updates
        # 注册新的更新次数缓冲区，初始值为 0
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    # 前向传播函数，更新阴影参数
    def forward(self, model):
        # 获取当前的衰减因子
        decay = self.decay

        # 如果更新次数有效，更新衰减因子
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        # 计算 1 减去衰减因子
        one_minus_decay = 1.0 - decay

        # 在不计算梯度的情况下执行以下操作
        with torch.no_grad():
            # 获取模型的参数字典
            m_param = dict(model.named_parameters())
            # 获取当前缓冲区中的阴影参数字典
            shadow_params = dict(self.named_buffers())

            # 遍历模型参数
            for key in m_param:
                # 只处理需要梯度的参数
                if m_param[key].requires_grad:
                    # 获取阴影参数名称
                    sname = self.m_name2s_name[key]
                    # 将阴影参数转换为与模型参数相同的类型
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    # 更新阴影参数
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    # 确保该参数不在名称映射中
                    assert not key in self.m_name2s_name

    # 将阴影参数复制回模型的函数
    def copy_to(self, model):
        # 获取模型参数和阴影参数的字典
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        # 遍历模型参数
        for key in m_param:
            # 只处理需要梯度的参数
            if m_param[key].requires_grad:
                # 将阴影参数的数据复制到模型参数
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                # 确保该参数不在名称映射中
                assert not key in self.m_name2s_name

    # 存储当前参数以便稍后恢复的函数
    def store(self, parameters):
        """
        保存当前参数以便稍后恢复。
        参数:
          parameters: 可迭代的 `torch.nn.Parameter`；要临时存储的参数。
        """
        # 将当前参数的副本存储在列表中
        self.collected_params = [param.clone() for param in parameters]
    # 定义一个恢复参数的方法
    def restore(self, parameters):
        # 文档字符串，说明此方法的作用及参数
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        # 遍历收集的参数和传入的参数，进行一一对应
        for c_param, param in zip(self.collected_params, parameters):
            # 将收集的参数数据复制到当前参数
            param.data.copy_(c_param.data)
```

# `.\cogvideo-finetune\sat\sgm\modules\encoders\modules.py`

```py
# 导入数学库
import math
# 导入上下文管理器的空上下文
from contextlib import nullcontext
# 导入部分函数的工具
from functools import partial
# 导入类型提示相关的类型
from typing import Dict, List, Optional, Tuple, Union

# 导入 Kornia 图像处理库
import kornia
# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 einops 库的重排列和重复函数
from einops import rearrange, repeat
# 导入 OmegaConf 库的 ListConfig
from omegaconf import ListConfig
# 导入 PyTorch 的检查点工具
from torch.utils.checkpoint import checkpoint
# 导入 Hugging Face 的 T5 模型和分词器
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
)

# 从自定义工具模块导入多个函数
from ...util import (
    append_dims,
    autocast,
    count_params,
    default,
    disabled_train,
    expand_dims_like,
    instantiate_from_config,
)


# 定义一个抽象的嵌入模型类，继承自 PyTorch 的 nn.Module
class AbstractEmbModel(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类初始化方法
        super().__init__()
        # 初始化是否可训练的标志
        self._is_trainable = None
        # 初始化 UCG 速率
        self._ucg_rate = None
        # 初始化输入键
        self._input_key = None

    # 返回是否可训练的属性
    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    # 返回 UCG 速率属性，可能是浮点数或张量
    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    # 返回输入键属性
    @property
    def input_key(self) -> str:
        return self._input_key

    # 设置是否可训练的属性
    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    # 设置 UCG 速率属性
    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    # 设置输入键属性
    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    # 删除是否可训练的属性
    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    # 删除 UCG 速率属性
    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    # 删除输入键属性
    @input_key.deleter
    def input_key(self):
        del self._input_key


# 定义一个通用条件器类，继承自 PyTorch 的 nn.Module
class GeneralConditioner(nn.Module):
    # 定义输出维度到键的映射
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    # 定义键到拼接维度的映射
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}
    # 初始化方法，接受嵌入模型配置及相关参数
        def __init__(self, emb_models: Union[List, ListConfig], cor_embs=[], cor_p=[]):
            # 调用父类初始化方法
            super().__init__()
            # 用于存储嵌入模型实例的列表
            embedders = []
            # 遍历每个嵌入模型的配置
            for n, embconfig in enumerate(emb_models):
                # 根据配置实例化嵌入模型
                embedder = instantiate_from_config(embconfig)
                # 确保实例是 AbstractEmbModel 的子类
                assert isinstance(
                    embedder, AbstractEmbModel
                ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
                # 设置嵌入模型是否可训练
                embedder.is_trainable = embconfig.get("is_trainable", False)
                # 设置嵌入模型的 ucg_rate 参数
                embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
                # 如果模型不可训练
                if not embedder.is_trainable:
                    # 禁用训练方法
                    embedder.train = disabled_train
                    # 将模型参数的 requires_grad 属性设为 False
                    for param in embedder.parameters():
                        param.requires_grad = False
                    # 将模型设置为评估模式
                    embedder.eval()
                # 打印嵌入模型的初始化信息
                print(
                    f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                    f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
                )
    
                # 检查配置中是否有 input_key，并赋值给嵌入模型
                if "input_key" in embconfig:
                    embedder.input_key = embconfig["input_key"]
                # 检查配置中是否有 input_keys，并赋值给嵌入模型
                elif "input_keys" in embconfig:
                    embedder.input_keys = embconfig["input_keys"]
                # 如果都没有，抛出 KeyError
                else:
                    raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")
    
                # 设置嵌入模型的 legacy_ucg_value 参数
                embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
                # 如果有 legacy_ucg_val，则初始化随机状态
                if embedder.legacy_ucg_val is not None:
                    embedder.ucg_prng = np.random.RandomState()
    
                # 将嵌入模型添加到列表中
                embedders.append(embedder)
            # 将嵌入模型列表转换为 nn.ModuleList
            self.embedders = nn.ModuleList(embedders)
    
            # 如果有 cor_embs，确保 cor_p 的长度正确
            if len(cor_embs) > 0:
                assert len(cor_p) == 2 ** len(cor_embs)
            # 设置相关嵌入和参数
            self.cor_embs = cor_embs
            self.cor_p = cor_p
    
        # 获取 UCG 值的方法，可能会基于概率进行赋值
        def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
            # 确保 legacy_ucg_val 不为 None
            assert embedder.legacy_ucg_val is not None
            # 获取 ucg_rate 参数
            p = embedder.ucg_rate
            # 获取 legacy_ucg_val 值
            val = embedder.legacy_ucg_val
            # 遍历 batch 中的输入数据
            for i in range(len(batch[embedder.input_key])):
                # 根据概率选择是否替换为 legacy_ucg_val
                if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[embedder.input_key][i] = val
            # 返回更新后的 batch
            return batch
    
        # 获取 UCG 值的方法，基于条件进行赋值
        def surely_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict, cond_or_not) -> Dict:
            # 确保 legacy_ucg_val 不为 None
            assert embedder.legacy_ucg_val is not None
            # 获取 legacy_ucg_val 值
            val = embedder.legacy_ucg_val
            # 遍历 batch 中的输入数据
            for i in range(len(batch[embedder.input_key])):
                # 如果条件为真，则替换为 legacy_ucg_val
                if cond_or_not[i]:
                    batch[embedder.input_key][i] = val
            # 返回更新后的 batch
            return batch
    
        # 获取单个嵌入的方法
        def get_single_embedding(
            self,
            embedder,
            batch,
            output,
            cond_or_not: Optional[np.ndarray] = None,
            force_zero_embeddings: Optional[List] = None,
    ):
        # 根据 embedder 是否可训练选择适当的上下文管理器
        embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
        # 进入上下文管理器以控制梯度计算
        with embedding_context():
            # 检查 embedder 是否有输入键属性并且不为 None
            if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                # 检查 embedder 的遗留 UCG 值是否不为 None
                if embedder.legacy_ucg_val is not None:
                    # 如果条件为 None，获取可能的 UCG 值
                    if cond_or_not is None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    # 否则，确保获取 UCG 值
                    else:
                        batch = self.surely_get_ucg_val(embedder, batch, cond_or_not)
                # 使用指定的输入键从 batch 中获取嵌入输出
                emb_out = embedder(batch[embedder.input_key])
            # 检查 embedder 是否有输入键列表
            elif hasattr(embedder, "input_keys"):
                # 解包 batch 中的输入键以获取嵌入输出
                emb_out = embedder(*[batch[k] for k in embedder.input_keys])
        # 确保嵌入输出是张量或序列类型
        assert isinstance(
            emb_out, (torch.Tensor, list, tuple)
        ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
        # 如果嵌入输出不是列表或元组，则将其转换为列表
        if not isinstance(emb_out, (list, tuple)):
            emb_out = [emb_out]
        # 遍历嵌入输出中的每个嵌入
        for emb in emb_out:
            # 根据嵌入的维度获取对应的输出键
            out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
            # 如果 UCG 率大于 0 且没有遗留 UCG 值
            if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                # 如果条件为 None，随机应用 UCG 率
                if cond_or_not is None:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device)),
                            emb,
                        )
                        * emb
                    )
                # 否则，基于条件应用 UCG
                else:
                    emb = (
                        expand_dims_like(
                            torch.tensor(1 - cond_or_not, dtype=emb.dtype, device=emb.device),
                            emb,
                        )
                        * emb
                    )
            # 如果输入键在强制零嵌入列表中，将嵌入置为零
            if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                emb = torch.zeros_like(emb)
            # 如果输出中已有该输出键，则拼接嵌入
            if out_key in output:
                output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
            # 否则，直接保存嵌入
            else:
                output[out_key] = emb
        # 返回最终输出字典
        return output
    # 定义一个前向传播函数，接收批次数据和强制零嵌入列表
        def forward(self, batch: Dict, force_zero_embeddings: Optional[List] = None) -> Dict:
            # 初始化输出字典
            output = dict()
            # 如果没有强制零嵌入，初始化为空列表
            if force_zero_embeddings is None:
                force_zero_embeddings = []
    
            # 如果相关嵌入列表不为空
            if len(self.cor_embs) > 0:
                # 获取批次大小
                batch_size = len(batch[list(batch.keys())[0]])
                # 根据相关概率随机选择索引
                rand_idx = np.random.choice(len(self.cor_p), size=(batch_size,), p=self.cor_p)
                # 遍历相关嵌入索引
                for emb_idx in self.cor_embs:
                    # 计算条件是否满足
                    cond_or_not = rand_idx % 2
                    # 更新随机索引
                    rand_idx //= 2
                    # 获取单个嵌入并更新输出字典
                    output = self.get_single_embedding(
                        self.embedders[emb_idx],
                        batch,
                        output=output,
                        cond_or_not=cond_or_not,
                        force_zero_embeddings=force_zero_embeddings,
                    )
    
            # 遍历所有嵌入
            for i, embedder in enumerate(self.embedders):
                # 如果当前索引在相关嵌入列表中，则跳过
                if i in self.cor_embs:
                    continue
                # 获取单个嵌入并更新输出字典
                output = self.get_single_embedding(
                    embedder, batch, output=output, force_zero_embeddings=force_zero_embeddings
                )
            # 返回最终输出字典
            return output
    
        # 定义获取无条件条件的函数
        def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
            # 如果没有强制无条件嵌入，初始化为空列表
            if force_uc_zero_embeddings is None:
                force_uc_zero_embeddings = []
            # 初始化无条件生成率列表
            ucg_rates = list()
            # 遍历所有嵌入，保存其生成率并将生成率设置为零
            for embedder in self.embedders:
                ucg_rates.append(embedder.ucg_rate)
                embedder.ucg_rate = 0.0
            # 保存当前相关嵌入和概率
            cor_embs = self.cor_embs
            cor_p = self.cor_p
            # 清空相关嵌入和概率
            self.cor_embs = []
            self.cor_p = []
    
            # 计算条件输出
            c = self(batch_c)
            # 计算无条件输出
            uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)
    
            # 恢复每个嵌入的生成率
            for embedder, rate in zip(self.embedders, ucg_rates):
                embedder.ucg_rate = rate
            # 恢复相关嵌入和概率
            self.cor_embs = cor_embs
            self.cor_p = cor_p
    
            # 返回条件输出和无条件输出
            return c, uc
# 定义一个名为 FrozenT5Embedder 的类，继承自 AbstractEmbModel
class FrozenT5Embedder(AbstractEmbModel):
    """使用 T5 变换器编码器处理文本"""

    # 初始化方法，设置模型的基本参数
    def __init__(
        self,
        model_dir="google/t5-v1_1-xxl",  # 模型目录，默认是 T5 模型
        device="cuda",                   # 设备设置，默认使用 GPU
        max_length=77,                   # 输入文本的最大长度
        freeze=True,                     # 是否冻结模型参数，默认是冻结
        cache_dir=None,                  # 缓存目录，默认无
    ):
        super().__init__()               # 调用父类的初始化方法
        # 检查模型目录是否为默认 T5 模型
        if model_dir is not "google/t5-v1_1-xxl":
            # 从指定目录加载 tokenizer 和 transformer 模型
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir)
        else:
            # 从指定目录加载 tokenizer 和 transformer 模型，同时指定缓存目录
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir, cache_dir=cache_dir)
        # 设置设备
        self.device = device
        # 设置最大输入长度
        self.max_length = max_length
        # 如果需要冻结模型参数，调用 freeze 方法
        if freeze:
            self.freeze()

    # 定义冻结方法
    def freeze(self):
        # 将 transformer 设置为评估模式
        self.transformer = self.transformer.eval()

        # 遍历模型参数，设置为不需要梯度更新
        for param in self.parameters():
            param.requires_grad = False

    # @autocast 注解（注释掉的），用于自动混合精度计算
    def forward(self, text):
        # 使用 tokenizer 对输入文本进行编码，返回批处理编码结果
        batch_encoding = self.tokenizer(
            text,
            truncation=True,                # 超出最大长度时进行截断
            max_length=self.max_length,     # 设置最大长度
            return_length=True,             # 返回编码长度
            return_overflowing_tokens=False, # 不返回溢出的 tokens
            padding="max_length",           # 填充到最大长度
            return_tensors="pt",           # 返回 PyTorch 张量
        )
        # 将输入 id 转移到指定设备
        tokens = batch_encoding["input_ids"].to(self.device)
        # 在禁用自动混合精度的上下文中进行前向传播
        with torch.autocast("cuda", enabled=False):
            # 使用 transformer 进行前向传播，获取输出
            outputs = self.transformer(input_ids=tokens)
        # 获取 transformer 输出的最后隐藏状态
        z = outputs.last_hidden_state
        # 返回最后隐藏状态
        return z

    # 定义编码方法，直接调用 forward 方法
    def encode(self, text):
        return self(text)  # 将输入文本传递给 forward 方法进行编码
```