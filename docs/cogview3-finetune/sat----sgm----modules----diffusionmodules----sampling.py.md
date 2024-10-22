# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\sampling.py`

```
# 部分代码移植自 https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

# 从 typing 模块导入字典和联合类型
from typing import Dict, Union

# 导入 PyTorch 库
import torch
# 从 omegaconf 模块导入配置相关的类
from omegaconf import ListConfig, OmegaConf
# 导入 tqdm 库用于显示进度条
from tqdm import tqdm

# 从相对路径模块导入采样相关的工具函数
from ...modules.diffusionmodules.sampling_utils import (
    get_ancestral_step,  # 获取祖先步骤
    linear_multistep_coeff,  # 线性多步骤系数
    to_d,  # 转换为 d
    to_neg_log_sigma,  # 转换为负对数sigma
    to_sigma,  # 转换为 sigma
)
# 从相对路径模块导入离散化工具
from ...modules.diffusionmodules.discretizer import generate_roughly_equally_spaced_steps
# 从相对路径模块导入工具函数
from ...util import append_dims, default, instantiate_from_config

# 定义默认引导器配置
DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}

# 定义用于生成引导嵌入的函数
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    参考文献: https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            在这些时间步生成嵌入向量
        embedding_dim (`int`, *可选*, 默认为 512):
            生成的嵌入的维度
        dtype:
            生成嵌入的数据类型

    Returns:
        `torch.FloatTensor`: 形状为 `(len(timesteps), embedding_dim)` 的嵌入向量
    """
    # 确保输入张量是一个一维张量
    assert len(w.shape) == 1
    # 将输入乘以 1000.0
    w = w * 1000.0

    # 计算嵌入维度的一半
    half_dim = embedding_dim // 2
    # 计算基础嵌入的系数
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    # 生成嵌入基础，转换为指数形式并调整为目标设备和数据类型
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb).to(w.device).to(w.dtype)
    # 生成最终的嵌入向量
    emb = w.to(dtype)[:, None] * emb[None, :]
    # 将正弦和余弦值连接在一起
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度为奇数，进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    # 确保生成的嵌入形状与预期一致
    assert emb.shape == (w.shape[0], embedding_dim)
    # 返回生成的嵌入向量
    return emb

# 定义基础扩散采样器类
class BaseDiffusionSampler:
    # 初始化采样器
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],  # 离散化配置
        num_steps: Union[int, None] = None,  # 采样步数，默认为 None
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,  # 引导器配置，默认为 None
        cfg_cond_scale: Union[int, None] = None,  # 条件缩放参数，默认为 None
        cfg_cond_embed_dim: Union[int, None] = 256,  # 条件嵌入维度，默认为 256
        verbose: bool = False,  # 是否显示详细信息
        device: str = "cuda",  # 设备类型，默认为 CUDA
    ):
        # 设置采样步数
        self.num_steps = num_steps
        # 实例化离散化配置
        self.discretization = instantiate_from_config(discretization_config)
        # 实例化引导器配置
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )

        # 设置条件参数
        self.cfg_cond_scale = cfg_cond_scale
        self.cfg_cond_embed_dim = cfg_cond_embed_dim
        
        # 设置详细模式和设备
        self.verbose = verbose
        self.device = device

    # 准备采样循环的函数
    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        # 生成 sigma 值
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        # 默认使用条件
        uc = default(uc, cond)

        # 根据 sigma 计算 x 的调整
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        # 获取 sigma 的数量
        num_sigmas = len(sigmas)

        # 创建新的一维张量 s_in，初始值为 1
        s_in = x.new_ones([x.shape[0]]).float()

        # 返回调整后的 x 和其他参数
        return x, s_in, sigmas, num_sigmas, cond, uc
    # 定义去噪函数，接受输入x、去噪器denoiser、噪声水平sigma、条件cond和无条件uc
        def denoise(self, x, denoiser, sigma, cond, uc):
            # 检查条件缩放系数是否不为None
            if self.cfg_cond_scale is not None:
                # 获取输入批次的大小
                batch_size = x.shape[0]
                # 创建与批次大小相同的全1张量，并乘以条件缩放系数，生成缩放嵌入
                scale_emb = guidance_scale_embedding(torch.ones(batch_size, device=x.device) * self.cfg_cond_scale, embedding_dim=self.cfg_cond_embed_dim, dtype=x.dtype)
                # 使用去噪器处理输入，传入缩放嵌入
                denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), scale_emb=scale_emb)
            else:
                # 若无条件缩放系数，直接使用去噪器处理输入
                denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
            # 对去噪后的结果进行进一步引导处理
            denoised = self.guider(denoised, sigma)
            # 返回最终去噪结果
            return denoised
    
        # 定义生成sigma的函数，接受sigma数量num_sigmas
        def get_sigma_gen(self, num_sigmas):
            # 创建一个范围生成器，从0到num_sigmas-1
            sigma_generator = range(num_sigmas - 1)
            # 如果启用了详细输出
            if self.verbose:
                # 打印分隔线和采样设置信息
                print("#" * 30, " Sampling setting ", "#" * 30)
                print(f"Sampler: {self.__class__.__name__}")
                print(f"Discretization: {self.discretization.__class__.__name__}")
                print(f"Guider: {self.guider.__class__.__name__}")
                # 使用tqdm包装生成器以显示进度条
                sigma_generator = tqdm(
                    sigma_generator,
                    total=num_sigmas,
                    desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
                )
            # 返回sigma生成器
            return sigma_generator
# 定义一个单步扩散采样器类，继承自基本扩散采样器
class SingleStepDiffusionSampler(BaseDiffusionSampler):
    # 定义采样步骤方法，未实现
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        # 抛出未实现错误，表明该方法需在子类中实现
        raise NotImplementedError

    # 定义欧拉步骤方法，用于计算下一个状态
    def euler_step(self, x, d, dt):
        # 返回更新后的状态，基于当前状态、导数和时间增量
        return x + dt * d


# 定义 EDM 采样器类，继承自单步扩散采样器
class EDMSampler(SingleStepDiffusionSampler):
    # 初始化 EDM 采样器的参数
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # 设置采样器的参数
        self.s_churn = s_churn  # 变化率
        self.s_tmin = s_tmin    # 最小时间
        self.s_tmax = s_tmax    # 最大时间
        self.s_noise = s_noise  # 噪声强度

    # 定义采样步骤方法
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        # 计算调整后的 sigma 值
        sigma_hat = sigma * (gamma + 1.0)
        # 如果 gamma 大于 0，加入噪声
        if gamma > 0:
            # 生成与 x 形状相同的随机噪声
            eps = torch.randn_like(x) * self.s_noise
            # 更新 x 的值，加入噪声
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        # 去噪，得到去噪后的结果
        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        # 计算导数
        d = to_d(x, sigma_hat, denoised)
        # 计算时间增量
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        # 执行欧拉步骤，更新 x
        euler_step = self.euler_step(x, d, dt)
        # 进行可能的修正步骤，得到最终的 x
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        # 返回更新后的 x
        return x

    # 定义调用方法
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        # 准备采样循环所需的参数
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        # 遍历 sigma 值
        for i in self.get_sigma_gen(num_sigmas):
            # 计算 gamma 值
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            # 执行采样步骤，更新 x
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        # 返回最终的 x
        return x


# 定义 DDIM 采样器类，继承自单步扩散采样器
class DDIMSampler(SingleStepDiffusionSampler):
    # 初始化 DDIM 采样器的参数
    def __init__(
        self, s_noise=0.1, *args, **kwargs
    ):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # 设置噪声强度
        self.s_noise = s_noise

    # 定义采样步骤方法
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, s_noise=0.0):

        # 去噪，得到去噪后的结果
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        # 计算导数
        d = to_d(x, sigma, denoised)
        # 计算时间增量
        dt = append_dims(next_sigma * (1 - s_noise**2)**0.5 - sigma, x.ndim)

        # 计算欧拉步骤，加入噪声
        euler_step = x + dt * d + s_noise * append_dims(next_sigma, x.ndim) * torch.randn_like(x)

        # 进行可能的修正步骤，得到最终的 x
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        # 返回更新后的 x
        return x
    # 定义一个可调用的类方法，接收去噪器、输入数据、条件及其他参数
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        # 准备采样循环，返回处理后的数据和相关参数
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
    
        # 遍历生成的 sigma 值
        for i in self.get_sigma_gen(num_sigmas):
            # 执行采样步骤，更新输入数据 x
            x = self.sampler_step(
                s_in * sigmas[i],    # 当前 sigma 乘以输入信号
                s_in * sigmas[i + 1],# 下一个 sigma 乘以输入信号
                denoiser,            # 传递去噪器
                x,                   # 当前数据
                cond,                # 条件信息
                uc,                  # 可选的额外条件
                self.s_noise,        # 传递噪声信息
            )
    
        # 返回最终处理后的数据
        return x
# 定义一个继承自 SingleStepDiffusionSampler 的类 AncestralSampler
class AncestralSampler(SingleStepDiffusionSampler):
    # 初始化方法，设定默认参数 eta 和 s_noise
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # 设置 eta 属性
        self.eta = eta
        # 设置 s_noise 属性
        self.s_noise = s_noise
        # 定义噪声采样器，生成与输入形状相同的随机噪声
        self.noise_sampler = lambda x: torch.randn_like(x)

    # 定义 ancestral_euler_step 方法，用于执行欧拉步长
    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        # 计算偏导数 d
        d = to_d(x, sigma, denoised)
        # 将 sigma_down 和 sigma 的差值扩展到 x 的维度
        dt = append_dims(sigma_down - sigma, x.ndim)

        # 返回欧拉步长的结果
        return self.euler_step(x, d, dt)

    # 定义 ancestral_step 方法，执行采样步骤
    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        # 根据条件选择更新 x 的值
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,  # 检查 next_sigma 是否大于 0
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),  # 更新 x 的值
            x,  # 保持原值
        )
        # 返回更新后的 x
        return x

    # 定义调用方法，使得类可以被调用
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        # 准备采样循环，获取必要的输入
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        # 遍历 sigma 生成器，进行采样步骤
        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],  # 当前 sigma 值
                s_in * sigmas[i + 1],  # 下一个 sigma 值
                denoiser,  # 去噪器
                x,  # 当前 x 值
                cond,  # 条件
                uc,  # 额外条件
            )

        # 返回最终的 x 值
        return x


# 定义一个继承自 BaseDiffusionSampler 的类 LinearMultistepSampler
class LinearMultistepSampler(BaseDiffusionSampler):
    # 初始化方法，设定默认的 order 参数
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # 设置 order 属性
        self.order = order

    # 定义调用方法，使得类可以被调用
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        # 准备采样循环，获取必要的输入
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        # 初始化一个列表 ds 用于存储导数
        ds = []
        # 将 sigmas 从 GPU 移到 CPU，并转换为 numpy 数组
        sigmas_cpu = sigmas.detach().cpu().numpy()
        # 遍历 sigma 生成器
        for i in self.get_sigma_gen(num_sigmas):
            # 计算当前的 sigma
            sigma = s_in * sigmas[i]
            # 使用去噪器处理当前输入
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            # 使用引导函数对去噪结果进行处理
            denoised = self.guider(denoised, sigma)
            # 计算导数 d
            d = to_d(x, sigma, denoised)
            # 将导数添加到列表 ds
            ds.append(d)
            # 如果 ds 的长度超过 order，移除最早的元素
            if len(ds) > self.order:
                ds.pop(0)
            # 计算当前的阶数
            cur_order = min(i + 1, self.order)
            # 计算当前阶数的线性多步系数
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            # 更新 x 值
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        # 返回最终的 x 值
        return x


# 定义一个继承自 EDMSampler 的类 EulerEDMSampler
class EulerEDMSampler(EDMSampler):
    # 定义可能的校正步骤方法
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        # 返回 euler_step，表示不进行额外的校正
        return euler_step


# 定义一个继承自 EDMSampler 的类 HeunEDMSampler
class HeunEDMSampler(EDMSampler):
    # 定义可能的校正步骤方法
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
    ):
        # 如果下一个噪声水平的总和小于一个非常小的阈值
        if torch.sum(next_sigma) < 1e-14:
            # 如果所有噪声水平为0，保存网络评估的结果
            return euler_step
        else:
            # 使用去噪器对当前步进行去噪处理
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            # 将去噪后的结果转换为新数据
            d_new = to_d(euler_step, next_sigma, denoised)
            # 计算当前数据与新数据的平均值
            d_prime = (d + d_new) / 2.0

            # 如果噪声水平不为0，则应用修正
            x = torch.where(
                # 检查噪声水平是否大于0，决定是否修正
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            # 返回修正后的结果
            return x
# 定义一个 Euler 祖先采样器类，继承自 AncestralSampler
class EulerAncestralSampler(AncestralSampler):
    # 定义采样步骤的方法，接受多个参数
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        # 获取下一个采样步的 sigma 值
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        # 使用去噪器对当前输入进行去噪
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        # 使用 Euler 方法更新 x 的值
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        # 应用祖先步骤更新 x 的值
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        # 返回更新后的 x
        return x


# 定义一个 DPMPP2S 祖先采样器类，继承自 AncestralSampler
class DPMPP2SAncestralSampler(AncestralSampler):
    # 获取变量的方法，计算相关参数
    def get_variables(self, sigma, sigma_down):
        # 将 sigma 和 sigma_down 转换为负对数形式
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        # 计算时间间隔 h
        h = t_next - t
        # 计算 s 值
        s = t + 0.5 * h
        # 返回计算的参数
        return h, s, t, t_next

    # 获取乘法因子的方法
    def get_mult(self, h, s, t, t_next):
        # 计算各个乘法因子
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        # 返回所有乘法因子
        return mult1, mult2, mult3, mult4

    # 采样步骤的方法，执行多个计算步骤
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        # 获取下一个采样步的 sigma 值
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        # 对输入进行去噪
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        # 使用 Euler 方法更新 x 的值
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        # 检查 sigma_down 是否接近于零
        if torch.sum(sigma_down) < 1e-14:
            # 如果噪声级别为 0，则保存网络评估
            x = x_euler
        else:
            # 获取变量 h, s, t, t_next
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            # 获取乘法因子，并调整维度
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            # 更新 x 的值
            x2 = mult[0] * x - mult[1] * denoised
            # 对 x2 进行去噪
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            # 计算最终的 x 值
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # 如果噪声级别不为 0，则应用校正
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        # 最终应用祖先步骤更新 x
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        # 返回更新后的 x
        return x


# 定义一个 DPMPP2M 采样器类，继承自 BaseDiffusionSampler
class DPMPP2MSampler(BaseDiffusionSampler):
    # 获取变量的方法，计算相关参数
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        # 将 sigma 和 next_sigma 转换为负对数形式
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        # 计算时间间隔 h
        h = t_next - t

        # 如果提供了 previous_sigma，则进行额外计算
        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            # 如果没有提供，则返回 h 和 t 值
            return h, None, t, t_next

    # 获取乘法因子的方法
    def get_mult(self, h, r, t, t_next, previous_sigma):
        # 计算基础乘法因子
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        # 如果提供了 previous_sigma，则计算额外的乘法因子
        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            # 返回基本的乘法因子
            return mult1, mult2

    # 采样步骤的方法，执行多个计算步骤
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
        # 使用去噪器对输入数据进行去噪，返回去噪后的结果
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        # 获取当前和下一个噪声级别相关的变量
        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        # 计算多重系数，扩展维度以匹配输入数据的维度
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        # 计算标准化后的输出
        x_standard = mult[0] * x - mult[1] * denoised
        # 检查之前的去噪结果是否存在或下一噪声级别是否接近零
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # 如果噪声级别为零或处于第一步，返回标准化结果和去噪结果
            return x_standard, denoised
        else:
            # 计算去噪后的数据修正值
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            # 计算高级输出
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # 如果噪声级别不为零且不是第一步，应用修正
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        # 返回最终输出和去噪结果
        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        # 准备采样循环，包括输入数据和条件信息的处理
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        # 遍历噪声级别生成器
        for i in self.get_sigma_gen(num_sigmas):
            # 在每个步骤中执行采样，更新去噪结果
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

        # 返回最终的去噪结果
        return x
# 定义一个将输入信号传递到去噪器的函数
def relay_to_d(x, sigma, denoised, image, step, total_step):
    # 计算模糊度的变化量
    blurring_d = (denoised - image) / total_step
    # 根据模糊度和当前步长更新去噪图像
    blurring_denoised = image + blurring_d * step
    # 计算当前信号与去噪信号的差异，标准化为 sigma 的维度
    d = (x - blurring_denoised) / append_dims(sigma, x.ndim)
    # 返回计算得到的差异和模糊度变化
    return d, blurring_d
    

# 定义一个线性中继EDM采样器，继承自EulerEDMSampler
class LinearRelayEDMSampler(EulerEDMSampler):
    # 初始化函数，设定部分步数
    def __init__(self, partial_num_steps=20, *args, **kwargs):
        # 调用父类初始化方法
        super().__init__(*args, **kwargs)
        # 设置部分步数
        self.partial_num_steps = partial_num_steps

    # 定义采样调用方法
    def __call__(self, denoiser, image, randn, cond, uc=None, num_steps=None):
        # 克隆随机数以保持不变
        randn_unit = randn.clone()
        # 准备采样循环，获取相关参数
        randn, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            randn, cond, uc, num_steps
        )
        # 初始化 x 为 None
        x = None

        # 遍历生成的 sigma 值
        for i in self.get_sigma_gen(num_sigmas):
            # 如果当前步数小于总步数减去部分步数，继续下一次循环
            if i < self.num_steps - self.partial_num_steps:
                continue
            # 如果 x 还未初始化，则根据图像和随机数计算初始值
            if x is None:
                x = image + randn_unit * append_dims(s_in * sigmas[i], len(randn_unit.shape))

            # 计算 gamma 值，控制采样过程中的噪声
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            # 进行一次采样步骤
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
                step=i - self.num_steps + self.partial_num_steps,
                image=image,
                index=self.num_steps - i,
            )

        # 返回最终的图像
        return x

    # 定义欧拉步骤的计算方法
    def euler_step(self, x, d, dt, blurring_d):
        # 更新 x 的值
        return x + dt * d + blurring_d

    # 定义采样步骤的计算方法
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0, step=None, image=None, index=None):
        # 计算 sigma_hat，考虑 gamma 的影响
        sigma_hat = sigma * (gamma + 1.0)
        # 如果 gamma 大于 0，添加噪声
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        # 使用去噪器去噪当前图像
        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        # 计算 beta_t，控制去噪过程
        beta_t = next_sigma / sigma_hat * index / self.partial_num_steps - (index - 1) / self.partial_num_steps
        # 更新 x 的值，结合去噪结果
        x = x * append_dims(next_sigma / sigma_hat, x.ndim) + denoised * append_dims(1 - next_sigma / sigma_hat + beta_t, x.ndim) - image * append_dims(beta_t, x.ndim)
        # 返回更新后的图像
        return x
    

# 定义零信噪比DDIM采样器，继承自SingleStepDiffusionSampler
class ZeroSNRDDIMSampler(SingleStepDiffusionSampler):
    # 初始化函数，设定是否使用条件生成
    def __init__(
        self,
        do_cfg=True,
        *args,
        **kwargs,
    ):
        # 调用父类初始化方法
        super().__init__(*args, **kwargs)
        # 设置条件生成标志
        self.do_cfg = do_cfg

    # 准备采样循环的参数
    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        # 计算累积的 alpha 值，并获取对应的索引
        alpha_cumprod_sqrt, indices = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device, return_idx=True
        )
        # 如果 uc 为 None，则使用 cond
        uc = default(uc, cond)

        # 获取 sigma 的数量
        num_sigmas = len(alpha_cumprod_sqrt)

        # 初始化 s_in 为全 1 向量
        s_in = x.new_ones([x.shape[0]])

        # 返回准备好的参数
        return x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, indices
    # 定义去噪函数，接受输入数据和其他参数
        def denoise(self, x, denoiser, alpha_cumprod_sqrt, cond, uc, i=None, idx=None):
            # 初始化额外的模型输入字典
            additional_model_inputs = {}
            # 如果启用 CFG，准备包含索引的输入
            if self.do_cfg:
                additional_model_inputs['idx'] = torch.cat([x.new_ones([x.shape[0]]) * idx] * 2)
            # 否则只准备单个索引输入
            else:
                additional_model_inputs['idx'] = torch.cat([x.new_ones([x.shape[0]]) * idx])
            # 使用去噪器处理准备好的输入和额外参数，得到去噪后的结果
            denoised = denoiser(*self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs)
            # 使用引导器进一步处理去噪后的结果
            denoised = self.guider(denoised, alpha_cumprod_sqrt, step=i, num_steps=self.num_steps)
            # 返回去噪后的结果
            return denoised
    
    # 定义采样步骤函数，执行去噪和更新过程
        def sampler_step(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, denoiser, x, cond, uc=None, i=None, idx=None, return_denoised=False):
            # 调用去噪函数，并转换结果为浮点型
            denoised = self.denoise(x, denoiser, alpha_cumprod_sqrt, cond, uc, i, idx).to(torch.float32)
            # 如果达到最后一步，返回去噪结果
            if i == self.num_steps - 1:
                if return_denoised:
                    return denoised, denoised
                return denoised
    
            # 计算当前步骤的 a_t 值
            a_t = ((1-next_alpha_cumprod_sqrt**2)/(1-alpha_cumprod_sqrt**2))**0.5
            # 计算当前步骤的 b_t 值
            b_t = next_alpha_cumprod_sqrt - alpha_cumprod_sqrt * a_t
    
            # 更新 x 的值，结合去噪后的结果
            x = append_dims(a_t, x.ndim) * x + append_dims(b_t, x.ndim) * denoised
            # 根据需要返回去噪结果
            if return_denoised:
                return x, denoised
            return x
    
    # 定义可调用函数，用于处理采样和去噪流程
        def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
            # 准备采样循环所需的输入数据
            x, s_in, alpha_cumprod_sqrts, num_sigmas, cond, uc, indices = self.prepare_sampling_loop(
                x, cond, uc, num_steps
            )
    
            # 根据 sigma 生成器逐步执行采样
            for i in self.get_sigma_gen(num_sigmas):
                x = self.sampler_step(
                    s_in * alpha_cumprod_sqrts[i],
                    s_in * alpha_cumprod_sqrts[i + 1],
                    denoiser,
                    x,
                    cond,
                    uc,
                    i=i,
                    idx=indices[self.num_steps-i-1],
                )
    
            # 返回最终的结果
            return x
```