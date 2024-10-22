# `.\diffusers\pipelines\free_init_utils.py`

```py
# 版权声明，指定该文件由 HuggingFace 团队版权所有，所有权利保留
# 根据 Apache 许可证 2.0 版进行授权；用户需遵循许可条款
# 提供许可证的获取地址
#
# 除非适用的法律要求或书面同意，否则该软件以 "原样" 基础提供，不提供任何形式的担保或条件
# 详见许可证中有关权限和限制的条款

# 导入数学模块
import math
# 从 typing 模块导入 Tuple 和 Union 类型
from typing import Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的 FFT 模块
import torch.fft as fft

# 从 utils.torch_utils 导入 randn_tensor 函数
from ..utils.torch_utils import randn_tensor


# 定义一个混入类 FreeInitMixin
class FreeInitMixin:
    r"""FreeInit 的混入类."""

    # 启用 FreeInit 机制的方法
    def enable_free_init(
        self,
        num_iters: int = 3,  # 默认的 FreeInit 噪声重新初始化迭代次数
        use_fast_sampling: bool = False,  # 是否使用快速采样，默认为 False
        method: str = "butterworth",  # 选择过滤方法，默认为 butterworth
        order: int = 4,  # butterworth 方法的过滤器阶数，默认值为 4
        spatial_stop_frequency: float = 0.25,  # 空间维度的归一化截止频率，默认值为 0.25
        temporal_stop_frequency: float = 0.25,  # 时间维度的归一化截止频率，默认值为 0.25
    ):
        """启用 FreeInit 机制，参考文献为 https://arxiv.org/abs/2312.07537.

        此实现已根据 [官方仓库](https://github.com/TianxingWu/FreeInit) 进行了调整.

        参数:
            num_iters (`int`, *可选*, 默认值为 `3`):
                FreeInit 噪声重新初始化的迭代次数.
            use_fast_sampling (`bool`, *可选*, 默认值为 `False`):
                是否以牺牲质量来加速采样过程，如果设置为 `True`，启用文中提到的 "粗到细采样" 策略.
            method (`str`, *可选*, 默认值为 `butterworth`):
                用于 FreeInit 低通滤波器的过滤方法，必须为 `butterworth`、`ideal` 或 `gaussian` 之一.
            order (`int`, *可选*, 默认值为 `4`):
                在 `butterworth` 方法中使用的滤波器阶数，较大的值会导致 `ideal` 方法行为，而较小的值会导致 `gaussian` 方法行为.
            spatial_stop_frequency (`float`, *可选*, 默认值为 `0.25`):
                空间维度的归一化截止频率，值必须在 0 到 1 之间，原实现中称为 `d_s`.
            temporal_stop_frequency (`float`, *可选*, 默认值为 `0.25`):
                时间维度的归一化截止频率，值必须在 0 到 1 之间，原实现中称为 `d_t`.
        """
        # 设置 FreeInit 迭代次数
        self._free_init_num_iters = num_iters
        # 设置是否使用快速采样
        self._free_init_use_fast_sampling = use_fast_sampling
        # 设置过滤方法
        self._free_init_method = method
        # 设置过滤器阶数
        self._free_init_order = order
        # 设置空间截止频率
        self._free_init_spatial_stop_frequency = spatial_stop_frequency
        # 设置时间截止频率
        self._free_init_temporal_stop_frequency = temporal_stop_frequency
    # 禁用 FreeInit 机制（如果已启用）
        def disable_free_init(self):
            """Disables the FreeInit mechanism if enabled."""
            # 将 FreeInit 迭代次数设置为 None，表示禁用
            self._free_init_num_iters = None
    
        @property
        # 属性，检查 FreeInit 是否启用
        def free_init_enabled(self):
            # 返回是否存在 FreeInit 迭代次数且不为 None
            return hasattr(self, "_free_init_num_iters") and self._free_init_num_iters is not None
    
        # 获取 FreeInit 频率滤波器
        def _get_free_init_freq_filter(
            self,
            shape: Tuple[int, ...],  # 输入形状，包含时间、高度、宽度
            device: Union[str, torch.dtype],  # 设备类型
            filter_type: str,  # 滤波器类型
            order: float,  # 滤波器阶数
            spatial_stop_frequency: float,  # 空间停止频率
            temporal_stop_frequency: float,  # 时间停止频率
        ) -> torch.Tensor:
            r"""Returns the FreeInit filter based on filter type and other input conditions."""
    
            # 提取时间、高度和宽度维度
            time, height, width = shape[-3], shape[-2], shape[-1]
            # 初始化全零的掩码张量
            mask = torch.zeros(shape)
    
            # 如果空间或时间停止频率为零，返回全零掩码
            if spatial_stop_frequency == 0 or temporal_stop_frequency == 0:
                return mask
    
            # 根据不同滤波器类型定义掩码函数
            if filter_type == "butterworth":
    
                # Butterworth 滤波器的掩码函数
                def retrieve_mask(x):
                    return 1 / (1 + (x / spatial_stop_frequency**2) ** order)
            elif filter_type == "gaussian":
    
                # Gaussian 滤波器的掩码函数
                def retrieve_mask(x):
                    return math.exp(-1 / (2 * spatial_stop_frequency**2) * x)
            elif filter_type == "ideal":
    
                # 理想滤波器的掩码函数
                def retrieve_mask(x):
                    return 1 if x <= spatial_stop_frequency * 2 else 0
            else:
                # 如果滤波器类型未实现，抛出异常
                raise NotImplementedError("`filter_type` must be one of gaussian, butterworth or ideal")
    
            # 遍历时间、高度和宽度，计算掩码值
            for t in range(time):
                for h in range(height):
                    for w in range(width):
                        # 计算距离平方，用于掩码函数
                        d_square = (
                            ((spatial_stop_frequency / temporal_stop_frequency) * (2 * t / time - 1)) ** 2
                            + (2 * h / height - 1) ** 2
                            + (2 * w / width - 1) ** 2
                        )
                        # 根据距离平方更新掩码值
                        mask[..., t, h, w] = retrieve_mask(d_square)
    
            # 将掩码张量转移到指定设备
            return mask.to(device)
    
        # 应用频率滤波器
        def _apply_freq_filter(self, x: torch.Tensor, noise: torch.Tensor, low_pass_filter: torch.Tensor) -> torch.Tensor:
            r"""Noise reinitialization."""
            # 对输入进行快速傅里叶变换（FFT）
            x_freq = fft.fftn(x, dim=(-3, -2, -1))
            # 将频谱中心移到频谱的中心
            x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
            # 对噪声进行快速傅里叶变换（FFT）
            noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
            # 将噪声频谱中心移到频谱的中心
            noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))
    
            # 频率混合操作
            high_pass_filter = 1 - low_pass_filter  # 计算高通滤波器
            x_freq_low = x_freq * low_pass_filter  # 低通滤波器作用于输入
            noise_freq_high = noise_freq * high_pass_filter  # 高通滤波器作用于噪声
            # 在频域中混合
            x_freq_mixed = x_freq_low + noise_freq_high  
    
            # 逆快速傅里叶变换（IFFT）
            x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))  # 将混合频谱中心移回
            x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real  # 还原到时域并取实部
    
            # 返回混合后的结果
            return x_mixed
    
        # 应用 FreeInit 机制
        def _apply_free_init(
            self,
            latents: torch.Tensor,  # 输入的潜变量
            free_init_iteration: int,  # FreeInit 迭代次数
            num_inference_steps: int,  # 推理步骤数量
            device: torch.device,  # 设备类型
            dtype: torch.dtype,  # 数据类型
            generator: torch.Generator,  # 随机数生成器
    # 方法体开始
        ):
            # 如果是第一次初始化
            if free_init_iteration == 0:
                # 克隆初始噪声，保存在属性中
                self._free_init_initial_noise = latents.detach().clone()
            else:
                # 获取当前潜在变量的形状
                latent_shape = latents.shape
    
                # 定义过滤器的形状，保留第一维
                free_init_filter_shape = (1, *latent_shape[1:])
                # 获取自由初始化频率过滤器
                free_init_freq_filter = self._get_free_init_freq_filter(
                    shape=free_init_filter_shape,  # 过滤器的形状
                    device=device,  # 设备
                    filter_type=self._free_init_method,  # 过滤器类型
                    order=self._free_init_order,  # 过滤器阶数
                    spatial_stop_frequency=self._free_init_spatial_stop_frequency,  # 空间停止频率
                    temporal_stop_frequency=self._free_init_temporal_stop_frequency,  # 时间停止频率
                )
    
                # 获取当前扩散时间步
                current_diffuse_timestep = self.scheduler.config.num_train_timesteps - 1
                # 创建与潜在变量数量相同的扩散时间步张量
                diffuse_timesteps = torch.full((latent_shape[0],), current_diffuse_timestep).long()
    
                # 向潜在变量添加噪声
                z_t = self.scheduler.add_noise(
                    original_samples=latents,  # 原始潜在样本
                    noise=self._free_init_initial_noise,  # 添加的噪声
                    timesteps=diffuse_timesteps.to(device)  # 转移到设备
                ).to(dtype=torch.float32)  # 转换数据类型
    
                # 创建随机张量
                z_rand = randn_tensor(
                    shape=latent_shape,  # 随机张量的形状
                    generator=generator,  # 随机数生成器
                    device=device,  # 设备
                    dtype=torch.float32,  # 数据类型
                )
                # 应用频率过滤器于潜在变量
                latents = self._apply_freq_filter(z_t, z_rand, low_pass_filter=free_init_freq_filter)
                # 转换潜在变量数据类型
                latents = latents.to(dtype)
    
            # 进行粗到细采样以加速推理（可能导致质量降低）
            if self._free_init_use_fast_sampling:
                # 计算推理步骤的数量
                num_inference_steps = max(
                    1, int(num_inference_steps / self._free_init_num_iters * (free_init_iteration + 1))  # 逐步减少推理步骤
                )
    
            # 如果推理步骤大于0
            if num_inference_steps > 0:
                # 设置调度器的时间步
                self.scheduler.set_timesteps(num_inference_steps, device=device)
    
            # 返回潜在变量和调度器的时间步
            return latents, self.scheduler.timesteps
```