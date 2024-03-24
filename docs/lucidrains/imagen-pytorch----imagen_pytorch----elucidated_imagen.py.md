# `.\lucidrains\imagen-pytorch\imagen_pytorch\elucidated_imagen.py`

```py
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 从 random 模块中导入 random 函数
from random import random
# 从 functools 模块中导入 partial 函数
from functools import partial
# 从 contextlib 模块中导入 contextmanager 和 nullcontext
from contextlib import contextmanager, nullcontext
# 从 typing 模块中导入 List 和 Union
from typing import List, Union
# 从 collections 模块中导入 namedtuple
from collections import namedtuple
# 从 tqdm.auto 模块中导入 tqdm 函数
from tqdm.auto import tqdm

# 导入 torch 库
import torch
# 从 torch.nn 模块中导入 functional 模块
import torch.nn.functional as F
# 从 torch 模块中导入 nn 模块
from torch import nn
# 从 torch.cuda.amp 模块中导入 autocast 函数
from torch.cuda.amp import autocast
# 从 torch.nn.parallel 模块中导入 DistributedDataParallel 类
from torch.nn.parallel import DistributedDataParallel
# 从 torchvision.transforms 模块中导入 T 别名
import torchvision.transforms as T

# 导入 kornia.augmentation 模块
import kornia.augmentation as K

# 从 einops 模块中导入 rearrange、repeat 和 reduce 函数
from einops import rearrange, repeat, reduce

# 从 imagen_pytorch.imagen_pytorch 模块中导入各种函数和类
from imagen_pytorch.imagen_pytorch import (
    GaussianDiffusionContinuousTimes,
    Unet,
    NullUnet,
    first,
    exists,
    identity,
    maybe,
    default,
    cast_tuple,
    cast_uint8_images_to_float,
    eval_decorator,
    pad_tuple_to_length,
    resize_image_to,
    calc_all_frame_dims,
    safe_get_tuple_index,
    right_pad_dims_to,
    module_device,
    normalize_neg_one_to_one,
    unnormalize_zero_to_one,
    compact,
    maybe_transform_dict_key
)

# 从 imagen_pytorch.imagen_video 模块中导入 Unet3D、resize_video_to 和 scale_video_time 函数
from imagen_pytorch.imagen_video import (
    Unet3D,
    resize_video_to,
    scale_video_time
)

# 从 imagen_pytorch.t5 模块中导入 t5_encode_text、get_encoded_dim 和 DEFAULT_T5_NAME 常量
from imagen_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

# 定义常量 Hparams_fields
Hparams_fields = [
    'num_sample_steps',
    'sigma_min',
    'sigma_max',
    'sigma_data',
    'rho',
    'P_mean',
    'P_std',
    'S_churn',
    'S_tmin',
    'S_tmax',
    'S_noise'
]

# 创建命名元组 Hparams
Hparams = namedtuple('Hparams', Hparams_fields)

# 定义辅助函数 log
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 主类 ElucidatedImagen
class ElucidatedImagen(nn.Module):
    # 初始化方法
    def __init__(
        self,
        unets,
        *,
        image_sizes,                                # 用于级联 ddpm 的图像大小
        text_encoder_name = DEFAULT_T5_NAME,
        text_embed_dim = None,
        channels = 3,
        cond_drop_prob = 0.1,
        random_crop_sizes = None,
        resize_mode = 'nearest',
        temporal_downsample_factor = 1,
        resize_cond_video_frames = True,
        lowres_sample_noise_level = 0.2,            # 低分辨率采样噪声级别
        per_sample_random_aug_noise_level = False,  # 是否在每个批次元素上接收随机增强噪声值
        condition_on_text = True,
        auto_normalize_img = True,                  # 是否自动归一化图像
        dynamic_thresholding = True,
        dynamic_thresholding_percentile = 0.95,     # 动态阈值百分位数
        only_train_unet_number = None,
        lowres_noise_schedule = 'linear',
        num_sample_steps = 32,                      # 采样步数
        sigma_min = 0.002,                          # 最小噪声水平
        sigma_max = 80,                             # 最大噪声水平
        sigma_data = 0.5,                           # 数据分布的标准差
        rho = 7,                                    # 控制采样计划
        P_mean = -1.2,                              # 训练时噪声抽取的对数正态分布均值
        P_std = 1.2,                                # 训练时噪声抽取的对数正态分布标准差
        S_churn = 80,                               # 随机采样参数
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
    # 强制取消条件性
    def force_unconditional_(self):
        self.condition_on_text = False
        self.unconditional = True

        for unet in self.unets:
            unet.cond_on_text = False
    # 返回属性 device 的值
    @property
    def device(self):
        return self._temp.device

    # 获取指定编号的 UNet 模型
    def get_unet(self, unet_number):
        # 确保 unet_number 在有效范围内
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        # 如果 self.unets 是 nn.ModuleList 类型，则转换为列表
        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            # 删除属性 'unets'
            delattr(self, 'unets')
            self.unets = unets_list

        # 如果 index 不等于正在训练的 UNet 索引，则将 UNet 移动到指定设备
        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.unet_being_trained_index = index
        return self.unets[index]

    # 将所有 UNet 模型重置到同一设备上
    def reset_unets_all_one_device(self, device = None):
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        self.unet_being_trained_index = -1

    # 使用上下文管理器将指定 UNet 移动到 GPU 上
    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        cpu = torch.device('cpu')

        devices = [module_device(unet) for unet in self.unets]

        self.unets.to(cpu)
        unet.to(self.device)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # 重写 state_dict 函数
    def state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    # 重写 load_state_dict 函数
    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # 动态阈值
    def threshold_x_start(self, x_start, dynamic_threshold = True):
        if not dynamic_threshold:
            return x_start.clamp(-1., 1.)

        s = torch.quantile(
            rearrange(x_start, 'b ... -> b (...)').abs(),
            self.dynamic_thresholding_percentile,
            dim = -1
        )

        s.clamp_(min = 1.)
        s = right_pad_dims_to(x_start, s)
        return x_start.clamp(-s, s) / s

    # 衍生的预处理参数 - 表 1
    def c_skip(self, sigma_data, sigma):
        return (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)

    def c_out(self, sigma_data, sigma):
        return sigma * sigma_data * (sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma_data, sigma):
        return 1 * (sigma ** 2 + sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # 预处理网络输出
    def preconditioned_network_forward(
        self,
        unet_forward,
        noised_images,
        sigma,
        *,
        sigma_data,
        clamp = False,
        dynamic_threshold = True,
        **kwargs
    ):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = self.right_pad_dims_to_datatype(sigma)

        net_out = unet_forward(
            self.c_in(sigma_data, padded_sigma) * noised_images,
            self.c_noise(sigma),
            **kwargs
        )

        out = self.c_skip(sigma_data, padded_sigma) * noised_images +  self.c_out(sigma_data, padded_sigma) * net_out

        if not clamp:
            return out

        return self.threshold_x_start(out, dynamic_threshold)

    # 采样
    # 采样计划
    def sample_schedule(
        self,
        num_sample_steps,
        rho,
        sigma_min,
        sigma_max
    ):
        N = num_sample_steps
        inv_rho = 1 / rho

        # 生成一个包含 num_sample_steps 个元素的张量，设备为 self.device，数据类型为 torch.float32
        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        # 计算每个步骤的 sigma 值
        sigmas = (sigma_max ** inv_rho + steps / (N - 1) * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho

        # 在 sigmas 张量的末尾填充一个值为 0 的元素，用于表示最后一个步骤的 sigma 值为 0
        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def one_unet_sample(
        self,
        unet,
        shape,
        *,
        unet_number,
        clamp = True,
        dynamic_threshold = True,
        cond_scale = 1.,
        use_tqdm = True,
        inpaint_videos = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        init_images = None,
        skip_steps = None,
        sigma_min = None,
        sigma_max = None,
        **kwargs
    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        texts: List[str] = None,
        text_masks = None,
        text_embeds = None,
        cond_images = None,
        cond_video_frames = None,
        post_cond_video_frames = None,
        inpaint_videos = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        init_images = None,
        skip_steps = None,
        sigma_min = None,
        sigma_max = None,
        video_frames = None,
        batch_size = 1,
        cond_scale = 1.,
        lowres_sample_noise_level = None,
        start_at_unet_number = 1,
        start_image_or_video = None,
        stop_at_unet_number = None,
        return_all_unet_outputs = False,
        return_pil_images = False,
        use_tqdm = True,
        use_one_unet_in_gpu = True,
        device = None,
    # training

    # 计算损失权重
    def loss_weight(self, sigma_data, sigma):
        return (sigma ** 2 + sigma_data ** 2) * (sigma * sigma_data) ** -2

    # 生成服从指定均值和标准差的噪声分布
    def noise_distribution(self, P_mean, P_std, batch_size):
        return (P_mean + P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(
        self,
        images, # 重命名为 images 或 video
        unet: Union[Unet, Unet3D, NullUnet, DistributedDataParallel] = None,
        texts: List[str] = None,
        text_embeds = None,
        text_masks = None,
        unet_number = None,
        cond_images = None,
        **kwargs
```