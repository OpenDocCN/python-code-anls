# `.\lucidrains\diffusion-policy\diffusion_policy\diffusion_policy.py`

```py
import math
from pathlib import Path
from random import random
from functools import partial
from multiprocessing import cpu_count

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
from torch.special import expm1
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T, utils

from beartype import beartype

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm

from ema_pytorch import EMA

from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs
)

# helpers functions

# 检查变量是否存在
def exists(x):
    return x is not None

# 返回输入值
def identity(x):
    return x

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 检查一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 安全地进行除法运算
def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)

# 生成数据集的循环迭代器
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 检查一个数是否有整数平方根
def has_int_squareroot(num):
    num_sqrt = math.sqrt(num)
    return int(num_sqrt) == num_sqrt

# 将一个数分成若干组
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 将图像转换为指定类型
def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# 创建序列模块
def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# normalize and unnormalize image

# 归一化图像
def normalize_img(x):
    return x * 2 - 1

# 反归一化图像
def unnormalize_img(x):
    return (x + 1) * 0.5

# 标准化有噪声图像的方差（如果比例不为1）
def normalize_img_variance(x, eps = 1e-5):
    std = reduce(x, 'b c h w -> b 1 1 1', partial(torch.std, unbiased = False))
    return x / std.clamp(min = eps)

# helper functions

# 计算对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 将右侧维度填充到与左侧相同的维度
def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# noise schedules

# 简单线性调度
def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

# 余弦调度
def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

# sigmoid调度
def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# 将gamma转换为alpha、sigma或logsnr

# 将gamma转换为alpha和sigma
def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

# 将gamma转换为logsnr
def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)

# gaussian diffusion

# 扩散策略类
class DiffusionPolicy(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        *,
        timesteps = 1000,
        use_ddim = True,
        noise_schedule = 'sigmoid',
        objective = 'v',
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        train_prob_self_cond = 0.9,
        scale = 1.                      # this will be set to < 1. for better convergence when training on higher resolution images
        # 调用父类的构造函数
        super().__init__()
        # 设置模型和通道数
        self.model = model
        self.channels = self.model.channels

        # 确保目标是预测 x0 或者噪声
        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective

        # 设置图像大小
        self.image_size = model.image_size

        # 根据噪声调度设置不同的 gamma 调度函数
        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # 根据图像尺寸调整噪声大小
        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale
        self.maybe_normalize_img_variance = normalize_img_variance if scale < 1 else identity

        # 设置 gamma 调度函数的参数
        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        # 设置采样时间步数和是否使用 DDIM
        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # 根据论文提出的方法，将时间差加到 time_next 上，以修复自条件不足和在采样时间步数 < 400 时降低 FID 的问题
        self.time_difference = time_difference

        # 训练过程中自条件的概率
        self.train_prob_self_cond = train_prob_self_cond

        # 最小 SNR 损失权重
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    # 返回设备信息
    @property
    def device(self):
        return next(self.model.parameters()).device

    # 获取采样时间步数
    def get_sampling_timesteps(self, batch, *, device):
        # 在设备上生成时间步数
        times = torch.linspace(1., 0., self.timesteps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    # 禁用梯度计算
    @torch.no_grad()
    # 从 DDPM 模型中采样生成图像
    def ddpm_sample(self, shape, time_difference = None):
        # 获取批次大小和设备信息
        batch, device = shape[0], self.device

        # 设置时间差，默认为 None
        time_difference = default(time_difference, self.time_difference)

        # 获取采样时间步骤对
        time_pairs = self.get_sampling_timesteps(batch, device = device)

        # 生成随机噪声图像
        img = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        # 遍历时间步骤对
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            # 添加时间延迟
            time_next = (time_next - self.time_difference).clamp(min = 0.)

            noise_cond = time

            # 获取预测的 x0
            maybe_normalized_img = self.maybe_normalize_img_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, noise_cond, x_start, last_latents, return_latents = True)

            # 获取 log(snr)
            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            # 获取 alpha 和 sigma
            alpha, sigma = gamma_to_alpha_sigma(gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next, self.scale)

            # 计算 x0 和噪声
            if self.objective == 'x0':
                x_start = model_output
            elif self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)
            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # 限制 x0 的取值范围
            x_start.clamp_(-1., 1.)

            # 推导后验均值和方差
            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # 生成噪声
            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            # 更新图像
            img = mean + (0.5 * log_variance).exp() * noise

        # 返回未归一化的图像
        return unnormalize_img(img)

    # 禁用梯度计算
    @torch.no_grad()
    # 从给定形状中获取批次和设备信息
    def ddim_sample(self, shape, time_difference = None):
        batch, device = shape[0], self.device

        # 设置时间差，默认为None
        time_difference = default(time_difference, self.time_difference)

        # 获取采样时间步骤
        time_pairs = self.get_sampling_timesteps(batch, device = device)

        # 生成符合正态分布的随机张量
        img = torch.randn(shape, device = device)

        x_start = None
        last_latents = None

        # 遍历时间对
        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # 获取时间和噪声水平
            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            # 对gamma进行填充
            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            # 将gamma转换为alpha和sigma
            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)

            # 添加时间延迟
            times_next = (times_next - time_difference).clamp(min = 0.)

            # 预测x0
            maybe_normalized_img = self.maybe_normalize_img_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, times, x_start, last_latents, return_latents = True)

            # 计算x0和噪声
            if self.objective == 'x0':
                x_start = model_output
            elif self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)
            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # 限制x0的范围在[-1, 1]之间
            x_start.clamp_(-1., 1.)

            # 获取预测的噪声
            pred_noise = safe_div(img - alpha * x_start, sigma)

            # 计算下一个x
            img = x_start * alpha_next + pred_noise * sigma_next

        # 返回未标准化的图像
        return unnormalize_img(img)

    # 无需梯度计算
    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        # 根据是否使用DDIM选择采样函数
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))
    # 定义一个前向传播函数，接受图像和其他参数
    def forward(self, img, *args, **kwargs):
        # 解包图像的形状和设备信息
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 断言图像的高度和宽度必须为指定的图像大小
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # 生成随机时间采样
        times = torch.zeros((batch,), device=device).float().uniform_(0, 1.)

        # 将图像转换为比特表示
        img = normalize_img(img)

        # 生成噪声样本
        noise = torch.randn_like(img)

        # 计算 gamma 值
        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(img, gamma)
        alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)

        # 添加噪声到图像
        noised_img = alpha * img + sigma * noise

        # 可能对图像进行归一化处理
        noised_img = self.maybe_normalize_img_variance(noised_img)

        # 在论文中，他们必须使用非常高的概率进行潜在的自我条件，高达 90% 的时间
        # 稍微有点缺点
        self_cond = self_latents = None

        if random() < self.train_prob_self_cond:
            with torch.no_grad():
                model_output, self_latents = self.model(noised_img, times, return_latents=True)
                self_latents = self_latents.detach()

                if self.objective == 'x0':
                    self_cond = model_output

                elif self.objective == 'eps':
                    self_cond = safe_div(noised_img - sigma * model_output, alpha)

                elif self.objective == 'v':
                    self_cond = alpha * noised_img - sigma * model_output

                self_cond.clamp_(-1., 1.)
                self_cond = self_cond.detach()

        # 预测并进行梯度下降步骤
        pred = self.model(noised_img, times, self_cond, self_latents)

        if self.objective == 'eps':
            target = noise

        elif self.objective == 'x0':
            target = img

        elif self.objective == 'v':
            target = alpha * noise - sigma * img

        # 计算损失
        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # 最小信噪比损失权重
        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=self.min_snr_gamma)

        if self.objective == 'eps':
            loss_weight = maybe_clipped_snr / snr

        elif self.objective == 'x0':
            loss_weight = maybe_clipped_snr

        elif self.objective == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        return (loss * loss_weight).mean()
# dataset classes

# 定义 Dataset 类，继承自 torch.utils.data.Dataset
class Dataset(Dataset):
    # 初始化函数
    def __init__(
        self,
        folder,  # 数据集文件夹路径
        image_size,  # 图像大小
        exts = ['jpg', 'jpeg', 'png', 'tiff'],  # 图像文件扩展名列表
        augment_horizontal_flip = False,  # 是否进行水平翻转增强
        convert_image_to = None  # 图像转换函数
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # 获取文件夹中指定扩展名的所有文件路径
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # 部分应用转换函数
        maybe_convert_fn = partial(convert_image_to, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # 图像转换操作序列
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    # 返回数据集长度
    def __len__(self):
        return len(self.paths)

    # 获取指定索引处的数据
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

# 定义 Trainer 类
@beartype
class Trainer(object):
    # 初始化函数
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,  # 扩散模型
        folder,  # 数据集文件夹路径
        *,
        train_batch_size = 16,  # 训练批量大小
        gradient_accumulate_every = 1,  # 梯度累积步数
        augment_horizontal_flip = True,  # 是否进行水平翻转增强
        train_lr = 1e-4,  # 训练学习率
        train_num_steps = 100000,  # 训练步数
        max_grad_norm = 1.,  # 梯度裁剪阈值
        ema_update_every = 10,  # EMA 更新频率
        ema_decay = 0.995,  # EMA 衰减率
        betas = (0.9, 0.99),  # Adam 优化器的 beta 参数
        save_and_sample_every = 1000,  # 保存和采样频率
        num_samples = 25,  # 采样数量
        results_folder = './results',  # 结果保存文件夹路径
        amp = False,  # 是否使用混合精度训练
        mixed_precision_type = 'fp16',  # 混合精度类型
        split_batches = True,  # 是否拆分批次
        convert_image_to = None  # 图像转换函数
    ):
        super().__init__()

        # 初始化加速器
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no',
            kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        # 设置扩散模型
        self.model = diffusion_model

        # 检查采样数量是否有整数平方根
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # 数据集和数据加载器

        # 创建数据集对象
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # 创建数据加载器
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        # 准备数据加载器
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # 优化器

        # 创建 Adam 优化器
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = betas)

        # 定期记录结果到文件夹

        self.results_folder = Path(results_folder)

        if self.accelerator.is_local_main_process:
            self.results_folder.mkdir(exist_ok = True)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        # 步数计数器状态

        self.step = 0

        # 准备模型、数据加载器、优化器与加速器

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    # 保存模型
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step + 1,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    # 加载指定里程碑的模型数据
    def load(self, milestone):
        # 从文件中加载模型数据
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        # 获取未加速的模型对象
        model = self.accelerator.unwrap_model(self.model)
        # 加载模型的状态字典
        model.load_state_dict(data['model'])

        # 设置当前训练步数
        self.step = data['step']
        # 加载优化器的状态字典
        self.opt.load_state_dict(data['opt'])

        # 如果是主进程，则加载指数移动平均模型的状态字典
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        # 如果加速器和数据中都存在缩放器状态字典，则加载缩放器的状态字典
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # 训练模型
    def train(self):
        # 获取加速器和设备
        accelerator = self.accelerator
        device = accelerator.device

        # 使用 tqdm 显示训练进度
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            # 在未达到训练步数上限前循环训练
            while self.step < self.train_num_steps:

                total_loss = 0.

                # 根据梯度累积次数循环执行训练步骤
                for _ in range(self.gradient_accumulate_every):
                    # 获取下一个数据批次并发送到设备
                    data = next(self.dl).to(device)

                    # 使用自动混合精度计算模型损失
                    with accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    # 反向传播计算梯度
                    accelerator.backward(loss)

                # 更新进度条显示当前损失值
                pbar.set_description(f'loss: {total_loss:.4f}')

                # 等待所有进程完成当前步骤
                accelerator.wait_for_everyone()
                # 对模型参数进行梯度裁剪
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # 执行优化器的一步更新
                self.opt.step()
                # 清空梯度
                self.opt.zero_grad()

                # 等待所有进程完成当前步骤
                accelerator.wait_for_everyone()

                # 在每个本地主进程上保存里程碑，仅在全局主进程上采样
                if accelerator.is_local_main_process:
                    milestone = self.step // self.save_and_sample_every
                    save_and_sample = self.step != 0 and self.step % self.save_and_sample_every == 0
                    
                    if accelerator.is_main_process:
                        # 将指数移动平均模型发送到设备
                        self.ema.to(device)
                        # 更新指数移动平均模型
                        self.ema.update()

                        if save_and_sample:
                            # 将指数移动平均模型设置为评估模式
                            self.ema.ema_model.eval()

                            with torch.no_grad():
                                # 将样本数量分组并生成样本图像
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                            all_images = torch.cat(all_images_list, dim = 0)
                            # 保存生成的样本图像
                            utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                    if save_and_sample:
                        # 保存当前里程碑的模型数据
                        self.save(milestone)

                # 更新训练步数并更新进度条
                self.step += 1
                pbar.update(1)

        # 打印训练完成信息
        accelerator.print('training complete')
```