# `.\lucidrains\phenaki-pytorch\phenaki_pytorch\cvivit_trainer.py`

```py
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 从 random 模块中导入 choice 函数
from random import choice
# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从 shutil 模块中导入 rmtree 函数
from shutil import rmtree

# 从 beartype 模块中导入 beartype 装饰器
from beartype import beartype

# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn 模块
from torch import nn
# 从 torch.utils.data 模块中导入 Dataset, DataLoader, random_split 类
from torch.utils.data import Dataset, DataLoader, random_split

# 从 torchvision.transforms 模块中导入 T 别名
import torchvision.transforms as T
# 从 torchvision.datasets 模块中导入 ImageFolder 类
from torchvision.datasets import ImageFolder
# 从 torchvision.utils 模块中导入 make_grid, save_image 函数
from torchvision.utils import make_grid, save_image

# 从 einops 模块中导入 rearrange 函数
from einops import rearrange

# 从 phenaki_pytorch.optimizer 模块中导入 get_optimizer 函数
from phenaki_pytorch.optimizer import get_optimizer

# 从 ema_pytorch 模块中导入 EMA 类
from ema_pytorch import EMA

# 从 phenaki_pytorch.cvivit 模块中导入 CViViT 类
from phenaki_pytorch.cvivit import CViViT
# 从 phenaki_pytorch.data 模块中导入 ImageDataset, VideoDataset, video_tensor_to_gif 函数
from phenaki_pytorch.data import ImageDataset, VideoDataset, video_tensor_to_gif

# 从 accelerate 模块中导入 Accelerator 类

# helpers

# 定义 exists 函数，判断值是否存在
def exists(val):
    return val is not None

# 定义 noop 函数，空函数
def noop(*args, **kwargs):
    pass

# 定义 cycle 函数，循环生成数据
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 定义 cast_tuple 函数，将参数转换为元组
def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

# 定义 yes_or_no 函数，询问用户是否为是或否
def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

# 定义 accum_log 函数，累积日志信息
def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# main trainer class

# 使用 beartype 装饰器定义 CViViTTrainer 类
@beartype
class CViViTTrainer(nn.Module):
    # 初始化方法
    def __init__(
        self,
        vae: CViViT,
        *,
        num_train_steps,
        batch_size,
        folder,
        train_on_images = False,
        num_frames = 17,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        discr_max_grad_norm = None,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        use_ema = True,
        ema_beta = 0.995,
        ema_update_after_step = 0,
        ema_update_every = 1,
        apply_grad_penalty_every = 4,
        accelerate_kwargs: dict = dict()
    ):
        # 调用父类的构造函数
        super().__init__()
        # 获取 VAE 模型的图像大小
        image_size = vae.image_size

        # 初始化加速器
        self.accelerator = Accelerator(**accelerate_kwargs)

        # 设置 VAE 模型
        self.vae = vae

        # 是否使用指数移动平均
        self.use_ema = use_ema
        # 如果是主进程且使用指数移动平均
        if self.is_main and use_ema:
            # 初始化指数移动平均 VAE 模型
            self.ema_vae = EMA(vae, update_after_step = ema_update_after_step, update_every = ema_update_every)

        # 注册缓冲区 'steps'，用于记录训练步数
        self.register_buffer('steps', torch.Tensor([0]))

        # 设置训练步数、批量大小和梯度累积步数
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # 获取所有参数、判别器参数和 VAE 参数
        all_parameters = set(vae.parameters())
        discr_parameters = set(vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters

        self.vae_parameters = vae_parameters

        # 获取优化器
        self.optim = get_optimizer(vae_parameters, lr = lr, wd = wd)
        self.discr_optim = get_optimizer(discr_parameters, lr = lr, wd = wd)

        # 设置梯度裁剪阈值
        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # 创建数据集
        dataset_klass = ImageDataset if train_on_images else VideoDataset
        if train_on_images:
            self.ds = ImageDataset(folder, image_size)
        else:
            self.ds = VideoDataset(folder, image_size, num_frames = num_frames)

        # 划分验证集
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # 创建数据加载器
        self.dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True
        )

        # 准备加速器
        (
            self.vae,
            self.optim,
            self.discr_optim,
            self.dl
        ) = self.accelerator.prepare(
            self.vae,
            self.optim,
            self.discr_optim,
            self.dl
        )

        # 创建数据加载器迭代器
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        # 设置模型保存频率和结果保存频率
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        # 设置应用梯度惩罚的频率
        self.apply_grad_penalty_every = apply_grad_penalty_every

        # 设置结果文件夹
        self.results_folder = Path(results_folder)

        # 如果结果文件夹不为空且确认清除之前的实验检查点和结果
        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        # 创建��果文件夹
        self.results_folder.mkdir(parents = True, exist_ok = True)

    # 保存模型
    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model = self.accelerator.get_state_dict(self.vae),
            optim = self.optim.state_dict(),
            discr_optim = self.discr_optim.state_dict()
        )
        torch.save(pkg, path)

    # 加载模型
    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        vae = self.accelerator.unwrap_model(self.vae)
        vae.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])
        self.discr_optim.load_state_dict(pkg['discr_optim'])

    # 打印信息
    def print(self, msg):
        self.accelerator.print(msg)

    # 获取设备
    @property
    def device(self):
        return self.accelerator.device

    # 是否分布式训练
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    # 检查当前进程是否为主进程
    def is_main(self):
        return self.accelerator.is_main_process

    # 检查当前进程是否为本地主进程
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # 训练函数，接受一个日志函数作为参数，默认为一个空函数
    def train(self, log_fn = noop):
        # 获取 VAE 模型参数的设备信息
        device = next(self.vae.parameters()).device

        # 在训练步数未达到指定步数之前循环执行训练步骤
        while self.steps < self.num_train_steps:
            # 执行单个训练步骤，返回日志信息
            logs = self.train_step()
            # 调用日志函数记录日志信息
            log_fn(logs)

        # 打印训练完成信息
        self.print('training complete')
```