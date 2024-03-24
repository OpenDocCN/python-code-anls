# `.\lucidrains\magvit2-pytorch\magvit2_pytorch\trainer.py`

```
# 导入必要的库
from pathlib import Path
from functools import partial
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset, random_split
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
import pytorch_warmup as warmup

from beartype import beartype
from beartype.typing import Optional, Literal, Union, Type

from magvit2_pytorch.optimizer import get_optimizer

from magvit2_pytorch.magvit2_pytorch import VideoTokenizer

from magvit2_pytorch.data import (
    VideoDataset,
    ImageDataset,
    DataLoader,
    video_tensor_to_gif
)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from einops import rearrange

from ema_pytorch import EMA

from pytorch_custom_utils import auto_unwrap_model

# 定义常量

VideosOrImagesLiteral = Union[
    Literal['videos'],
    Literal['images']
]

ConstantLRScheduler = partial(LambdaLR, lr_lambda = lambda step: 1.)

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

# 定义辅助函数

def exists(v):
    return v is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

# 定义类

@auto_unwrap_model()
class VideoTokenizerTrainer:
    @beartype
    def __init__(
        self,
        model: VideoTokenizer,
        *,
        batch_size: int,
        num_train_steps: int,
        learning_rate: float = 1e-5,
        grad_accum_every: int = 1,
        apply_gradient_penalty_every: int = 4,
        max_grad_norm: Optional[float] = None,
        dataset: Optional[Dataset] = None,
        dataset_folder: Optional[str] = None,
        dataset_type: VideosOrImagesLiteral = 'videos',
        checkpoints_folder = './checkpoints',
        results_folder = './results',
        random_split_seed = 42,
        valid_frac = 0.05,
        validate_every_step = 100,
        checkpoint_every_step = 100,
        num_frames = 17,
        use_wandb_tracking = False,
        discr_start_after_step = 0.,
        warmup_steps = 1000,
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        dataset_kwargs: dict = dict()
    @contextmanager
    @beartype
    def trackers(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        hps: Optional[dict] = None
    ):
        assert self.use_wandb_tracking

        self.accelerator.init_trackers(project_name, config = hps)

        if exists(run_name):
            self.accelerator.trackers[0].run.name = run_name

        yield
        self.accelerator.end_training()

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step)

    @property
    def device(self):
        return self.model.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    @property
    def ema_tokenizer(self):
        return self.ema_model.ema_model

    def tokenize(self, *args, **kwargs):
        return self.ema_tokenizer.tokenize(*args, **kwargs)
    # 保存模型参数到指定路径
    def save(self, path, overwrite = True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 如果 overwrite 为 False，则要求路径不存在
        assert overwrite or not path.exists()

        # 构建保存的模型参数字典
        pkg = dict(
            model = self.model.state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            discr_optimizer = self.discr_optimizer.state_dict(),
            warmup = self.warmup.state_dict(),
            scheduler = self.scheduler.state_dict(),
            discr_warmup = self.discr_warmup.state_dict(),
            discr_scheduler = self.discr_scheduler.state_dict(),
            step = self.step
        )

        # 保存多尺度判别器优化器的参数
        for ind, opt in enumerate(self.multiscale_discr_optimizers):
            pkg[f'multiscale_discr_optimizer_{ind}'] = opt.state_dict()

        # 使用 torch.save 保存模型参数到指定路径
        torch.save(pkg, str(path))

    # 加载模型参数
    def load(self, path):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 要求路径存在
        assert path.exists()

        # 加载模型参数字典
        pkg = torch.load(str(path))

        # 加载模型参数到对应的模型、优化器等对象中
        self.model.load_state_dict(pkg['model'])
        self.ema_model.load_state_dict(pkg['ema_model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.discr_optimizer.load_state_dict(pkg['discr_optimizer'])
        self.warmup.load_state_dict(pkg['warmup'])
        self.scheduler.load_state_dict(pkg['scheduler'])
        self.discr_warmup.load_state_dict(pkg['discr_warmup'])
        self.discr_scheduler.load_state_dict(pkg['discr_scheduler'])

        # 加载多尺度判别器优化器的参数
        for ind, opt in enumerate(self.multiscale_discr_optimizers):
            opt.load_state_dict(pkg[f'multiscale_discr_optimizer_{ind}'])

        # 加载步数
        self.step = pkg['step']

    # 执行验证步骤
    @torch.no_grad()
    def valid_step(
        self,
        dl_iter,
        save_recons = True,
        num_save_recons = 1
    ):
        # 将 EMA 模型设置为评估模式
        self.ema_model.eval()

        # 初始化重建损失
        recon_loss = 0.
        ema_recon_loss = 0.

        # 初始化有效视频和重建视频列表
        valid_videos = []
        recon_videos = []

        # 循环执行梯度��积次数
        for _ in range(self.grad_accum_every):
            # 从数据迭代器中获取有效视频数据
            valid_video, = next(dl_iter)
            valid_video = valid_video.to(self.device)

            # 使用自动混合精度计算损失
            with self.accelerator.autocast():
                loss, _ = self.model(valid_video, return_recon_loss_only = True)
                ema_loss, ema_recon_video = self.ema_model(valid_video, return_recon_loss_only = True)

            # 累积重建损失
            recon_loss += loss / self.grad_accum_every
            ema_recon_loss += ema_loss / self.grad_accum_every

            # 调整视频维度
            if valid_video.ndim == 4:
                valid_video = rearrange(valid_video, 'b c h w -> b c 1 h w')

            # 将有效视频和重建视频添加到列表中
            valid_videos.append(valid_video.cpu())
            recon_videos.append(ema_recon_video.cpu())

        # 记录验证重建损失和 EMA 重建损失
        self.log(
            valid_recon_loss = recon_loss.item(),
            valid_ema_recon_loss = ema_recon_loss.item()
        )

        # 打印验证重建损失和 EMA 重建损失
        self.print(f'validation recon loss {recon_loss:.3f}')
        self.print(f'validation EMA recon loss {ema_recon_loss:.3f}')

        # 如果需要保存重建视频
        if not save_recons:
            return

        # 合并有效视频和重建视频
        valid_videos = torch.cat(valid_videos)
        recon_videos = torch.cat(recon_videos)

        # 将重建视频像素值限制在 0 到 1 之间
        recon_videos.clamp_(min = 0., max = 1.)

        # 选择指定数量的有效视频和重建视频
        valid_videos, recon_videos = map(lambda t: t[:num_save_recons], (valid_videos, recon_videos))

        # 重排有效视频和重建视频的维度
        real_and_recon = rearrange([valid_videos, recon_videos], 'n b c f h w -> c f (b h) (n w)')

        # 生成 GIF 文件保存路径
        validate_step = self.step // self.validate_every_step
        sample_path = str(self.results_folder / f'sampled.{validate_step}.gif')

        # 将视频张量保存为 GIF 文件
        video_tensor_to_gif(real_and_recon, str(sample_path))

        # 打印保存的样本路径
        self.print(f'sample saved to {str(sample_path)}')
    # 定义训练方法
    def train(self):

        # 获取当前步数
        step = self.step

        # 创建数据加载器的循环迭代器
        dl_iter = cycle(self.dataloader)
        valid_dl_iter = cycle(self.valid_dataloader)

        # 当步数小于总训练步数时循环执行以下操作
        while step < self.num_train_steps:
            # 打印当前步数
            self.print(f'step {step}')

            # 执行训练步骤
            self.train_step(dl_iter)

            # 等待

            # 如果是主进程且当前步数是验证间隔的倍数时
            if self.is_main and not (step % self.validate_every_step):
                # 执行验证步骤
                self.valid_step(valid_dl_iter)

            # 等待

            # 如果是主进程且当前步数是保存检查点间隔的倍数时
            if self.is_main and not (step % self.checkpoint_every_step):
                # 计算检查点编号
                checkpoint_num = step // self.checkpoint_every_step
                # 检查点路径
                checkpoint_path = self.checkpoints_folder / f'checkpoint.{checkpoint_num}.pt'
                # 保存检查点
                self.save(str(checkpoint_path))

            # 等待

            # 步数加一
            step += 1
```