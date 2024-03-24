# `.\lucidrains\meshgpt-pytorch\meshgpt_pytorch\trainer.py`

```py
# 导入必要的库
from pathlib import Path
from functools import partial
from packaging import version
from contextlib import nullcontext, contextmanager

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

# 导入自定义的工具函数和类
from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule,
    add_wandb_tracker_contextmanager
)

# 导入加速库
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# 导入类型检查相关库
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Type, List

# 导入指数移动平均库
from ema_pytorch import EMA

# 导入数据处理相关函数
from meshgpt_pytorch.data import custom_collate

# 导入版本号
from meshgpt_pytorch.version import __version__

# 导入 MeshGPT 相关模型
from meshgpt_pytorch.meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
)

# 常量定义
DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

# 辅助函数定义

# 判断变量是否存在
def exists(v):
    return v is not None

# 返回默认值
def default(v, d):
    return v if exists(v) else d

# 判断是否可以整除
def divisible_by(num, den):
    return (num % den) == 0

# 生成数据循环迭代器
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 删除字典中指定的键
def maybe_del(d: dict, *keys):
    for key in keys:
        if key not in d:
            continue

        del d[key]

# 自动编码器训练器类定义

# 添加 WandB 追踪上下文管理器
@add_wandb_tracker_contextmanager()
class MeshAutoencoderTrainer(Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        model: MeshAutoencoder,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int,
        val_dataset: Optional[Dataset] = None,
        val_every: int = 100,
        val_num_batches: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.,
        max_grad_norm: Optional[float] = None,
        ema_kwargs: dict = dict(),
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        checkpoint_every = 1000,
        checkpoint_folder = './checkpoints',
        data_kwargs: Tuple[str, ...] = ['vertices', 'faces', 'face_edges'],
        warmup_steps = 1000,
        use_wandb_tracking = False
    # 初始化函数，继承父类的初始化方法
    def __init__(
        self,
        model,
        dataset,
        learning_rate,
        batch_size,
        optimizer_kwargs = {},
        scheduler = None,
        scheduler_kwargs = {},
        warmup_steps = 0,
        max_grad_norm = 1.0,
        grad_accum_every = 1,
        num_train_steps = None,
        checkpoint_every = None,
        checkpoint_folder = 'checkpoints',
        ema_kwargs = {},
        val_dataset = None,
        val_every = 1000,
        val_num_batches = 10,
        data_kwargs = {}
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 实验追踪器
        self.use_wandb_tracking = use_wandb_tracking

        # 如果使用 wandb 追踪
        if use_wandb_tracking:
            # 设置加速器参数中的日志记录方式为 'wandb'
            accelerator_kwargs['log_with'] = 'wandb'

        # 如果加速器参数中没有 'kwargs_handlers'
        if 'kwargs_handlers' not in accelerator_kwargs:
            # 设置加速器参数中的 'kwargs_handlers' 为默认的 DDP 参数
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        # 初始化加速器
        self.accelerator = Accelerator(**accelerator_kwargs)

        # 设置模型
        self.model = model

        # 如果是主进程
        if self.is_main:
            # 初始化 EMA 模型
            self.ema_model = EMA(model, **ema_kwargs)

        # 初始化优化器
        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator = self.accelerator,
            optimizer = get_adam_optimizer(model.parameters(), lr = learning_rate, wd = weight_decay, **optimizer_kwargs),
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs,
            warmup_steps = warmup_steps,
            max_grad_norm = max_grad_norm
        )

        # 初始化数据加载器
        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            drop_last = True,
            collate_fn = partial(custom_collate, pad_id = model.pad_id)
        )

        # 是否需要验证
        self.should_validate = exists(val_dataset)

        # 如果需要验证
        if self.should_validate:
            # 确保验证数据集不为空
            assert len(val_dataset) > 0, 'your validation dataset is empty'

            # 设置验证频率和验证批次数
            self.val_every = val_every
            self.val_num_batches = val_num_batches

            # 初始化验证数据加载器
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size = batch_size,
                shuffle = True,
                drop_last = True,
                collate_fn = partial(custom_collate, pad_id = model.pad_id)
            )

        # 如果数据集具有 'data_kwargs' 属性且不为空
        if hasattr(dataset, 'data_kwargs') and exists(dataset.data_kwargs):
            # 确保数据参数是字符串列表
            assert is_bearable(dataset.data_kwargs, List[str])
            self.data_kwargs = dataset.data_kwargs
        else:
            self.data_kwargs = data_kwargs

        # 准备模型和数据加载器
        (
            self.model,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader
        )

        # 设置梯度累积步数和训练步数
        self.grad_accum_every = grad_accum_every
        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))

        # 设置检查点保存频率和文件夹
        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

    # 获取 EMA tokenizer
    @property
    def ema_tokenizer(self):
        return self.ema_model.ema_model

    # 分词方法
    def tokenize(self, *args, **kwargs):
        return self.ema_tokenizer.tokenize(*args, **kwargs)

    # 日志记录方法
    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step.item())

    # 获取设备
    @property
    def device(self):
        return self.unwrapped_model.device

    # 是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 获取未包装的模型
    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    # 是否为本地主进程
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # 等待方法
    def wait(self):
        return self.accelerator.wait_for_everyone()

    # 打印方法
    def print(self, msg):
        return self.accelerator.print(msg)

    # 保存方法
    def save(self, path, overwrite = True):
        path = Path(path)
        # 如果覆盖或路径不存在
        assert overwrite or not path.exists()

        # 保存模型、EMA 模型、优化器等信息到文件
        pkg = dict(
            model = self.unwrapped_model.state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            version = __version__,
            step = self.step.item(),
            config = self.unwrapped_model._config
        )

        torch.save(pkg, str(path))
    # 加载模型参数
    def load(self, path):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 断言路径存在
        assert path.exists()

        # 加载模型参数
        pkg = torch.load(str(path))

        # 检查模型版本是否与当前包版本匹配
        if version.parse(__version__) != version.parse(pkg['version']):
            self.print(f'loading saved mesh autoencoder at version {pkg["version"]}, but current package version is {__version__}')

        # 加载模型参数
        self.model.load_state_dict(pkg['model'])
        self.ema_model.load_state_dict(pkg['ema_model'])
        self.optimizer.load_state_dict(pkg['optimizer'])

        # 加载步数
        self.step.copy_(pkg['step'])

    # 获取下一个要传递给 forward 方法的数据
    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        # 获取下一个数据
        data = next(dl_iter)

        # 根据数据类型创建传递给 forward 方法的参数字典
        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        elif isinstance(data, dict):
            forward_kwargs = data

        # 删除不需要的键
        maybe_del(forward_kwargs, 'texts', 'text_embeds')
        return forward_kwargs

    # 前向传播方法
    def forward(self):
        # 获取当前步数
        step = self.step.item()
        # 创建数据加载器迭代器
        dl_iter = cycle(self.dataloader)

        # 如果是主进程且需要验证
        if self.is_main and self.should_validate:
            val_dl_iter = cycle(self.val_dataloader)

        # 循环训练步数
        while step < self.num_train_steps:

            # 对于每个梯度累积步数
            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                # 获取下一个要传递给 forward 方法的参数
                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():

                    # 执行模型前向传播
                    total_loss, (recon_loss, commit_loss) = self.model(
                        **forward_kwargs,
                        return_loss_breakdown = True
                    )

                    # 反向传播
                    self.accelerator.backward(total_loss / self.grad_accum_every)

            # 打印重建损失和压缩损失
            self.print(f'recon loss: {recon_loss.item():.3f} | commit loss: {commit_loss.sum().item():.3f}')

            # 记录损失
            self.log(
                total_loss = total_loss.item(),
                commit_loss = commit_loss.sum().item(),
                recon_loss = recon_loss.item()
            )

            # 更新优化器
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 更新步数
            step += 1
            self.step.add_(1)

            # 等待
            self.wait()

            # 如果是主进程，更新 EMA 模型
            if self.is_main:
                self.ema_model.update()

            # 等待
            self.wait()

            # 如果是主进程且需要验证，并且步数是验证间隔的倍数
            if self.is_main and self.should_validate and divisible_by(step, self.val_every):

                total_val_recon_loss = 0.
                self.ema_model.eval()

                num_val_batches = self.val_num_batches * self.grad_accum_every

                # 验证模型
                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():

                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)

                        val_loss, (val_recon_loss, val_commit_loss) = self.ema_model(
                            **forward_kwargs,
                            return_loss_breakdown = True
                        )

                        total_val_recon_loss += (val_recon_loss / num_val_batches)

                # 打印验证重建损失
                self.print(f'valid recon loss: {total_val_recon_loss:.3f}')

                # 记录验证损失
                self.log(val_loss = total_val_recon_loss)

            # 等待
            self.wait()

            # 如果是主进程且步数是保存检查点间隔的倍数
            if self.is_main and divisible_by(step, self.checkpoint_every):
                checkpoint_num = step // self.checkpoint_every
                self.save(self.checkpoint_folder / f'mesh-autoencoder.ckpt.{checkpoint_num}.pt')

            # 等待
            self.wait()

        # 训练完成
        self.print('training complete')
# mesh transformer trainer

# 添加 WandB跟踪上下文管理器
@add_wandb_tracker_contextmanager()
class MeshTransformerTrainer(Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        model: MeshTransformer,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.,
        max_grad_norm: Optional[float] = 0.5,
        val_dataset: Optional[Dataset] = None,
        val_every = 1,
        val_num_batches = 5,
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        checkpoint_every = 1000,
        checkpoint_folder = './checkpoints',
        data_kwargs: Tuple[str, ...] = ['vertices', 'faces', 'face_edges', 'texts'],
        warmup_steps = 1000,
        use_wandb_tracking = False
    ):
        super().__init__()

        # 实验跟踪器

        # 设置是否使用WandB跟踪
        self.use_wandb_tracking = use_wandb_tracking

        # 如果使用WandB跟踪，则设置加速器参数中的日志记录方式为'wandb'
        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        # 如果加速器参数中没有'kwargs_handlers'，则添加默认的DDP参数处理器
        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        # 创建加速器对象
        self.accelerator = Accelerator(**accelerator_kwargs)

        # 设置模型
        self.model = model

        # 获取Adam优化器
        optimizer = get_adam_optimizer(
            model.parameters(),
            lr = learning_rate,
            wd = weight_decay,
            filter_by_requires_grad = True,
            **optimizer_kwargs
        )

        # 设置优化器和学习率调度器
        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator = self.accelerator,
            optimizer = optimizer,
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs,
            warmup_steps = warmup_steps,
            max_grad_norm = max_grad_norm
        )

        # 创建数据加载器
        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            drop_last = True,
            collate_fn = partial(custom_collate, pad_id = model.pad_id)
        )

        # 是否需要验证
        self.should_validate = exists(val_dataset)

        # 如果需要验证
        if self.should_validate:
            assert len(val_dataset) > 0, 'your validation dataset is empty'

            self.val_every = val_every
            self.val_num_batches = val_num_batches

            # 创建验证数据加载器
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size = batch_size,
                shuffle = True,
                drop_last = True,
                collate_fn = partial(custom_collate, pad_id = model.pad_id)
            )

        # 如果数据集有'data_kwargs'属性且存在
        if hasattr(dataset, 'data_kwargs') and exists(dataset.data_kwargs):
            assert is_bearable(dataset.data_kwargs, List[str])
            self.data_kwargs = dataset.data_kwargs
        else:
            self.data_kwargs = data_kwargs

        # 准备模型和数据加载器
        (
            self.model,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader
        )

        # 设置梯度累积次数、训练步数、注册缓冲区
        self.grad_accum_every = grad_accum_every
        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))

        # 设置检查点保存频率和文件夹路径
        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

    # 日志记录函数
    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step.item())

    # 设备属性
    @property
    def device(self):
        return self.unwrapped_model.device

    # 是否为主进程属性
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 未包装模型属性
    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    # 是否为本地主进程属性
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # 等待函数
    def wait(self):
        return self.accelerator.wait_for_everyone()
    # 打印消息，调用加速器的打印方法
    def print(self, msg):
        return self.accelerator.print(msg)

    # 获取下一个要传递给前向传播的数据，并返回关键字参数字典
    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        # 获取下一个数据
        data = next(dl_iter)

        # 如果数据是元组，则将数据关键字与数据值组成字典
        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        # 如果数据是字典，则直接使用该字典
        elif isinstance(data, dict):
            forward_kwargs = data

        return forward_kwargs

    # 保存模型和优化器状态到指定路径
    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        # 构建要保存的数据包
        pkg = dict(
            model = self.unwrapped_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            step = self.step.item(),
            version = __version__
        )

        # 使用torch保存数据包到指定路径
        torch.save(pkg, str(path))

    # 从指定路径加载模型和优化器状态
    def load(self, path):
        path = Path(path)
        assert path.exists()

        # 加载数据包
        pkg = torch.load(str(path))

        # 检查加载的模型版本与当前包版本是否一致
        if version.parse(__version__) != version.parse(pkg['version']):
            self.print(f'loading saved mesh transformer at version {pkg["version"]}, but current package version is {__version__}')

        # 加载模型和优化器状态
        self.model.load_state_dict(pkg['model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])

    # 模型的前向传播方法
    def forward(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        # 如果需要验证，则创建验证数据迭代器
        if self.should_validate:
            val_dl_iter = cycle(self.val_dataloader)

        # 循环训练步数
        while step < self.num_train_steps:

            # 对于每个梯度累积步数
            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                # 获取下一个要传递给前向传播的数据关键字参数
                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                # 使用自动混合精度进行前向传播
                with self.accelerator.autocast(), maybe_no_sync():
                    loss = self.model(**forward_kwargs)

                    # 反向传播
                    self.accelerator.backward(loss / self.grad_accum_every)

            self.print(f'loss: {loss.item():.3f}')

            # 记录损失
            self.log(loss = loss.item())

            # 更新优化器
            self.optimizer.step()
            self.optimizer.zero_grad()

            step += 1
            self.step.add_(1)

            self.wait()

            # 如果是主进程且需要验证，并且当前步数是验证间隔的倍数
            if self.is_main and self.should_validate and divisible_by(step, self.val_every):

                total_val_loss = 0.
                self.unwrapped_model.eval()

                num_val_batches = self.val_num_batches * self.grad_accum_every

                # 验证损失计算
                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():

                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)

                        val_loss = self.unwrapped_model(**forward_kwargs)

                        total_val_loss += (val_loss / num_val_batches)

                self.print(f'valid recon loss: {total_val_loss:.3f}')

                # 记录验证损失
                self.log(val_loss = total_val_loss)

            self.wait()

            # 如果是主进程且当前步数是保存检查点间隔的倍数
            if self.is_main and divisible_by(step, self.checkpoint_every):
                checkpoint_num = step // self.checkpoint_every
                self.save(self.checkpoint_folder / f'mesh-transformer.ckpt.{checkpoint_num}.pt')

            self.wait()

        self.print('training complete')
```