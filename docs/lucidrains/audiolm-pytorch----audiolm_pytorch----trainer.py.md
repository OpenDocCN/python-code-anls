# `.\lucidrains\audiolm-pytorch\audiolm_pytorch\trainer.py`

```py
# 导入所需的库
import re
import copy
from math import sqrt
from datetime import timedelta
from random import choice
from pathlib import Path
from shutil import rmtree
from functools import partial
from collections import Counter
from contextlib import contextmanager, nullcontext

# 导入类型提示相关的库
from beartype.typing import Union, List, Optional, Tuple, Type
from typing_extensions import Annotated

# 导入 beartype 相关的库
from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is

# 导入 PyTorch 相关的库
import torch
import torchaudio
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.utils.data import Dataset, DataLoader, random_split

# 导入 pytorch_warmup 库
import pytorch_warmup as warmup

# 导入 einops 库
from einops import rearrange

# 导入 audiolm_pytorch 相关的库
from audiolm_pytorch.optimizer import get_optimizer
import wandb
from ema_pytorch import EMA
from audiolm_pytorch.soundstream import SoundStream
from audiolm_pytorch.encodec import EncodecWrapper
from audiolm_pytorch.audiolm_pytorch import (
    SemanticTransformer,
    SemanticTransformerWrapper,
    CoarseTransformer,
    CoarseTransformerWrapper,
    FineTransformer,
    FineTransformerWrapper,
    FairseqVQWav2Vec,
    HubertWithKmeans
)

# 导入 audiolm_pytorch 中的数据处理相关的库
from audiolm_pytorch.data import SoundDataset, get_dataloader
from audiolm_pytorch.utils import AudioConditionerBase

# 导入 audiolm_pytorch 版本相关的库
from audiolm_pytorch.version import __version__
from packaging import version

# 导入 accelerate 相关的库
from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.tracking import WandBTracker

# 常量定义

DEFAULT_SAMPLE_RATE = 16000

# 定义 ConstantLRScheduler 为 LambdaLR 的部分应用
ConstantLRScheduler = partial(LambdaLR, lr_lambda = lambda step: 1.)

# 确保只有一个 Trainer 实例化

ONE_TRAINER_INSTANTIATED = False

def check_one_trainer():
    global ONE_TRAINER_INSTANTIATED
    assert not ONE_TRAINER_INSTANTIATED, 'only one Trainer can be instantiated at a time for training'
    ONE_TRAINER_INSTANTIATED = True

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(find_unused_parameters = True)

# 用于自动将数据从数据集传递到变换器包装器的关键字

DATASET_FIELD_TYPE_CONFIG = dict(
    raw_wave = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
    text = List[str],
    text_embeds = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim == 3]
    ],
)

# 辅助函数

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def dict_values_to_device(d: dict, device):
    out = {}
    for k, v in d.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out

# 自动将数据传递到模块关键字参数路由函数

def has_duplicates(tup):
    counts = dict(Counter(tup))
    return any(filter(lambda count: count > 1, counts.values()))

def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)

def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.
    # 假设文件名格式类似于"/path/to/semantic.transformer.20000.pt"，表示训练步数为2万步。在这种情况下返回20000
    """
    # 使用正则表达式查找文件路径中的数字部分，并返回结果列表
    results = re.findall(r'\d+', str(checkpoint_path))

    # 如果结果列表为空，则返回0
    if len(results) == 0:
        return 0

    # 返回结果列表中最后一个元素（即最后一个数字）
    return int(results[-1])
# 定义一个带有调度器和热身启动的优化器类
class OptimizerWithWarmupSchedule(nn.Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        warmup_steps: int = 0
    ):
        super().__init__()
        # 创建一个线性热身启动对象
        self.warmup = warmup.LinearWarmup(optimizer, warmup_period = warmup_steps)

        # 如果调度器存在，则使用给定的调度器，否则使用常数学习率调度器
        if exists(scheduler):
            self.scheduler = scheduler(optimizer, **scheduler_kwargs)
        else:
            self.scheduler = ConstantLRScheduler(optimizer)

        self.optimizer = optimizer

        # 准备优化器和调度器
        self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)
        self.accelerator = accelerator

    # 返回状态字典
    def state_dict(self):
        return dict(
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            warmup = self.warmup.state_dict()
        )

    # 加载状态字典
    def load_state_dict(self, pkg):
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.scheduler.load_state_dict(pkg['scheduler'])
        self.warmup.load_state_dict(pkg['warmup'])

    # 清零梯度
    def zero_grad(self):
        self.optimizer.zero_grad()

    # 执行优化步骤
    def step(self):
        self.optimizer.step()

        # 如果优化步骤未被跳过，则执行调度器步骤
        if not self.accelerator.optimizer_step_was_skipped:
            with self.warmup.dampening():
                self.scheduler.step()

# 主训练器类
class SoundStreamTrainer(nn.Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        soundstream: SoundStream,
        *,
        num_train_steps: int,
        batch_size: int,
        data_max_length: int = None,
        data_max_length_seconds: Union[int, float] = None,
        folder: str = None,
        dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        lr: float = 2e-4,
        grad_accum_every: int = 4,
        wd: float = 0.,
        warmup_steps: int = 1000,
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        discr_warmup_steps: Optional[int] = None,
        discr_scheduler: Optional[Type[_LRScheduler]] = None,
        discr_scheduler_kwargs: dict = dict(),
        max_grad_norm: float = 0.5,
        discr_max_grad_norm: float = None,
        save_results_every: int = 100,
        save_model_every: int = 1000,
        log_losses_every: int = 1,
        results_folder: str = './results',
        valid_frac: float = 0.05,
        random_split_seed: int = 42,
        use_ema: bool = True,
        ema_beta: float = 0.995,
        ema_update_after_step: int = 500,
        ema_update_every: int = 10,
        apply_grad_penalty_every: int = 4,
        dl_num_workers: int = 0,
        accelerator: Optional[Accelerator] = None,
        accelerate_kwargs: dict = dict(),
        init_process_group_timeout_seconds = 1800,
        dataloader_drop_last = True,
        split_batches = False,
        use_wandb_tracking = False,
        force_clear_prev_results: bool = None  # set to True | False to skip the prompt
    @property
    def ema_tokenizer(self):
        return self.ema_soundstream.ema_model

    # 对音频进行标记化处理
    def tokenize(self, audio):
        return ema_tokenizer.tokenize(audio)

    # 将模型设置为指数移动平均模型
    def set_model_as_ema_model_(self):
        """ this will force the main 'online' model to have same parameters as the exponentially moving averaged model """
        assert self.use_ema
        self.ema_soundstream.ema_model.load_state_dict(self.soundstream.state_dict())
    # 保存模型参数到指定路径
    def save(self, path):
        # 构建包含模型参数、优化器状态、配置信息等的字典
        pkg = dict(
            model = self.accelerator.get_state_dict(self.soundstream),
            optim = self.optim.state_dict(),
            config = self.unwrapped_soundstream._configs,
            discr_optim = self.discr_optim.state_dict(),
            version = __version__
        )

        # 如果使用指数移动平均模型，保存其参数
        if self.use_ema:
            pkg['ema_model'] = self.ema_soundstream.state_dict()

        # 遍历多尺度鉴别器优化器，保存其参数
        for key, _ in self.multiscale_discriminator_iter():
            discr_optim = getattr(self, key)
            pkg[key] = discr_optim.state_dict()

        # 保存整个包含模型参数的字典到指定路径
        torch.save(pkg, path)

    # 获取未包装的声音流模型
    @property
    def unwrapped_soundstream(self):
        return self.accelerator.unwrap_model(self.soundstream)

    # 加载模型参数
    def load(self, path):
        path = Path(path)
        assert path.exists()
        # 加载模型参数字典
        pkg = torch.load(str(path), map_location = 'cpu')

        # 如果加载的是旧版本，进行特殊处理

        if len(pkg.keys()) > 20:
            self.unwrapped_soundstream.load_state_dict(pkg)

            if self.use_ema:
                self.ema_soundstream.ema_model.load_state_dict(pkg)
            return

        # 检查版本

        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')

        # 否则正常加载模型参数

        self.unwrapped_soundstream.load_state_dict(pkg['model'])

        if self.use_ema:
            assert 'ema_model' in pkg
            self.ema_soundstream.load_state_dict(pkg['ema_model'])

        self.optim.load_state_dict(pkg['optim'])
        self.discr_optim.load_state_dict(pkg['discr_optim'])

        for key, _ in self.multiscale_discriminator_iter():
            discr_optim = getattr(self, key)
            discr_optim.load_state_dict(pkg[key])

        # + 1 以从下一步开始，避免覆盖最后一个检查点

        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    # 遍历多尺度鉴别器
    def multiscale_discriminator_iter(self):
        for ind, discr in enumerate(self.unwrapped_soundstream.discriminators):
            yield f'multiscale_discr_optimizer_{ind}', discr

    # 遍历多尺度鉴别器优化器
    def multiscale_discriminator_optim_iter(self):
        for name, _ in self.multiscale_discriminator_iter():
            yield name, getattr(self, name)

    # 打印消息
    def print(self, msg):
        self.accelerator.print(msg)

    # 记录日志
    def log(self, **logs_as_kwargs):
        self.accelerator.log(logs_as_kwargs, step = self.steps.item())

    # 使用wandb跟踪器
    @contextmanager
    def wandb_tracker(self, project, run = None, hps = None):
        assert self.use_wandb_tracking, '`use_wandb_tracking` must be set to True on SoundStreamTrainer'

        hps = default(hps, self.tracker_hps)

        self.accelerator.init_trackers(project, config = None)

        if exists(run):
            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), self.accelerator.trackers)
            assert exists(wandb_tracker)

            wandb_tracker.run.name = run

        yield

        self.accelerator.end_training()

    # 获取设备
    @property
    def device(self):
        return self.accelerator.device

    # 是否分布式训练
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    # 是否主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 是否本地主进程
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # 训练模型
    def train(self, log_fn = noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
# 语义转换器训练器

class SemanticTransformerTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]],
        transformer: SemanticTransformer,
        *,
        num_train_steps,
        batch_size,
        audio_conditioner: Optional[AudioConditionerBase] = None,
        dataset: Optional[Dataset] = None,
        valid_dataset: Optional[Dataset] = None,
        data_max_length = None,
        data_max_length_seconds = None,
        folder = None,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        init_process_group_timeout_seconds = 1800,
        use_wandb_tracking = False,
        split_batches = False,
        drop_last = False,
        force_clear_prev_results = None,
        average_valid_loss_over_grad_accum_every: bool = True, # if False, valid loss on a single batch
    # 保存模型参数到指定路径
    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.transformer),
            optim = self.optim.state_dict(),
            version = __version__
        )
        torch.save(pkg, path)

    # 从指定路径加载模型参数
    def load(self, path):
        transformer = self.accelerator.unwrap_model(self.transformer)
        pkg = transformer.load(path)
        # 特定于训练器的操作
        self.optim.load_state_dict(pkg['optim'])

        # + 1 to start from the next step and avoid overwriting the last checkpoint
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)


    # 打印消息
    def print(self, msg):
        self.accelerator.print(msg)

    # 生成结果
    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # 将数据元组转换为关键字参数
    def data_tuple_to_kwargs(self, data):
        if not exists(self.ds_fields):
            self.ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
            assert not has_duplicates(self.ds_fields), 'dataset fields must not have duplicate field names'

        return dict(zip(self.ds_fields, data))

    @contextmanager
    def wandb_tracker(self, project, run = None, hps = None):
        assert self.use_wandb_tracking, '`use_wandb_tracking` must be set to True on SemanticTransformerTrainer'

        hps = default(hps, self.tracker_hps)

        self.accelerator.init_trackers(project, config = None)

        if exists(run):
            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), self.accelerator.trackers)
            assert exists(wandb_tracker)

            wandb_tracker.run.name = run

        yield

        self.accelerator.end_training()
    # 定义训练步骤函数
    def train_step(self):
        # 获取设备信息
        device = self.device

        # 获取当前步数
        steps = int(self.steps.item())

        # 设置 Transformer 模型为训练模式
        self.transformer.train()

        # 初始化日志字典
        logs = {}

        # 更新 Transformer 模型
        for i in range(self.grad_accum_every):
            # 判断是否为最后一次迭代
            is_last = i == (self.grad_accum_every - 1)
            # 根据是否为最后一次迭代选择上下文管理器
            context = partial(self.accelerator.no_sync, self.train_wrapper) if not is_last else nullcontext

            # 将数据转换为关键字参数
            data_kwargs = self.data_tuple_to_kwargs(next(self.dl_iter))

            # 使用自动混合精度和上下文管理器进行训练
            with self.accelerator.autocast(), context():
                # 计算损失
                loss = self.train_wrapper(**data_kwargs, return_loss = True)

                # 反向传播
                self.accelerator.backward(loss / self.grad_accum_every)

            # 累积损失日志
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        # 根据最大梯度范数对梯度进行裁剪
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        # 更新优化器
        self.optim.step()
        self.optim.zero_grad()

        # 打印日志
        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # 每隔一段时间对结果进行采样
        self.accelerator.wait_for_everyone()

        # 如果是主进程且满足保存结果的条件
        if self.is_main and not (steps % self.save_results_every):
            # 初始化验证损失
            valid_loss = 0
            # 获取未包装的模型
            unwrapped_model = self.accelerator.unwrap_model(self.train_wrapper)

            # 计算平均验证损失
            for _ in range(self.average_valid_loss_over_grad_accum_every):
                data_kwargs = self.data_tuple_to_kwargs(next(self.valid_dl_iter))
                data_kwargs = dict_values_to_device(data_kwargs, unwrapped_model.device)

                with torch.inference_mode():
                    unwrapped_model.eval()
                    valid_loss += unwrapped_model(**data_kwargs, return_loss = True)

            valid_loss = valid_loss.clone() # 避免推理模���到非推理模式的错误
            valid_loss /= self.average_valid_loss_over_grad_accum_every

            # 打印验证损失日志
            self.print(f'{steps}: valid loss {valid_loss}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # 每隔一段时间保存模型
        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'semantic.transformer.{steps}.pt')
            self.save(model_path)
            if self.use_wandb_tracking:
                wandb.save(model_path)
            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.accelerator.wait_for_everyone()

        # 更新步数
        self.steps.add_(1)
        return logs

    # 训练函数
    def train(self, log_fn = noop):

        # 循环训练直到达到指定步数
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        # 打印训练完成信息
        self.print('training complete')
# 定义粗糙变换器训练器类
class CoarseTransformerTrainer(nn.Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        transformer: CoarseTransformer,  # 粗糙变换器对象
        codec: Union[SoundStream, EncodecWrapper],  # 编解码器对象
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]],  # 可选的音频向量化器对象
        *,
        num_train_steps,  # 训练步数
        batch_size,  # 批量大小
        audio_conditioner: Optional[AudioConditionerBase] = None,  # 可选的音频调节器对象
        dataset: Optional[Dataset] = None,  # 可选的数据集对象
        valid_dataset: Optional[Dataset] = None,  # 可选的验证数据集对象
        ds_fields: Tuple[str, ...] = ('raw_wave', 'raw_wave_for_codec', 'text'),  # 数据集字段元组
        data_max_length = None,  # 数据最大长度
        data_max_length_seconds = None,  # 数据最大长度（秒）
        folder = None,  # 文件夹路径
        lr = 3e-4,  # 学习率
        grad_accum_every = 1,  # 梯度累积频率
        wd = 0.,  # 权重衰减
        max_grad_norm = 0.5,  # 最大梯度范数
        valid_frac = 0.05,  # 验证集比例
        random_split_seed = 42,  # 随机拆分种子
        save_results_every = 100,  # 每隔多少步保存结果
        save_model_every = 1000,  # 每隔多少步保存模型
        results_folder = './results',  # 结果文件夹路径
        accelerate_kwargs: dict = dict(),  # 加速参数字典
        init_process_group_timeout_seconds = 1800,  # 初始化进程组超时时间（秒）
        split_batches = False,  # 是否拆分批次
        drop_last = False,  # 是否丢弃最后一批
        force_clear_prev_results = None,  # 强制清除之前的结果
        use_wandb_tracking = False,  # 是否使用WandB跟踪
        average_valid_loss_over_grad_accum_every: bool = True,  # 是否在梯度累积频率上平均验证损失
    # 保存方法
    def save(self, path):
        # 封装模型、优化器状态字典和版本信息，保存到指定路径
        pkg = dict(
            model = self.accelerator.get_state_dict(self.transformer),
            optim = self.optim.state_dict(),
            version = __version__
        )
        torch.save(pkg, path)

    # 加载方法
    def load(self, path):
        # 解封装模型，加载模型状态字典和优化器状态字典
        transformer = self.accelerator.unwrap_model(self.transformer)
        pkg = transformer.load(path)
        # 加载训练器特定内容
        self.optim.load_state_dict(pkg['optim'])

        # 从下一步开始，避免覆盖最后一个检查点
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    # 打印方法
    def print(self, msg):
        # 打印消息
        self.accelerator.print(msg)

    # 生成方法
    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    # WandB跟踪器上下文管理器
    @contextmanager
    def wandb_tracker(self, project, run = None, hps = None):
        assert self.use_wandb_tracking, '`use_wandb_tracking` must be set to True on CoarseTransformerTrainer'

        hps = default(hps, self.tracker_hps)

        self.accelerator.init_trackers(project, config = None)

        if exists(run):
            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), self.accelerator.trackers)
            assert exists(wandb_tracker)

            wandb_tracker.run.name = run

        yield

        self.accelerator.end_training()  

    # 设备属性
    @property
    def device(self):
        return self.accelerator.device

    # 是否分布式属性
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    # 是否主进程属性
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 是否本地主进程属性
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process
    # 定义训练步骤函数
    def train_step(self):
        # 获取设备信息
        device = self.device

        # 获取当前步数
        steps = int(self.steps.item())

        # 设置 Transformer 模型为训练模式
        self.transformer.train()

        # 初始化日志字典
        logs = {}

        # 更新 Transformer 模型
        for i in range(self.grad_accum_every):
            # 判断是否是最后一次迭代
            is_last = i == (self.grad_accum_every - 1)
            # 根据是否是最后一次迭代选择上下文管理器
            context = partial(self.accelerator.no_sync, self.train_wrapper) if not is_last else nullcontext

            # 从数据加载器迭代器中获取数据关键字参数
            data_kwargs = dict(zip(self.ds_fields, next(self.dl_iter)))

            # 在自动混合精度下，执行训练包装器
            with self.accelerator.autocast(), context():
                loss = self.train_wrapper(
                    **data_kwargs,
                    return_loss = True
                )

                # 反向传播并计算梯度
                self.accelerator.backward(loss / self.grad_accum_every)

            # 累积损失日志
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        # 如果存在最大梯度范数限制，则进行梯度裁剪
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        # 更新优化器参数
        self.optim.step()
        self.optim.zero_grad()

        # 记录日志
        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # 定期采样结果

        self.accelerator.wait_for_everyone()

        # 如果是主进程且满足保存结果的条件
        if self.is_main and not (steps % self.save_results_every):
            valid_loss = 0
            unwrapped_model = self.accelerator.unwrap_model(self.train_wrapper)

            # 计算平均验证损失
            for i in range(self.average_valid_loss_over_grad_accum_every):
                data_kwargs = dict(zip(self.ds_fields, next(self.valid_dl_iter)))
                data_kwargs = dict_values_to_device(data_kwargs, unwrapped_model.device)

                with torch.no_grad():
                    unwrapped_model.eval()

                    valid_loss += unwrapped_model(
                        **data_kwargs,
                        return_loss = True
                    )

            valid_loss = valid_loss.clone() # 避免推理模式到非推理模式的错误
            valid_loss /= self.average_valid_loss_over_grad_accum_every

            # 记录验证损失日志
            self.print(f'{steps}: valid loss {valid_loss}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # 定期保存模型
        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'coarse.transformer.{steps}.pt')
            self.save(model_path)
            if self.use_wandb_tracking:
                wandb.save(model_path)
            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.accelerator.wait_for_everyone()

        # 更新步数
        self.steps.add_(1)
        return logs

    # 训练函数
    def train(self, log_fn = noop):

        # 在未达到训练步数之前循环执行训练步骤
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        # 打印训练完成信息
        self.print('training complete')
# 定义一个 FineTransformerTrainer 类，用于训练 FineTransformer 模型
class FineTransformerTrainer(nn.Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        transformer: FineTransformer,  # 接收 FineTransformer 模型
        codec: Union[SoundStream, EncodecWrapper],  # 接收音频流或编码器包装器
        *,
        num_train_steps,  # 训练步数
        batch_size,  # 批量大小
        audio_conditioner: Optional[AudioConditionerBase] = None,  # 可选的音频调节器
        dataset: Optional[Dataset] = None,  # 可选的数据集
        valid_dataset: Optional[Dataset] = None,  # 可选的验证数据集
        data_max_length = None,  # 数据最大长度
        data_max_length_seconds = None,  # 数据最大长度（秒）
        dataset_normalize = False,  # 是否对数据集进行归一化
        folder = None,  # 文件夹路径
        lr = 3e-4,  # 学习率
        grad_accum_every = 1,  # 梯度累积频率
        wd = 0.,  # 权重衰减
        max_grad_norm = 0.5,  # 最大梯度范数
        valid_frac = 0.05,  # 验证集比例
        random_split_seed = 42,  # 随机拆分种子
        save_results_every = 100,  # 每隔多少步保存结果
        save_model_every = 1000,  # 每隔多少步保存模型
        results_folder = './results',  # 结果保存文件夹路径
        accelerate_kwargs: dict = dict(),  # 加速参数
        init_process_group_timeout_seconds = 1800,  # 初始化进程组超时时间（秒）
        split_batches = False,  # 是否拆分批次
        drop_last = False,  # 是否丢弃最后一批次
        use_wandb_tracking = False,  # 是否使用 WandB 追踪
        force_clear_prev_results = None,  # 强制清除之前的结果
        average_valid_loss_over_grad_accum_every: bool = True,  # 是否在梯度累积频率上计算验证损失的平均值
    # 保存模型方法
    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.transformer),  # 获取模型状态字典
            optim = self.optim.state_dict(),  # 获取优化器状态字典
            version = __version__  # 版本信息
        )
        torch.save(pkg, path)  # 保存模型参数到指定路径

    # 加载模型方法
    def load(self, path):
        transformer = self.accelerator.unwrap_model(self.transformer)  # 解封装模型
        pkg = transformer.load(path)  # 加载模型参数
        # 特定于训练器的操作
        self.optim.load_state_dict(pkg['optim'])  # 加载优化器参数

        # + 1 to start from the next step and avoid overwriting the last checkpoint
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)  # 设置训练步数

    # 打印方法
    def print(self, msg):
        self.accelerator.print(msg)  # 打印消息

    # 生成方法
    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)  # 生成结果

    # WandB 追踪上下文管理器
    @contextmanager
    def wandb_tracker(self, project, run = None, hps = None):
        assert self.use_wandb_tracking, '`use_wandb_tracking` must be set to True on FineTransformerTrainer'  # 断言是否启用 WandB 追踪

        hps = default(hps, self.tracker_hps)  # 设置超参数

        self.accelerator.init_trackers(project, config = None)  # 初始化追踪器

        if exists(run):
            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), self.accelerator.trackers)  # 查找 WandB 追踪器
            assert exists(wandb_tracker)  # 断言是否存在 WandB 追踪器

            wandb_tracker.run.name = run  # 设置运行名称

        yield  # 生成结果

        self.accelerator.end_training()  # 结束训练

    # 设备属性
    @property
    def device(self):
        return self.accelerator.device  # 返回设备

    # 是否分布式属性
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)  # 判断是否分布式

    # 是否主进程属性
    @property
    def is_main(self):
        return self.accelerator.is_main_process  # 判断是否主进程

    # 是否本地主进程属性
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process  # 判断是否本地主进程

    # 数据元组转关键字参数方法
    def data_tuple_to_kwargs(self, data):
        if not exists(self.ds_fields):
            self.ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)  # 确定数据类型
            assert not has_duplicates(self.ds_fields), 'dataset fields must not have duplicate field names'  # 断言数据字段不能有重复字段名

        return dict(zip(self.ds_fields, data))  # 返回数据关键字参数
    # 定义训练步骤函数
    def train_step(self):
        # 获取设备信息
        device = self.device

        # 获取当前步数
        steps = int(self.steps.item())

        # 设置 Transformer 模型为训练模式
        self.transformer.train()

        # 初始化日志字典
        logs = {}

        # 更新 Transformer 模型
        for i in range(self.grad_accum_every):
            # 判断是否是最后一次迭代
            is_last = i == (self.grad_accum_every - 1)
            # 根据是否是最后一次迭代选择上下文管理器
            context = partial(self.accelerator.no_sync, self.train_wrapper) if not is_last else nullcontext

            # 将数据转换为关键字参数
            data_kwargs = self.data_tuple_to_kwargs(next(self.dl_iter))

            # 使用自动混合精度和上下文管理器执行训练
            with self.accelerator.autocast(), context():
                # 计算损失
                loss = self.train_wrapper(**data_kwargs, return_loss = True)

                # 反向传播
                self.accelerator.backward(loss / self.grad_accum_every)

            # 累积损失日志
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        # 如果存在最大梯度范数，则进行梯度裁剪
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        # 更新优化器
        self.optim.step()
        self.optim.zero_grad()

        # 打印日志
        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # 定期采样结果
        self.accelerator.wait_for_everyone()

        # 如果是主进程且满足保存结果条件
        if self.is_main and not (steps % self.save_results_every):
            # 获取未包装的模型
            unwrapped_model = self.accelerator.unwrap_model(self.train_wrapper)
            valid_loss = 0

            # 计算验证集损失
            for i in range(self.average_valid_loss_over_grad_accum_every):
                data_kwargs = self.data_tuple_to_kwargs(next(self.valid_dl_iter))
                data_kwargs = dict_values_to_device(data_kwargs, unwrapped_model.device)

                with torch.inference_mode():
                    unwrapped_model.eval()
                    valid_loss += unwrapped_model(**data_kwargs, return_loss = True)

            valid_loss = valid_loss.clone() # 避免推理模式到非推理模式的错误
            valid_loss /= self.average_valid_loss_over_grad_accum_every

            # 打印验证集损失
            self.print(f'{steps}: valid loss {valid_loss}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # 定期保存模型
        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'fine.transformer.{steps}.pt')
            self.save(model_path)
            if self.use_wandb_tracking:
                wandb.save(model_path)
            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.accelerator.wait_for_everyone()

        # 更新步数
        self.steps.add_(1)
        return logs

    # 训练函数
    def train(self, log_fn = noop):

        # 循环执行训练步骤
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        # 训练完成后打印信息
        self.print('training complete')
```