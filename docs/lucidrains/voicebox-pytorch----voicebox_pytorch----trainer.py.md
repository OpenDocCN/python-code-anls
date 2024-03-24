# `.\lucidrains\voicebox-pytorch\voicebox_pytorch\trainer.py`

```py
# 导入正则表达式模块
import re
# 从路径模块中导入 Path 类
from pathlib import Path
# 从 shutil 模块中导入 rmtree 函数
from shutil import rmtree
# 从 functools 模块中导入 partial 函数
from functools import partial
# 从 contextlib 模块中导入 nullcontext 上下文管理器
from contextlib import nullcontext

# 导入 beartype 模块中的 beartype 装饰器
from beartype import beartype

# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn 模块
from torch import nn
# 从 torch.optim.lr_scheduler 模块中导入 CosineAnnealingLR 类
from torch.optim.lr_scheduler import CosineAnnealingLR
# 从 torch.utils.data 模块中导入 Dataset 类和 random_split 函数
from torch.utils.data import Dataset, random_split

# 从 voicebox_pytorch.voicebox_pytorch 模块中导入 ConditionalFlowMatcherWrapper 类
from voicebox_pytorch.voicebox_pytorch import ConditionalFlowMatcherWrapper
# 从 voicebox_pytorch.data 模块中导入 get_dataloader 函数
from voicebox_pytorch.data import get_dataloader
# 从 voicebox_pytorch.optimizer 模块中导入 get_optimizer 函数

from voicebox_pytorch.optimizer import get_optimizer

# 从 accelerate 模块中导入 Accelerator 类和 DistributedType 类
from accelerate import Accelerator, DistributedType
# 从 accelerate.utils 模块中导入 DistributedDataParallelKwargs 类
from accelerate.utils import DistributedDataParallelKwargs

# helpers

# 定义一个函数，判断值是否存在
def exists(val):
    return val is not None

# 定义一个空函数，不做任何操作
def noop(*args, **kwargs):
    pass

# 定义一个循环生成器函数，用于循环遍历数据集
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 定义一个函数，将输入转换为元组
def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

# 定义一个函数，询问用户是或否的问题
def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

# 定义一个函数，累积日志信息
def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# 定义一个函数，从检查点文件名中获取训练步数
def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/voicebox.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    # 使用正则表达式查找文件名中的数字
    results = re.findall(r'\d+', str(checkpoint_path)

    # 如果没有找到数字，则返回 0
    if len(results) == 0:
        return 0

    # 返回最后一个找到的数字
    return int(results[-1])

# 定义一个 VoiceBoxTrainer 类，继承自 nn.Module
class VoiceBoxTrainer(nn.Module):
    # 使用 beartype 装饰器对初始化方法进行类型检查
    @beartype
    def __init__(
        self,
        cfm_wrapper: ConditionalFlowMatcherWrapper,
        *,
        batch_size,
        dataset: Dataset,
        num_train_steps = None,
        num_warmup_steps = None,
        num_epochs = None,
        lr = 3e-4,
        initial_lr = 1e-5,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        log_every = 10,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        force_clear_prev_results = None,
        split_batches = False,
        drop_last = False,
        accelerate_kwargs: dict = dict(),
        ):
        # 调用父类的构造函数
        super().__init__()

        # 设置分布式数据并行的参数
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)

        # 初始化加速器
        self.accelerator = Accelerator(
            kwargs_handlers = [ddp_kwargs],
            split_batches = split_batches,
            **accelerate_kwargs
        )

        # 设置模型包装器
        self.cfm_wrapper = cfm_wrapper

        # 注册缓冲区
        self.register_buffer('steps', torch.Tensor([0]))

        # 设置批量大小和梯度累积步数
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # 初始化优化器
        self.optim = get_optimizer(
            cfm_wrapper.parameters(),
            lr = lr,
            wd = wd
        )

        self.lr = lr
        self.initial_lr = initial_lr

        # 设置最大梯度范数
        self.max_grad_norm = max_grad_norm

        # 创建数据集
        self.ds = dataset

        # 划分验证集
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        assert len(self.ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        assert exists(num_train_steps) or exists(num_epochs), 'either num_train_steps or num_epochs must be specified'

        if exists(num_epochs):
            self.num_train_steps = len(dataset) // batch_size * num_epochs
        else:
            self.num_train_steps = num_train_steps
        self.scheduler = CosineAnnealingLR(self.optim, T_max=self.num_train_steps)
        self.num_warmup_steps = num_warmup_steps if exists(num_warmup_steps) else 0
        
        # 初始化数据加载器
        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)
        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        # 使用加速器准备模型、优化器、调度器和数据加载器
        (
            self.cfm_wrapper,
            self.optim,
            self.scheduler,
            self.dl
        ) = self.accelerator.prepare(
            self.cfm_wrapper,
            self.optim,
            self.scheduler,
            self.dl
        )

        # 初始化数据加载器迭代器
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        # 设置日志、保存模型和保存结果的频率
        self.log_every = log_every
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        # 设置结果文件夹路径
        self.results_folder = Path(results_folder)

        # 如果是主进程并且需要清除之前的结果，则清除结果文件夹
        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        # 创建结果文件夹
        self.results_folder.mkdir(parents = True, exist_ok = True)
        
        # 设置超参数
        hps = {
            "num_train_steps": self.num_train_steps,
            "num_warmup_steps": self.num_warmup_steps,
            "learning_rate": self.lr,
            "initial_learning_rate": self.initial_lr,
            "wd": wd
        }
        # 初始化加速器的跟踪器
        self.accelerator.init_trackers("voicebox", config=hps)

    # 保存模型的方法
    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.cfm_wrapper),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict()
        )
        # 保存模型参数、优化器状态和调度器状态到指定路径
        torch.save(pkg, path)
    # 加载模型参数和优化器状态
    def load(self, path):
        # 解封装模型
        cfm_wrapper = self.accelerator.unwrap_model(self.cfm_wrapper)
        # 加载模型参数
        pkg = cfm_wrapper.load(path)

        # 加载优化器状态
        self.optim.load_state_dict(pkg['optim'])
        # 加载调度器状态
        self.scheduler.load_state_dict(pkg['scheduler'])

        # 从下一步开始，避免覆盖最后一个检查点
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    # 打印消息
    def print(self, msg):
        self.accelerator.print(msg)

    # 生成结果
    def generate(self, *args, **kwargs):
        return self.cfm_wrapper.generate(*args, **kwargs)

    # 获取设备
    @property
    def device(self):
        return self.accelerator.device

    # 是否分布式
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

    # 热身
    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr
    
    # 训练步骤
    def train_step(self):
        steps = int(self.steps.item())

        self.cfm_wrapper.train()
        
        # 根据调度表调整学习率
        
        if steps < self.num_warmup_steps:
            # 应用热身

            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        else:
            # 热身期后，开始应用学习率退火
            
            self.scheduler.step()

        # 日志

        logs = {}

        # 训练步骤

        for grad_accum_step in range(self.grad_accum_every):
            is_last = grad_accum_step == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.cfm_wrapper) if not is_last else nullcontext

            wave, = next(self.dl_iter)

            with self.accelerator.autocast(), context():
                loss = self.cfm_wrapper(wave)

                self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.cfm_wrapper.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # 日志

        if not steps % self.log_every:
            self.print(f"{steps}: loss: {logs['loss']:0.3f}")

        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # 每隔一段时间采样结果

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_results_every):
            wave, = next(self.valid_dl_iter)
            unwrapped_model = self.accelerator.unwrap_model(self.cfm_wrapper)

            with torch.inference_mode():
                unwrapped_model.eval()

                wave = wave.to(unwrapped_model.device)
                valid_loss = unwrapped_model(wave)

                self.print(f'{steps}: valid loss {valid_loss:0.3f}')
                self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # 每隔一段时间保存模型

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'voicebox.{steps}.pt')
            self.save(model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    # 训练
    def train(self, log_fn = noop):
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
        self.accelerator.end_training()
```