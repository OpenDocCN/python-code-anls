# `.\lucidrains\musiclm-pytorch\musiclm_pytorch\trainer.py`

```py
# 导入必要的库
import copy
from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
from functools import wraps, partial

from typing_extensions import Annotated

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is
from beartype.typing import Union, List, Optional, Tuple, Callable

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from lion_pytorch import Lion
from musiclm_pytorch import MuLaN
from einops import rearrange
from accelerate import Accelerator

# 用于自动将数据从数据集发出到变换器包装器的关键字的路由

DATASET_FIELD_TYPE_CONFIG = dict(
    wavs = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
    raw_texts = List[str],
    texts = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.long and t.ndim == 2]
    ],
)

# 辅助函数

def exists(val):
    return val is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def noop(*args, **kwargs):
    pass

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

# 自动将数据路由到模块关键字参数的函数

def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
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

# 优化器函数

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params

# 数据加载器函数

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)

# 语义变换器训练器

@beartype
class MuLaNTrainer(nn.Module):
    # 初始化函数，接受多个参数，设置默认值和必须参数
    def __init__(
        self,
        mulan: MuLaN,
        dataset: Dataset,
        *,
        num_train_steps = None,
        batch_size,
        data_max_length = None,
        folder = None,
        lr = 3e-4,
        grad_accum_every = 1,
        betas = (0.9, 0.99),
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        use_lion = False,
        force_clear_prev_results = None  # set to True | False to skip the prompt
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 断言批处理大小大于1，用于对比学习（最好尽可能大）
        assert batch_size > 1, 'batch size must be greater than 1 for contrastive learning (but ideally as large as possible)'

        # 初始化加速器
        self.accelerator = Accelerator(**accelerate_kwargs)

        # 设置参数
        self.mulan = mulan
        self.register_buffer('steps', torch.Tensor([0]))
        self.num_train_steps = default(num_train_steps, len(dataset)) # 默认为1个epoch
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # 选择优化器
        optim_klass = Lion if use_lion else Adam
        self.optim = optim_klass(mulan.parameters(), lr = lr, betas = betas)

        # 设置最大梯度范数
        self.max_grad_norm = max_grad_norm
        self.data_max_length = data_max_length

        # 创建数据集
        self.ds = dataset
        self.ds_fields = None

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
        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, pad_to_longest = False, drop_last = True)
        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, pad_to_longest = False, drop_last = True)

        # 准备加速器
        (
            self.mulan,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.mulan,
            self.optim,
            self.dl,
            self.valid_dl
        )

        # 创建数据加载器迭代器
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        # 设置模型保存频率
        self.save_model_every = save_model_every

        # 设置超参数
        hps = dict(
            num_train_steps = num_train_steps,
            data_max_length = data_max_length,
            learning_rate = lr
        )

        # 初始化跟踪器
        self.accelerator.init_trackers("mulan", config = hps)

        # 设置结果文件夹
        self.results_folder = Path(results_folder)

        # 清除之前的结果
        if force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

        # 将模型移动到设备
        self.mulan.to(self.device)

    # 保存模型
    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.mulan),
            optim = self.optim.state_dict()
        )
        torch.save(pkg, path)

    # 加载模型
    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        mulan = self.accelerator.unwrap_model(self.mulan)
        mulan.load_state_dict(pkg['model'])
        self.optim.load_state_dict(pkg['optim'])
    # 打印消息，调用加速器对象的打印方法
    def print(self, msg):
        self.accelerator.print(msg)

    # 返回加速器对象的设备属性
    @property
    def device(self):
        return self.accelerator.device

    # 返回是否为分布式训练
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    # 返回是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 返回是否为本地主进程
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # 将数据元组转换为关键字参数
    def data_tuple_to_kwargs(self, data):
        # 如果数据字段不存在，则根据数据和数据集字段类型配置确定数据字段
        if not exists(self.ds_fields):
            self.ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
            assert not has_duplicates(self.ds_fields), 'dataset fields must not have duplicate field names'

        # 将数据字段和数据组成字典
        data_kwargs =  dict(zip(self.ds_fields, data))

        # 截取音频数据长度
        wavs = data_kwargs['wavs']
        data_kwargs.update(wavs = wavs[..., :self.data_max_length])

        return data_kwargs

    # 训练步骤
    def train_step(self):
        # 获取设备
        device = self.device

        # 获取步数
        steps = int(self.steps.item())

        # 模型训练
        self.mulan.train()

        # 日志
        logs = {}

        # 更新生成器
        for _ in range(self.grad_accum_every):
            data_kwargs = self.data_tuple_to_kwargs(next(self.dl_iter))

            loss = self.mulan(**data_kwargs)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        # 梯度裁剪
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.mulan.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # 打印日志
        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step = steps)

        # 定期保存模型
        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'mulan.{steps}.pt')
            self.save(model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    # 训练方法
    def train(self, log_fn: Callable = noop):

        # 循环训练步骤
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
```