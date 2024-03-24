# `.\lucidrains\imagen-pytorch\imagen_pytorch\trainer.py`

```py
# 导入必要的库
import os
from math import ceil
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler

import pytorch_warmup as warmup

from imagen_pytorch.imagen_pytorch import Imagen, NullUnet
from imagen_pytorch.elucidated_imagen import ElucidatedImagen
from imagen_pytorch.data import cycle

from imagen_pytorch.version import __version__
from packaging import version

import numpy as np

from ema_pytorch import EMA

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 返回值或默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 将值转换为元组
def cast_tuple(val, length = 1):
    if isinstance(val, list):
        val = tuple(val)

    return val if isinstance(val, tuple) else ((val,) * length)

# 查找第一个满足条件的元素的索引
def find_first(fn, arr):
    for ind, el in enumerate(arr):
        if fn(el):
            return ind
    return -1

# 选择并弹出指定键的值
def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

# 根据键的条件分组字典
def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

# 检查字符串是否以指定前缀开头
def string_begins_with(prefix, str):
    return str.startswith(prefix)

# 根据键的前缀分组字典
def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

# 根据前缀分组字典并修剪键
def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs

# 将数字分成组
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# URL转换为文���系统、存储桶、路径 - 用于将检查点保存到云端

def url_to_bucket(url):
    if '://' not in url:
        return url

    _, suffix = url.split('://')

    if prefix in {'gs', 's3'}:
        return suffix.split('/')[0]
    else:
        raise ValueError(f'storage type prefix "{prefix}" is not supported yet')

# 装饰器

# 模型评估装饰器
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 转换为Torch张量装饰器
def cast_torch_tensor(fn, cast_fp16 = False):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        device = kwargs.pop('_device', model.device)
        cast_device = kwargs.pop('_cast_device', True)

        should_cast_fp16 = cast_fp16 and model.cast_half_at_training

        kwargs_keys = kwargs.keys()
        all_args = (*args, *kwargs.values())
        split_kwargs_index = len(all_args) - len(kwargs_keys)
        all_args = tuple(map(lambda t: torch.from_numpy(t) if exists(t) and isinstance(t, np.ndarray) else t, all_args))

        if cast_device:
            all_args = tuple(map(lambda t: t.to(device) if exists(t) and isinstance(t, torch.Tensor) else t, all_args))

        if should_cast_fp16:
            all_args = tuple(map(lambda t: t.half() if exists(t) and isinstance(t, torch.Tensor) and t.dtype != torch.bool else t, all_args))

        args, kwargs_values = all_args[:split_kwargs_index], all_args[split_kwargs_index:]
        kwargs = dict(tuple(zip(kwargs_keys, kwargs_values)))

        out = fn(model, *args, **kwargs)
        return out
    return inner
# 定义一个函数，将可迭代对象按照指定大小分割成子列表
def split_iterable(it, split_size):
    accum = []
    # 遍历可迭代对象，根据指定大小分割成子列表
    for ind in range(ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index: (start_index + split_size)])
    return accum

# 定义一个函数，根据不同类型的输入进行分割操作
def split(t, split_size = None):
    # 如果未指定分割大小，则直接返回输入
    if not exists(split_size):
        return t

    # 如果输入是 torch.Tensor 类型，则按照指定大小在指定维度上进行分割
    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim = 0)

    # 如果输入是可迭代对象，则调用 split_iterable 函数进行分割
    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    # 其他情况返回类型错误
    return TypeError

# 定义一个函数，查找满足条件的第一个元素
def find_first(cond, arr):
    # 遍历数组，找到满足条件的第一个元素并返回
    for el in arr:
        if cond(el):
            return el
    return None

# 定义一个函数，将参数和关键字参数按照指定大小分割成子列表
def split_args_and_kwargs(*args, split_size = None, **kwargs):
    # 将所有参数和关键字参数合并成一个列表
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)
    # 找到第一个是 torch.Tensor 类型的参数
    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
    assert exists(first_tensor)

    # 获取第一个 tensor 的大小作为 batch_size
    batch_size = len(first_tensor)
    split_size = default(split_size, batch_size)
    num_chunks = ceil(batch_size / split_size)

    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len

    # 对所有参数和关键字参数进行分割操作
    split_all_args = [split(arg, split_size = split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else ((arg,) * num_chunks) for arg in all_args]
    chunk_sizes = num_to_groups(batch_size, split_size)

    # 遍历分割后的结果，生成分块大小比例和分块后的参数和关键字参数
    for (chunk_size, *chunked_all_args) in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)

# 定义一个装饰器函数，用于对输入的函数进行分块处理
def imagen_sample_in_chunks(fn):
    @wraps(fn)
    def inner(self, *args, max_batch_size = None, **kwargs):
        # 如果未指定最大批处理大小，则直接调用原函数
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)

        # 如果是无条件的训练，则根据最大批处理大小分块处理
        if self.imagen.unconditional:
            batch_size = kwargs.get('batch_size')
            batch_sizes = num_to_groups(batch_size, max_batch_size)
            outputs = [fn(self, *args, **{**kwargs, 'batch_size': sub_batch_size}) for sub_batch_size in batch_sizes]
        else:
            # 否则根据参数和关键字参数进行分块处理
            outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs)]

        # 如果输出是 torch.Tensor 类型，则按照指定维��拼接
        if isinstance(outputs[0], torch.Tensor):
            return torch.cat(outputs, dim = 0)

        # 否则对输出进行拼接处理
        return list(map(lambda t: torch.cat(t, dim = 0), list(zip(*outputs))))

    return inner

# 定义一个函数，用于恢复模型的部分参数
def restore_parts(state_dict_target, state_dict_from):
    for name, param in state_dict_from.items():

        if name not in state_dict_target:
            continue

        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
        else:
            print(f"layer {name}({param.size()} different than target: {state_dict_target[name].size()}")

    return state_dict_target

# 定义一个类，用于图像生成的训练
class ImagenTrainer(nn.Module):
    locked = False

    def __init__(
        self,
        imagen = None,
        imagen_checkpoint_path = None,
        use_ema = True,
        lr = 1e-4,
        eps = 1e-8,
        beta1 = 0.9,
        beta2 = 0.99,
        max_grad_norm = None,
        group_wd_params = True,
        warmup_steps = None,
        cosine_decay_max_steps = None,
        only_train_unet_number = None,
        fp16 = False,
        precision = None,
        split_batches = True,
        dl_tuple_output_keywords_names = ('images', 'text_embeds', 'text_masks', 'cond_images'),
        verbose = True,
        split_valid_fraction = 0.025,
        split_valid_from_train = False,
        split_random_seed = 42,
        checkpoint_path = None,
        checkpoint_every = None,
        checkpoint_fs = None,
        fs_kwargs: dict = None,
        max_checkpoints_keep = 20,
        **kwargs
    # 准备训练器，确保训练器尚未准备好，设置只训练的 UNet 编号，并将 prepared 标记为 True
    def prepare(self):
        assert not self.prepared, f'The trainer is allready prepared'
        self.validate_and_set_unet_being_trained(self.only_train_unet_number)
        self.prepared = True
    # 计算属性

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

    @property
    def unwrapped_unet(self):
        return self.accelerator.unwrap_model(self.unet_being_trained)

    # 优化器辅助函数

    def get_lr(self, unet_number):
        self.validate_unet_number(unet_number)
        unet_index = unet_number - 1

        optim = getattr(self, f'optim{unet_index}')

        return optim.param_groups[0]['lr']

    # 仅允许同时训练一个 UNet 的函数

    def validate_and_set_unet_being_trained(self, unet_number = None):
        if exists(unet_number):
            self.validate_unet_number(unet_number)

        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you cannot only train on one unet at a time. you will need to save the trainer into a checkpoint, and resume training on a new unet'

        self.only_train_unet_number = unet_number
        self.imagen.only_train_unet_number = unet_number

        if not exists(unet_number):
            return

        self.wrap_unet(unet_number)

    def wrap_unet(self, unet_number):
        if hasattr(self, 'one_unet_wrapped'):
            return

        unet = self.imagen.get_unet(unet_number)
        unet_index = unet_number - 1

        optimizer = getattr(self, f'optim{unet_index}')
        scheduler = getattr(self, f'scheduler{unet_index}')

        if self.train_dl:
            self.unet_being_trained, self.train_dl, optimizer = self.accelerator.prepare(unet, self.train_dl, optimizer)
        else:
            self.unet_being_trained, optimizer = self.accelerator.prepare(unet, optimizer)

        if exists(scheduler):
            scheduler = self.accelerator.prepare(scheduler)

        setattr(self, f'optim{unet_index}', optimizer)
        setattr(self, f'scheduler{unet_index}', scheduler)

        self.one_unet_wrapped = True

    # 由于没有每个优化器单独的 gradscaler，对 accelerator 进行修改

    def set_accelerator_scaler(self, unet_number):
        def patch_optimizer_step(accelerated_optimizer, method):
            def patched_step(*args, **kwargs):
                accelerated_optimizer._accelerate_step_called = True
                return method(*args, **kwargs)
            return patched_step

        unet_number = self.validate_unet_number(unet_number)
        scaler = getattr(self, f'scaler{unet_number - 1}')

        self.accelerator.scaler = scaler
        for optimizer in self.accelerator._optimizers:
            optimizer.scaler = scaler
            optimizer._accelerate_step_called = False
            optimizer._optimizer_original_step_method = optimizer.optimizer.step
            optimizer._optimizer_patched_step_method = patch_optimizer_step(optimizer, optimizer.optimizer.step)

    # 辅助打印函数

    def print(self, msg):
        if not self.is_main:
            return

        if not self.verbose:
            return

        return self.accelerator.print(msg)

    # 验证 UNet 编号

    def validate_unet_number(self, unet_number = None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        assert 0 < unet_number <= self.num_unets, f'unet number should be in between 1 and {self.num_unets}'
        return unet_number

    # 训练步骤数
    # 返回指定 U-Net 编号的训练步数
    def num_steps_taken(self, unet_number = None):
        # 如果只有一个 U-Net，则默认使用编号为 1
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        # 返回指定 U-Net 的训练步数
        return self.steps[unet_number - 1].item()

    # 打印未训练的 U-Net
    def print_untrained_unets(self):
        print_final_error = False

        # 遍历训练步数和 U-Net 对象，检查是否未训练
        for ind, (steps, unet) in enumerate(zip(self.steps.tolist(), self.imagen.unets)):
            if steps > 0 or isinstance(unet, NullUnet):
                continue

            # 打印未训练的 U-Net 编号
            self.print(f'unet {ind + 1} has not been trained')
            print_final_error = True

        # 如果存在未训练的 U-Net，则打印提示信息
        if print_final_error:
            self.print('when sampling, you can pass stop_at_unet_number to stop early in the cascade, so it does not try to generate with untrained unets')

    # 数据相关函数

    # 添加训练数据加载器
    def add_train_dataloader(self, dl = None):
        if not exists(dl):
            return

        # 确保训练数据加载器未添加过
        assert not exists(self.train_dl), 'training dataloader was already added'
        assert not self.prepared, f'You need to add the dataset before preperation'
        self.train_dl = dl

    # 添加验证数据加载器
    def add_valid_dataloader(self, dl):
        if not exists(dl):
            return

        # 确保验证数据加载器未添加过
        assert not exists(self.valid_dl), 'validation dataloader was already added'
        assert not self.prepared, f'You need to add the dataset before preperation'
        self.valid_dl = dl

    # 添加训练数据集
    def add_train_dataset(self, ds = None, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        # 确保训练数据加载器未添加过
        assert not exists(self.train_dl), 'training dataloader was already added'

        # 如果需要从训练数据集中分割验证数据集
        valid_ds = None
        if self.split_valid_from_train:
            # 计算训练数据集和验证数据集的大小
            train_size = int((1 - self.split_valid_fraction) * len(ds)
            valid_size = len(ds) - train_size

            # 随机分割数据集
            ds, valid_ds = random_split(ds, [train_size, valid_size], generator = torch.Generator().manual_seed(self.split_random_seed))
            self.print(f'training with dataset of {len(ds)} samples and validating with randomly splitted {len(valid_ds)} samples')

        # 创建数据加载器并添加训练数据加载器
        dl = DataLoader(ds, batch_size = batch_size, **dl_kwargs)
        self.add_train_dataloader(dl)

        # 如果不需要从训练数据集中分割验证数据集，则直接返回
        if not self.split_valid_from_train:
            return

        # 添加验证数据集
        self.add_valid_dataset(valid_ds, batch_size = batch_size, **dl_kwargs)

    # 添加验证数据集
    def add_valid_dataset(self, ds, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        # 确保验证数据加载器未添加过
        assert not exists(self.valid_dl), 'validation dataloader was already added'

        # 创建数据加载器并添加验证数据加载器
        dl = DataLoader(ds, batch_size = batch_size, **dl_kwargs)
        self.add_valid_dataloader(dl)

    # 创建训练数据迭代器
    def create_train_iter(self):
        assert exists(self.train_dl), 'training dataloader has not been registered with the trainer yet'

        if exists(self.train_dl_iter):
            return

        self.train_dl_iter = cycle(self.train_dl)

    # 创建验证数据迭代器
    def create_valid_iter(self):
        assert exists(self.valid_dl), 'validation dataloader has not been registered with the trainer yet'

        if exists(self.valid_dl_iter):
            return

        self.valid_dl_iter = cycle(self.valid_dl)

    # 训练步骤
    def train_step(self, *, unet_number = None, **kwargs):
        if not self.prepared:
            self.prepare()
        self.create_train_iter()

        kwargs = {'unet_number': unet_number, **kwargs}
        loss = self.step_with_dl_iter(self.train_dl_iter, **kwargs)
        self.update(unet_number = unet_number)
        return loss

    # 验证步骤
    @torch.no_grad()
    @eval_decorator
    def valid_step(self, **kwargs):
        if not self.prepared:
            self.prepare()
        self.create_valid_iter()
        context = self.use_ema_unets if kwargs.pop('use_ema_unets', False) else nullcontext
        with context():
            loss = self.step_with_dl_iter(self.valid_dl_iter, **kwargs)
        return loss
    # 使用 dl_iter 迭代器获取下一个数据元组
    def step_with_dl_iter(self, dl_iter, **kwargs):
        dl_tuple_output = cast_tuple(next(dl_iter))
        # 将数据元组转换为字典
        model_input = dict(list(zip(self.dl_tuple_output_keywords_names, dl_tuple_output)))
        # 调用 forward 方法计算损失
        loss = self.forward(**{**kwargs, **model_input})
        return loss

    # 检查点函数

    # 获取所有按照时间排序的检查点文件
    @property
    def all_checkpoints_sorted(self):
        glob_pattern = os.path.join(self.checkpoint_path, '*.pt')
        checkpoints = self.fs.glob(glob_pattern)
        sorted_checkpoints = sorted(checkpoints, key = lambda x: int(str(x).split('.')[-2]), reverse = True)
        return sorted_checkpoints

    # 从检查点文件夹加载模型
    def load_from_checkpoint_folder(self, last_total_steps = -1):
        if last_total_steps != -1:
            filepath = os.path.join(self.checkpoint_path, f'checkpoint.{last_total_steps}.pt')
            self.load(filepath)
            return

        sorted_checkpoints = self.all_checkpoints_sorted

        if len(sorted_checkpoints) == 0:
            self.print(f'no checkpoints found to load from at {self.checkpoint_path}')
            return

        last_checkpoint = sorted_checkpoints[0]
        self.load(last_checkpoint)

    # 保存到检查点文件夹
    def save_to_checkpoint_folder(self):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        total_steps = int(self.steps.sum().item())
        filepath = os.path.join(self.checkpoint_path, f'checkpoint.{total_steps}.pt')

        self.save(filepath)

        if self.max_checkpoints_keep <= 0:
            return

        sorted_checkpoints = self.all_checkpoints_sorted
        checkpoints_to_discard = sorted_checkpoints[self.max_checkpoints_keep:]

        for checkpoint in checkpoints_to_discard:
            self.fs.rm(checkpoint)

    # 保存和加载函数

    # 保存模型到指定路径
    def save(
        self,
        path,
        overwrite = True,
        without_optim_and_sched = False,
        **kwargs
    ):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        fs = self.fs

        assert not (fs.exists(path) and not overwrite)

        self.reset_ema_unets_all_one_device()

        # 构建保存对象
        save_obj = dict(
            model = self.imagen.state_dict(),
            version = __version__,
            steps = self.steps.cpu(),
            **kwargs
        )

        save_optim_and_sched_iter = range(0, self.num_unets) if not without_optim_and_sched else tuple()

        # 保存优化器和调度器状态
        for ind in save_optim_and_sched_iter:
            scaler_key = f'scaler{ind}'
            optimizer_key = f'optim{ind}'
            scheduler_key = f'scheduler{ind}'
            warmup_scheduler_key = f'warmup{ind}'

            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)
            warmup_scheduler = getattr(self, warmup_scheduler_key)

            if exists(scheduler):
                save_obj = {**save_obj, scheduler_key: scheduler.state_dict()}

            if exists(warmup_scheduler):
                save_obj = {**save_obj, warmup_scheduler_key: warmup_scheduler.state_dict()}

            save_obj = {**save_obj, scaler_key: scaler.state_dict(), optimizer_key: optimizer.state_dict()}

        if self.use_ema:
            save_obj = {**save_obj, 'ema': self.ema_unets.state_dict()}

        # 确定是否存在 imagen 配置
        if hasattr(self.imagen, '_config'):
            self.print(f'this checkpoint is commandable from the CLI - "imagen --model {str(path)} \"<prompt>"')
            save_obj = {
                **save_obj,
                'imagen_type': 'elucidated' if self.is_elucidated else 'original',
                'imagen_params': self.imagen._config
            }

        # 保存到指定路径
        with fs.open(path, 'wb') as f:
            torch.save(save_obj, f)

        self.print(f'checkpoint saved to {path}')
    # 加载模型参数和优化器状态
    def load(self, path, only_model = False, strict = True, noop_if_not_exist = False):
        # 获取文件系统对象
        fs = self.fs

        # 如果文件不存在且设置了不执行操作，则打印消息并返回
        if noop_if_not_exist and not fs.exists(path):
            self.print(f'trainer checkpoint not found at {str(path)}')
            return

        # 断言文件存在，否则抛出异常
        assert fs.exists(path), f'{path} does not exist'

        # 重置所有 EMA 模型到同一设备上
        self.reset_ema_unets_all_one_device()

        # 避免在主进程中使用 Accelerate 时产生额外的 GPU 内存使用
        with fs.open(path) as f:
            # 加载模型参数和优化器状态
            loaded_obj = torch.load(f, map_location='cpu')

        # 检查加载的模型版本是否与当前包版本一致
        if version.parse(__version__) != version.parse(loaded_obj['version']):
            self.print(f'loading saved imagen at version {loaded_obj["version"]}, but current package version is {__version__}')

        try:
            # 加载模型参数
            self.imagen.load_state_dict(loaded_obj['model'], strict = strict)
        except RuntimeError:
            print("Failed loading state dict. Trying partial load")
            # 尝试部分加载模型参数
            self.imagen.load_state_dict(restore_parts(self.imagen.state_dict(),
                                                      loaded_obj['model']))

        # 如果只加载模型参数，则返回加载的对象
        if only_model:
            return loaded_obj

        # 复制加载的步数
        self.steps.copy_(loaded_obj['steps'])

        # 遍历所有 U-Net 模型
        for ind in range(0, self.num_unets):
            scaler_key = f'scaler{ind}'
            optimizer_key = f'optim{ind}'
            scheduler_key = f'scheduler{ind}'
            warmup_scheduler_key = f'warmup{ind}'

            # 获取对应的 scaler、optimizer、scheduler 和 warmup_scheduler
            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)
            warmup_scheduler = getattr(self, warmup_scheduler_key)

            # 如果 scheduler 存在且在加载对象中有对应的键，则加载其状态
            if exists(scheduler) and scheduler_key in loaded_obj:
                scheduler.load_state_dict(loaded_obj[scheduler_key])

            # 如果 warmup_scheduler 存在且在加载对象中���对应的键，则加载其状态
            if exists(warmup_scheduler) and warmup_scheduler_key in loaded_obj:
                warmup_scheduler.load_state_dict(loaded_obj[warmup_scheduler_key])

            # 如果 optimizer 存在，则尝试加载其状态
            if exists(optimizer):
                try:
                    optimizer.load_state_dict(loaded_obj[optimizer_key])
                    scaler.load_state_dict(loaded_obj[scaler_key])
                except:
                    self.print('could not load optimizer and scaler, possibly because you have turned on mixed precision training since the last run. resuming with new optimizer and scalers')

        # 如果使用 EMA，则加载 EMA 模型参数
        if self.use_ema:
            assert 'ema' in loaded_obj
            try:
                self.ema_unets.load_state_dict(loaded_obj['ema'], strict = strict)
            except RuntimeError:
                print("Failed loading state dict. Trying partial load")
                self.ema_unets.load_state_dict(restore_parts(self.ema_unets.state_dict(),
                                                             loaded_obj['ema']))

        # 打印加载成功的消息，并返回加载的对象
        self.print(f'checkpoint loaded from {path}')
        return loaded_obj

    # 获取所有 EMA 模型
    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    # 获取指定编号的 EMA 模型
    def get_ema_unet(self, unet_number = None):
        # 如果不使用 EMA，则返回
        if not self.use_ema:
            return

        # 验证并获取正确的 U-Net 编号
        unet_number = self.validate_unet_number(unet_number)
        index = unet_number - 1

        # 如果 unets 是 nn.ModuleList，则转换为列表并更新 ema_unets
        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.ema_unets]
            delattr(self, 'ema_unets')
            self.ema_unets = unets_list

        # 将当前训练的 EMA 模型移到指定设备上
        if index != self.ema_unet_being_trained_index:
            for unet_index, unet in enumerate(self.ema_unets):
                unet.to(self.device if unet_index == index else 'cpu')

        # 更新当前训练的 EMA 模型索引，并返回对应的 EMA 模型
        self.ema_unet_being_trained_index = index
        return self.ema_unets[index]

    # 重置所有 EMA 模型到指定设备上
    def reset_ema_unets_all_one_device(self, device = None):
        # 如果不使用 EMA，则返回
        if not self.use_ema:
            return

        # 获取默认设备
        device = default(device, self.device)
        # 将所有 EMA 模型转移到指定设备上
        self.ema_unets = nn.ModuleList([*self.ema_unets])
        self.ema_unets.to(device)

        # 重置当前训练的 EMA 模型索引
        self.ema_unet_being_trained_index = -1

    # 禁用梯度计算
    @torch.no_grad()
    # 定义一个上下文管理器，用于控制是否使用指数移动平均的 U-Net 模型
    @contextmanager
    def use_ema_unets(self):
        # 如果不使用指数移动平均模型，则直接返回输出
        if not self.use_ema:
            output = yield
            return output

        # 重置所有 U-Net 模型为同一设备上的指数移动平均模型
        self.reset_ema_unets_all_one_device()
        self.imagen.reset_unets_all_one_device()

        # 将 U-Net 模型设置为评估模式
        self.unets.eval()

        # 保存可训练的 U-Net 模型，然后将指数移动平均模型用于采样
        trainable_unets = self.imagen.unets
        self.imagen.unets = self.unets

        output = yield

        # 恢复原始的训练 U-Net 模型
        self.imagen.unets = trainable_unets

        # 将指数移动平均模型的 U-Net 恢复到原始设备
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        return output

    # 打印 U-Net 模型的设备信息
    def print_unet_devices(self):
        self.print('unet devices:')
        for i, unet in enumerate(self.imagen.unets):
            device = next(unet.parameters()).device
            self.print(f'\tunet {i}: {device}')

        # 如果不使用指数移动平均模型，则直接返回
        if not self.use_ema:
            return

        self.print('\nema unet devices:')
        for i, ema_unet in enumerate(self.ema_unets):
            device = next(ema_unet.parameters()).device
            self.print(f'\tema unet {i}: {device}')

    # 重写状态字典函数

    def state_dict(self, *args, **kwargs):
        # 重置所有 U-Net 模型为同一设备上的指数移动平均模型
        self.reset_ema_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        # 重置所有 U-Net 模型为同一设备上的指数移动平均模型
        self.reset_ema_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # 编码文本函数

    def encode_text(self, text, **kwargs):
        return self.imagen.encode_text(text, **kwargs)

    # 前向传播函数和梯度更新步骤

    def update(self, unet_number = None):
        unet_number = self.validate_unet_number(unet_number)
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        index = unet_number - 1
        unet = self.unet_being_trained

        optimizer = getattr(self, f'optim{index}')
        scaler = getattr(self, f'scaler{index}')
        scheduler = getattr(self, f'scheduler{index}')
        warmup_scheduler = getattr(self, f'warmup{index}')

        # 在加速器上设置梯度缩放器，因为我们每个 U-Net 管理一个

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(unet.parameters(), self.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        if self.use_ema:
            ema_unet = self.get_ema_unet(unet_number)
            ema_unet.update()

        # 调度器，如果需要

        maybe_warmup_context = nullcontext() if not exists(warmup_scheduler) else warmup_scheduler.dampening()

        with maybe_warmup_context:
            if exists(scheduler) and not self.accelerator.optimizer_step_was_skipped: # 推荐在文档中
                scheduler.step()

        self.steps += F.one_hot(torch.tensor(unet_number - 1, device = self.steps.device), num_classes = len(self.steps))

        if not exists(self.checkpoint_path):
            return

        total_steps = int(self.steps.sum().item())

        if total_steps % self.checkpoint_every:
            return

        self.save_to_checkpoint_folder()

    @torch.no_grad()
    @cast_torch_tensor
    @imagen_sample_in_chunks
    def sample(self, *args, **kwargs):
        context = nullcontext if  kwargs.pop('use_non_ema', False) else self.use_ema_unets

        self.print_untrained_unets()

        if not self.is_main:
            kwargs['use_tqdm'] = False

        with context():
            output = self.imagen.sample(*args, device = self.device, **kwargs)

        return output

    @partial(cast_torch_tensor, cast_fp16 = True)
    def forward(
        self,
        *args,
        unet_number = None,
        max_batch_size = None,
        **kwargs
        ):
        # 验证并修正 UNet 编号
        unet_number = self.validate_unet_number(unet_number)
        # 验证并设置正在训练的 UNet 编号
        self.validate_and_set_unet_being_trained(unet_number)
        # 设置加速器缩放器
        self.set_accelerator_scaler(unet_number)

        # 断言只有训练指定 UNet 编号或者没有指定 UNet 编号
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, f'you can only train unet #{self.only_train_unet_number}'

        # 初始化总损失
        total_loss = 0.

        # 将参数和关键字参数按照最大批处理大小拆分
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            # 使用加速器自动转换
            with self.accelerator.autocast():
                # 计算损失
                loss = self.imagen(*chunked_args, unet = self.unet_being_trained, unet_number = unet_number, **chunked_kwargs)
                # 损失乘以分块大小比例
                loss = loss * chunk_size_frac

            # 累加总损失
            total_loss += loss.item()

            # 如果处于训练状态，进行反向传播
            if self.training:
                self.accelerator.backward(loss)

        # 返回总损失
        return total_loss
```