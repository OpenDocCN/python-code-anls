# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\trainer.py`

```py
# 导入必要的库
import time
import copy
from pathlib import Path
from math import ceil
from functools import partial, wraps
from contextlib import nullcontext
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

# 导入自定义模块
from dalle2_pytorch.dalle2_pytorch import Decoder, DiffusionPrior
from dalle2_pytorch.optimizer import get_optimizer
from dalle2_pytorch.version import __version__
from packaging import version

# 导入第三方库
import pytorch_warmup as warmup
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedType
import numpy as np

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 将值转换为元组
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 从字典中选择指定键的值并弹出这些键
def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

# 根据条件将字典分组
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

# 根据键的前缀将字典分组
def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

# 根据前缀将字典分组并修剪键
def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs

# 将数字分成若干组
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 装饰器

# 将函数参数转换为 torch 张量
def cast_torch_tensor(fn):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        device = kwargs.pop('_device', next(model.parameters()).device)
        cast_device = kwargs.pop('_cast_device', True)
        cast_deepspeed_precision = kwargs.pop('_cast_deepspeed_precision', True)

        kwargs_keys = kwargs.keys()
        all_args = (*args, *kwargs.values())
        split_kwargs_index = len(all_args) - len(kwargs_keys)
        all_args = tuple(map(lambda t: torch.from_numpy(t) if exists(t) and isinstance(t, np.ndarray) else t, all_args))

        if cast_device:
            all_args = tuple(map(lambda t: t.to(device) if exists(t) and isinstance(t, torch.Tensor) else t, all_args))

        if cast_deepspeed_precision:
            try:
                accelerator = model.accelerator
                if accelerator is not None and accelerator.distributed_type == DistributedType.DEEPSPEED:
                    cast_type_map = {
                        "fp16": torch.half,
                        "bf16": torch.bfloat16,
                        "no": torch.float
                    }
                    precision_type = cast_type_map[accelerator.mixed_precision]
                    all_args = tuple(map(lambda t: t.to(precision_type) if exists(t) and isinstance(t, torch.Tensor) else t, all_args))
            except AttributeError:
                # Then this model doesn't have an accelerator
                pass

        args, kwargs_values = all_args[:split_kwargs_index], all_args[split_kwargs_index:]
        kwargs = dict(tuple(zip(kwargs_keys, kwargs_values)))

        out = fn(model, *args, **kwargs)
        return out
    return inner

# 梯度累积函数

# 将可迭代对象分割成指定大小的子集
def split_iterable(it, split_size):
    accum = []
    for ind in range(ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index: (start_index + split_size)])
    return accum

# 如果未提供分割大小，则返回原始对象
def split(t, split_size = None):
    if not exists(split_size):
        return t
    # 检查输入是否为 torch.Tensor 类型
    if isinstance(t, torch.Tensor):
        # 如果是，则按照指定维度和大小拆分张量
        return t.split(split_size, dim=0)

    # 检查输入是否为可迭代对象
    if isinstance(t, Iterable):
        # 如果是，则调用自定义函数 split_iterable() 拆分可迭代对象
        return split_iterable(t, split_size)

    # 如果输入既不是 torch.Tensor 也不是可迭代对象，则返回类型错误
    return TypeError
# 在给定条件下，查找数组中第一个满足条件的元素并返回
def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

# 将位置参数和关键字参数拆分成一个包含所有参数值的元组，并计算参数的长度
def split_args_and_kwargs(*args, split_size = None, **kwargs):
    # 将所有参数值组合成一个元组
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)
    # 查找第一个是 torch.Tensor 类型的参数
    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
    # 断言第一个参数存在
    assert exists(first_tensor)

    # 获取第一个参数的长度作为批量大小
    batch_size = len(first_tensor)
    # 如果未指定拆分大小，则默认为批量大小
    split_size = default(split_size, batch_size)
    # 计算拆分后的块数
    num_chunks = ceil(batch_size / split_size)

    # 计算关键字参数的长度和键名
    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    # 计算关键字参数在拆分后的参数中的索引位置
    split_kwargs_index = len_all_args - dict_len

    # 对所有参数进行拆分，如果参数是 torch.Tensor 或可迭代对象，则按拆分大小进行拆分，否则复制参数值
    split_all_args = [split(arg, split_size = split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else ((arg,) * num_chunks) for arg in all_args]
    # 计算每个块的大小
    chunk_sizes = tuple(map(len, split_all_args[0]))

    # 遍历每个块，将参数和关键字参数拆分成块，并生成块大小的比例和拆分后的参数
    for (chunk_size, *chunked_all_args) in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)

# 扩散先验训练器

# 将函数分块处理
def prior_sample_in_chunks(fn):
    @wraps(fn)
    def inner(self, *args, max_batch_size = None, **kwargs):
        # 如果未指定最大批量大小，则直接调用函数
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)

        # 拆分参数并调用函数，将结果拼接在一起
        outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs)]
        return torch.cat(outputs, dim = 0)
    return inner

# 扩散先验训练器类
class DiffusionPriorTrainer(nn.Module):
    def __init__(
        self,
        diffusion_prior,
        accelerator = None,
        use_ema = True,
        lr = 3e-4,
        wd = 1e-2,
        eps = 1e-6,
        max_grad_norm = None,
        group_wd_params = True,
        warmup_steps = None,
        cosine_decay_max_steps = None,
        **kwargs
    # 初始化函数，设置一些成员变量和参数
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 断言确保传入的参数是 DiffusionPrior 类型的对象
        assert isinstance(diffusion_prior, DiffusionPrior)

        # 将参数按照前缀 'ema_' 分组并去除前缀，返回未分组的参数和 ema 参数
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)
        # 将参数按照前缀 'accelerator_' 分组并去除前缀，返回未分组的参数和 accelerator 参数
        accelerator_kwargs, kwargs = groupby_prefix_and_trim('accelerator_', kwargs)

        # 如果 accelerator 不存在，则根据参数创建一个 Accelerator 对象
        if not exists(accelerator):
            accelerator = Accelerator(**accelerator_kwargs)

        # 设置一些有用的成员变量

        self.accelerator = accelerator
        self.text_conditioned = diffusion_prior.condition_on_text_encodings

        # 设置设备

        self.device = accelerator.device
        diffusion_prior.to(self.device)

        # 保存模型

        self.diffusion_prior = diffusion_prior

        # 混合精度检查

        if (
            exists(self.accelerator) 
            and self.accelerator.distributed_type == DistributedType.DEEPSPEED 
            and self.diffusion_prior.clip is not None
            ):
            # 确保 clip 使用正确的精度，否则 deepspeed 会报错
            cast_type_map = {
                "fp16": torch.half,
                "bf16": torch.bfloat16,
                "no": torch.float
            }
            precision_type = cast_type_map[accelerator.mixed_precision]
            assert precision_type == torch.float, "DeepSpeed currently only supports float32 precision when using on the fly embedding generation from clip"
            self.diffusion_prior.clip.to(precision_type)

        # 优化器设置

        self.optim_kwargs = dict(lr=lr, wd=wd, eps=eps, group_wd_params=group_wd_params)

        # 根据参数创建优化器
        self.optimizer = get_optimizer(
            self.diffusion_prior.parameters(),
            **self.optim_kwargs,
            **kwargs
        )

        # 如果存在 cosine_decay_max_steps，则使用 CosineAnnealingLR 调度器，否则使用 LambdaLR 调度器
        if exists(cosine_decay_max_steps):
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max = cosine_decay_max_steps)
        else:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda = lambda _: 1.0)
        
        # 如果存在 warmup_steps，则使用 LinearWarmup 调度器
        self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period = warmup_steps) if exists(warmup_steps) else None

        # 如果使用 HFA，则分发模型
        self.diffusion_prior, self.optimizer, self.scheduler = self.accelerator.prepare(self.diffusion_prior, self.optimizer, self.scheduler)

        # 指数移动平均设置

        self.use_ema = use_ema

        if self.use_ema:
            self.ema_diffusion_prior = EMA(self.accelerator.unwrap_model(self.diffusion_prior), **ema_kwargs)

        # 如果需要梯度裁剪

        self.max_grad_norm = max_grad_norm

        # 内部跟踪步数

        self.register_buffer('step', torch.tensor([0], device = self.device))

    # 实用函数

    def save(self, path, overwrite = True, **kwargs):

        # 只在主进程上保存
        if self.accelerator.is_main_process:
            print(f"Saving checkpoint at step: {self.step.item()}")
            path = Path(path)
            assert not (path.exists() and not overwrite)
            path.parent.mkdir(parents = True, exist_ok = True)

            # FIXME: LambdaLR 由于 pickling 问题无法保存
            save_obj = dict(
                optimizer = self.optimizer.state_dict(),
                scheduler = self.scheduler.state_dict(),
                warmup_scheduler = self.warmup_scheduler,
                model = self.accelerator.unwrap_model(self.diffusion_prior).state_dict(),
                version = version.parse(__version__),
                step = self.step,
                **kwargs
            )

            # 如果使用指数移动平均，则保存相关参数
            if self.use_ema:
                save_obj = {
                    **save_obj,
                    'ema': self.ema_diffusion_prior.state_dict(),
                    'ema_model': self.ema_diffusion_prior.ema_model.state_dict() # 为了方便只保存 ema 模型
                }

            # 保存模型
            torch.save(save_obj, str(path))
    def load(self, path_or_state, overwrite_lr = True, strict = True):
        """
        Load a checkpoint of a diffusion prior trainer.

        Will load the entire trainer, including the optimizer and EMA.

        Params:
            - path_or_state (str | torch): a path to the DiffusionPriorTrainer checkpoint file
            - overwrite_lr (bool): wether or not to overwrite the stored LR with the LR specified in the new trainer
            - strict (bool): kwarg for `torch.nn.Module.load_state_dict`, will force an exact checkpoint match

        Returns:
            loaded_obj (dict): The loaded checkpoint dictionary
        """

        # all processes need to load checkpoint. no restriction here
        if isinstance(path_or_state, str):
            path = Path(path_or_state)
            assert path.exists()
            loaded_obj = torch.load(str(path), map_location=self.device)

        elif isinstance(path_or_state, dict):
            loaded_obj = path_or_state

        if version.parse(__version__) != loaded_obj['version']:
            print(f'loading saved diffusion prior at version {loaded_obj["version"]} but current package version is at {__version__}')

        # unwrap the model when loading from checkpoint
        self.accelerator.unwrap_model(self.diffusion_prior).load_state_dict(loaded_obj['model'], strict = strict)
        self.step.copy_(torch.ones_like(self.step, device=self.device) * loaded_obj['step'].to(self.device))

        self.optimizer.load_state_dict(loaded_obj['optimizer'])
        self.scheduler.load_state_dict(loaded_obj['scheduler'])

        # set warmupstep
        if exists(self.warmup_scheduler):
            self.warmup_scheduler.last_step = self.step.item()

        # ensure new lr is used if different from old one
        if overwrite_lr:
            new_lr = self.optim_kwargs["lr"]

            for group in self.optimizer.param_groups:
                group["lr"] = new_lr if group["lr"] > 0.0 else 0.0

        if self.use_ema:
            assert 'ema' in loaded_obj
            self.ema_diffusion_prior.load_state_dict(loaded_obj['ema'], strict = strict)
            # below might not be necessary, but I had a suspicion that this wasn't being loaded correctly
            self.ema_diffusion_prior.ema_model.load_state_dict(loaded_obj["ema_model"])

        return loaded_obj

    # model functionality

    def update(self):

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.diffusion_prior.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        # accelerator will ocassionally skip optimizer steps in a "dynamic loss scaling strategy"
        if not self.accelerator.optimizer_step_was_skipped:
            sched_context = self.warmup_scheduler.dampening if exists(self.warmup_scheduler) else nullcontext
            with sched_context():
                self.scheduler.step()

        if self.use_ema:
            self.ema_diffusion_prior.update()

        self.step += 1

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def p_sample_loop(self, *args, **kwargs):
        model = self.ema_diffusion_prior.ema_model if self.use_ema else self.diffusion_prior
        return model.p_sample_loop(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def sample(self, *args, **kwargs):
        model = self.ema_diffusion_prior.ema_model if self.use_ema else self.diffusion_prior
        return model.sample(*args, **kwargs)

    @torch.no_grad()
    def sample_batch_size(self, *args, **kwargs):
        model = self.ema_diffusion_prior.ema_model if self.use_ema else self.diffusion_prior
        return model.sample_batch_size(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    # 调用加速器对象的unwrap_model方法，将扩散先验解包后调用clip对象的embed_text方法，返回结果
    def embed_text(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.diffusion_prior).clip.embed_text(*args, **kwargs)

    # 使用装饰器将函数参数转换为torch张量
    def forward(
        self,
        *args,
        max_batch_size = None,
        **kwargs
    ):
        # 初始化总损失为0
        total_loss = 0.

        # 将参数和关键字参数按照指定大小分块，遍历每个分块
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            # 使用加速器对象的autocast方法进行自动混合精度计算
            with self.accelerator.autocast():
                # 调用扩散先验函数，传入分块参数和关键字参数，计算损失
                loss = self.diffusion_prior(*chunked_args, **chunked_kwargs)
                # 将损失乘以分块大小比例
                loss = loss * chunk_size_frac

            # 将损失值加到总损失中
            total_loss += loss.item()

            # 如果处于训练状态，使用加速器对象的backward方法进行反向传播
            if self.training:
                self.accelerator.backward(loss)

        # 返回总损失值
        return total_loss
# 解码器训练器

# 定义一个装饰器函数，用于将输入数据分成多个批次进行处理
def decoder_sample_in_chunks(fn):
    @wraps(fn)
    def inner(self, *args, max_batch_size = None, **kwargs):
        # 如果未指定最大批次大小，则直接调用原始函数
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)

        # 如果解码器是无条件的，则将批次大小分组成多个子批次进行处理
        if self.decoder.unconditional:
            batch_size = kwargs.get('batch_size')
            batch_sizes = num_to_groups(batch_size, max_batch_size)
            outputs = [fn(self, *args, **{**kwargs, 'batch_size': sub_batch_size}) for sub_batch_size in batch_sizes]
        else:
            # 如果解码器是有条件的，则将输入数据分成多个子块进行处理
            outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs)]

        # 将所有子批次或子块的输出拼接在一起
        return torch.cat(outputs, dim = 0)
    return inner

# 定义解码器训练器类
class DecoderTrainer(nn.Module):
    def __init__(
        self,
        decoder,
        accelerator = None,
        dataloaders = None,
        use_ema = True,
        lr = 1e-4,
        wd = 1e-2,
        eps = 1e-8,
        warmup_steps = None,
        cosine_decay_max_steps = None,
        max_grad_norm = 0.5,
        amp = False,
        group_wd_params = True,
        **kwargs
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言确保decoder是Decoder类型的实例
        assert isinstance(decoder, Decoder)
        # 将参数中以'ema_'开头的参数分组并去除前缀，返回两个字典
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)

        # 设置加速器，默认为Accelerator
        self.accelerator = default(accelerator, Accelerator)

        # 获取decoder中包含的unet数量
        self.num_unets = len(decoder.unets)

        # 设置是否使用指数移动平均
        self.use_ema = use_ema
        # 初始化ema_unets为一个空的ModuleList
        self.ema_unets = nn.ModuleList([])

        # 设置是否使用混合精度训练
        self.amp = amp

        # 可以对每个unet进行学习率、权重衰减等参数的细致定制

        # 将lr, wd, eps, warmup_steps, cosine_decay_max_steps映射为长度为num_unets的元组
        lr, wd, eps, warmup_steps, cosine_decay_max_steps = map(partial(cast_tuple, length = self.num_unets), (lr, wd, eps, warmup_steps, cosine_decay_max_steps))

        # 断言确保所有unet的学习率都不超过1e-2
        assert all([unet_lr <= 1e-2 for unet_lr in lr]), 'your learning rate is too high, recommend sticking with 1e-4, at most 5e-4'

        # 初始化优化器、调度器和预热调度器列表
        optimizers = []
        schedulers = []
        warmup_schedulers = []

        # 遍历decoder中的unets以及对应的lr, wd, eps, warmup_steps, cosine_decay_max_steps
        for unet, unet_lr, unet_wd, unet_eps, unet_warmup_steps, unet_cosine_decay_max_steps in zip(decoder.unets, lr, wd, eps, warmup_steps, cosine_decay_max_steps):
            # 如果unet是nn.Identity类型，则添加None到列表中
            if isinstance(unet, nn.Identity):
                optimizers.append(None)
                schedulers.append(None)
                warmup_schedulers.append(None)
            else:
                # 获取unet的参数，初始化优化器
                optimizer = get_optimizer(
                    unet.parameters(),
                    lr = unet_lr,
                    wd = unet_wd,
                    eps = unet_eps,
                    group_wd_params = group_wd_params,
                    **kwargs
                )

                optimizers.append(optimizer)

                # 初始化调度器和预热调度器
                if exists(unet_cosine_decay_max_steps):
                    scheduler = CosineAnnealingLR(optimizer, T_max = unet_cosine_decay_max_steps)
                else:
                    scheduler = LambdaLR(optimizer, lr_lambda = lambda step: 1.0)

                warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period = unet_warmup_steps) if exists(unet_warmup_steps) else None
                warmup_schedulers.append(warmup_scheduler)

                schedulers.append(scheduler)

            # 如果使用指数移动平均，则将unet添加到ema_unets中
            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

        # 如果需要梯度裁剪
        self.max_grad_norm = max_grad_norm

        # 注册一个名为steps的缓冲区，值为长度为num_unets的全零张量
        self.register_buffer('steps', torch.tensor([0] * self.num_unets))

        # 如果使用的分布式类型是DEEPSPEED且decoder中有clip参数
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED and decoder.clip is not None:
            # 确保clip使用正确的精度，否则会出错
            cast_type_map = {
                "fp16": torch.half,
                "bf16": torch.bfloat16,
                "no": torch.float
            }
            precision_type = cast_type_map[accelerator.mixed_precision]
            assert precision_type == torch.float, "DeepSpeed currently only supports float32 precision when using on the fly embedding generation from clip"
            clip = decoder.clip
            clip.to(precision_type)

        # 准备decoder和optimizers
        decoder, *optimizers = list(self.accelerator.prepare(decoder, *optimizers))

        self.decoder = decoder

        # 准备数据加载器

        train_loader = val_loader = None
        if exists(dataloaders):
            train_loader, val_loader = self.accelerator.prepare(dataloaders["train"], dataloaders["val"])

        self.train_loader = train_loader
        self.val_loader = val_loader

        # 存储优化器

        for opt_ind, optimizer in zip(range(len(optimizers)), optimizers):
            setattr(self, f'optim{opt_ind}', optimizer)

        # 存储调度器

        for sched_ind, scheduler in zip(range(len(schedulers)), schedulers):
            setattr(self, f'sched{sched_ind}', scheduler)

        # 存储预热调度器

        self.warmup_schedulers = warmup_schedulers

    # 验证并返回unet的编号
    def validate_and_return_unet_number(self, unet_number = None):
        # 如果只有一个unet，则默认unet_number为1
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        # 断言确保unet_number存在且在1到num_unets之间
        assert exists(unet_number) and 1 <= unet_number <= self.num_unets
        return unet_number
    # 返回指定 UNet 编号已经执行的步数
    def num_steps_taken(self, unet_number = None):
        # 验证并返回 UNet 编号
        unet_number = self.validate_and_return_unet_number(unet_number)
        # 返回指定 UNet 编号已经执行的步数
        return self.steps[unet_number - 1].item()

    # 保存模型状态到指定路径
    def save(self, path, overwrite = True, **kwargs):
        # 转换路径为 Path 对象
        path = Path(path)
        # 断言路径不存在或者可以覆盖
        assert not (path.exists() and not overwrite)
        # 创建父目录
        path.parent.mkdir(parents = True, exist_ok = True)

        # 构建保存对象字典
        save_obj = dict(
            model = self.accelerator.unwrap_model(self.decoder).state_dict(),
            version = __version__,
            steps = self.steps.cpu(),
            **kwargs
        )

        # 遍历 UNet 数量
        for ind in range(0, self.num_unets):
            optimizer_key = f'optim{ind}'
            scheduler_key = f'sched{ind}'

            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)

            optimizer_state_dict = optimizer.state_dict() if exists(optimizer) else None
            scheduler_state_dict = scheduler.state_dict() if exists(scheduler) else None

            # 更新保存对象字典
            save_obj = {**save_obj, optimizer_key: optimizer_state_dict, scheduler_key: scheduler_state_dict}

        # 如果使用 EMA，更新保存对象字典
        if self.use_ema:
            save_obj = {**save_obj, 'ema': self.ema_unets.state_dict()}

        # 保存模型状态到指定路径
        self.accelerator.save(save_obj, str(path))

    # 加载模型状态
    def load_state_dict(self, loaded_obj, only_model = False, strict = True):
        # 检查版本是否匹配
        if version.parse(__version__) != version.parse(loaded_obj['version']):
            self.accelerator.print(f'loading saved decoder at version {loaded_obj["version"]}, but current package version is {__version__}')

        # 加载模型状态
        self.accelerator.unwrap_model(self.decoder).load_state_dict(loaded_obj['model'], strict = strict)
        self.steps.copy_(loaded_obj['steps'])

        # 如果只加载模型状态，直接返回加载的对象
        if only_model:
            return loaded_obj

        # 遍历 UNet 数量，加载优化器和调度器状态
        for ind, last_step in zip(range(0, self.num_unets), self.steps.tolist()):

            optimizer_key = f'optim{ind}'
            optimizer = getattr(self, optimizer_key)

            scheduler_key = f'sched{ind}'
            scheduler = getattr(self, scheduler_key)

            warmup_scheduler = self.warmup_schedulers[ind]

            if exists(optimizer):
                optimizer.load_state_dict(loaded_obj[optimizer_key])

            if exists(scheduler):
                scheduler.load_state_dict(loaded_obj[scheduler_key])

            if exists(warmup_scheduler):
                warmup_scheduler.last_step = last_step

        # 如果使用 EMA，加载 EMA 模型状态
        if self.use_ema:
            assert 'ema' in loaded_obj
            self.ema_unets.load_state_dict(loaded_obj['ema'], strict = strict)

    # 加载模型状态
    def load(self, path, only_model = False, strict = True):
        # 转换路径为 Path 对象
        path = Path(path)
        # 断言路径存在
        assert path.exists()

        # 加载模型状态
        loaded_obj = torch.load(str(path), map_location = 'cpu')

        # 调用 load_state_dict 方法加载模型状态
        self.load_state_dict(loaded_obj, only_model = only_model, strict = strict)

        return loaded_obj

    # 返回 EMA 模型列表
    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    # 增加步数
    def increment_step(self, unet_number):
        # 断言 UNet 编号在有效范围内
        assert 1 <= unet_number <= self.num_unets

        # 转换 UNet 编号为张量
        unet_index_tensor = torch.tensor(unet_number - 1, device = self.steps.device)
        # 增加步数
        self.steps += F.one_hot(unet_index_tensor, num_classes = len(self.steps))
    # 更新模型参数
    def update(self, unet_number = None):
        # 验证并返回UNET编号
        unet_number = self.validate_and_return_unet_number(unet_number)
        index = unet_number - 1

        # 获取对应的优化器和调度器
        optimizer = getattr(self, f'optim{index}')
        scheduler = getattr(self, f'sched{index}')

        # 如果存在最大梯度范数，则对解码器参数进行梯度裁剪
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)  # Automatically unscales gradients

        # 执行优化器的步骤和梯度清零操作
        optimizer.step()
        optimizer.zero_grad()

        # 获取热身调度器，并根据是否存在进行相应操作
        warmup_scheduler = self.warmup_schedulers[index]
        scheduler_context = warmup_scheduler.dampening if exists(warmup_scheduler) else nullcontext

        # 在上下文中执行调度器的步骤
        with scheduler_context():
            scheduler.step()

        # 如果使用指数移动平均模型，则更新模型
        if self.use_ema:
            ema_unet = self.ema_unets[index]
            ema_unet.update()

        # 增加步数
        self.increment_step(unet_number)

    # 生成样本
    @torch.no_grad()
    @cast_torch_tensor
    @decoder_sample_in_chunks
    def sample(self, *args, **kwargs):
        distributed = self.accelerator.num_processes > 1
        base_decoder = self.accelerator.unwrap_model(self.decoder)

        was_training = base_decoder.training
        base_decoder.eval()

        # 根据是否使用EMA模型进行采样
        if kwargs.pop('use_non_ema', False) or not self.use_ema:
            out = base_decoder.sample(*args, **kwargs, distributed = distributed)
            base_decoder.train(was_training)
            return out

        # 切换为指数移动平均UNET进行采样
        trainable_unets = self.accelerator.unwrap_model(self.decoder).unets
        base_decoder.unets = self.unets                  # swap in exponential moving averaged unets for sampling

        output = base_decoder.sample(*args, **kwargs, distributed = distributed)

        base_decoder.unets = trainable_unets             # restore original training unets

        # 将EMA模型UNET转回原始设备
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        base_decoder.train(was_training)
        return output

    # 嵌入文本
    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_text(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.decoder).clip.embed_text(*args, **kwargs)

    # 嵌入图像
    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_image(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.decoder).clip.embed_image(*args, **kwargs)

    # 前向传播
    @cast_torch_tensor
    def forward(
        self,
        *args,
        unet_number = None,
        max_batch_size = None,
        return_lowres_cond_image=False,
        **kwargs
    ):
        # 验证并返回UNET编号
        unet_number = self.validate_and_return_unet_number(unet_number)

        total_loss = 0.
        cond_images = []
        # 将参数拆分为指定大小的块，并进行处理
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            with self.accelerator.autocast():
                # 调用解码器进行前向传播，计算损失
                loss_obj = self.decoder(*chunked_args, unet_number = unet_number, return_lowres_cond_image=return_lowres_cond_image, **chunked_kwargs)
                # 如果需要返回低分辨率条件图像，则提取出来
                if return_lowres_cond_image:
                    loss, cond_image = loss_obj
                else:
                    loss = loss_obj
                    cond_image = None
                loss = loss * chunk_size_frac
                if cond_image is not None:
                    cond_images.append(cond_image)

            total_loss += loss.item()

            # 如果处于训练状态，则进行反向传播
            if self.training:
                self.accelerator.backward(loss)

        # 如果需要返回低分辨率条件图像，则返回总损失和条件图像的张量
        if return_lowres_cond_image:
            return total_loss, torch.stack(cond_images)
        else:
            return total_loss
```