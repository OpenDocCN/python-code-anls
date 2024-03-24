# `.\lucidrains\ema-pytorch\ema_pytorch\post_hoc_ema.py`

```py
# 导入必要的模块
from pathlib import Path
from copy import deepcopy
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList

import numpy as np

from beartype import beartype
from beartype.typing import Set, Tuple, Optional

# 检查值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回数组的第一个元素
def first(arr):
    return arr[0]

# 获取模块的设备
def get_module_device(m: Module):
    return next(m.parameters()).device

# 在原地复制张量
def inplace_copy(tgt: Tensor, src: Tensor, *, auto_move_device = False):
    if auto_move_device:
        src = src.to(tgt.device)

    tgt.copy_(src)

# 在原地执行线性插值
def inplace_lerp(tgt: Tensor, src: Tensor, weight, *, auto_move_device = False):
    if auto_move_device:
        src = src.to(tgt.device)

    tgt.lerp_(src, weight)

# 将相对标准差转换为 gamma
def sigma_rel_to_gamma(sigma_rel):
    t = sigma_rel ** -2
    return np.roots([1, 7, 16 - t, 12 - t]).real.max().item()

# EMA 模块，使用论文 https://arxiv.org/abs/2312.02696 中的超参数
class KarrasEMA(Module):
    """
    exponential moving average module that uses hyperparameters from the paper https://arxiv.org/abs/2312.02696
    can either use gamma or sigma_rel from paper
    """

    @beartype
    def __init__(
        self,
        model: Module,
        sigma_rel: Optional[float] = None,
        gamma: Optional[float] = None,
        ema_model: Optional[Module] = None,           # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        update_every: int = 100,
        frozen: bool = False,
        param_or_buffer_names_no_ema: Set[str] = set(),
        ignore_names: Set[str] = set(),
        ignore_startswith_names: Set[str] = set(),
        allow_different_devices = False               # if the EMA model is on a different device (say CPU), automatically move the tensor
    ):
        super().__init__()

        assert exists(sigma_rel) ^ exists(gamma), 'either sigma_rel or gamma is given. gamma is derived from sigma_rel as in the paper, then beta is dervied from gamma'

        if exists(sigma_rel):
            gamma = sigma_rel_to_gamma(sigma_rel)

        self.gamma = gamma
        self.frozen = frozen

        self.online_model = [model]

        # ema model

        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = deepcopy(model)
            except Exception as e:
                print(f'Error: While trying to deepcopy model: {e}')
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        self.ema_model.requires_grad_(False)

        # parameter and buffer names

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if torch.is_floating_point(param) or torch.is_complex(param)}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if torch.is_floating_point(buffer) or torch.is_complex(buffer)}

        # tensor update functions

        self.inplace_copy = partial(inplace_copy, auto_move_device = allow_different_devices)
        self.inplace_lerp = partial(inplace_lerp, auto_move_device = allow_different_devices)

        # updating hyperparameters

        self.update_every = update_every

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        # whether to manage if EMA model is kept on a different device

        self.allow_different_devices = allow_different_devices

        # init and step states

        self.register_buffer('initted', torch.tensor(False))
        self.register_buffer('step', torch.tensor(0))

    @property
    def model(self):
        return first(self.online_model)
    
    @property
    # 计算 beta 值，用于更新移动平均模型
    def beta(self):
        return (1 - 1 / (self.step + 1)) ** (1 + self.gamma)

    # 调用 EMA 模型的 eval 方法
    def eval(self):
        return self.ema_model.eval()
    
    # 将 EMA 模型恢复到指定设备上
    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    # 获取模型的参数迭代器
    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    # 获取模型的缓冲区迭代器
    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    # 从原模型复制参数到 EMA 模型
    def copy_params_from_model_to_ema(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(ma_params.data, current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(ma_buffers.data, current_buffers.data)

    # 从 EMA 模型复制参数到原模型
    def copy_params_from_ema_to_model(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(current_params.data, ma_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(current_buffers.data, ma_buffers.data)

    # 更新步数并执行移动平均更新
    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True))

        self.update_moving_average(self.ema_model, self.model)

    # 迭代所有 EMA 模型的参数和缓冲区
    def iter_all_ema_params_and_buffers(self):
        for name, ma_params in self.get_params_iter(self.ema_model):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                continue

            yield ma_params

        for name, ma_buffer in self.get_buffers_iter(self.ema_model):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                continue

            yield ma_buffer

    # 更新移动平均模型
    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        if self.frozen:
            return

        copy, lerp = self.inplace_copy, self.inplace_lerp
        current_decay = self.beta

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                copy(ma_params.data, current_params.data)
                continue

            lerp(ma_params.data, current_params.data, 1. - current_decay)

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                copy(ma_buffer.data, current_buffer.data)
                continue

            lerp(ma_buffer.data, current_buffer.data, 1. - current_decay)
    # 定义一个特殊方法 __call__，使得对象可以像函数一样被调用
    def __call__(self, *args, **kwargs):
        # 调用 ema_model 对象，并传入参数
        return self.ema_model(*args, **kwargs)
# 后验EMA包装器

# 解决将所有检查点组合成新合成的EMA的权重，以达到所需的gamma
# 算法3从论文中复制，用torch重新实现

# 计算两个张量的点乘
def p_dot_p(t_a, gamma_a, t_b, gamma_b):
    t_ratio = t_a / t_b
    t_exp = torch.where(t_a < t_b , gamma_b , -gamma_a)
    t_max = torch.maximum(t_a , t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio ** t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den

# 解决权重
def solve_weights(t_i, gamma_i, t_r, gamma_r):
    rv = lambda x: x.double().reshape(-1, 1)
    cv = lambda x: x.double().reshape(1, -1)
    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
    b = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
    return torch.linalg.solve(A, b)

# 后验EMA类
class PostHocEMA(Module):

    # 初始化函数
    @beartype
    def __init__(
        self,
        model: Module,
        sigma_rels: Optional[Tuple[float, ...]] = None,
        gammas: Optional[Tuple[float, ...]] = None,
        checkpoint_every_num_steps: int = 1000,
        checkpoint_folder: str = './post-hoc-ema-checkpoints',
        **kwargs
    ):
        super().__init__()
        assert exists(sigma_rels) ^ exists(gammas)

        if exists(sigma_rels):
            gammas = tuple(map(sigma_rel_to_gamma, sigma_rels))

        assert len(gammas) > 1, 'at least 2 ema models with different gammas in order to synthesize new ema models of a different gamma'
        assert len(set(gammas)) == len(gammas), 'calculated gammas must be all unique'

        self.gammas = gammas
        self.num_ema_models = len(gammas)

        self._model = [model]
        self.ema_models = ModuleList([KarrasEMA(model, gamma = gamma, **kwargs) for gamma in gammas])

        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        self.checkpoint_every_num_steps = checkpoint_every_num_steps
        self.ema_kwargs = kwargs

    # 返回模型
    @property
    def model(self):
        return first(self._model)

    # 返回步数
    @property
    def step(self):
        return first(self.ema_models).step

    # 返回设备
    @property
    def device(self):
        return self.step.device

    # 从EMA复制参数到模型
    def copy_params_from_ema_to_model(self):
        for ema_model in self.ema_models:
            ema_model.copy_params_from_model_to_ema()

    # 更新EMA模型
    def update(self):
        for ema_model in self.ema_models:
            ema_model.update()

        if not (self.step.item() % self.checkpoint_every_num_steps):
            self.checkpoint()

    # 创建检查点
    def checkpoint(self):
        step = self.step.item()

        for ind, ema_model in enumerate(self.ema_models):
            filename = f'{ind}.{step}.pt'
            path = self.checkpoint_folder / filename

            pkg = deepcopy(ema_model).half().state_dict()
            torch.save(pkg, str(path))

    # 合成EMA模型
    @beartype
    def synthesize_ema_model(
        self,
        gamma: Optional[float] = None,
        sigma_rel: Optional[float] = None,
        step: Optional[int] = None,
    # 定义一个返回 KarrasEMA 对象的函数，参数包括 gamma 和 sigma_rel
    def __call__(self, gamma: Optional[float] = None, sigma_rel: Optional[float] = None) -> KarrasEMA:
        # 断言 gamma 和 sigma_rel 只能存在一个
        assert exists(gamma) ^ exists(sigma_rel)
        # 获取设备信息
        device = self.device

        # 如果存在 sigma_rel，则根据 sigma_rel 转换为 gamma
        if exists(sigma_rel):
            gamma = sigma_rel_to_gamma(sigma_rel)

        # 创建一个合成的 EMA 模型对象
        synthesized_ema_model = KarrasEMA(
            model = self.model,
            gamma = gamma,
            **self.ema_kwargs
        )

        synthesized_ema_model

        # 获取所有检查点

        gammas = []
        timesteps = []
        checkpoints = [*self.checkpoint_folder.glob('*.pt')]

        # 遍历检查点文件，获取 gamma 和 timestep
        for file in checkpoints:
            gamma_ind, timestep = map(int, file.stem.split('.'))
            gamma = self.gammas[gamma_ind]

            gammas.append(gamma)
            timesteps.append(timestep)

        # 设置步数为最大 timestep
        step = default(step, max(timesteps))
        # 断言步数小于等于最大 timestep
        assert step <= max(timesteps), f'you can only synthesize for a timestep that is less than the max timestep {max(timesteps)}'

        # 与算法 3 对齐

        gamma_i = Tensor(gammas, device = device)
        t_i = Tensor(timesteps, device = device)

        gamma_r = Tensor([gamma], device = device)
        t_r = Tensor([step], device = device)

        # 使用最小二乘法解出将所有检查点组合成合成检查点的权重

        weights = solve_weights(t_i, gamma_i, t_r, gamma_r)
        weights = weights.squeeze(-1)

        # 逐个使用权重将所有检查点相加到合成模型中

        tmp_ema_model = KarrasEMA(
            model = self.model,
            gamma = gamma,
            **self.ema_kwargs
        )

        for ind, (checkpoint, weight) in enumerate(zip(checkpoints, weights.tolist())):
            is_first = ind == 0

            # 将检查点加载到临时 EMA 模型中

            ckpt_state_dict = torch.load(str(checkpoint))
            tmp_ema_model.load_state_dict(ckpt_state_dict)

            # 将加权检查点添加到合成模型中

            for ckpt_tensor, synth_tensor in zip(tmp_ema_model.iter_all_ema_params_and_buffers(), synthesized_ema_model.iter_all_ema_params_and_buffers()):
                if is_first:
                    synth_tensor.zero_()

                synth_tensor.add_(ckpt_tensor * weight)

        # 返回合成模型

        return synthesized_ema_model

    # 调用函数，返回所有 EMA 模型的结果
    def __call__(self, *args, **kwargs):
        return tuple(ema_model(*args, **kwargs) for ema_model in self.ema_models)
```