# `.\lucidrains\ema-pytorch\ema_pytorch\ema_pytorch.py`

```py
# 导入深拷贝函数 deepcopy 和 partial 函数
from copy import deepcopy
from functools import partial

# 导入 torch 库
import torch
# 从 torch 库中导入 nn, Tensor 模块
from torch import nn, Tensor
# 从 torch.nn 模块中导入 Module 类
from torch.nn import Module

# 导入 beartype 库
from beartype import beartype
# 从 beartype.typing 模块中导入 Set, Optional 类型
from beartype.typing import Set, Optional

# 定义函数 exists，用于检查值是否存在
def exists(val):
    return val is not None

# 定义函数 get_module_device，用于获取模块的设备信息
def get_module_device(m: Module):
    return next(m.parameters()).device

# 定义函数 inplace_copy，用于原地复制张量数据
def inplace_copy(tgt: Tensor, src: Tensor, *, auto_move_device = False):
    if auto_move_device:
        src = src.to(tgt.device)

    tgt.copy_(src)

# 定义函数 inplace_lerp，用于原地线性插值
def inplace_lerp(tgt: Tensor, src: Tensor, weight, *, auto_move_device = False):
    if auto_move_device:
        src = src.to(tgt.device)

    tgt.lerp_(src, weight)

# 定义 EMA 类，实现模型的指数移动平均阴影
class EMA(Module):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 2/3.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    # 使用 beartype 装饰器，对初始化函数进行类型检查
    @beartype
    def __init__(
        self,
        model: Module,
        ema_model: Optional[Module] = None,           # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
        inv_gamma = 1.0,
        power = 2 / 3,
        min_value = 0.0,
        param_or_buffer_names_no_ema: Set[str] = set(),
        ignore_names: Set[str] = set(),
        ignore_startswith_names: Set[str] = set(),
        include_online_model = True,                  # set this to False if you do not wish for the online model to be saved along with the ema model (managed externally)
        allow_different_devices = False               # if the EMA model is on a different device (say CPU), automatically move the tensor
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化 beta 属性
        self.beta = beta

        # 判断是否冻结模型
        self.is_frozen = beta == 1.

        # 是否在模块树中包含在线模型，以便 state_dict 也保存它
        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model] # hack

        # EMA 模型
        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = deepcopy(model)
            except Exception as e:
                print(f'Error: While trying to deepcopy model: {e}')
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        self.ema_model.requires_grad_(False)

        # 参数和缓冲区的名称
        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if torch.is_floating_point(param) or torch.is_complex(param)}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if torch.is_floating_point(buffer) or torch.is_complex(buffer)}

        # 张量更新函数
        self.inplace_copy = partial(inplace_copy, auto_move_device = allow_different_devices)
        self.inplace_lerp = partial(inplace_lerp, auto_move_device = allow_different_devices)

        # 更新超参数
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        # 是否管理 EMA 模型是否保留在不同设备上
        self.allow_different_devices = allow_different_devices

        # 初始化和步骤状态
        self.register_buffer('initted', torch.tensor(False))
        self.register_buffer('step', torch.tensor(0))

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    def eval(self):
        return self.ema_model.eval()
    
    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(ma_params.data, current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(ma_buffers.data, current_buffers.data)

    def copy_params_from_ema_to_model(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(current_params.data, ma_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(current_buffers.data, ma_buffers.data)
    # 获取当前的衰减值
    def get_current_decay(self):
        # 计算当前的 epoch，确保不小于 0
        epoch = (self.step - self.update_after_step - 1).clamp(min=0.)
        # 根据公式计算衰减值
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        # 如果 epoch 小于等于 0，则返回 0
        if epoch.item() <= 0:
            return 0.

        # 返回计算得到的衰减值，确保在一定范围内
        return value.clamp(min=self.min_value, max=self.beta).item()

    # 更新操作
    def update(self):
        # 获取当前步数
        step = self.step.item()
        # 步数加一
        self.step += 1

        # 如果步数不是更新频率的倍数，则直接返回
        if (step % self.update_every) != 0:
            return

        # 如果步数小于等于更新之后的步数，则将模型参数拷贝到指数移动平均模型中
        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        # 如果模型还未初始化，则将模型参数拷贝到指数移动平均模型中，并标记为已初始化
        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True))

        # 更新指数移动平均模型
        self.update_moving_average(self.ema_model, self.model)

    # 更新指数移动平均模型
    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        # 如果模型被冻结，则直接返回
        if self.is_frozen:
            return

        # 获取拷贝和线性插值函数
        copy, lerp = self.inplace_copy, self.inplace_lerp
        # 获取当前的衰减值
        current_decay = self.get_current_decay()

        # 遍历当前模型和指数移动平均模型的参数
        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            # 如果参数名在忽略列表中，则跳过
            if name in self.ignore_names:
                continue

            # 如果参数名以忽略列表中的前缀开头，则跳过
            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            # 如果参数名在不进行指数移动平均的列表中，则直接拷贝参数值
            if name in self.param_or_buffer_names_no_ema:
                copy(ma_params.data, current_params.data)
                continue

            # 对参数进行线性插值
            lerp(ma_params.data, current_params.data, 1. - current_decay)

        # 遍历当前模型和指数移动平均模型的缓冲区
        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            # 如果缓冲区名在忽略列表中，则跳过
            if name in self.ignore_names:
                continue

            # 如果缓冲区名以忽略列表中的前缀开头，则跳过
            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            # 如果缓冲区名在不进行指数移动平均的列表中，则直接拷贝缓冲区值
            if name in self.param_or_buffer_names_no_ema:
                copy(ma_buffer.data, current_buffer.data)
                continue

            # 对缓冲区进行线性插值
            lerp(ma_buffer.data, current_buffer.data, 1. - current_decay)

    # 调用函数，返回指数移动平均模型的结果
    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
```