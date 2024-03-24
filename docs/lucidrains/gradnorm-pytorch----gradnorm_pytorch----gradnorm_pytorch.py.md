# `.\lucidrains\gradnorm-pytorch\gradnorm_pytorch\gradnorm_pytorch.py`

```
# 导入必要的库
from functools import cache, partial
import torch
import torch.distributed as dist
from torch.autograd import grad
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList, Parameter
from einops import rearrange, repeat
from accelerate import Accelerator
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Union, List, Dict, Tuple, NamedTuple

# 辅助函数

# 检查变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 张量辅助函数

# 计算张量的 L1 范数
def l1norm(t, dim = -1):
    return F.normalize(t, p = 1, dim = dim)

# 分布式计算辅助函数

# 判断是否处于分布式环境
@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

# 如果处于分布式环境，则计算张量的均值
def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# 主类

class GradNormLossWeighter(Module):
    @beartype
    def __init__(
        self,
        *,
        num_losses: Optional[int] = None,
        loss_weights: Optional[Union[
            List[float],
            Tensor
        ]] = None,
        loss_names: Optional[Tuple[str, ...]] = None,
        learning_rate = 1e-4,
        restoring_force_alpha = 0.,
        grad_norm_parameters: Optional[Parameter] = None,
        accelerator: Optional[Accelerator] = None,
        frozen = False,
        initial_losses_decay = 1.,
        update_after_step = 0.,
        update_every = 1.
    ):
        super().__init__()
        assert exists(num_losses) or exists(loss_weights)

        if exists(loss_weights):
            if isinstance(loss_weights, list):
                loss_weights = torch.tensor(loss_weights)

            num_losses = default(num_losses, loss_weights.numel())
        else:
            loss_weights = torch.ones((num_losses,), dtype = torch.float32)

        assert len(loss_weights) == num_losses
        assert num_losses > 1, 'only makes sense if you have multiple losses'
        assert loss_weights.ndim == 1, 'loss weights must be 1 dimensional'

        self.accelerator = accelerator
        self.num_losses = num_losses
        self.frozen = frozen

        self.loss_names = loss_names
        assert not exists(loss_names) or len(loss_names) == num_losses

        assert restoring_force_alpha >= 0.

        self.alpha = restoring_force_alpha
        self.has_restoring_force = self.alpha > 0

        self._grad_norm_parameters = [grad_norm_parameters] # hack

        # 损失权重，可以是学习得到的或静态的

        self.register_buffer('loss_weights', loss_weights)

        self.learning_rate = learning_rate

        # 初始损失
        # 如果初始损失衰减设置为小于1，则会对初始损失进行 EMA 平滑处理

        assert 0 <= initial_losses_decay <= 1.
        self.initial_losses_decay = initial_losses_decay

        self.register_buffer('initial_losses', torch.zeros(num_losses))

        # 用于在最后重新归一化损失权重

        self.register_buffer('loss_weights_sum', self.loss_weights.sum())

        # 用于梯度累积

        self.register_buffer('loss_weights_grad', torch.zeros_like(loss_weights), persistent = False)

        # 步数，用于可能的调度等

        self.register_buffer('step', torch.tensor(0.))

        # 可以较少频繁更新，以节省计算资源

        self.update_after_step = update_after_step
        self.update_every = update_every

        self.register_buffer('initted', torch.tensor(False))

    @property
    def grad_norm_parameters(self):
        return self._grad_norm_parameters[0]

    def backward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @beartype
    # 定义一个 forward 方法，用于前向传播
    def forward(
        self,
        losses: Union[
            Dict[str, Tensor],    # 损失值可以是字典类型，键为字符串，值为张量
            List[Tensor],         # 损失值可以是张量列表
            Tuple[Tensor],        # 损失值可以是元组中的张量
            Tensor                # 损失值可以是单个张量
        ],
        activations: Optional[Tensor] = None,     # 激活值，默认为 None，在论文中，他们使用了从骨干层次的倒数第二个参数的梯度范数。但这也可以是激活值（例如，共享的图像被馈送到多个鉴别器）
        freeze = False,                           # 可以选择在前向传播时冻结可学习的损失权重
        scale = 1.,                               # 缩放因子，默认为 1
        grad_step = True,                         # 是否进行梯度步骤，默认为 True
        **backward_kwargs                          # 其他后向传播参数
```