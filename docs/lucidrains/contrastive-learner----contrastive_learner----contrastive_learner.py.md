# `.\lucidrains\contrastive-learner\contrastive_learner\contrastive_learner.py`

```py
# 导入必要的库
import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import resnet50
from kornia import augmentation as augs
from kornia import filters

# 辅助函数

# 定义一个返回输入的函数
def identity(x): return x

# 如果输入值为None，则返回默认值
def default(val, def_val):
    return def_val if val is None else val

# 将输入张量展平
def flatten(t):
    return t.reshape(t.shape[0], -1)

# 安全地在指定维度上连接张量
def safe_concat(arr, el, dim=0):
    if arr is None:
        return el
    return torch.cat((arr, el), dim=dim)

# 单例装饰器，用于缓存结果
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

# 损失函数

# 对比损失函数
def contrastive_loss(queries, keys, temperature = 0.1):
    b, device = queries.shape[0], queries.device
    logits = queries @ keys.t()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits /= temperature
    return F.cross_entropy(logits, torch.arange(b, device=device))

# NT-Xent损失函数
def nt_xent_loss(queries, keys, temperature = 0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)
    logits /= temperature

    labels = torch.cat(((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)), dim=0)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    loss /= n
    return loss

# 数据增强工具

# 随机应用数据增强函数
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# 指数移动平均

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# 更新移动平均值
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# 隐藏层提取器类

class OutputHiddenLayer(nn.Module):
    def __init__(self, net, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self._register_hook()

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _register_hook(self):
        def hook(_, __, output):
            self.hidden = output

        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(hook)

    def forward(self, x):
        if self.layer == -1:
            return self.net(x)

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

# 主类

class ContrastiveLearner(nn.Module):
    # 初始化函数，设置模型参数和属性
    def __init__(self, net, image_size, hidden_layer = -2, project_hidden = True, project_dim=128, augment_both=True, use_nt_xent_loss=False, augment_fn = None, use_bilinear = False, use_momentum = False, momentum_value = 0.999, key_encoder = None, temperature = 0.1):
        # 调用父类的初始化函数
        super().__init__()
        # 创建输出隐藏层对象
        self.net = OutputHiddenLayer(net, layer=hidden_layer)

        # 默认数据增强操作
        DEFAULT_AUG = nn.Sequential(
            RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            augs.RandomGrayscale(p=0.2),
            augs.RandomHorizontalFlip(),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            augs.RandomResizedCrop((image_size, image_size))
        )

        # 设置数据增强操作
        self.augment = default(augment_fn, DEFAULT_AUG)

        # 是否同时对两个数据进行增强
        self.augment_both = augment_both

        # 设置温度参数和是否使用 NT-Xent 损失函数
        self.temperature = temperature
        self.use_nt_xent_loss = use_nt_xent_loss

        # 是否对隐藏层进行投影
        self.project_hidden = project_hidden
        self.projection = None
        self.project_dim = project_dim

        # 是否使用双线性插值
        self.use_bilinear = use_bilinear
        self.bilinear_w = None

        # 是否使用动量方法
        self.use_momentum = use_momentum
        self.ema_updater = EMA(momentum_value)
        self.key_encoder = key_encoder

        # 用于累积查询和键
        self.queries = None
        self.keys = None

        # 发送一个模拟图像张量以实例化参数
        self.forward(torch.randn(1, 3, image_size, image_size))

    # 获取键编码器对象
    @singleton('key_encoder')
    def _get_key_encoder(self):
        key_encoder = copy.deepcopy(self.net)
        key_encoder._register_hook()
        return key_encoder

    # 获取双线性插值矩阵
    @singleton('bilinear_w')
    def _get_bilinear(self, hidden):
        _, dim = hidden.shape
        return nn.Parameter(torch.eye(dim, device=device, dtype=dtype)).to(hidden)

    # 获取投影函数
    @singleton('projection')
    def _get_projection_fn(self, hidden):
        _, dim = hidden.shape

        return nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim, self.project_dim, bias = False)
        ).to(hidden)

    # 重置移动平均值
    def reset_moving_average(self):
        assert self.use_momentum, 'must be using momentum method for key encoder'
        del self.key_encoder
        self.key_encoder = None

    # 更新移动平均值
    def update_moving_average(self):
        assert self.key_encoder is not None, 'key encoder has not been created yet'
        self.key_encoder = update_moving_average(self.ema_updater, self.key_encoder, self.net)

    # 计算损失函数
    def calculate_loss(self):
        assert self.queries is not None and self.keys is not None, 'no queries or keys accumulated'
        loss_fn = nt_xent_loss if self.use_nt_xent_loss else contrastive_loss
        loss = loss_fn(self.queries, self.keys, temperature = self.temperature)
        self.queries = self.keys = None
        return loss

    # 前向传播函数
    def forward(self, x, accumulate = False):
        # 获取输入张量的形状和设备信息
        b, c, h, w, device = *x.shape, x.device
        transform_fn = self.augment if self.augment_both else noop

        # 获取查询编码器
        query_encoder = self.net
        queries = query_encoder(transform_fn(x))

        # 获取键编码器
        key_encoder = self.net if not self.use_momentum else self._get_key_encoder()
        keys = key_encoder(self.augment(x))

        if self.use_momentum:
            keys = keys.detach()

        queries, keys = map(flatten, (queries, keys))

        if self.use_bilinear:
            W = self._get_bilinear(keys)
            keys = (W @ keys.t()).t()

        project_fn = self._get_projection_fn(queries) if self.project_hidden else identity
        queries, keys = map(project_fn, (queries, keys))

        self.queries = safe_concat(self.queries, queries)
        self.keys = safe_concat(self.keys, keys)

        return self.calculate_loss() if not accumulate else None
```