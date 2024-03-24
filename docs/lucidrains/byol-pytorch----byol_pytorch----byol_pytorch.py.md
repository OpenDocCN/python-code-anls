# `.\lucidrains\byol-pytorch\byol_pytorch\byol_pytorch.py`

```
# 导入必要的库
import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T

# 辅助函数

# 如果值为 None，则返回默认值
def default(val, def_val):
    return def_val if val is None else val

# 将张量展平为二维张量
def flatten(t):
    return t.reshape(t.shape[0], -1)

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

# 获取模块所在设备
def get_module_device(module):
    return next(module.parameters()).device

# 设置模型参数是否需要梯度
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# 根据是否分布式训练返回不同的批归一化层
def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

# 损失函数

# 计算余弦相似度损失
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# 数据增强工具

# 随机应用函数 fn 进行数据增强
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

    # 更新移动平均值
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# 更新移动平均值
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# 用于投影器和预测器的 MLP 类

# 创建多层感知机
def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

# 创建 SimSiam 模型的多层感知机
def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False)
    )

# 用于基础神经网络的包装类
# 管理隐藏层输出并将其传递到投影器和预测器网络中

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False, sync_batchnorm = None):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

    # 查找指定层
    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None
    # 在 forward 方法中的 hook 函数，用于获取隐藏层输出并保存到 self.hidden 字典中
    def _hook(self, _, input, output):
        # 获取输入数据的设备信息
        device = input[0].device
        # 将输出数据扁平化后保存到 self.hidden 字典中
        self.hidden[device] = flatten(output)

    # 注册 hook 函数到指定的隐藏层
    def _register_hook(self):
        # 查找指定的隐藏层
        layer = self._find_layer()
        # 断言找到隐藏层
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        # 注册 forward hook 函数到隐藏层
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    # 获取投影器对象
    @singleton('projector')
    def _get_projector(self, hidden):
        # 获取隐藏层的维度信息
        _, dim = hidden.shape
        # 根据是否使用 SimSiamMLP 创建 MLP 或 SimSiamMLP 对象
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        # 创建投影器对象
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, sync_batchnorm = self.sync_batchnorm)
        # 将投影器对象移动到隐藏层所在设备
        return projector.to(hidden)

    # 获取输入数据的表示
    def get_representation(self, x):
        # 如果指定的隐藏层为最后一层，则直接返回网络输出
        if self.layer == -1:
            return self.net(x)

        # 如果 hook 函数未注册，则注册 hook 函数
        if not self.hook_registered:
            self._register_hook()

        # 清空 self.hidden 字典
        self.hidden.clear()
        # 前向传播输入数据，并获取隐藏层输出
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        # 断言隐藏层输出不为空
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    # 前向传播方法
    def forward(self, x, return_projection = True):
        # 获取输入数据的表示
        representation = self.get_representation(x)

        # 如果不需要返回投影结果，则直接返回表示
        if not return_projection:
            return representation

        # 获取投影器对象
        projector = self._get_projector(representation)
        # 对表示进行投影
        projection = projector(representation)
        return projection, representation
# 主类 BYOL，继承自 nn.Module
class BYOL(nn.Module):
    # 初始化函数
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        sync_batchnorm = None
    ):
        super().__init__()
        self.net = net

        # 默认的 SimCLR 数据增强
        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        # 设置数据增强函数
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        # 在线编码器
        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer = hidden_layer,
            use_simsiam_mlp = not use_momentum,
            sync_batchnorm = sync_batchnorm
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        # 在线预测器
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # 获取网络设备并将 wrapper 设置为相同设备
        device = get_module_device(net)
        self.to(device)

        # 发送一个模拟图像张量以实例化单例参数
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    # 获取目标编码器的单例函数
    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    # 重置移动平均
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    # 更新移动平均
    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    # 前向传播函数
    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)

        # 获取两个增强后的图像
        image_one, image_two = self.augment1(x), self.augment2(x)

        # 拼接两个图像
        images = torch.cat((image_one, image_two), dim = 0)

        # 获取在线编码器的投影和预测
        online_projections, _ = self.online_encoder(images)
        online_predictions = self.online_predictor(online_projections)

        online_pred_one, online_pred_two = online_predictions.chunk(2, dim = 0)

        with torch.no_grad():
            # 获取目标编码器
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder

            target_projections, _ = target_encoder(images)
            target_projections = target_projections.detach()

            target_proj_one, target_proj_two = target_projections.chunk(2, dim = 0)

        # 计算损失
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
```