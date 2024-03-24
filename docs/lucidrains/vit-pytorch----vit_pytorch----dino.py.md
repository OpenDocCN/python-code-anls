# `.\lucidrains\vit-pytorch\vit_pytorch\dino.py`

```
# 导入所需的库
import copy
import random
from functools import wraps, partial

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 如果值存在，则返回该值，否则返回默认值
def default(val, default):
    return val if exists(val) else default

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

# 损失函数（论文中的算法1）

def loss_fn(
    teacher_logits,
    student_logits,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim = -1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim = -1)
    return - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()

# 数据增强工具类

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

# MLP类用于投影器和预测器

class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        norm = x.norm(dim = 1, keepdim = True).clamp(min = eps)
        return x / norm

class MLP(nn.Module):
    def __init__(self, dim, dim_out, num_layers, hidden_size = 256):
        super().__init__()

        layers = []
        dims = (dim, *((hidden_size,) * (num_layers - 1)))

        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == (len(dims) - 1)

            layers.extend([
                nn.Linear(layer_dim_in, layer_dim_out),
                nn.GELU() if not is_last else nn.Identity()
            ])

        self.net = nn.Sequential(
            *layers,
            L2Norm(),
            nn.Linear(hidden_size, dim_out)
        )

    def forward(self, x):
        return self.net(x)

# 用于基础神经网络的包装类
# 将管理隐藏层输出的拦截并将其传递到投影器和预测器网络中

class NetWrapper(nn.Module):
    def __init__(self, net, output_dim, projection_hidden_size, projection_num_layers, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_hidden_size = projection_hidden_size
        self.projection_num_layers = projection_num_layers
        self.output_dim = output_dim

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None
    # 定义一个私有方法，用于在 forward hook 中保存隐藏层的输出
    def _hook(self, _, input, output):
        # 获取输入数据的设备信息
        device = input[0].device
        # 将隐藏层的输出展平并保存到字典中
        self.hidden[device] = output.flatten(1)

    # 注册 forward hook，用于捕获隐藏层的输出
    def _register_hook(self):
        # 查找指定的隐藏层
        layer = self._find_layer()
        # 断言找到了隐藏层
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        # 注册 forward hook
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    # 获取投影器，用于将隐藏层的输出投影到指定维度
    @singleton('projector')
    def _get_projector(self, hidden):
        # 获取隐藏层输出的维度
        _, dim = hidden.shape
        # 创建 MLP 投影器
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector.to(hidden)

    # 获取输入数据的隐藏层输出
    def get_embedding(self, x):
        # 如果隐藏层为最后一层，则直接返回网络的输出
        if self.layer == -1:
            return self.net(x)

        # 如果 hook 没有注册，则注册 hook
        if not self.hook_registered:
            self._register_hook()

        # 清空隐藏层输出字典
        self.hidden.clear()
        # 前向传播获取隐藏层输出
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        # 断言隐藏层输出不为空
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    # 网络的前向传播，可选择是否返回投影后的输出
    def forward(self, x, return_projection = True):
        # 获取输入数据的隐藏层输出
        embed = self.get_embedding(x)
        # 如果不需要返回投影后的输出，则直接返回隐藏层输出
        if not return_projection:
            return embed

        # 获取投影器并对隐藏层输出进行投影
        projector = self._get_projector(embed)
        return projector(embed), embed
# 主类定义
class Dino(nn.Module):
    # 初始化函数
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_hidden_size = 256,
        num_classes_K = 65336,
        projection_layers = 4,
        student_temp = 0.9,
        teacher_temp = 0.04,
        local_upper_crop_scale = 0.4,
        global_lower_crop_scale = 0.5,
        moving_average_decay = 0.9,
        center_moving_average_decay = 0.9,
        augment_fn = None,
        augment_fn2 = None
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置网络
        self.net = net

        # 默认的 BYOL 数据增强
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
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        # 设置数据增强函数
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, DEFAULT_AUG)

        # 设置局部和全局裁剪
        self.local_crop = T.RandomResizedCrop((image_size, image_size), scale = (0.05, local_upper_crop_scale))
        self.global_crop = T.RandomResizedCrop((image_size, image_size), scale = (global_lower_crop_scale, 1.))

        # 设置学生编码器
        self.student_encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer = hidden_layer)

        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)

        # 注册缓冲区
        self.register_buffer('teacher_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_centers',  torch.zeros(1, num_classes_K))

        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # 获取网络设备并将包装器设置为相同设备
        device = get_module_device(net)
        self.to(device)

        # 发送一个模拟图像张量以实例化单例参数
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    # 获取教师编码器的单例函数
    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    # 重置移动平均值
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    # 更新移动平均值
    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.last_teacher_centers)
        self.teacher_centers.copy_(new_teacher_centers)

    # 前向传播函数
    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True,
        student_temp = None,
        teacher_temp = None
        ):
        # 如果需要返回嵌入向量，则调用学生编码器并返回结果
        if return_embedding:
            return self.student_encoder(x, return_projection = return_projection)

        # 对输入数据进行两种不同的数据增强
        image_one, image_two = self.augment1(x), self.augment2(x)

        # 对增强后的图像进行局部裁剪
        local_image_one, local_image_two   = self.local_crop(image_one),  self.local_crop(image_two)
        # 对增强后的图像进行全局裁剪
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        # 使用学生编码器对局部裁剪后的图像进行编码
        student_proj_one, _ = self.student_encoder(local_image_one)
        student_proj_two, _ = self.student_encoder(local_image_two)

        # 使用torch.no_grad()上下文管理器，获取教师编码器并对全局裁剪后的图像进行编码
        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_proj_one, _ = teacher_encoder(global_image_one)
            teacher_proj_two, _ = teacher_encoder(global_image_two)

        # 部分应用损失函数，设置学生温度、教师温度和教师中心
        loss_fn_ = partial(
            loss_fn,
            student_temp = default(student_temp, self.student_temp),
            teacher_temp = default(teacher_temp, self.teacher_temp),
            centers = self.teacher_centers
        )

        # 计算教师投影的平均值，并将其复制到最后的教师中心
        teacher_logits_avg = torch.cat((teacher_proj_one, teacher_proj_two)).mean(dim = 0)
        self.last_teacher_centers.copy_(teacher_logits_avg)

        # 计算损失，取两个损失函数的平均值
        loss = (loss_fn_(teacher_proj_one, student_proj_two) + loss_fn_(teacher_proj_two, student_proj_one)) / 2
        return loss
```