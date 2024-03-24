# `.\lucidrains\pixel-level-contrastive-learning\pixel_level_contrastive_learning\pixel_level_contrastive_learning.py`

```
# 导入数学库
import math
# 导入复制库
import copy
# 导入随机库
import random
# 导入wraps和partial函数
from functools import wraps, partial
# 从数学库中导入floor函数
from math import floor

# 导入torch库
import torch
# 从torch中导入nn和einsum模块
from torch import nn, einsum
# 从torch.nn中导入functional模块
import torch.nn.functional as F

# 从kornia库中导入augmentation、filters和color模块
from kornia import augmentation as augs
from kornia import filters, color

# 从einops库中导入rearrange函数
from einops import rearrange

# 辅助函数

# 返回输入的张量
def identity(t):
    return t

# 如果输入值为None，则返回默认值
def default(val, def_val):
    return def_val if val is None else val

# 根据概率返回True或False
def rand_true(prob):
    return random.random() < prob

# 缓存装饰器，用于缓存计算结果
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

# 随机生成cutout的坐标和比例
def cutout_coordinates(image, ratio_range = (0.6, 0.8)):
    _, _, orig_h, orig_w = image.shape

    ratio_lo, ratio_hi = ratio_range
    random_ratio = ratio_lo + random.random() * (ratio_hi - ratio_lo)
    w, h = floor(random_ratio * orig_w), floor(random_ratio * orig_h)
    coor_x = floor((orig_w - w) * random.random())
    coor_y = floor((orig_h - h) * random.random())
    return ((coor_y, coor_y + h), (coor_x, coor_x + w)), random_ratio

# 对cutout后的图像进行插值缩放
def cutout_and_resize(image, coordinates, output_size = None, mode = 'nearest'):
    shape = image.shape
    output_size = default(output_size, shape[2:])
    (y0, y1), (x0, x1) = coordinates
    cutout_image = image[:, :, y0:y1, x0:x1]
    return F.interpolate(cutout_image, size = output_size, mode = mode)

# 数据增强工具

# 随机应用函数
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

# 损失函数

# 计算损失函数
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# 类

# 多层感知器
class MLP(nn.Module):
    def __init__(self, chan, chan_out = 256, inner_dim = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chan, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, chan_out)
        )

    def forward(self, x):
        return self.net(x)

# 卷积多层感知器
class ConvMLP(nn.Module):
    def __init__(self, chan, chan_out = 256, inner_dim = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, chan_out, 1)
        )

    def forward(self, x):
        return self.net(x)

# 空间金字塔池化
class PPM(nn.Module):
    # 初始化函数，设置网络的参数
    def __init__(
        self,
        *,
        chan,
        num_layers = 1,
        gamma = 2):
        # 调用父类的初始化函数
        super().__init__()
        # 设置网络的 gamma 参数
        self.gamma = gamma

        # 根据 num_layers 的值选择不同的转换网络
        if num_layers == 0:
            # 如果 num_layers 为 0，则使用恒等映射
            self.transform_net = nn.Identity()
        elif num_layers == 1:
            # 如果 num_layers 为 1，则使用一个卷积层
            self.transform_net = nn.Conv2d(chan, chan, 1)
        elif num_layers == 2:
            # 如果 num_layers 为 2，则使用两个卷积层和批归一化层
            self.transform_net = nn.Sequential(
                nn.Conv2d(chan, chan, 1),
                nn.BatchNorm2d(chan),
                nn.ReLU(),
                nn.Conv2d(chan, chan, 1)
            )
        else:
            # 如果 num_layers 不是 0、1 或 2，则抛出数值错误
            raise ValueError('num_layers must be one of 0, 1, or 2')

    # 前向传播函数，定义网络的计算流程
    def forward(self, x):
        # 对输入张量 x 进行维度扩展
        xi = x[:, :, :, :, None, None]
        xj = x[:, :, None, None, :, :]
        # 计算相似度矩阵，使用余弦相似度并进行非负化和幂运算
        similarity = F.relu(F.cosine_similarity(xi, xj, dim = 1)) ** self.gamma

        # 对输入张量 x 进行变换
        transform_out = self.transform_net(x)
        # 使用 einsum 函数将相似度矩阵和变换后的张量进行乘积和重组
        out = einsum('b x y h w, b c h w -> b c x y', similarity, transform_out)
        # 返回计算结果
        return out
# 一个用于基础神经网络的包装类
# 将管理隐藏层输出的拦截并将其传递到投影器和预测器网络中

class NetWrapper(nn.Module):
    def __init__(
        self,
        *,
        net,
        projection_size,
        projection_hidden_size,
        layer_pixel = -2,
        layer_instance = -2
    ):
        super().__init__()
        self.net = net
        self.layer_pixel = layer_pixel
        self.layer_instance = layer_instance

        self.pixel_projector = None
        self.instance_projector = None

        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden_pixel = None
        self.hidden_instance = None
        self.hook_registered = False

    # 查找指定层
    def _find_layer(self, layer_id):
        if type(layer_id) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(layer_id, None)
        elif type(layer_id) == int:
            children = [*self.net.children()]
            return children[layer_id]
        return None

    # 钩子函数，用于拦截像素层输出
    def _hook_pixel(self, _, __, output):
        setattr(self, 'hidden_pixel', output)

    # 钩子函数，用于拦截实例层输出
    def _hook_instance(self, _, __, output):
        setattr(self, 'hidden_instance', output)

    # 注册钩子函数
    def _register_hook(self):
        pixel_layer = self._find_layer(self.layer_pixel)
        instance_layer = self._find_layer(self.layer_instance)

        assert pixel_layer is not None, f'hidden layer ({self.layer_pixel}) not found'
        assert instance_layer is not None, f'hidden layer ({self.layer_instance}) not found'

        pixel_layer.register_forward_hook(self._hook_pixel)
        instance_layer.register_forward_hook(self._hook_instance)
        self.hook_registered = True

    # 获取像素投影器
    @singleton('pixel_projector')
    def _get_pixel_projector(self, hidden):
        _, dim, *_ = hidden.shape
        projector = ConvMLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    # 获取实例投影器
    @singleton('instance_projector')
    def _get_instance_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    # 获取表示
    def get_representation(self, x):
        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden_pixel = self.hidden_pixel
        hidden_instance = self.hidden_instance
        self.hidden_pixel = None
        self.hidden_instance = None
        assert hidden_pixel is not None, f'hidden pixel layer {self.layer_pixel} never emitted an output'
        assert hidden_instance is not None, f'hidden instance layer {self.layer_instance} never emitted an output'
        return hidden_pixel, hidden_instance

    # 前向传播
    def forward(self, x):
        pixel_representation, instance_representation = self.get_representation(x)
        instance_representation = instance_representation.flatten(1)

        pixel_projector = self._get_pixel_projector(pixel_representation)
        instance_projector = self._get_instance_projector(instance_representation)

        pixel_projection = pixel_projector(pixel_representation)
        instance_projection = instance_projector(instance_representation)
        return pixel_projection, instance_projection

# 主类

class PixelCL(nn.Module):
    # 初始化函数，设置模型参数和数据增强方式等
    def __init__(
        self,
        net,
        image_size,
        hidden_layer_pixel = -2,
        hidden_layer_instance = -2,
        projection_size = 256,
        projection_hidden_size = 2048,
        augment_fn = None,
        augment_fn2 = None,
        prob_rand_hflip = 0.25,
        moving_average_decay = 0.99,
        ppm_num_layers = 1,
        ppm_gamma = 2,
        distance_thres = 0.7,
        similarity_temperature = 0.3,
        alpha = 1.,
        use_pixpro = True,
        cutout_ratio_range = (0.6, 0.8),
        cutout_interpolate_mode = 'nearest',
        coord_cutout_interpolate_mode = 'bilinear'
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 默认的数据增强方式
        DEFAULT_AUG = nn.Sequential(
            RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            augs.RandomGrayscale(p=0.2),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            augs.RandomSolarize(p=0.5),
            augs.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        )

        # 设置数据增强方式
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)
        self.prob_rand_hflip = prob_rand_hflip

        # 在线编码器
        self.online_encoder = NetWrapper(
            net = net,
            projection_size = projection_size,
            projection_hidden_size = projection_hidden_size,
            layer_pixel = hidden_layer_pixel,
            layer_instance = hidden_layer_instance
        )

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.distance_thres = distance_thres
        self.similarity_temperature = similarity_temperature
        self.alpha = alpha

        self.use_pixpro = use_pixpro

        # 如果使用像素级处理
        if use_pixpro:
            self.propagate_pixels = PPM(
                chan = projection_size,
                num_layers = ppm_num_layers,
                gamma = ppm_gamma
            )

        self.cutout_ratio_range = cutout_ratio_range
        self.cutout_interpolate_mode = cutout_interpolate_mode
        self.coord_cutout_interpolate_mode = coord_cutout_interpolate_mode

        # 实例级别预测器
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

    # 重置移动平均值
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    # 更新移动平均值
    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
```