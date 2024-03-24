# `.\lucidrains\vit-pytorch\vit_pytorch\es_vit.py`

```
# 导入所需的库
import copy
import random
from functools import wraps, partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms as T
from einops import rearrange, reduce, repeat

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
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

# 张量相关的辅助函数

# 对张量取对数
def log(t, eps = 1e-20):
    return torch.log(t + eps)

# 损失函数

# 视图损失函数
def view_loss_fn(
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
    return - (teacher_probs * log(student_probs, eps)).sum(dim = -1).mean()

# 区域损失函数
def region_loss_fn(
    teacher_logits,
    student_logits,
    teacher_latent,
    student_latent,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim = -1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim = -1)

    sim_matrix = einsum('b i d, b j d -> b i j', student_latent, teacher_latent)
    sim_indices = sim_matrix.max(dim = -1).indices
    sim_indices = repeat(sim_indices, 'b n -> b n k', k = teacher_probs.shape[-1])
    max_sim_teacher_probs = teacher_probs.gather(1, sim_indices)

    return - (max_sim_teacher_probs * log(student_probs, eps)).sum(dim = -1).mean()

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

# MLP 类用于投影器和预测器

# L2范数
class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        return F.normalize(x, dim = 1, eps = eps)

# 多层感知机
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
# 将管理隐藏层输出的拦截
# 创建一个包装器类，用于将输入传递到投影器和预测器网络中
class NetWrapper(nn.Module):
    def __init__(self, net, output_dim, projection_hidden_size, projection_num_layers, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.view_projector = None
        self.region_projector = None
        self.projection_hidden_size = projection_hidden_size
        self.projection_num_layers = projection_num_layers
        self.output_dim = output_dim

        self.hidden = {}
        self.hook_registered = False

    # 查找指定的层
    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    # 钩子函数，用于获取隐藏层输出
    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output

    # 注册钩子函数
    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    # 获取视图投影器
    @singleton('view_projector')
    def _get_view_projector(self, hidden):
        dim = hidden.shape[1]
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector.to(hidden)

    # 获取区域投影器
    @singleton('region_projector')
    def _get_region_projector(self, hidden):
        dim = hidden.shape[1]
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector.to(hidden)

    # 获取嵌入向量
    def get_embedding(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    # 前向传播函数
    def forward(self, x, return_projection = True):
        region_latents = self.get_embedding(x)
        global_latent = reduce(region_latents, 'b c h w -> b c', 'mean')

        if not return_projection:
            return global_latent, region_latents

        view_projector = self._get_view_projector(global_latent)
        region_projector = self._get_region_projector(region_latents)

        region_latents = rearrange(region_latents, 'b c h w -> b (h w) c')

        return view_projector(global_latent), region_projector(region_latents), region_latents

# 主类
class EsViTTrainer(nn.Module):
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
    # 定义一个继承自父类的子类，初始化网络
    ):
        super().__init__()
        self.net = net

        # 默认的 BYOL 数据增强

        DEFAULT_AUG = torch.nn.Sequential(
            # 随机应用颜色抖动
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            # 随机转换为灰度图像
            T.RandomGrayscale(p=0.2),
            # 随机水平翻转
            T.RandomHorizontalFlip(),
            # 随机应用高斯模糊
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            # 归一化
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        # 初始化两种数据增强方式
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, DEFAULT_AUG)

        # 定义局部和全局裁剪
        self.local_crop = T.RandomResizedCrop((image_size, image_size), scale = (0.05, local_upper_crop_scale))
        self.global_crop = T.RandomResizedCrop((image_size, image_size), scale = (global_lower_crop_scale, 1.))

        # 初始化学生编码器
        self.student_encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer = hidden_layer)

        # 初始化教师编码器和指数移动平均更新器
        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)

        # 注册缓冲区，用于存储教师视图中心和区域中心
        self.register_buffer('teacher_view_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_view_centers',  torch.zeros(1, num_classes_K))

        self.register_buffer('teacher_region_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_region_centers',  torch.zeros(1, num_classes_K))

        # 初始化教师中心指数移动平均更新器
        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # 获取网络设备并将包装器设备设置为相同
        device = get_module_device(net)
        self.to(device)

        # 发送一个模拟图像张量以实例化单例参数
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    # 使用装饰器创建单例模式，获取教师编码器
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

        new_teacher_view_centers = self.teacher_centering_ema_updater.update_average(self.teacher_view_centers, self.last_teacher_view_centers)
        self.teacher_view_centers.copy_(new_teacher_view_centers)

        new_teacher_region_centers = self.teacher_centering_ema_updater.update_average(self.teacher_region_centers, self.last_teacher_region_centers)
        self.teacher_region_centers.copy_(new_teacher_region_centers)

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

        # 对增强后的数据进行局部裁剪和全局裁剪
        local_image_one, local_image_two   = self.local_crop(image_one),  self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        # 使用学生编码器对局部裁剪后的数据进行编码
        student_view_proj_one, student_region_proj_one, student_latent_one = self.student_encoder(local_image_one)
        student_view_proj_two, student_region_proj_two, student_latent_two = self.student_encoder(local_image_two)

        # 使用torch.no_grad()上下文管理器，获取教师编码器的结果
        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_view_proj_one, teacher_region_proj_one, teacher_latent_one = teacher_encoder(global_image_one)
            teacher_view_proj_two, teacher_region_proj_two, teacher_latent_two = teacher_encoder(global_image_two)

        # 部分函数调用，设置视图级别损失函数和区域级别损失函数的参数
        view_loss_fn_ = partial(
            view_loss_fn,
            student_temp = default(student_temp, self.student_temp),
            teacher_temp = default(teacher_temp, self.teacher_temp),
            centers = self.teacher_view_centers
        )

        region_loss_fn_ = partial(
            region_loss_fn,
            student_temp = default(student_temp, self.student_temp),
            teacher_temp = default(teacher_temp, self.teacher_temp),
            centers = self.teacher_region_centers
        )

        # 计算视图级别损失
        teacher_view_logits_avg = torch.cat((teacher_view_proj_one, teacher_view_proj_two)).mean(dim = 0)
        self.last_teacher_view_centers.copy_(teacher_view_logits_avg)

        teacher_region_logits_avg = torch.cat((teacher_region_proj_one, teacher_region_proj_two)).mean(dim = (0, 1))
        self.last_teacher_region_centers.copy_(teacher_region_logits_avg)

        view_loss = (view_loss_fn_(teacher_view_proj_one, student_view_proj_two) \
                   + view_loss_fn_(teacher_view_proj_two, student_view_proj_one)) / 2

        # 计算区域级别损失
        region_loss = (region_loss_fn_(teacher_region_proj_one, student_region_proj_two, teacher_latent_one, student_latent_two) \
                     + region_loss_fn_(teacher_region_proj_two, student_region_proj_one, teacher_latent_two, student_latent_one)) / 2

        # 返回视图级别损失和区域级别损失的平均值
        return (view_loss + region_loss) / 2
```