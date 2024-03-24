# `.\lucidrains\pi-GAN-pytorch\pi_gan_pytorch\pi_gan_pytorch.py`

```
# 导入所需的库
import math
from pathlib import Path
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from tqdm import trange
from PIL import Image
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as T

# 导入自定义模块
from pi_gan_pytorch.coordconv import CoordConv
from pi_gan_pytorch.nerf import get_image_from_nerf_model
from einops import rearrange, repeat

# 检查是否有可用的 CUDA 设备
assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# 定义一些辅助函数

def exists(val):
    return val is not None

def leaky_relu(p = 0.2):
    return nn.LeakyReLU(p)

def to_value(t):
    return t.clone().detach().item()

def get_module_device(module):
    return next(module.parameters()).device

# 定义损失函数

def gradient_penalty(images, output, weight = 10):
    batch_size, device = images.shape[0], images.device
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    l2 = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
    return weight * l2

# 定义正弦激活函数

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# 定义 Siren 层

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, gamma = None, beta = None):
        out =  F.linear(x, self.weight, self.bias)

        # FiLM modulation

        if exists(gamma):
            out = out * gamma

        if exists(beta):
            out = out + beta

        out = self.activation(out)
        return out

# 定义映射网络

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 0.1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class MappingNetwork(nn.Module):
    def __init__(self, *, dim, dim_out, depth = 3, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(dim, dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

        self.to_gamma = nn.Linear(dim, dim_out)
        self.to_beta = nn.Linear(dim, dim_out)

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        x = self.net(x)
        return self.to_gamma(x), self.to_beta(x)

# 定义 Siren 网络

class SirenNet(nn.Module):
    # 初始化神经网络模型
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个空的神经网络层列表
        self.layers = nn.ModuleList([])

        # 循环创建指定数量的 Siren 层
        for ind in range(num_layers):
            # 判断是否是第一层
            is_first = ind == 0
            # 根据是否是第一层选择不同的参数
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            # 将创建的 Siren 层添加到神经网络层列表中
            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        # 创建最后一层 Siren 层
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    # 前向传播函数
    def forward(self, x, gamma, beta):
        # 遍历神经网络层列表，依次进行前向传播
        for layer in self.layers:
            x = layer(x, gamma, beta)
        # 返回最后一层的前向传播结果
        return self.last_layer(x)
# 定义 Siren 生成器类
class SirenGenerator(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,
        dim_hidden,
        siren_num_layers = 8
    ):
        super().__init__()

        # 创建映射网络对象
        self.mapping = MappingNetwork(
            dim = dim,
            dim_out = dim_hidden
        )

        # 创建 Siren 网络对象
        self.siren = SirenNet(
            dim_in = 3,
            dim_hidden = dim_hidden,
            dim_out = dim_hidden,
            num_layers = siren_num_layers
        )

        # 创建输出 alpha 的线性层
        self.to_alpha = nn.Linear(dim_hidden, 1)

        # 创建 Siren 网络对象用于生成 RGB
        self.to_rgb_siren = Siren(
            dim_in = dim_hidden,
            dim_out = dim_hidden
        )

        # 创建输出 RGB 的线性层
        self.to_rgb = nn.Linear(dim_hidden, 3)

    # 前向传播函数
    def forward(self, latent, coors, batch_size = 8192):
        # 获取 gamma 和 beta
        gamma, beta = self.mapping(latent)

        outs = []
        # 分批处理坐标
        for coor in coors.split(batch_size):
            # 重排 gamma 和 beta 的维度
            gamma_, beta_ = map(lambda t: rearrange(t, 'n -> () n'), (gamma, beta))
            # 使用 Siren 网络生成 x
            x = self.siren(coor, gamma_, beta_)
            # 生成 alpha
            alpha = self.to_alpha(x)

            # 使用 Siren 网络生成 RGB
            x = self.to_rgb_siren(x, gamma, beta)
            rgb = self.to_rgb(x)
            # 拼接 RGB 和 alpha
            out = torch.cat((rgb, alpha), dim = -1)
            outs.append(out)

        return torch.cat(outs)

# 定义生成器类
class Generator(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        image_size,
        dim,
        dim_hidden,
        siren_num_layers
    ):
        super().__init__()
        self.dim = dim
        self.image_size = image_size

        # 创建 Siren 生成器对象
        self.nerf_model = SirenGenerator(
            dim = dim,
            dim_hidden = dim_hidden,
            siren_num_layers = siren_num_layers
        )

    # 设置图像尺寸
    def set_image_size(self, image_size):
        self.image_size = image_size

    # 前向传播函数
    def forward(self, latents):
        image_size = self.image_size
        device, b = latents.device, latents.shape[0]

        # 从 Siren 生成器模型获取生成的图像
        generated_images = get_image_from_nerf_model(
            self.nerf_model,
            latents,
            image_size,
            image_size
        )

        return generated_images

# 定义判别器块类
class DiscriminatorBlock(nn.Module):
    # 初始化函数
    def __init__(self, dim, dim_out):
        super().__init__()
        # 创建 CoordConv 层
        self.res = CoordConv(dim, dim_out, kernel_size = 1, stride = 2)

        # 创建网络序列
        self.net = nn.Sequential(
            CoordConv(dim, dim_out, kernel_size = 3, padding = 1),
            leaky_relu(),
            CoordConv(dim_out, dim_out, kernel_size = 3, padding = 1),
            leaky_relu()
        )

        # 下采样层
        self.down = nn.AvgPool2d(2)

    # 前向传播函数
    def forward(self, x):
        res = self.res(x)
        x = self.net(x)
        x = self.down(x)
        x = x + res
        return x

# 定义判别器类
class Discriminator(nn.Module):
    # 初始化函数
    def __init__(
        self,
        image_size,
        init_chan = 64,
        max_chan = 400,
        init_resolution = 32,
        add_layer_iters = 10000
    ):
        # 调用父类的构造函数
        super().__init__()
        # 计算图像大小的对数值
        resolutions = math.log2(image_size)
        # 断言图像大小必须是2的幂
        assert resolutions.is_integer(), 'image size must be a power of 2'
        # 断言初始分辨率必须是2的幂
        assert math.log2(init_resolution).is_integer(), 'initial resolution must be power of 2'

        # 将对数值转换为整数
        resolutions = int(resolutions)
        # 计算层数
        layers = resolutions - 1

        # 计算通道数列表
        chans = list(reversed(list(map(lambda t: 2 ** (11 - t), range(layers))))
        # 将通道数限制在最大通道数以内
        chans = list(map(lambda n: min(max_chan, n), chans))
        # 添加初始通道数到通道数列表
        chans = [init_chan, *chans]
        # 获取最终通道数
        final_chan = chans[-1]

        # 初始化 from_rgb_layers 和 layers
        self.from_rgb_layers = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        self.image_size = image_size
        self.resolutions = list(map(lambda t: 2 ** (7 - t), range(layers)))

        # 遍历分辨率、输入通道数、输出通道数，创建 from_rgb_layer 和 DiscriminatorBlock
        for resolution, in_chan, out_chan in zip(self.resolutions, chans[:-1], chans[1:]):

            from_rgb_layer = nn.Sequential(
                CoordConv(3, in_chan, kernel_size = 1),
                leaky_relu()
            ) if resolution >= init_resolution else None

            self.from_rgb_layers.append(from_rgb_layer)

            self.layers.append(DiscriminatorBlock(
                dim = in_chan,
                dim_out = out_chan
            ))

        # 创建最终卷积层
        self.final_conv = CoordConv(final_chan, 1, kernel_size = 2)

        # 初始化 alpha、resolution 和 iterations
        self.add_layer_iters = add_layer_iters
        self.register_buffer('alpha', torch.tensor(0.))
        self.register_buffer('resolution', torch.tensor(init_resolution))
        self.register_buffer('iterations', torch.tensor(0.))

    # 增加分辨率
    def increase_resolution_(self):
        if self.resolution >= self.image_size:
            return

        self.alpha += self.alpha + (1 - self.alpha)
        self.iterations.fill_(0.)
        self.resolution *= 2

    # 更新迭代次数
    def update_iter_(self):
        self.iterations += 1
        self.alpha -= (1 / self.add_layer_iters)
        self.alpha.clamp_(min = 0.)

    # 前向传播函数
    def forward(self, img):
        x = img

        for resolution, from_rgb, layer in zip(self.resolutions, self.from_rgb_layers, self.layers):
            if self.resolution < resolution:
                continue

            if self.resolution == resolution:
                x = from_rgb(x)

            if bool(resolution == (self.resolution // 2)) and bool(self.alpha > 0):
                x_down = F.interpolate(img, scale_factor = 0.5)
                x = x * (1 - self.alpha) + from_rgb(x_down) * self.alpha

            x = layer(x)

        out = self.final_conv(x)
        return out
# 定义 piGAN 类
class piGAN(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        dim,
        init_resolution = 32,
        generator_dim_hidden = 256,
        siren_num_layers = 6,
        add_layer_iters = 10000
    ):
        super().__init__()
        self.dim = dim

        # 初始化生成器 G
        self.G = Generator(
            image_size = image_size,
            dim = dim,
            dim_hidden = generator_dim_hidden,
            siren_num_layers = siren_num_layers
        )

        # 初始化判别器 D
        self.D = Discriminator(
            image_size = image_size,
            add_layer_iters = add_layer_iters,
            init_resolution = init_resolution
        )

# 定义数据集相关函数

# 无限循环迭代器
def cycle(iterable):
    while True:
        for i in iterable:
            yield i

# 调整图像大小至最小尺寸
def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

# 图像数据集类
class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        transparent = False,
        aug_prob = 0.,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'
        self.create_transform(image_size)

    # 创建图像转换函数
    def create_transform(self, image_size):
        self.transform = T.Compose([
            T.Lambda(partial(resize_to_minimum_size, image_size)),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# 训练器类

# 生成器采样函数
def sample_generator(G, batch_size):
    dim = G.dim
    rand_latents = torch.randn(batch_size, dim).cuda()
    return G(rand_latents)

class Trainer(nn.Module):
    def __init__(
        self,
        *,
        gan,
        folder,
        add_layers_iters = 10000,
        batch_size = 8,
        gradient_accumulate_every = 4,
        sample_every = 100,
        log_every = 10,
        num_train_steps = 50000,
        lr_gen = 5e-5,
        lr_discr = 4e-4,
        target_lr_gen = 1e-5,
        target_lr_discr = 1e-4,
        lr_decay_span = 10000
    ):
        super().__init__()
        gan.D.add_layer_iters = add_layers_iters
        self.add_layers_iters = add_layers_iters

        # 将 gan 移至 GPU
        self.gan = gan.cuda()

        # 初始化判别器和生成器的优化器
        self.optim_D = Adam(self.gan.D.parameters(), betas=(0, 0.9), lr = lr_discr)
        self.optim_G = Adam(self.gan.G.parameters(), betas=(0, 0.9), lr = lr_gen)

        # 定义判别器和生成器的学习率衰减函数
        D_decay_fn = lambda i: max(1 - i / lr_decay_span, 0) + (target_lr_discr / lr_discr) * min(i / lr_decay_span, 1)
        G_decay_fn = lambda i: max(1 - i / lr_decay_span, 0) + (target_lr_gen / lr_gen) * min(i / lr_decay_span, 1)

        # 初始化判别器和生成器的学习率调度器
        self.sched_D = LambdaLR(self.optim_D, D_decay_fn)
        self.sched_G = LambdaLR(self.optim_G, G_decay_fn)

        self.iterations = 0
        self.batch_size = batch_size
        self.num_train_steps = num_train_steps

        self.log_every = log_every
        self.sample_every = sample_every
        self.gradient_accumulate_every = gradient_accumulate_every

        # 初始化数据集和数据加载器
        self.dataset = ImageDataset(folder = folder, image_size = gan.D.resolution.item())
        self.dataloader = cycle(DataLoader(self.dataset, batch_size = batch_size, shuffle = True, drop_last = True))

        self.last_loss_D = 0
        self.last_loss_G = 0
    # 定义每一步训练的操作
    def step(self):
        # 获取GAN模型的判别器D、生成器G、批量大小batch_size、维度dim、梯度累积次数accumulate_every
        D, G, batch_size, dim, accumulate_every = self.gan.D, self.gan.G, self.batch_size, self.gan.dim, self.gradient_accumulate_every

        # 设置适当的图像大小
        if self.iterations % self.add_layers_iters == 0:
            if self.iterations != 0:
                D.increase_resolution_()

            # 获取图像大小
            image_size = D.resolution.item()
            G.set_image_size(image_size)
            self.dataset.create_transform(image_size)

        # 是否应用梯度惩罚
        apply_gp = self.iterations % 4 == 0

        # 训练判别器
        D.train()
        loss_D = 0

        for _ in range(accumulate_every):
            # 获取下一个批量图像数据
            images = next(self.dataloader)
            images = images.cuda().requires_grad_()
            real_out = D(images)

            # 生成假图像
            fake_imgs = sample_generator(G, batch_size)
            fake_out = D(fake_imgs.clone().detach())

            # 计算梯度惩罚
            divergence = (F.relu(1 + real_out) + F.relu(1 - fake_out)).mean()
            loss = divergence

            if apply_gp:
                gp = gradient_penalty(images, real_out)
                self.last_loss_gp = to_value(gp)
                loss = loss + gp

            (loss / accumulate_every).backward()
            loss_D += to_value(divergence) / accumulate_every

        self.last_loss_D = loss_D

        self.optim_D.step()
        self.optim_D.zero_grad()

        # 训练生成器
        G.train()
        loss_G = 0

        for _ in range(accumulate_every):
            fake_out = sample_generator(G, batch_size)
            loss = D(fake_out).mean()
            (loss / accumulate_every).backward()
            loss_G += to_value(loss) / accumulate_every

        self.last_loss_G = loss_G

        self.optim_G.step()
        self.optim_G.zero_grad()

        # 更新调度器
        self.sched_D.step()
        self.sched_G.step()

        self.iterations += 1
        D.update_iter_()

    # 前向传播函数
    def forward(self):
        for _ in trange(self.num_train_steps):
            self.step()

            # 每隔一定步数打印损失信息
            if self.iterations % self.log_every == 0:
                print(f'I: {self.gan.D.resolution.item()} | D: {self.last_loss_D:.2f} | G: {self.last_loss_G:.2f} | GP: {self.last_loss_gp:.2f}')

            # 每隔一定步数保存生成的图像
            if self.iterations % self.sample_every == 0:
                i = self.iterations // self.sample_every
                imgs = sample_generator(self.gan.G, 4)
                imgs.clamp_(0., 1.)
                save_image(imgs, f'./{i}.png', nrow=2)
```