# `stable-diffusion-webui\modules\rng.py`

```py
# 导入 torch 库
import torch

# 从 modules 模块中导入 devices、rng_philox、shared
from modules import devices, rng_philox, shared

# 使用给定种子生成服从正态分布的随机数张量
def randn(seed, shape, generator=None):
    """Generate a tensor with random numbers from a normal distribution using seed.

    Uses the seed parameter to set the global torch seed; to generate more with that seed, use randn_like/randn_without_seed."""

    # 设置全局随机数种子为给定种子
    manual_seed(seed)

    # 根据配置选择随机数生成源，生成指定形状的张量
    if shared.opts.randn_source == "NV":
        return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)

    # 根据配置选择随机数生成源，生成指定形状的张量
    if shared.opts.randn_source == "CPU" or devices.device.type == 'mps':
        return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)

    # 根据配置选择随机数生成源，生成指定形状的张量
    return torch.randn(shape, device=devices.device, generator=generator)


# 使用给定种子生成服从正态分布的随机数张量，不改变全局随机数生成器
def randn_local(seed, shape):
    """Generate a tensor with random numbers from a normal distribution using seed.

    Does not change the global random number generator. You can only generate the seed's first tensor using this function."""

    # 根据配置选择随机数生成源，生成指定形状的张量
    if shared.opts.randn_source == "NV":
        rng = rng_philox.Generator(seed)
        return torch.asarray(rng.randn(shape), device=devices.device)

    # 根据配置选择随机数生成源，生成指定形状的张量
    local_device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
    local_generator = torch.Generator(local_device).manual_seed(int(seed))
    return torch.randn(shape, device=local_device, generator=local_generator).to(devices.device)


# 使用先前初始化的生成器生成服从正态分布的随机数张量
def randn_like(x):
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""

    # 根据配置选择随机数生成源，生成与输入张量相同形状的张量
    if shared.opts.randn_source == "NV":
        return torch.asarray(nv_rng.randn(x.shape), device=x.device, dtype=x.dtype)

    # 根据配置选择随机数生成源，生成与输入张量相同形状的张量
    if shared.opts.randn_source == "CPU" or x.device.type == 'mps':
        return torch.randn_like(x, device=devices.cpu).to(x.device)

    # 根据配置选择随机数生成源，生成与输入张量相同形状的张量
    return torch.randn_like(x)


# 生成服从正态分布的随机数张量，不使用种子
def randn_without_seed(shape, generator=None):
    """使用先前初始化的生成器从正态分布中生成具有随机数的张量。

    使用randn()或manual_seed()来初始化生成器。"""

    # 如果随机数源为"NV"，则使用生成器或nv_rng生成服从正态分布的随机数张量
    if shared.opts.randn_source == "NV":
        return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)

    # 如果随机数源为"CPU"或设备类型为'mps'，则在CPU上生成服从正态分布的随机数张量，并转移到指定设备
    if shared.opts.randn_source == "CPU" or devices.device.type == 'mps':
        return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)

    # 在指定设备上生成服从正态分布的随机数张量
    return torch.randn(shape, device=devices.device, generator=generator)
# 设置全局随机数生成器，使用指定的种子
def manual_seed(seed):
    # 如果随机数来源为 NV，则创建 NV 随机数生成器对象
    if shared.opts.randn_source == "NV":
        global nv_rng
        nv_rng = rng_philox.Generator(seed)
        return

    # 使用指定种子设置 PyTorch 的随机数生成器
    torch.manual_seed(seed)


# 创建随机数生成器对象
def create_generator(seed):
    # 如果随机数来源为 NV，则返回 NV 随机数生成器对象
    if shared.opts.randn_source == "NV":
        return rng_philox.Generator(seed)

    # 根据不同的随机数来源选择设备
    device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
    # 创建 PyTorch 随机数生成器对象，并使用指定种子
    generator = torch.Generator(device).manual_seed(int(seed))
    return generator


# 实现球面线性插值函数
# 参考 https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    # 对低维度向量进行归一化
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    # 计算向量点乘
    dot = (low_norm*high_norm).sum(1)

    # 如果点乘结果接近于 1，则直接返回线性插值结果
    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    # 计算角度和正弦值
    omega = torch.acos(dot)
    so = torch.sin(omega)
    # 计算球面线性插值结果
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


# 定义图像随机数生成器类
class ImageRNG:
    def __init__(self, shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0):
        # 初始化图像随机数生成器对象的属性
        self.shape = tuple(map(int, shape))
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w

        # 根据种子创建随机数生成器对象列表
        self.generators = [create_generator(seed) for seed in seeds]

        # 标记是否为第一次使用
        self.is_first = True
    # 定义一个方法，用于生成第一层噪声
    def first(self):
        # 根据条件确定噪声的形状
        noise_shape = self.shape if self.seed_resize_from_h <= 0 or self.seed_resize_from_w <= 0 else (self.shape[0], int(self.seed_resize_from_h) // 8, int(self.seed_resize_from_w // 8))

        # 初始化一个空列表用于存储噪声数据
        xs = []

        # 遍历种子和生成器的组合
        for i, (seed, generator) in enumerate(zip(self.seeds, self.generators)):
            subnoise = None
            # 如果存在子种子并且子种子强度不为0，则生成子噪声
            if self.subseeds is not None and self.subseed_strength != 0:
                subseed = 0 if i >= len(self.subseeds) else self.subseeds[i]
                subnoise = randn(subseed, noise_shape)

            # 根据噪声形状生成噪声数据
            if noise_shape != self.shape:
                noise = randn(seed, noise_shape)
            else:
                noise = randn(seed, self.shape, generator=generator)

            # 如果存在子噪声，则进行插值
            if subnoise is not None:
                noise = slerp(self.subseed_strength, noise, subnoise)

            # 如果噪声形状不等于原始形状，则进行裁剪和填充
            if noise_shape != self.shape:
                x = randn(seed, self.shape, generator=generator)
                dx = (self.shape[2] - noise_shape[2]) // 2
                dy = (self.shape[1] - noise_shape[1]) // 2
                w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
                h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
                tx = 0 if dx < 0 else dx
                ty = 0 if dy < 0 else dy
                dx = max(-dx, 0)
                dy = max(-dy, 0)

                x[:, ty:ty + h, tx:tx + w] = noise[:, dy:dy + h, dx:dx + w]
                noise = x

            # 将生成的噪声数据添加到列表中
            xs.append(noise)

        # 获取共享选项中的eta_noise_seed_delta值，如果存在则更新生成器
        eta_noise_seed_delta = shared.opts.eta_noise_seed_delta or 0
        if eta_noise_seed_delta:
            self.generators = [create_generator(seed + eta_noise_seed_delta) for seed in self.seeds]

        # 返回所有噪声数据的张量
        return torch.stack(xs).to(shared.device)
    # 定义一个方法用于获取下一个数据点
    def next(self):
        # 如果是第一次调用next方法
        if self.is_first:
            # 将is_first标记为False，并返回第一个数据点
            self.is_first = False
            return self.first()

        # 初始化一个空列表用于存储生成的数据点
        xs = []
        # 遍历所有的生成器
        for generator in self.generators:
            # 使用指定生成器生成一个数据点，并添加到xs列表中
            x = randn_without_seed(self.shape, generator=generator)
            xs.append(x)

        # 将所有生成的数据点堆叠成一个张量，并将其转移到指定设备上
        return torch.stack(xs).to(shared.device)
# 将全局随机数生成器设备的 randn 函数赋值给 devices 对象的 randn 属性
devices.randn = randn
# 将本地随机数生成器设备的 randn_local 函数赋值给 devices 对象的 randn_local 属性
devices.randn_local = randn_local
# 将全局随机数生成器设备的 randn_like 函数赋值给 devices 对象的 randn_like 属性
devices.randn_like = randn_like
# 将不使用种子的全局随机数生成器设备的 randn_without_seed 函数赋值给 devices 对象的 randn_without_seed 属性
devices.randn_without_seed = randn_without_seed
# 将手动设置种子的函数 manual_seed 赋值给 devices 对象的 manual_seed 属性
devices.manual_seed = manual_seed
```