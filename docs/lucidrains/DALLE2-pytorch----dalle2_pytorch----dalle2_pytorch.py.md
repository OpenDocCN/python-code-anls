# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\dalle2_pytorch.py`

```
# 导入数学库
import math
# 导入随机数库
import random
# 导入进度条库
from tqdm.auto import tqdm
# 导入偏函数库
from functools import partial, wraps
# 导入上下文管理库
from contextlib import contextmanager
# 导入命名元组库
from collections import namedtuple
# 导入路径库
from pathlib import Path

# 导入 PyTorch 库
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn, einsum
import torchvision.transforms as T

# 导入 einops 库
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# 导入 kornia 库
from kornia.filters import gaussian_blur2d
import kornia.augmentation as K

# 导入 dalle2_pytorch 库
from dalle2_pytorch.tokenizer import tokenizer
from dalle2_pytorch.vqgan_vae import NullVQGanVAE, VQGanVAE

# 导入 resize_right 库
from resize_right import resize

# 导入旋转嵌入库
from rotary_embedding_torch import RotaryEmbedding

# 导入 x-clip 库
from x_clip import CLIP
from coca_pytorch import CoCa

# 常量定义
NAT = 1. / math.log(2.)

# 定义命名元组 UnetOutput
UnetOutput = namedtuple('UnetOutput', ['pred', 'var_interp_frac_unnormalized'])

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回输入值
def identity(t, *args, **kwargs):
    return t

# 返回列表的第一个元素
def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

# 可选函数装饰器
def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

# 默认值函数
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 将值转换为元组
def cast_tuple(val, length = None, validate = True):
    if isinstance(val, list):
        val = tuple(val)

    out = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length) and validate:
        assert len(out) == length

    return out

# 获取模块的设备
def module_device(module):
    if isinstance(module, nn.Identity):
        return 'cpu' # 无关紧要
    return next(module.parameters()).device

# 初始化权重为零
def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

# 空上下文管理器
@contextmanager
def null_context(*args, **kwargs):
    yield

# 模型评估装饰器
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 判断是否为浮点数类型
def is_float_dtype(dtype):
    return any([dtype == float_dtype for float_dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)])

# 判断是否为字符串列表
def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# 将元组填充到指定长度
def pad_tuple_to_length(t, length, fillvalue = None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))

# 检查点辅助函数

def make_checkpointable(fn, **kwargs):
    if isinstance(fn, nn.ModuleList):
        return [maybe(make_checkpointable)(el, **kwargs) for el in fn]

    condition = kwargs.pop('condition', None)

    if exists(condition) and not condition(fn):
        return fn

    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# 控制 CLIP 冻结的函数

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)

# 张量辅助函数

# 对数函数
def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

# L2 归一化函数
def l2norm(t):
    return F.normalize(t, dim = -1)

# 调整图像大小函数
def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    nearest = False,
    **kwargs
):
    orig_image_size = image.shape[-1]
    # 如果原始图像大小与目标图像大小相同，则直接返回原始图像
    if orig_image_size == target_image_size:
        return image

    # 如果不使用最近邻插值，则计算缩放因子并调整图像大小
    if not nearest:
        scale_factors = target_image_size / orig_image_size
        out = resize(image, scale_factors=scale_factors, **kwargs)
    # 如果使用最近邻插值，则使用最近邻插值方法调整图像大小
    else:
        out = F.interpolate(image, target_image_size, mode='nearest')

    # 如果指定了范围限制，则对输出图像进行范围限制
    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    # 返回调整后的图像
    return out
# 图像归一化函数
# DDPMS 期望图像在 -1 到 1 的范围内
# 但 CLIP 可能不同

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

# CLIP 相关适配器

EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings'])
EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])

class BaseClipAdapter(nn.Module):
    def __init__(self, clip, **kwargs):
        super().__init__()
        self.clip = clip
        self.overrides = kwargs

    def validate_and_resize_image(self, image):
        image_size = image.shape[-1]
        assert image_size >= self.image_size, f'you are passing in an image of size {image_size} but CLIP requires the image size to be at least {self.image_size}'
        return resize_image_to(image, self.image_size)

    @property
    def dim_latent(self):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def image_channels(self):
        raise NotImplementedError

    @property
    def max_text_len(self):
        raise NotImplementedError

    def embed_text(self, text):
        raise NotImplementedError

    def embed_image(self, image):
        raise NotImplementedError

class XClipAdapter(BaseClipAdapter):
    @property
    def dim_latent(self):
        return self.clip.dim_latent

    @property
    def image_size(self):
        return self.clip.image_size

    @property
    def image_channels(self):
        return self.clip.image_channels

    @property
    def max_text_len(self):
        return self.clip.text_seq_len

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]
        text_mask = text != 0
        encoder_output = self.clip.text_transformer(text)

        encoder_output_is_cls = encoder_output.ndim == 3

        text_cls, text_encodings = (encoder_output[:, 0], encoder_output[:, 1:]) if encoder_output_is_cls else (encoder_output, None)
        text_embed = self.clip.to_text_latent(text_cls)

        if exists(text_encodings):
            text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)

        return EmbeddedText(l2norm(text_embed), text_encodings)

    @torch.no_grad()
    def embed_image(self, image):
        image = self.validate_and_resize_image(image)
        encoder_output = self.clip.visual_transformer(image)
        image_cls, image_encodings = encoder_output[:, 0], encoder_output[:, 1:]
        image_embed = self.clip.to_visual_latent(image_cls)
        return EmbeddedImage(l2norm(image_embed), image_encodings)

class CoCaAdapter(BaseClipAdapter):
    @property
    def dim_latent(self):
        return self.clip.dim

    @property
    def image_size(self):
        assert 'image_size' in self.overrides
        return self.overrides['image_size']

    @property
    def image_channels(self):
        assert 'image_channels' in self.overrides
        return self.overrides['image_channels']

    @property
    def max_text_len(self):
        assert 'max_text_len' in self.overrides
        return self.overrides['max_text_len']

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]
        text_mask = text != 0
        text_embed, text_encodings = self.clip.embed_text(text)
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        return EmbeddedText(text_embed, text_encodings)

    @torch.no_grad()
    def embed_image(self, image):
        image = self.validate_and_resize_image(image)
        image_embed, image_encodings = self.clip.embed_image(image)
        return EmbeddedImage(image_embed, image_encodings)

class OpenAIClipAdapter(BaseClipAdapter):
    def __init__(
        self,
        name = 'ViT-B/32'
    ): 
        # 导入 clip 模块
        import clip
        # 加载 OpenAI 的 CLIP 模型和预处理函数
        openai_clip, preprocess = clip.load(name)
        # 调用父类的构造函数，初始化 CLIP 模型
        super().__init__(openai_clip)
        # 设置结束符号的 ID，用于处理 0 也是 '!' 的情况
        self.eos_id = 49407 

        # 获取文本注意力最终层
        text_attention_final = self.find_layer('ln_final')

        # 设置潜在维度
        self.dim_latent_ = text_attention_final.weight.shape[0]
        # 注册前向钩子
        self.handle = text_attention_final.register_forward_hook(self._hook)

        # 获取 CLIP 模型的归一化函数
        self.clip_normalize = preprocess.transforms[-1]
        # 标记是否已清除
        self.cleared = False

    # 查找指定层
    def find_layer(self,  layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    # 清除钩子
    def clear(self):
        if self.cleared:
            return

        self.handle()

    # 钩子函数
    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    # 获取潜在维度
    @property
    def dim_latent(self):
        return self.dim_latent_

    # 获取图像大小
    @property
    def image_size(self):
        return self.clip.visual.input_resolution

    # 获取图像通道数
    @property
    def image_channels(self):
        return 3

    # 获取最大文本长度
    @property
    def max_text_len(self):
        return self.clip.context_length

    # 嵌入文本
    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]

        # 判断是否为结束符号
        is_eos_id = (text == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        text_mask = text_mask & (text != 0)
        assert not self.cleared

        # 编码文本
        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        del self.text_encodings
        return EmbeddedText(l2norm(text_embed.float()), text_encodings.float())

    # 嵌入图像
    @torch.no_grad()
    def embed_image(self, image):
        assert not self.cleared
        # 验证和调整图像大小
        image = self.validate_and_resize_image(image)
        image = self.clip_normalize(image)
        # 编码图像
        image_embed = self.clip.encode_image(image)
        return EmbeddedImage(l2norm(image_embed.float()), None)
class OpenClipAdapter(BaseClipAdapter):
    # OpenClipAdapter 类继承自 BaseClipAdapter 类
    def __init__(
        self,
        name = 'ViT-B/32',
        pretrained = 'laion400m_e32'
    ):
        # 导入 open_clip 模块
        import open_clip
        # 创建 OpenCLIP 模型和预处理方法
        clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained = pretrained)

        # 调用父类的构造函数，传入 clip 模型
        super().__init__(clip)
        # 设置结束符 ID
        self.eos_id = 49407

        # 查找文本注意力最终层
        text_attention_final = self.find_layer('ln_final')
        # 获取潜在维度
        self._dim_latent = text_attention_final.weight.shape[0]

        # 注册 forward hook
        self.handle = text_attention_final.register_forward_hook(self._hook)
        # 获取 CLIP 模型的归一化方法
        self.clip_normalize = preprocess.transforms[-1]
        # 标记是否已清除
        self.cleared = False

    # 查找指定层
    def find_layer(self,  layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    # 清除方法
    def clear(self):
        if self.cleared:
            return

        self.handle()

    # 钩子方法
    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def image_size(self):
        # 获取图像尺寸
        image_size = self.clip.visual.image_size
        if isinstance(image_size, tuple):
            return max(image_size)
        return image_size

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length

    @torch.no_grad()
    def embed_text(self, text):
        # 截取文本长度
        text = text[..., :self.max_text_len]

        # 创建文本掩码
        is_eos_id = (text == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        text_mask = text_mask & (text != 0)
        assert not self.cleared

        # 编码文本
        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        del self.text_encodings
        return EmbeddedText(l2norm(text_embed.float()), text_encodings.float())

    @torch.no_grad()
    def embed_image(self, image):
        assert not self.cleared
        # 验证并调整图像大小
        image = self.validate_and_resize_image(image)
        image = self.clip_normalize(image)
        image_embed = self.clip.encode_image(image)
        return EmbeddedImage(l2norm(image_embed.float()), None)

# 分类器自由指导函数

# 创建概率掩码
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# 高斯扩散辅助函数

# 提取函数
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 平均扁平函数
def meanflat(x):
    return x.mean(dim = tuple(range(1, len(x.shape))))

# 正态 KL 散度
def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

# 近��标准正态 CDF
def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(((2.0 / math.pi) ** 0.5) * (x + 0.044715 * (x ** 3)))

# 离散化高斯对数似然
def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres = 0.999):
    assert x.shape == means.shape == log_scales.shape

    # 修正 nan 梯度
    eps = 1e-12 if x.dtype == torch.float32 else 1e-3

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus, eps = eps)
    log_one_minus_cdf_min = log(1. - cdf_min, eps = eps)
    cdf_delta = cdf_plus - cdf_min
    # 使用 torch.where 函数根据条件选择不同的操作
    # 如果 x 小于 -thres，则返回 log_cdf_plus
    # 如果 x 大于 thres，则返回 log_one_minus_cdf_min
    # 否则返回 log(cdf_delta, eps = eps)
    log_probs = torch.where(x < -thres,
        log_cdf_plus,
        torch.where(x > thres,
            log_one_minus_cdf_min,
            log(cdf_delta, eps = eps)))

    # 返回计算得到的 log_probs
    return log_probs
# 定义一个余弦调度函数，根据给定的时间步数和参数s生成一组beta值
def cosine_beta_schedule(timesteps, s = 0.008):
    # 计算总步数
    steps = timesteps + 1
    # 在0到timesteps之间生成均匀间隔的值，作为x
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    # 根据余弦函数计算alpha的累积乘积
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    # 将alpha的累积乘积除以第一个元素，得到归一化后的值
    alphas_cumprod = alphas_cumprod / first(alphas_cumprod)
    # 计算beta值
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # 将beta值限制在0到0.999之间
    return torch.clip(betas, 0, 0.999)


# 定义一个线性调度函数，根据给定的时间步数生成一组beta值
def linear_beta_schedule(timesteps):
    # 计算比例尺
    scale = 1000 / timesteps
    # 计算起始beta值
    beta_start = scale * 0.0001
    # 计算结束beta值
    beta_end = scale * 0.02
    # 在起始和结束之间生成均匀间隔的值，作为beta值
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


# 定义一个二次调度函数，根据给定的时间步数生成一组beta值
def quadratic_beta_schedule(timesteps):
    # 计算比例尺
    scale = 1000 / timesteps
    # 计算起始beta值
    beta_start = scale * 0.0001
    # 计算结束beta值
    beta_end = scale * 0.02
    # 在起始和结束之间生成均匀间隔的值，然后取平方，作为beta值
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype = torch.float64) ** 2


# 定义一个sigmoid调度函数，根据给定的时间步数生成一组beta值
def sigmoid_beta_schedule(timesteps):
    # 计算比例尺
    scale = 1000 / timesteps
    # 计算起始beta值
    beta_start = scale * 0.0001
    # 计算结束beta值
    beta_end = scale * 0.02
    # 在-6到6之间生成均匀间隔的值，作为betas
    betas = torch.linspace(-6, 6, timesteps, dtype = torch.float64)
    # 对betas应用sigmoid函数，然后乘以结束和起始之间的差值，再加上起始值，得到最终的beta值
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# 定义一个噪声调度器类
class NoiseScheduler(nn.Module):
    # 初始化函数，设置参数和计算beta值
    def __init__(self, *, beta_schedule, timesteps, loss_type, p2_loss_weight_gamma = 0., p2_loss_weight_k = 1):
        # 调用父类的初始化函数
        super().__init__()

        # 根据不同的beta调度方式计算beta值
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        # 计算alphas值
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        # 获取时间步数并设置为类属性
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # 根据损失类型选择损失函数
        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        # 设置损失类型和损失函数为类属性
        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # 注册缓冲区辅助函数，将double类型转换为float类型
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # 注册各种缓冲区
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 计算后验分布的方差
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 设置是否进行p2损失重新加权的标志和p2损失权重
        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    # 生成随机时间步
    def sample_random_times(self, batch):
        return torch.randint(0, self.num_timesteps, (batch,), device = self.betas.device, dtype = torch.long)

    # 计算后验分布
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 从q分布中采样
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    # 计算给定时间点 t 的速度 v
    def calculate_v(self, x_start, t, noise = None):
        # 使用累积平方根 alpha 乘以噪声，减去累积平方根 1-alpha 乘以起始位置 x_start
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    # 从起始位置 x_from 到目标时间 to_t 的采样
    def q_sample_from_to(self, x_from, from_t, to_t, noise = None):
        shape = x_from.shape
        noise = default(noise, lambda: torch.randn_like(x_from))

        # 提取累积平方根 alpha 和 1-alpha
        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)

        # 计算采样结果
        return x_from * (alpha_next / alpha) + noise * (sigma_next * alpha - sigma * alpha_next) / alpha

    # 根据速度 v 预测起始位置
    def predict_start_from_v(self, x_t, t, v):
        # 使用累积平方根 alpha 乘以当前位置 x_t，减去累积平方根 1-alpha 乘以速度 v
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    # 根据噪声预测起始位置
    def predict_start_from_noise(self, x_t, t, noise):
        # 使用倒数累积平方根 alpha 乘以当前位置 x_t，减去倒数累积平方根 alpha-1 乘以噪声
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # 根据起始位置和当前位置预测噪声
    def predict_noise_from_start(self, x_t, t, x0):
        # 使用倒数累积平方根 alpha 乘以当前位置 x_t 减去起始位置 x0，再除以倒数累积平方根 alpha-1
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # 对损失进行 P2 重加权
    def p2_reweigh_loss(self, loss, times):
        # 如果没有 P2 损失重加权，则直接返回原始损失
        if not self.has_p2_loss_reweighting:
            return loss
        # 返回损失乘以 P2 损失权重
        return loss * extract(self.p2_loss_weight, times, loss.shape)
# 重新排列图像为序列

class RearrangeToSequence(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = rearrange(x, 'b c ... -> b ... c')  # 重新排列输入张量的维度
        x, ps = pack([x], 'b * c')  # 打包张量

        x = self.fn(x)  # 使用给定的函数处理张量

        x, = unpack(x, ps, 'b * c')  # 解包张量
        x = rearrange(x, 'b ... c -> b c ...')  # 重新排列输出张量的维度
        return x

# 扩散先验

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5, fp16_eps = 1e-3, stable = False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim = -1, keepdim = True).detach()

        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5, fp16_eps = 1e-3, stable = False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim = 1, keepdim = True).detach()

        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 多层感知机

class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        expansion_factor = 2.,
        depth = 2,
        norm = False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.SiLU(),
            norm_fn()
        )]

        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                norm_fn()
            ))

        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())

# 因果变换器的相对位置偏差

class RelPosBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)
    # 前向传播函数，接受输入参数 i, j 和 device
    def forward(self, i, j, *, device):
        # 生成一个从 0 到 i-1 的长整型张量，使用指定设备
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        # 生成一个从 0 到 j-1 的长整型张量，使用指定设备
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        # 计算相对位置矩阵，即 k_pos 和 q_pos 的差值
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        # 将相对位置矩阵映射到指定的桶中，使用 self._relative_position_bucket 方法
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        # 计算相对位置注意力偏置，使用 self.relative_attention_bias 方法
        values = self.relative_attention_bias(rp_bucket)
        # 重新排列结果张量的维度，将 'i j h' 转换为 'h i j'
        return rearrange(values, 'i j h -> h i j')
# 定义一个 SwiGLU 类，用于前向传播
class SwiGLU(nn.Module):
    """ 在 https://arxiv.org/abs/2204.0231 中成功使用 """
    def forward(self, x):
        # 将输入张量 x 按照最后一个维度分成两部分
        x, gate = x.chunk(2, dim = -1)
        # 返回经过门控线性单元激活函数处理后的结果
        return x * F.silu(gate)

# 定义一个 FeedForward 函数，用于创建前馈神经网络
def FeedForward(
    dim,
    mult = 4,
    dropout = 0.,
    post_activation_norm = False
):
    """ 后激活归一化 https://arxiv.org/abs/2110.09456 """

    # 计算内部维度
    inner_dim = int(mult * dim)
    # 返回一个包含多个层的神经网络模型
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# 定义一个 Attention 类，用于实现注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        rotary_emb = None,
        cosine_sim = True,
        cosine_sim_scale = 16
    ):
        super().__init__()
        # 初始化注意力机制的参数
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # 旋转嵌入

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # 添加空键/值以用于先验网络中的无分类器引导

        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # 是否使用余弦相似度

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # 计算查询/键的相似性

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # 相对位置编码（T5 风格）

        if exists(attn_bias):
            sim = sim + attn_bias

        # 掩码

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # 注意力

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        # 聚合值

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义一个 CausalTransformer 类，用于实现因果变换器
class CausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True
    ): 
        # 调用父类的构造函数
        super().__init__()
        # 如果需要进行输入层归一化，则初始化 LayerNorm 对象，否则使用 nn.Identity()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        # 初始化相对位置偏置对象
        self.rel_pos_bias = RelPosBias(heads = heads)

        # 如果需要旋转嵌入，则初始化 RotaryEmbedding 对象，否则为 None
        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        # 初始化多层 Transformer 模块
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 每层包含注意力机制和前馈神经网络
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        # 如果需要输出层归一化，则初始化 LayerNorm 对象，否则使用 nn.Identity()
        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        # 如果需要最终投影，则初始化线性层，否则使用 nn.Identity()
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        # 获取输入张量的长度和设备信息
        n, device = x.shape[1], x.device

        # 对输入张量进行初始归一化处理
        x = self.init_norm(x)

        # 计算注意力偏置
        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        # 遍历每一层 Transformer 模块
        for attn, ff in self.layers:
            # 执行注意力机制和前馈神经网络操作
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        # 对输出结果进行归一化处理
        out = self.norm(x)
        # 返回最终输出结果
        return self.project_out(out)
# 定义一个名为 DiffusionPriorNetwork 的神经网络模块
class DiffusionPriorNetwork(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        num_timesteps = None,
        num_time_embeds = 1,
        num_image_embeds = 1,
        num_text_embeds = 1,
        max_text_len = 256,
        self_cond = False,
        **kwargs
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置维度属性
        self.dim = dim

        # 设置时间嵌入、图像嵌入和文本嵌入的数量
        self.num_time_embeds = num_time_embeds
        self.num_image_embeds = num_image_embeds
        self.num_text_embeds = num_text_embeds

        # 将输入转换为文本嵌入
        self.to_text_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_text_embeds) if num_text_embeds > 1 else nn.Identity(),
            Rearrange('b (n d) -> b n d', n = num_text_embeds)
        )

        # 检查是否存在时间步长
        self.continuous_embedded_time = not exists(num_timesteps)

        # 将输入转换为时间嵌入
        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # 将输入转换为图像嵌入
        self.to_image_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_image_embeds) if num_image_embeds > 1 else nn.Identity(),
            Rearrange('b (n d) -> b n d', n = num_image_embeds)
        )

        # 学习查询向量
        self.learned_query = nn.Parameter(torch.randn(dim))
        # 创建因果变换器
        self.causal_transformer = CausalTransformer(dim = dim, **kwargs)

        # dalle1 学习的填充策略

        # 设置最大文本长度
        self.max_text_len = max_text_len

        # 创建空文本编码和空文本嵌入
        self.null_text_encodings = nn.Parameter(torch.randn(1, max_text_len, dim))
        self.null_text_embeds = nn.Parameter(torch.randn(1, num_text_embeds, dim))
        self.null_image_embed = nn.Parameter(torch.randn(1, dim))

        # 是否使用自我条件，Hinton 的团队的新 ddpm 技术

        self.self_cond = self_cond

    # 带有条件缩放的前向传播函数
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        # 调用前向传播函数
        logits = self.forward(*args, **kwargs)

        # 如果条件缩放为1���则直接返回logits
        if cond_scale == 1:
            return logits

        # 计算空logits
        null_logits = self.forward(*args, text_cond_drop_prob = 1., image_cond_drop_prob = 1, **kwargs)
        # 返回经过条件缩放后的logits
        return null_logits + (logits - null_logits) * cond_scale

    # 前向传播函数
    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        text_embed,
        text_encodings = None,
        self_cond = None,
        text_cond_drop_prob = 0.,
        image_cond_drop_prob = 0.
        ):
            # 解包图像嵌入的批次大小、维度、设备和数据类型
            batch, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype

            # 获取时间嵌入、图像嵌入和文本嵌入的数量
            num_time_embeds, num_image_embeds, num_text_embeds = self.num_time_embeds, self.num_image_embeds, self.num_text_embeds

            # 设置自身条件

            if self.self_cond:
                # 如果存在自身条件，则创建一个全零张量
                self_cond = default(self_cond, lambda: torch.zeros(batch, self.dim, device = device, dtype = dtype))
                self_cond = rearrange(self_cond, 'b d -> b 1 d')

            # 在第2.2节，最后一段
            # "... 包括编码文本、CLIP文本嵌入、扩散时间步嵌入、噪声CLIP图像嵌入、用于预测的最终嵌入"

            # 将文本嵌入转换为所需格式
            text_embed = self.to_text_embeds(text_embed)
            # 将图像嵌入转换为所需格式
            image_embed = self.to_image_embeds(image_embed)

            # 分类器自由引导掩码

            # 创建文本保留掩码
            text_keep_mask = prob_mask_like((batch,), 1 - text_cond_drop_prob, device = device)
            text_keep_mask = rearrange(text_keep_mask, 'b -> b 1 1')

            # 创建图像保留掩码
            image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device = device)
            image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

            # 使文本编码变为可选
            # 尽管论文似乎暗示它是存在的 <--

            if not exists(text_encodings):
                text_encodings = torch.empty((batch, 0, dim), device = device, dtype = dtype)
        
            # 创建一个掩码，用于检测文本编码中的填充
            mask = torch.any(text_encodings != 0., dim = -1)

            # 用学习填充令牌替换文本编码中的任何填充
            text_encodings = text_encodings[:, :self.max_text_len]
            mask = mask[:, :self.max_text_len]

            text_len = text_encodings.shape[-2]
            remainder = self.max_text_len - text_len

            if remainder > 0:
                text_encodings = F.pad(text_encodings, (0, 0, 0, remainder), value = 0.)
                mask = F.pad(mask, (0, remainder), value = False)

            # 使用空编码屏蔽文本编码
            null_text_encodings = self.null_text_encodings.to(text_encodings.dtype)

            text_encodings = torch.where(
                rearrange(mask, 'b n -> b n 1').clone() & text_keep_mask,
                text_encodings,
                null_text_encodings
            )

            # 使用空文本嵌入屏蔽文本嵌入
            null_text_embeds = self.null_text_embeds.to(text_embed.dtype)

            text_embed = torch.where(
                text_keep_mask,
                text_embed,
                null_text_embeds
            )

            # 使用空图像嵌入屏蔽图像嵌入
            null_image_embed = self.null_image_embed.to(image_embed.dtype)

            image_embed = torch.where(
                image_keep_mask,
                image_embed,
                null_image_embed
            )

            # 文本嵌入是否用于条件取决于是否文本编码可用于注意力（对于分类器自由引导，尽管从论文中看出先前的ddpm未使用，因为目标不同）
            # 但让我们做正确的事情

            if self.continuous_embedded_time:
                diffusion_timesteps = diffusion_timesteps.type(dtype)

            # 将时间嵌入转换为所需格式
            time_embed = self.to_time_embeds(diffusion_timesteps)

            # 重复学习的查询，以预测图像嵌入（每个DDPM时间步）
            learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

            if self.self_cond:
                learned_queries = torch.cat((self_cond, learned_queries), dim = -2)

            # 将各种嵌入拼接在一起
            tokens = torch.cat((
                text_encodings,
                text_embed,
                time_embed,
                image_embed,
                learned_queries
            ), dim = -2)

            # 注意力机制
            tokens = self.causal_transformer(tokens)

            # 获取学习的查询，应该预测图像嵌入（每个DDPM时间步）
            pred_image_embed = tokens[..., -1, :]

            return pred_image_embed
# 定义一个 DiffusionPrior 类，继承自 nn.Module
class DiffusionPrior(nn.Module):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        net,
        *,
        clip = None,  # 用于裁剪梯度的阈值
        image_embed_dim = None,  # 图像嵌入维度
        image_size = None,  # 图像尺寸
        image_channels = 3,  # 图像通道数，默认为3
        timesteps = 1000,  # 时间步数
        sample_timesteps = None,  # 采样时间步数
        cond_drop_prob = 0.,  # 条件丢弃概率
        text_cond_drop_prob = None,  # 文本条件丢弃概率
        image_cond_drop_prob = None,  # 图像条件丢弃概率
        loss_type = "l2",  # 损失类型，默认为 l2
        predict_x_start = True,  # 是否预测 x 的起始值
        predict_v = False,  # 是否预测速度
        beta_schedule = "cosine",  # beta 调度方式
        condition_on_text_encodings = True,  # 是否在文本编码上进行条件化，论文建议开启，但可以在 CLIP 预处理文本嵌入到图像嵌入训练中关闭
        sampling_clamp_l2norm = False,  # 是否在每个去噪迭代中对图像嵌入进行 l2 范数裁剪（类似于通常 DDPMs 的 -1 到 1 裁剪）
        sampling_final_clamp_l2norm = False,  # 是否对最终图像嵌入输出进行 l2 范数裁剪（这也适用于 DDPM 中的图像）
        training_clamp_l2norm = False,  # 是否在训练时对 l2 范数进行裁剪
        init_image_embed_l2norm = False,  # 是否初始化图像嵌入的 l2 范数
        image_embed_scale = None,  # 用于缩放 l2 范数的图像嵌入，使其更适合高斯扩散，由 Katherine (@crowsonkb) 在 https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132 中提出
        clip_adapter_overrides = dict()  # 用于覆盖 clip 适配器的字典
    ):
        # 调用父类的构造函数
        super().__init__()

        # 设置样本时间步数
        self.sample_timesteps = sample_timesteps

        # 创建噪声调度器对象
        self.noise_scheduler = NoiseScheduler(
            beta_schedule = beta_schedule,
            timesteps = timesteps,
            loss_type = loss_type
        )

        # 如果指定了 clip 参数
        if exists(clip):
            # 检查图像通道数是否与 clip 接受的通道数相同
            assert image_channels == clip.image_channels, f'channels of image ({image_channels}) should be equal to the channels that CLIP accepts ({clip.image_channels})'

            # 根据 clip 的类型进行适配
            if isinstance(clip, CLIP):
                clip = XClipAdapter(clip, **clip_adapter_overrides)
            elif isinstance(clip, CoCa):
                clip = CoCaAdapter(clip, **clip_adapter_overrides)

            # 断言 clip 是 BaseClipAdapter 类型
            assert isinstance(clip, BaseClipAdapter)
            # 冻结模型并设置为评估模式
            freeze_model_and_make_eval_(clip)
            self.clip = clip
        else:
            # 如果未指定 clip 参数，则需要指定图像嵌入维度
            assert exists(image_embed_dim), 'latent dimension must be given, if training prior network without CLIP given'
            self.clip = None

        # 设置网络和图像嵌入维度
        self.net = net
        self.image_embed_dim = default(image_embed_dim, lambda: clip.dim_latent)

        # 断言网络维度与图像嵌入维度相同
        assert net.dim == self.image_embed_dim, f'your diffusion prior network has a dimension of {net.dim}, but you set your image embedding dimension (keyword image_embed_dim) on DiffusionPrior to {self.image_embed_dim}'
        # 断言 clip 的潜在维度与图像嵌入维度相同
        assert not exists(clip) or clip.dim_latent == self.image_embed_dim, f'you passed in a CLIP to the diffusion prior with latent dimensions of {clip.dim_latent}, but your image embedding dimension (keyword image_embed_dim) for the DiffusionPrior was set to {self.image_embed_dim}'

        # 设置通道数
        self.channels = default(image_channels, lambda: clip.image_channels)

        # 设置文本条件丢弃概率和图像条件丢弃概率
        self.text_cond_drop_prob = default(text_cond_drop_prob, cond_drop_prob)
        self.image_cond_drop_prob = default(image_cond_drop_prob, cond_drop_prob)

        # 是否使用分类器指导
        self.can_classifier_guidance = self.text_cond_drop_prob > 0. and self.image_cond_drop_prob > 0.
        self.condition_on_text_encodings = condition_on_text_encodings

        # 在论文中，他们不预测噪声，而是直接为图像嵌入预测 x0，声称实验结果更好。我将提供两者。

        self.predict_x_start = predict_x_start
        self.predict_v = predict_v # 优先于 predict_x_start

        # @crowsonkb 的建议 - https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132

        # 设置图像嵌入缩放因子
        self.image_embed_scale = default(image_embed_scale, self.image_embed_dim ** 0.5)

        # 是否在采样时强制进行 l2norm，类似于裁剪去噪时的操作

        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.sampling_final_clamp_l2norm = sampling_final_clamp_l2norm

        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

        # 设备跟踪器

        self.register_buffer('_dummy', torch.tensor([True]), persistent = False)

    @property
    def device(self):
        # 返回设备信息
        return self._dummy.device

    # 对图像嵌入进行 l2norm 裁剪
    def l2norm_clamp_embed(self, image_embed):
        return l2norm(image_embed) * self.image_embed_scale
    # 计算预测的均值、后验方差和后验对数方差，以及起始值
    def p_mean_variance(self, x, t, text_cond, self_cond = None, clip_denoised = False, cond_scale = 1.):
        # 断言条件，如果条件不成立则抛出异常
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'
        
        # 使用网络进行预测，根据条件缩放和文本条件
        pred = self.net.forward_with_cond_scale(x, t, cond_scale = cond_scale, self_cond = self_cond, **text_cond)

        # 根据预测值选择起始值
        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t = t, v = pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)

        # 如果需要剪裁去噪后的值，并且不是预测 x 的起始值
        if clip_denoised and not self.predict_x_start:
            x_start.clamp_(-1., 1.)

        # 如果预测 x 的起始值并且采样剪裁 L2 范数
        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.image_embed_scale

        # 获取模型均值、后验方差和后验对数方差
        model_mean, posterior_variance, posterior_log_variance = self.noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # 生成样本
    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1.):
        # 获取输入 x 的形状和设备信息
        b, *_, device = *x.shape, x.device
        # 计算模型均值、模型方差和模型对数方差，以及起始值
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        # 生成噪声
        noise = torch.randn_like(x)
        # 当 t == 0 时不添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # 根据模型均值、模型对数方差和噪声生成预测值
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    # 循环生成样本
    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale = 1.):
        # 获取批量大小和设备信息
        batch, device = shape[0], self.device

        # 生成随机图像嵌入
        image_embed = torch.randn(shape, device = device)
        x_start = None # 用于自我条件

        # 如果初始化图像嵌入的 L2 范数
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        # 遍历时间步骤，生成样本
        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond = text_cond, self_cond = self_cond, cond_scale = cond_scale)

        # 如果采样最终剪裁 L2 范数并且预测 x 的起始值
        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    # 无梯度计算
    @torch.no_grad()
    # 定义一个函数，用于在动态图像生成中循环采样，支持不同维度的输入
    def p_sample_loop_ddim(self, shape, text_cond, *, timesteps, eta = 1., cond_scale = 1.):
        # 获取输入形状的相关信息
        batch, device, alphas, total_timesteps = shape[0], self.device, self.noise_scheduler.alphas_cumprod_prev, self.noise_scheduler.num_timesteps

        # 在指定时间范围内生成时间序列
        times = torch.linspace(-1., total_timesteps, steps = timesteps + 1)[:-1]

        # 将时间序列反转并转换为整数列表
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # 生成随机的图像嵌入向量
        image_embed = torch.randn(shape, device = device)

        x_start = None # 用于自条件生成

        # 如果需要对初始图像嵌入向量进行 L2 范数归一化
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        # 在时间序列上进行循环采样
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = alphas[time]
            alpha_next = alphas[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None

            # 使用条件信息生成预测结果
            pred = self.net.forward_with_cond_scale(image_embed, time_cond, self_cond = self_cond, cond_scale = cond_scale, **text_cond)

            # 推导 x0

            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(image_embed, t = time_cond, v = pred)
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(image_embed, t = time_cond, noise = pred)

            # 在可能预测噪声之前对 x0 进行裁剪

            if not self.predict_x_start:
                x_start.clamp_(-1., 1.)

            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            # 预测噪声

            pred_noise = self.noise_scheduler.predict_noise_from_start(image_embed, t = time_cond, x0 = x_start)

            if time_next < 0:
                image_embed = x_start
                continue

            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(image_embed) if time_next > 0 else 0.

            image_embed = x_start * alpha_next.sqrt() + \
                          c1 * noise + \
                          c2 * pred_noise

        # 如果需要对最终的图像嵌入向量进行 L2 范数归一化
        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    # 用于在动态图像生成中循环采样的函数，支持不同维度的输入
    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps = None, **kwargs):
        # 如果未指定时间步长，则使用默认值
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        # 根据是否为低维输入选择不同的采样函数
        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps = timesteps)

        # 对图像嵌入向量进行缩放处理并返回
        image_embed = normalized_image_embed / self.image_embed_scale
        return image_embed
    # 定义一个函数，计算损失值
    def p_losses(self, image_embed, times, text_cond, noise = None):
        # 如果没有提供噪声，则生成一个默认的噪声
        noise = default(noise, lambda: torch.randn_like(image_embed))

        # 使用噪声调度器生成噪声图像嵌入
        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        self_cond = None
        # 如果网络支持自身条件，并且随机数小于0.5
        if self.net.self_cond and random.random() < 0.5:
            # 使用网络生成自身条件
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        # 使用网络进行预测
        pred = self.net(
            image_embed_noisy,
            times,
            self_cond = self_cond,
            text_cond_drop_prob = self.text_cond_drop_prob,
            image_cond_drop_prob = self.image_cond_drop_prob,
            **text_cond
        )

        # 如果需要预测起始图像并且训练时使用L2范数约束
        if self.predict_x_start and self.training_clamp_l2norm:
            # 对预测结果进行L2范数约束
            pred = self.l2norm_clamp_embed(pred)

        # 如果需要预测速度
        if self.predict_v:
            # 计算目标速度
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        # 如果需要预测起始图像
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        # 计算损失值
        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss

    # 生成一个批次的图像
    @torch.no_grad()
    @eval_decorator
    def sample_batch_size(self, batch_size, text_cond, cond_scale = 1.):
        # 获取设备信息
        device = self.betas.device
        shape = (batch_size, self.image_embed_dim)

        # 生成随机噪声图像
        img = torch.randn(shape, device = device)

        # 对于每个时间步长，生成图像
        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc = 'sampling loop time step', total = self.noise_scheduler.num_timesteps):
            img = self.p_sample(img, torch.full((batch_size,), i, device = device, dtype = torch.long), text_cond = text_cond, cond_scale = cond_scale)
        return img

    # 生成样本
    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        text,
        num_samples_per_batch = 2,
        cond_scale = 1.,
        timesteps = None
    ):
        timesteps = default(timesteps, self.sample_timesteps)

        # 重复文本以匹配样本数
        text = repeat(text, 'b ... -> (b r) ...', r = num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        # 嵌入文本
        text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # 生成图像嵌入
        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), text_cond = text_cond, cond_scale = cond_scale, timesteps = timesteps)

        # 计算文本和图像之间的相似度
        text_embeds = text_cond['text_embed']
        text_embeds = rearrange(text_embeds, '(b r) d -> b r d', r = num_samples_per_batch)
        image_embeds = rearrange(image_embeds, '(b r) d -> b r d', r = num_samples_per_batch)
        text_image_sims = einsum('b r d, b r d -> b r', l2norm(text_embeds), l2norm(image_embeds)
        top_sim_indices = text_image_sims.topk(k = 1).indices
        top_sim_indices = repeat(top_sim_indices, 'b 1 -> b 1 d', d = image_embed_dim)
        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, 'b 1 d -> b d')

    # 前向传播函数
    def forward(
        self,
        text = None,
        image = None,
        text_embed = None,      # 允许在预处理的CLIP文本和图像嵌入上进行训练
        image_embed = None,
        text_encodings = None,  # 以及CLIP文本编码
        *args,
        **kwargs
        # 检查是否提供了文本或文本嵌入，二者必须有一个
        assert exists(text) ^ exists(text_embed), 'either text or text embedding must be supplied'
        # 检查是否提供了图像或图像嵌入，二者必须有一个
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        # 如果在初始化时指定了要在文本编码上进行条件化，则文本编码必须存在
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        # 如果提供了图像，则使用CLIP模型嵌入图像
        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # 根据传入的内容计算文本条件
        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        # 创建文本条件字典
        text_cond = dict(text_embed = text_embed)

        # 如果在文本编码上进行条件化，则文本编码必须存在
        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # 从ddpm中获取时间步条件
        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # 缩放图像嵌入
        image_embed *= self.image_embed_scale

        # 计算前向损失
        return self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)
# 定义一个最近邻上采样模块，将输入维度提升为指定的输出维度
def NearestUpsample(dim, dim_out = None):
    # 如果未指定输出维度，则默认与输入维度相同
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        # 使用最近邻插值方式上采样，比例为2
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        # 使用3x3卷积核进行卷积，将输入维度转换为输出维度
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

# 定义一个像素混洗上采样模块，用于解决棋盘伪影问题
class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        # 如果未指定输出维度，则默认与输入维度相同
        dim_out = default(dim_out, dim)
        # 使用1x1卷积核将输入维度转换为输出维度的4倍
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            # 进行卷积操作
            conv,
            # 使用SiLU激活函数
            nn.SiLU(),
            # 像素混洗操作，将通道数减少为原来的四分之一
            nn.PixelShuffle(2)
        )

        # 初始化卷积层的权重
        self.init_conv_(conv)

    # 初始化卷积层的权重
    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(self, x):
        return self.net(x)

# 定义一个下采样模块，采用最优的像素解开操作
def Downsample(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 显示这是最优的下采样方式
    # 在论文中被称为SP-conv，实际上是像素解开操作
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        # 像素解开操作，将每个像素分成4个像素
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        # 使用1x1卷积核将输入维度转换为输出维度
        nn.Conv2d(dim * 4, dim_out, 1)
    )

# 定义一个权重标准化的卷积层
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        flattened_weights = rearrange(weight, 'o ... -> o (...)')

        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')

        var = torch.var(flattened_weights, dim = -1, unbiased = False)
        var = rearrange(var, 'o -> o 1 1 1')

        weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# 定义一个正弦位置编码模块
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert is_float_dtype(dtype), 'input to sinusoidal pos emb must be a float type'

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device, dtype = dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1).type(dtype)

# 定义一个块模块
class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        weight_standardization = False
    ):
        super().__init__()
        conv_klass = nn.Conv2d if not weight_standardization else WeightStandardizedConv2d

        # 使用3x3卷积核进行卷积，将输入维度转换为输出维度
        self.project = conv_klass(dim, dim_out, 3, padding = 1)
        # 使用组归一化进行归一化
        self.norm = nn.GroupNorm(groups, dim_out)
        # 使用SiLU激活函数
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.project(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        groups = 8,
        weight_standardization = False,
        cosine_sim_cross_attn = False
    # 初始化函数，继承父类的初始化方法
    def __init__(
        super().__init__()

        # 初始化时间多层感知器为 None
        self.time_mlp = None

        # 如果时间条件维度存在
        if exists(time_cond_dim):
            # 创建时间多层感知器模型
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        # 初始化交叉注意力为 None
        self.cross_attn = None

        # 如果条件维度存在
        if exists(cond_dim):
            # 创建交叉注意力模型
            self.cross_attn = CrossAttention(
                dim = dim_out,
                context_dim = cond_dim,
                cosine_sim = cosine_sim_cross_attn
            )

        # 创建第一个块
        self.block1 = Block(dim, dim_out, groups = groups, weight_standardization = weight_standardization)
        # 创建第二个块
        self.block2 = Block(dim_out, dim_out, groups = groups, weight_standardization = weight_standardization)
        # 如果输入维度不等于输出维度，创建卷积层；否则创建恒等映射
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # 前向传播函数
    def forward(self, x, time_emb = None, cond = None):

        # 初始化缩放和平移为 None
        scale_shift = None
        # 如果时间多层感知器和时间嵌入都存在
        if exists(self.time_mlp) and exists(time_emb):
            # 通过时间多层感知器处理时间嵌入
            time_emb = self.time_mlp(time_emb)
            # 重新排列时间嵌入的维度
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # 将处理后的时间嵌入分成两部分，分别表示缩放和平移
            scale_shift = time_emb.chunk(2, dim = 1)

        # 使用第一个块处理输入数据
        h = self.block1(x, scale_shift = scale_shift)

        # 如果交叉注意力存在
        if exists(self.cross_attn):
            # 确保条件存在
            assert exists(cond)

            # 重新排列隐藏状态的维度
            h = rearrange(h, 'b c ... -> b ... c')
            # 打包隐藏状态
            h, ps = pack([h], 'b * c')

            # 使用交叉注意力处理隐藏状态
            h = self.cross_attn(h, context = cond) + h

            # 解包隐藏状态
            h, = unpack(h, ps, 'b * c')
            # 重新排列隐藏状态的维度
            h = rearrange(h, 'b ... c -> b c ...')

        # 使用第二个块处理隐藏状态
        h = self.block2(h)
        # 返回最终结果，加上残差连接
        return h + self.res_conv(x)
# 定义交叉注意力模块
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        norm_context = False,
        cosine_sim = False,
        cosine_sim_scale = 16
    ):
        super().__init__()
        self.cosine_sim = cosine_sim
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b h 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义线性注意力模块
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.GELU()
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        seq_len = x * y

        fmap = self.norm(fmap)
        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale
        v = l2norm(v)

        k, v = map(lambda t: t / math.sqrt(seq_len), (k, v))

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

# 定义交叉嵌入层模块
class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    # 初始化函数，继承父类的初始化方法
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 断言所有卷积核大小与步长的奇偶性相同
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        # 如果未指定输出维度，则与输入维度相同
        dim_out = default(dim_out, dim_in)

        # 对卷积核大小进行排序
        kernel_sizes = sorted(kernel_sizes)
        # 计算总共有多少个尺度
        num_scales = len(kernel_sizes)

        # 计算每个尺度的维度
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        # 最后一个尺度的维度为总维度减去前面各尺度的维度之和
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        # 创建卷积层列表
        self.convs = nn.ModuleList([])
        # 遍历卷积核大小和对应的尺度维度
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            # 将每个尺度的卷积层添加到列表中
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    # 前向传播函数
    def forward(self, x):
        # 对输入数据进行多尺度卷积操作，得到特征图元组
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        # 在通道维度上拼接特征图
        return torch.cat(fmaps, dim = 1)
class UpsampleCombiner(nn.Module):
    # 定义一个 UpsampleCombiner 类，继承自 nn.Module
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple()
    ):
        # 初始化函数，接受维度 dim 和一些可选参数
        super().__init__()
        # 调用父类的初始化函数
        assert len(dim_ins) == len(dim_outs)
        # 断言输入维度和输出维度的长度相等
        self.enabled = enabled
        # 设置是否启用的标志

        if not self.enabled:
            # 如果未启用
            self.dim_out = dim
            # 设置输出维度为输入维度
            return

        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        # 使用输入维度和输出维度创建 Block 对象列表
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)
        # 设置输出维度为输入维度加上所有输出维度之和

    def forward(self, x, fmaps = None):
        # 前向传播函数，接受输入 x 和特征图列表 fmaps，默认为 None
        target_size = x.shape[-1]
        # 获取输入 x 的最后一个维度大小

        fmaps = default(fmaps, tuple())
        # 如果 fmaps 为 None，则设置为空元组

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            # 如果未启用或者 fmaps 为空或者 fmap_convs 为空
            return x
            # 返回输入 x

        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        # 调整特征图大小为目标大小
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        # 对每个特征图应用对应的卷积操作
        return torch.cat((x, *outs), dim = 1)
        # 沿着指定维度拼接输入 x 和处理后的特征图列表

class Unet(nn.Module):
    # 定义一个 Unet 类，继承自 nn.Module
    def __init__(
        self,
        dim,
        *,
        image_embed_dim = None,
        text_embed_dim = None,
        cond_dim = None,
        num_image_tokens = 4,
        num_time_tokens = 2,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        channels_out = None,
        self_attn = False,
        attn_dim_head = 32,
        attn_heads = 16,
        lowres_cond = False,             # for cascading diffusion - https://cascaded-diffusion.github.io/
        lowres_noise_cond = False,       # for conditioning on low resolution noising, based on Imagen
        self_cond = False,               # set this to True to use the self-conditioning technique from - https://arxiv.org/abs/2208.04202
        sparse_attn = False,
        cosine_sim_cross_attn = False,
        cosine_sim_self_attn = False,
        attend_at_middle = True,         # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        cond_on_text_encodings = False,
        max_text_len = 256,
        cond_on_image_embeds = False,
        add_image_embeds_to_time = True, # alerted by @mhh0318 to a phrase in the paper - "Specifically, we modify the architecture described in Nichol et al. (2021) by projecting and adding CLIP embeddings to the existing timestep embedding"
        init_dim = None,
        init_conv_kernel_size = 7,
        resnet_groups = 8,
        resnet_weight_standardization = False,
        num_resnet_blocks = 2,
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        memory_efficient = False,
        scale_skip_connection = False,
        pixel_shuffle_upsample = True,
        final_conv_kernel_size = 1,
        combine_upsample_fmaps = False, # whether to combine the outputs of all upsample blocks, as in unet squared paper
        checkpoint_during_training = False,
        **kwargs
    # 定义初始化函数，接受一系列参数

    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        lowres_noise_cond,
        channels,
        channels_out,
        cond_on_image_embeds,
        cond_on_text_encodings,
    # 如果当前模型参数与输入参数相同，则返回当前模型
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_image_embeds == self.cond_on_image_embeds and \
            cond_on_text_encodings == self.cond_on_text_encodings and \
            lowres_noise_cond == self.lowres_noise_cond and \
            channels_out == self.channels_out:
            return self

        # 更新参数字典
        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            channels = channels,
            channels_out = channels_out,
            cond_on_image_embeds = cond_on_image_embeds,
            cond_on_text_encodings = cond_on_text_encodings,
            lowres_noise_cond = lowres_noise_cond
        )

        # 返回一个新的类实例，使用当前模型的局部变量和更新后的参数
        return self.__class__(**{**self._locals, **updated_kwargs})

    # 带有条件缩放的前向传播函数
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        # 调用前向传播函数获取 logits
        logits = self.forward(*args, **kwargs)

        # 如果条件缩放因子为1，则直接返回 logits
        if cond_scale == 1:
            return logits

        # 计算无条件 logits
        null_logits = self.forward(*args, text_cond_drop_prob = 1., image_cond_drop_prob = 1., **kwargs)
        # 返回加权后的 logits
        return null_logits + (logits - null_logits) * cond_scale

    # 前向传播函数
    def forward(
        self,
        x,
        time,
        *,
        image_embed,
        lowres_cond_img = None,
        lowres_noise_level = None,
        text_encodings = None,
        image_cond_drop_prob = 0.,
        text_cond_drop_prob = 0.,
        blur_sigma = None,
        blur_kernel_size = None,
        disable_checkpoint = False,
        self_cond = None
# 定义一个低分辨率条件器的类，继承自 nn.Module
class LowresConditioner(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        downsample_first = True,  # 是否先降采样
        use_blur = True,  # 是否使用模糊
        blur_prob = 0.5,  # 模糊概率
        blur_sigma = 0.6,  # 模糊标准差
        blur_kernel_size = 3,  # 模糊核大小
        use_noise = False,  # 是否使用噪声
        input_image_range = None,  # 输入图像范围
        normalize_img_fn = identity,  # 图像归一化函数
        unnormalize_img_fn = identity  # 图像反归一化函数
    ):
        super().__init__()  # 调用父类的初始化函数
        self.downsample_first = downsample_first  # 是否先降采样
        self.input_image_range = input_image_range  # 输入图像范围

        self.use_blur = use_blur  # 是否使用模糊
        self.blur_prob = blur_prob  # 模糊概率
        self.blur_sigma = blur_sigma  # 模糊标准差
        self.blur_kernel_size = blur_kernel_size  # 模糊核大小

        self.use_noise = use_noise  # 是否使用噪声
        self.normalize_img = normalize_img_fn  # 图像归一化函数
        self.unnormalize_img = unnormalize_img_fn  # 图像反归一化函数
        self.noise_scheduler = NoiseScheduler(beta_schedule = 'linear', timesteps = 1000, loss_type = 'l2') if use_noise else None  # 噪声调度器

    # 添加噪声到图像
    def noise_image(self, cond_fmap, noise_levels = None):
        assert exists(self.noise_scheduler)  # 断言噪声调度器存在

        batch = cond_fmap.shape[0]  # 批次大小
        cond_fmap = self.normalize_img(cond_fmap)  # 归一化图像

        random_noise_levels = default(noise_levels, lambda: self.noise_scheduler.sample_random_times(batch))  # 随机噪声级别
        cond_fmap = self.noise_scheduler.q_sample(cond_fmap, t = random_noise_levels, noise = torch.randn_like(cond_fmap))  # 添加噪声

        cond_fmap = self.unnormalize_img(cond_fmap)  # 反归一化图像
        return cond_fmap, random_noise_levels  # 返回添加噪声后的图像和随机噪声级别

    # 前向传播函数
    def forward(
        self,
        cond_fmap,
        *,
        target_image_size,  # 目标图像大小
        downsample_image_size = None,  # 降采样图像大小
        should_blur = True,  # 是否应该模糊
        blur_sigma = None,  # 模糊标准差
        blur_kernel_size = None  # 模糊核大小
    ):
        if self.downsample_first and exists(downsample_image_size):  # 如果先降采样且降采样图像大小存在
            cond_fmap = resize_image_to(cond_fmap, downsample_image_size, clamp_range = self.input_image_range, nearest = True)  # 调整图像大小

        # 模糊只有50%的概率应用
        # 参考 https://arxiv.org/abs/2106.15282 中的第3.1节

        if self.use_blur and should_blur and random.random() < self.blur_prob:  # 如果使用模糊且应该模糊且随机数小于模糊概率
            # 在训练时，模糊低分辨率条件图像

            blur_sigma = default(blur_sigma, self.blur_sigma)  # 默认模糊标准差
            blur_kernel_size = default(blur_kernel_size, self.blur_kernel_size)  # 默认模糊核大小

            # 允许在 lo 和 hi 浮点值之间绘制随机标准差

            if isinstance(blur_sigma, tuple):  # 如果模糊标准差是元组
                blur_sigma = tuple(map(float, blur_sigma))  # 转换为浮点数元组
                blur_sigma = random.uniform(*blur_sigma)  # 在范围内随机选择一个值

            # 允许在 lo 和 hi 整数值之间绘制随机核大小

            if isinstance(blur_kernel_size, tuple):  # 如果模糊核大小是元组
                blur_kernel_size = tuple(map(int, blur_kernel_size))  # 转换为整数元组
                kernel_size_lo, kernel_size_hi = blur_kernel_size  # 获取最小和最大值
                blur_kernel_size = random.randrange(kernel_size_lo, kernel_size_hi + 1)  # 在范围内随机选择一个值

            cond_fmap = gaussian_blur2d(cond_fmap, cast_tuple(blur_kernel_size, 2), cast_tuple(blur_sigma, 2))  # 二维高斯模糊

        # 调整到目标图像大小

        cond_fmap = resize_image_to(cond_fmap, target_image_size, clamp_range = self.input_image_range, nearest = True)  # 调整图像大小

        # 噪声调节，如在 Imagen 中所做
        # 作为 BSR 噪声的替代，并可能替换第一阶段的模糊

        random_noise_levels = None  # 随机噪声级别为空

        if self.use_noise:  # 如果使用噪声
            cond_fmap, random_noise_levels = self.noise_image(cond_fmap)  # 添加噪声

        # 返回条件特征图，以及增强噪声级别

        return cond_fmap, random_noise_levels  # 返回条件特征图和随机噪声级别

# 解码器类
class Decoder(nn.Module):
    # 初始化函数，设置各种参数和默认值
    def __init__(
        self,
        unet,
        *,
        clip = None,                               # 剪辑参数
        image_size = None,                         # 图像大小
        channels = 3,                              # 通道数
        vae = tuple(),                             # 变分自动编码器
        timesteps = 1000,                          # 时间步数
        sample_timesteps = None,                   # 采样时间步数
        image_cond_drop_prob = 0.1,                # 图像条件概率
        text_cond_drop_prob = 0.5,                 # 文本条件概率
        loss_type = 'l2',                          # 损失类型
        beta_schedule = None,                      # beta调度
        predict_x_start = False,                   # 预测x的起始点
        predict_v = False,                         # 预测v
        predict_x_start_for_latent_diffusion = False,  # 用于潜在扩散的预测x的起始点
        image_sizes = None,                        # 用于级联ddpm，每个阶段的图像大小
        random_crop_sizes = None,                  # 是否在级联中随机裁剪图像
        use_noise_for_lowres_cond = False,         # 是否在低分辨率条件下使用噪声
        use_blur_for_lowres_cond = True,           # 是否在低分辨率条件下使用模糊
        lowres_downsample_first = True,            # 级联ddpm - 先缩小分辨率，然后到下一个条件分辨率+模糊
        blur_prob = 0.5,                           # 训练时，高斯模糊仅应用50%的时间
        blur_sigma = 0.6,                          # 模糊sigma
        blur_kernel_size = 3,                      # 模糊核大小
        lowres_noise_sample_level = 0.2,           # 在样本时间为低分辨率条件使用0.2的噪声水平
        clip_denoised = True,                      # 剪辑去噪
        clip_x_start = True,                       # 剪辑x的起始点
        clip_adapter_overrides = dict(),           # 剪辑适配器覆盖
        learned_variance = True,                   # 学习方差
        learned_variance_constrain_frac = False,   # 学习方差约束分数
        vb_loss_weight = 0.001,                    # vb损失权重
        unconditional = False,                     # 为生成没有条件的图像设置为True
        auto_normalize_img = True,                 # 是否自动归一化图像
        use_dynamic_thres = False,                 # 是否使用动态阈值
        dynamic_thres_percentile = 0.95,           # 动态阈值百分位数
        p2_loss_weight_gamma = 0.,                 # p2损失权重
        p2_loss_weight_k = 1,                      # p2损失权重k
        ddim_sampling_eta = 0.                     # 确定性采样
    @property
    def device(self):
        return self._dummy.device

    @property
    def condition_on_text_encodings(self):
        return any([unet.cond_on_text_encodings for unet in self.unets if isinstance(unet, Unet)])

    # 获取指定编号的unet
    def get_unet(self, unet_number):
        assert 0 < unet_number <= self.num_unets
        index = unet_number - 1
        return self.unets[index]

    # 解析unet输出
    def parse_unet_output(self, learned_variance, output):
        var_interp_frac_unnormalized = None

        if learned_variance:
            output, var_interp_frac_unnormalized = output.chunk(2, dim = 1)

        return UnetOutput(output, var_interp_frac_unnormalized)

    # 上下文管理器，用于在GPU上处理一个unet
    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.get_unet(unet_number)

        # 设备
        cuda, cpu = torch.device('cuda'), torch.device('cpu')

        self.cuda()

        devices = [module_device(unet) for unet in self.unets]

        self.unets.to(cpu)
        unet.to(cuda)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)
    # 定义一个动态阈值函数，用于改进分类器自由引导设置中的夹紧操作
    def dynamic_threshold(self, x):
        """ proposed in https://arxiv.org/abs/2205.11487 as an improved clamping in the setting of classifier free guidance """
        
        # s 是阈值量
        # 静态阈值设定为 s = 1
        s = 1.
        # 如果使用动态阈值
        if self.use_dynamic_thres:
            # 计算 x 的绝对值的分位数，用于确定动态阈值
            s = torch.quantile(
                rearrange(x, 'b ... -> b (...)').abs(),
                self.dynamic_thres_percentile,
                dim = -1
            )

            # 夹紧阈值，确保不小于1
            s.clamp_(min = 1.)
            s = s.view(-1, *((1,) * (x.ndim - 1)))

        # 根据阈值夹紧 x，取值范围为 [-s, s]，然后归一化
        x = x.clamp(-s, s) / s
        return x

    # 计算模型的均值、后验方差和后验对数方差，用于生成样本
    def p_mean_variance(self, unet, x, t, image_embed, noise_scheduler, text_encodings = None, lowres_cond_img = None, self_cond = None, clip_denoised = True, predict_x_start = False, predict_v = False, learned_variance = False, cond_scale = 1., model_output = None, lowres_noise_level = None):
        # 断言条件，确保条件满足
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'the decoder was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        # 默认情况下，使用 unet 进行前向传播
        model_output = default(model_output, lambda: unet.forward_with_cond_scale(x, t, image_embed = image_embed, text_encodings = text_encodings, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, self_cond = self_cond, lowres_noise_level = lowres_noise_level))

        # 解析 unet 输出，获取预测值和方差插值比例
        pred, var_interp_frac_unnormalized = self.parse_unet_output(learned_variance, model_output)

        # 根据预测值选择不同的处理方式
        if predict_v:
            x_start = noise_scheduler.predict_start_from_v(x, t = t, v = pred)
        elif predict_x_start:
            x_start = pred
        else:
            x_start = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)

        # 如果需要对去噪后的结果进行夹紧
        if clip_denoised:
            x_start = self.dynamic_threshold(x_start)

        # 计算模型均值、后验方差和后验对数方差
        model_mean, posterior_variance, posterior_log_variance = noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)

        # 如果使用了学习的方差
        if learned_variance:
            # 根据网络预测的最大和最小对数 beta 值进行插值，计算后验对数方差和后验方差
            min_log = extract(noise_scheduler.posterior_log_variance_clipped, t, x.shape)
            max_log = extract(torch.log(noise_scheduler.betas), t, x.shape)
            var_interp_frac = unnormalize_zero_to_one(var_interp_frac_unnormalized)

            if self.learned_variance_constrain_frac:
                var_interp_frac = var_interp_frac.sigmoid()

            posterior_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
            posterior_variance = posterior_log_variance.exp()

        return model_mean, posterior_variance, posterior_log_variance, x_start

    # 生成样本，使用模型均值和后验方差
    @torch.no_grad()
    def p_sample(self, unet, x, t, image_embed, noise_scheduler, text_encodings = None, cond_scale = 1., lowres_cond_img = None, self_cond = None, predict_x_start = False, predict_v = False, learned_variance = False, clip_denoised = True, lowres_noise_level = None):
        b, *_, device = *x.shape, x.device
        # 计算模型均值、后验方差和后验对数方差
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(unet, x = x, t = t, image_embed = image_embed, text_encodings = text_encodings, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, self_cond = self_cond, clip_denoised = clip_denoised, predict_x_start = predict_x_start, predict_v = predict_v, noise_scheduler = noise_scheduler, learned_variance = learned_variance, lowres_noise_level = lowres_noise_level)
        noise = torch.randn_like(x)
        # 当 t == 0 时不添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    # 生成样本，使用模型均值和后验方差
    @torch.no_grad()
    # 定义一个函数，用于执行采样循环，生成图片
    def p_sample_loop_ddpm(
        self,
        unet,
        shape,
        image_embed,
        noise_scheduler,
        predict_x_start = False,
        predict_v = False,
        learned_variance = False,
        clip_denoised = True,
        lowres_cond_img = None,
        text_encodings = None,
        cond_scale = 1,
        is_latent_diffusion = False,
        lowres_noise_level = None,
        inpaint_image = None,
        inpaint_mask = None,
        inpaint_resample_times = 5
    ):
        # 获取设备信息
        device = self.device

        # 获取 batch 大小
        b = shape[0]
        # 生成随机噪声图片
        img = torch.randn(shape, device = device)

        x_start = None # for self-conditioning

        is_inpaint = exists(inpaint_image)
        resample_times = inpaint_resample_times if is_inpaint else 1

        if is_inpaint:
            # 对 inpaint_image 进行归一化处理
            inpaint_image = self.normalize_img(inpaint_image)
            # 将 inpaint_image 调整大小以匹配 shape[-1]
            inpaint_image = resize_image_to(inpaint_image, shape[-1], nearest = True)
            # 将 inpaint_mask 调整大小以匹配 shape[-1]
            inpaint_mask = rearrange(inpaint_mask, 'b h w -> b 1 h w').float()
            inpaint_mask = resize_image_to(inpaint_mask, shape[-1], nearest = True)
            inpaint_mask = inpaint_mask.bool()

        if not is_latent_diffusion:
            # 对 lowres_cond_img 进行归一化处理
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # 遍历时间步骤
        for time in tqdm(reversed(range(0, noise_scheduler.num_timesteps)), desc = 'sampling loop time step', total = noise_scheduler.num_timesteps):
            is_last_timestep = time == 0

            # 遍历重新采样次数
            for r in reversed(range(0, resample_times)):
                is_last_resample_step = r == 0

                # 生成时间步骤的张量
                times = torch.full((b,), time, device = device, dtype = torch.long)

                if is_inpaint:
                    # 根据 repaint 论文进行处理
                    noised_inpaint_image = noise_scheduler.q_sample(inpaint_image, t = times)
                    img = (img * ~inpaint_mask) + (noised_inpaint_image * inpaint_mask)

                self_cond = x_start if unet.self_cond else None

                # 执行采样操作
                img, x_start = self.p_sample(
                    unet,
                    img,
                    times,
                    image_embed = image_embed,
                    text_encodings = text_encodings,
                    cond_scale = cond_scale,
                    self_cond = self_cond,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_level = lowres_noise_level,
                    predict_x_start = predict_x_start,
                    predict_v = predict_v,
                    noise_scheduler = noise_scheduler,
                    learned_variance = learned_variance,
                    clip_denoised = clip_denoised
                )

                if is_inpaint and not (is_last_timestep or is_last_resample_step):
                    # 在 repaint 中，每个步骤最多重新噪声和重新采样 10 次
                    img = noise_scheduler.q_sample_from_to(img, times - 1, times)

        if is_inpaint:
            img = (img * ~inpaint_mask) + (inpaint_image * inpaint_mask)

        # 对生成的图片进行反归一化处理
        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @torch.no_grad()
    def p_sample_loop_ddim(
        self,
        unet,
        shape,
        image_embed,
        noise_scheduler,
        timesteps,
        eta = 1.,
        predict_x_start = False,
        predict_v = False,
        learned_variance = False,
        clip_denoised = True,
        lowres_cond_img = None,
        text_encodings = None,
        cond_scale = 1,
        is_latent_diffusion = False,
        lowres_noise_level = None,
        inpaint_image = None,
        inpaint_mask = None,
        inpaint_resample_times = 5
        # 解构 shape 变量，获取批次大小、设备、总时间步长、alpha 值、eta 值
        batch, device, total_timesteps, alphas, eta = shape[0], self.device, noise_scheduler.num_timesteps, noise_scheduler.alphas_cumprod, self.ddim_sampling_eta

        # 在 0 到总时间步长之间生成 timesteps + 2 个步长的时间点，并去除最后一个时间点
        times = torch.linspace(0., total_timesteps, steps = timesteps + 2)[:-1]

        # 将时间点列表反转，并转换为整数列表
        times = list(reversed(times.int().tolist()))
        # 生成时间点对列表
        time_pairs = list(zip(times[:-1], times[1:]))
        # 过滤出时间点对中第一个时间点大于第二个时间点的情况
        time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))

        # 检查是否存在 inpaint_image
        is_inpaint = exists(inpaint_image)
        # 如果存在 inpaint_image，则使用 inpaint_resample_times，否则为 1
        resample_times = inpaint_resample_times if is_inpaint else 1

        # 如果存在 inpaint_image，则对其进行归一化和调整大小，并生成对应的掩码
        if is_inpaint:
            inpaint_image = self.normalize_img(inpaint_image)
            inpaint_image = resize_image_to(inpaint_image, shape[-1], nearest = True)
            inpaint_mask = rearrange(inpaint_mask, 'b h w -> b 1 h w').float()
            inpaint_mask = resize_image_to(inpaint_mask, shape[-1], nearest = True)
            inpaint_mask = inpaint_mask.bool()

        # 生成随机噪声图像
        img = torch.randn(shape, device = device)

        # 初始化 x_start 为 None，用于自条件
        x_start = None

        # 如果不是潜在扩散，则对低分辨率条件图像进行归一化
        if not is_latent_diffusion:
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # 遍历时间点对
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # 判断是否为最后一个时间步
            is_last_timestep = time_next == 0

            # 反向遍历重采样次数
            for r in reversed(range(0, resample_times)):
                # 判断是否为最后一个重采样步骤
                is_last_resample_step = r == 0

                # 获取当前时间点和下一个时间点的 alpha 值
                alpha = alphas[time]
                alpha_next = alphas[time_next]

                # 生成当前时间点的条件
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

                # 如果存在 inpaint_image，则根据时间点和掩码生成噪声图像
                if is_inpaint:
                    noised_inpaint_image = noise_scheduler.q_sample(inpaint_image, t = time_cond)
                    img = (img * ~inpaint_mask) + (noised_inpaint_image * inpaint_mask)

                # 根据 unet 的 self_cond 属性确定是否使用自条件
                self_cond = x_start if unet.self_cond else None

                # 使用 unet 模型生成输出
                unet_output = unet.forward_with_cond_scale(img, time_cond, image_embed = image_embed, text_encodings = text_encodings, cond_scale = cond_scale, self_cond = self_cond, lowres_cond_img = lowres_cond_img, lowres_noise_level = lowres_noise_level)

                # 解析 unet 输出
                pred, _ = self.parse_unet_output(learned_variance, unet_output)

                # 预测 x0
                if predict_v:
                    x_start = noise_scheduler.predict_start_from_v(img, t = time_cond, v = pred)
                elif predict_x_start:
                    x_start = pred
                else:
                    x_start = noise_scheduler.predict_start_from_noise(img, t = time_cond, noise = pred)

                # 可能对 x0 进行裁剪
                if clip_denoised:
                    x_start = self.dynamic_threshold(x_start)

                # 预测噪声
                pred_noise = noise_scheduler.predict_noise_from_start(img, t = time_cond, x0 = x_start)

                # 计算 c1 和 c2
                c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
                noise = torch.randn_like(img) if not is_last_timestep else 0.

                # 更新图像
                img = x_start * alpha_next.sqrt() + \
                      c1 * noise + \
                      c2 * pred_noise

                # 如果存在 inpaint_image 且不是最后一个时间步或最后一个重采样步骤，则重新噪声和重采样
                if is_inpaint and not (is_last_timestep or is_last_resample_step):
                    time_next_cond = torch.full((batch,), time_next, device = device, dtype = torch.long)
                    img = noise_scheduler.q_sample_from_to(img, time_next_cond, time_cond)

        # 如果存在 inpaint_image，则将图像还原为原始图像
        if exists(inpaint_image):
            img = (img * ~inpaint_mask) + (inpaint_image * inpaint_mask)

        # 将图像还原为原始图像
        img = self.unnormalize_img(img)
        # 返回生成的图像
        return img

    # 禁用梯度
    @torch.no_grad()
    # 定义一个方法 p_sample_loop，接受可变数量的参数和关键字参数
    def p_sample_loop(self, *args, noise_scheduler, timesteps = None, **kwargs):
        # 获取噪声调度器的总时间步数
        num_timesteps = noise_scheduler.num_timesteps

        # 如果未指定时间步数，则使用默认值为总时间步数
        timesteps = default(timesteps, num_timesteps)
        # 断言指定的时间步数不超过总时间步数
        assert timesteps <= num_timesteps
        # 判断是否为动态维度
        is_ddim = timesteps < num_timesteps

        # 如果不是动态维度，则调用 p_sample_loop_ddpm 方法
        if not is_ddim:
            return self.p_sample_loop_ddpm(*args, noise_scheduler = noise_scheduler, **kwargs)

        # 如果是动态维度，则调用 p_sample_loop_ddim 方法
        return self.p_sample_loop_ddim(*args, noise_scheduler = noise_scheduler, timesteps = timesteps, **kwargs)
    # 定义一个函数，计算损失值
    def p_losses(self, unet, x_start, times, *, image_embed, noise_scheduler, lowres_cond_img = None, text_encodings = None, predict_x_start = False, predict_v = False, noise = None, learned_variance = False, clip_denoised = False, is_latent_diffusion = False, lowres_noise_level = None):
        # 设置默认的噪声函数
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 将输入归一化到[-1, 1]范围内
        if not is_latent_diffusion:
            x_start = self.normalize_img(x_start)
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # 获取带噪声的输入图像
        x_noisy = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)

        # 设置 UNet 的参数
        unet_kwargs = dict(
            image_embed = image_embed,
            text_encodings = text_encodings,
            lowres_cond_img = lowres_cond_img,
            lowres_noise_level = lowres_noise_level,
        )

        # 自我条件
        self_cond = None

        # 如果 UNet 具有自我条件属性且随机数小于0.5
        if unet.self_cond and random.random() < 0.5:
            with torch.no_grad():
                unet_output = unet(x_noisy, times, **unet_kwargs)
                self_cond, _ = self.parse_unet_output(learned_variance, unet_output)
                self_cond = self_cond.detach()

        # 前向传播获取模型预测
        unet_output = unet(
            x_noisy,
            times,
            **unet_kwargs,
            self_cond = self_cond,
            image_cond_drop_prob = self.image_cond_drop_prob,
            text_cond_drop_prob = self.text_cond_drop_prob,
        )

        pred, _ = self.parse_unet_output(learned_variance, unet_output)

        # 根据需求选择目标值
        if predict_v:
            target = noise_scheduler.calculate_v(x_start, times, noise)
        elif predict_x_start:
            target = x_start
        else:
            target = noise

        # 计算损失值
        loss = noise_scheduler.loss_fn(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        # 对损失值进行重新加权
        loss = noise_scheduler.p2_reweigh_loss(loss, times)

        loss = loss.mean()

        if not learned_variance:
            # 如果不使用学习的方差，则返回简单的损失值
            return loss

        # 如果学习方差，还包括额外的 kl 损失
        true_mean, _, true_log_variance_clipped = noise_scheduler.q_posterior(x_start = x_start, x_t = x_noisy, t = times)
        model_mean, _, model_log_variance, _ = self.p_mean_variance(unet, x = x_noisy, t = times, image_embed = image_embed, noise_scheduler = noise_scheduler, clip_denoised = clip_denoised, learned_variance = True, model_output = unet_output)

        # KL 损失
        detached_model_mean = model_mean.detach()
        kl = normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = meanflat(kl) * NAT

        # 解码器负对数似然
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means = detached_model_mean, log_scales = 0.5 * model_log_variance)
        decoder_nll = meanflat(decoder_nll) * NAT

        # 在第一个时间步返回解码器 NLL，否则返回 KL 散度
        vb_losses = torch.where(times == 0, decoder_nll, kl)

        # 对 vb 损失进行加权
        vb_loss = vb_losses.mean() * self.vb_loss_weight

        return loss + vb_loss

    # 禁止梯度计算
    @torch.no_grad()
    # 评估装饰器
    @eval_decorator
    # 定义一个名为sample的方法，用于生成样本
    def sample(
        self,
        image = None, # 图像输入，默认为None
        image_embed = None, # 图像嵌入，默认为None
        text = None, # 文本输入，默认为None
        text_encodings = None, # 文本编码，默认为None
        batch_size = 1, # 批处理大小，默认为1
        cond_scale = 1., # 条件比例，默认为1.0
        start_at_unet_number = 1, # 开始的UNET编号，默认为1
        stop_at_unet_number = None, # 结束的UNET编号，默认为None
        distributed = False, # 是否分布式，默认为False
        inpaint_image = None, # 修复图像，默认为None
        inpaint_mask = None, # 修复掩码，默认为None
        inpaint_resample_times = 5, # 修复重采样次数，默认为5
        one_unet_in_gpu_at_time = True # 是否一次在GPU上运行一个UNET，默认为True
    # 定义一个名为forward的方法，用于前向传播
    def forward(
        self,
        image, # 图像输入
        text = None, # 文本输入，默认为None
        image_embed = None, # 图像嵌入，默认为None
        text_encodings = None, # 文本编码，默认为None
        unet_number = None, # UNET编号，默认为None
        return_lowres_cond_image = False # 是否返回低分辨率的条件图像，用于调试上采样器的目的，默认为False
        ):
        # 断言语句，用于检查是否指定了要训练的 unet 编号，如果训练多个 unet，则必须指定要训练的 unet 编号
        assert not (self.num_unets > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {self.num_unets}, if you are training cascading DDPM (multiple unets)'
        # 如果未指定 unet 编号，则默认为 1
        unet_number = default(unet_number, 1)
        # 计算 unet 编号在列表中的索引
        unet_index = unet_number - 1

        # 获取指定编号的 unet 模型
        unet = self.get_unet(unet_number)

        # 获取对应 unet 编号的 VAE 模型、噪声调度器、低分辨率条件器、目标图像大小、预测 x 起始位置、预测速度、随机裁剪大小、学习的方差、图像的形状和设备
        vae                 = self.vaes[unet_index]
        noise_scheduler     = self.noise_schedulers[unet_index]
        lowres_conditioner  = self.lowres_conds[unet_index]
        target_image_size   = self.image_sizes[unet_index]
        predict_x_start     = self.predict_x_start[unet_index]
        predict_v           = self.predict_v[unet_index]
        random_crop_size    = self.random_crop_sizes[unet_index]
        learned_variance    = self.learned_variance[unet_index]
        b, c, h, w, device, = *image.shape, image.device

        # 断言语句，用于检查图像通道数是否与模型要求的通道数相同
        assert image.shape[1] == self.channels
        # 断言语句，用于检查图像的高度和宽度是否大于等于目标图像大小
        assert h >= target_image_size and w >= target_image_size

        # 生成一组随机时间步长
        times = torch.randint(0, noise_scheduler.num_timesteps, (b,), device = device, dtype = torch.long)

        # 如果未提供图像嵌入且不是无条件生成，则使用 CLIP 模型对图像进行嵌入
        if not exists(image_embed) and not self.unconditional:
            assert exists(self.clip), 'if you want to derive CLIP image embeddings automatically, you must supply `clip` to the decoder on init'
            image_embed, _ = self.clip.embed_image(image)

        # 如果提供了文本且未提供文本编码且不是无条件生成，则使用 CLIP 模型对文本进行嵌入
        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(self.clip), 'if you are passing in raw text, you need to supply `clip` to the decoder'
            _, text_encodings = self.clip.embed_text(text)

        # 断言语句，用于检查是否传入了文本编码
        assert not (self.condition_on_text_encodings and not exists(text_encodings)), 'text or text encodings must be passed into decoder if specified'
        # 断言语句，用于检查是否指定了不基于文本编码的解码器
        assert not (not self.condition_on_text_encodings and exists(text_encodings)), 'decoder specified not to be conditioned on text, yet it is presented'

        # 如果存在低分辨率条件器，则对图像进行低分辨率处理
        lowres_cond_img, lowres_noise_level = lowres_conditioner(image, target_image_size = target_image_size, downsample_image_size = self.image_sizes[unet_index - 1]) if exists(lowres_conditioner) else (None, None)
        # 调整图像大小为目标图像大小
        image = resize_image_to(image, target_image_size, nearest = True)

        # 如果存在随机裁剪大小，则对图像进行随机裁剪
        if exists(random_crop_size):
            aug = K.RandomCrop((random_crop_size, random_crop_size), p = 1.)

            # 确保低分辨率条件器和图像都以相同方式进行增强
            # 详细信息请参考 https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            image = aug(image)
            lowres_cond_img = aug(lowres_cond_img, params = aug._params)

        # 判断是否为潜在扩散模型
        is_latent_diffusion = not isinstance(vae, NullVQGanVAE)

        # 将 VAE 模型设置为评估模式，并禁用梯度计算
        vae.eval()
        with torch.no_grad():
            # 对图像进行编码
            image = vae.encode(image)
            # 对低分辨率条件图像进行编码
            lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

        # 计算损失
        losses = self.p_losses(unet, image, times, image_embed = image_embed, text_encodings = text_encodings, lowres_cond_img = lowres_cond_img, predict_x_start = predict_x_start, predict_v = predict_v, learned_variance = learned_variance, is_latent_diffusion = is_latent_diffusion, noise_scheduler = noise_scheduler, lowres_noise_level = lowres_noise_level)

        # 如果不返回低分辨率条件图像，则返回损失
        if not return_lowres_cond_image:
            return losses

        # 返回损失和低分辨率条件图像
        return losses, lowres_cond_img
# 主类定义

class DALLE2(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        prior,  # 先验模型
        decoder,  # 解码器
        prior_num_samples = 2  # 先验模型采样数量，默认为2
    ):
        super().__init__()
        # 断言先验模型和解码器的类型
        assert isinstance(prior, DiffusionPrior)
        assert isinstance(decoder, Decoder)
        # 初始化先验模型和解码器
        self.prior = prior
        self.decoder = decoder

        self.prior_num_samples = prior_num_samples  # 先验模型采样数量
        self.decoder_need_text_cond = self.decoder.condition_on_text_encodings  # 解码器是否需要文本编码

        self.to_pil = T.ToPILImage()  # 转换为 PIL 图像

    @torch.no_grad()
    @eval_decorator
    # 前向传播函数
    def forward(
        self,
        text,  # 文本输入
        cond_scale = 1.,  # 条件缩放
        prior_cond_scale = 1.,  # 先验条件缩放
        return_pil_images = False  # 是否返回 PIL 图像
    ):
        device = module_device(self)  # 获取设备
        one_text = isinstance(text, str) or (not is_list_str(text) and text.shape[0] == 1)  # 判断是否为单个文本

        if isinstance(text, str) or is_list_str(text):
            text = [text] if not isinstance(text, (list, tuple)) else text
            text = tokenizer.tokenize(text).to(device)  # 对文本进行标记化处理并移动到设备

        # 从先验模型中采样图像编码
        image_embed = self.prior.sample(text, num_samples_per_batch = self.prior_num_samples, cond_scale = prior_cond_scale)

        text_cond = text if self.decoder_need_text_cond else None  # 如果解码器需要文本编码，则传入文本编码，否则为None
        # 从解码器中采样图像
        images = self.decoder.sample(image_embed = image_embed, text = text_cond, cond_scale = cond_scale)

        if return_pil_images:
            images = list(map(self.to_pil, images.unbind(dim = 0)))  # 将图像转换为 PIL 图像

        if one_text:
            return first(images)  # 如果只有一个文本输入，则返回第一个图像

        return images  # 返回图像列表
```