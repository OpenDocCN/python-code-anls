# `.\lucidrains\DALLE-pytorch\dalle_pytorch\vae.py`

```py
# 导入所需的库
import io
import sys
import os
import requests
import PIL
import warnings
import hashlib
import urllib
import yaml
from pathlib import Path
from tqdm import tqdm
from math import sqrt, log
from packaging import version

# 导入第三方库
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import importlib

# 导入 PyTorch 库
import torch
from torch import nn
import torch.nn.functional as F

# 导入 einops 库
from einops import rearrange

# 导入 dalle_pytorch 库中的 distributed_utils 模块
from dalle_pytorch import distributed_utils

# 常量定义

CACHE_PATH = os.path.expanduser("~/.cache/dalle")

OPENAI_VAE_ENCODER_PATH = 'https://cdn.openai.com/dall-e/encoder.pkl'
OPENAI_VAE_DECODER_PATH = 'https://cdn.openai.com/dall-e/decoder.pkl'

VQGAN_VAE_PATH = 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'
VQGAN_VAE_CONFIG_PATH = 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'

# 辅助方法

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location = torch.device('cpu'))

def map_pixels(x, eps = 0.1):
    return (1 - 2 * eps) * x + eps

def unmap_pixels(x, eps = 0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)

def download(url, filename = None, root = CACHE_PATH):
    if (
            not distributed_utils.is_distributed
            or distributed_utils.backend.is_local_root_worker()
    ):
        os.makedirs(root, exist_ok = True)
    filename = default(filename, os.path.basename(url))

    download_target = os.path.join(root, filename)
    download_target_tmp = os.path.join(root, f'tmp.{filename}')

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if (
            distributed_utils.is_distributed
            and not distributed_utils.backend.is_local_root_worker()
            and not os.path.isfile(download_target)
    ):
        # 如果文件尚不存在，则等待根工作节点下载
        distributed_utils.backend.local_barrier()

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    os.rename(download_target_tmp, download_target)
    if (
            distributed_utils.is_distributed
            and distributed_utils.backend.is_local_root_worker()
    ):
        distributed_utils.backend.local_barrier()
    return download_target

def make_contiguous(module):
    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())

# 获取包版本信息

def get_pkg_version(pkg_name):
    from pkg_resources import get_distribution
    return get_distribution(pkg_name).version

# 预训练的 OpenAI 离散 VAE

class OpenAIDiscreteVAE(nn.Module):
    def __init__(self):
        super().__init__()
        assert version.parse(get_pkg_version('torch')) < version.parse('1.11.0'), 'torch version must be <= 1.10 in order to use OpenAI discrete vae'

        # 加载编码器和解码器模型
        self.enc = load_model(download(OPENAI_VAE_ENCODER_PATH))
        self.dec = load_model(download(OPENAI_VAE_DECODER_PATH))
        make_contiguous(self)

        self.channels = 3
        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192

    @torch.no_grad()
    def get_codebook_indices(self, img):
        # 映射像素值
        img = map_pixels(img)
        # 获取编码器的输出
        z_logits = self.enc.blocks(img)
        # 获取最大概率的索引
        z = torch.argmax(z_logits, dim = 1)
        return rearrange(z, 'b h w -> b (h w)')
    # 解码函数，将图像序列解码为图像
    def decode(self, img_seq):
        # 获取图像序列的形状
        b, n = img_seq.shape
        # 重新排列图像序列的形状，将其转换为二维图像
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(sqrt(n)))

        # 将图像序列转换为 one-hot 编码
        z = F.one_hot(img_seq, num_classes = self.num_tokens)
        # 重新排列 one-hot 编码的形状
        z = rearrange(z, 'b h w c -> b c h w').float()
        # 使用解码器解码 one-hot 编码的数据
        x_stats = self.dec(z).float()
        # 将解码后的数据映射回像素值范围
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        # 返回解码后的图像
        return x_rec

    # 前向传播函数，抛出未实现异常
    def forward(self, img):
        raise NotImplemented
# 从 Taming Transformers 论文中获取 VQGAN 模型
# https://arxiv.org/abs/2012.09841

# 从字符串中获取对象
def get_obj_from_str(string, reload=False):
    # 拆分字符串，获取模块和类名
    module, cls = string.rsplit(".", 1)
    if reload:
        # 导入模块并重新加载
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

# 根据配置实例化对象
def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# VQGAN VAE 类
class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None):
        super().__init__()

        if vqgan_model_path is None:
            model_filename = 'vqgan.1024.model.ckpt'
            config_filename = 'vqgan.1024.config.yml'
            download(VQGAN_VAE_CONFIG_PATH, config_filename)
            download(VQGAN_VAE_PATH, model_filename)
            config_path = str(Path(CACHE_PATH) / config_filename)
            model_path = str(Path(CACHE_PATH) / model_filename)
        else:
            model_path = vqgan_model_path
            config_path = vqgan_config_path

        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config["model"])

        state = torch.load(model_path, map_location = 'cpu')['state_dict']
        model.load_state_dict(state, strict = False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")

        self.model = model

        # 计算分辨率缩放因子 f
        f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]

        self.num_layers = int(log(f)/log(2))
        self.channels = 3
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed
        self.is_gumbel = isinstance(self.model, GumbelVQ)

        self._register_external_parameters()

    def _register_external_parameters(self):
        """为 DeepSpeed 分区注册外部参数"""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(
            self, self.model.quantize.embed.weight if self.is_gumbel else self.model.quantize.embedding.weight)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return rearrange(indices, 'b h w -> b (h w)', b=b)
        return rearrange(indices, '(b n) -> b n', b = b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = one_hot_indices @ self.model.quantize.embed.weight if self.is_gumbel \
            else (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        raise NotImplemented
```