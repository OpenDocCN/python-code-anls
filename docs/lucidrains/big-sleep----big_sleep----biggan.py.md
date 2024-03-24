# `.\lucidrains\big-sleep\big_sleep\biggan.py`

```py
# 导入所需的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import copy
import logging
import os
import shutil
import tempfile
from functools import wraps
from hashlib import sha256
import sys
from io import open

import boto3
import requests
from botocore.exceptions import ClientError
from tqdm import tqdm

# 尝试导入 Python 3 版本的 urllib.parse，如果失败则导入 Python 2 版本的 urlparse
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

# 尝试导入 Python 3 版本的 pathlib.Path，设置缓存路径为用户主目录下的 .pytorch_pretrained_biggan 文件夹
try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BIGGAN_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BIGGAN_CACHE',
                                                   Path.home() / '.pytorch_pretrained_biggan'))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BIGGAN_CACHE = os.getenv('PYTORCH_PRETRAINED_BIGGAN_CACHE',
                                              os.path.join(os.path.expanduser("~"), '.pytorch_pretrained_biggan'))

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# 预训练模型和配置文件的下载链接映射
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'biggan-deep-128': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin",
    'biggan-deep-256': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin",
    'biggan-deep-512': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin",
}

PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'biggan-deep-128': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-config.json",
    'biggan-deep-256': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-config.json",
    'biggan-deep-512': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-config.json",
}

WEIGHTS_NAME = 'pytorch_model.bin'  # 权重文件名
CONFIG_NAME = 'config.json'  # 配置文件名

# 将 URL 转换为哈希文件名的函数
def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename

# 将文件名转换为 URL 的函数
def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BIGGAN_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']

    return url, etag

# 缓存路径函数，根据输入的 URL 或文件名判断是下载文件还是返回本地文件路径
def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BIGGAN_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)
    # 如果 URL 方案是 'http', 'https', 's3' 中的一个，说明是 URL 地址，从缓存中获取数据（必要时下载）
    if parsed.scheme in ('http', 'https', 's3'):
        return get_from_cache(url_or_filename, cache_dir)
    # 如果是文件路径，并且文件存在
    elif os.path.exists(url_or_filename):
        return url_or_filename
    # 如果是文件路径，但文件不存在
    elif parsed.scheme == '':
        raise EnvironmentError("file {} not found".format(url_or_filename))
    # 其他情况，无法解析为 URL 或本地路径
    else:
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))
# 将完整的 S3 路径分割成存储桶名称和路径
def split_s3_path(url):
    # 解析 URL
    parsed = urlparse(url)
    # 检查是否存在 netloc 和 path
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    # 获取存储桶名称和 S3 路径
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # 移除路径开头的 '/'
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


# 用于包装 S3 请求的装饰器函数，以便创建更有用的错误消息
def s3_request(func):
    
    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            # 检查错误码是否为 404
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


# 检查 S3 对象的 ETag
@s3_request
def s3_etag(url):
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


# 从 S3 直接获取文件
@s3_request
def s3_get(url, temp_file):
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


# 从 HTTP 获取文件
def http_get(url, temp_file):
    # 发送 GET 请求
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    # 逐块写入文件
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # 过滤掉保持连接的新块
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


# 从缓存获取文件
def get_from_cache(url, cache_dir=None):
    # 如果未指定缓存目录，则使用默认缓存目录
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BIGGAN_CACHE
    # 如果是 Python 3 并且缓存目录是 Path 对象，则转换为字符串
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    # 如果缓存目录不存在，则创建
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # 如果 URL 是以 "s3://" 开头，则获取 ETag
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        # 发送 HEAD 请求获取 ETag
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError("HEAD request failed for url {} with status code {}"
                          .format(url, response.status_code))
        etag = response.headers.get("ETag")

    # 根据 URL 和 ETag 生成文件名
    filename = url_to_filename(url, etag)

    # 获取缓存路径
    cache_path = os.path.join(cache_dir, filename)
    # 检查缓存路径是否存在，如果不存在则执行下载操作
    if not os.path.exists(cache_path):
        # 在下载完成之前，先下载到临时文件，然后再复制到缓存目录中
        # 否则，如果下载被中断，会导致缓存条目损坏
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # 获取文件对象
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # 在关闭文件之前复制文件，因此需要刷新以避免截断
            temp_file.flush()
            # shutil.copyfileobj() 从当前位置开始复制，所以需要回到起始位置
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            # 创建元数据，包括 URL 和 ETag
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w', encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    # 返回缓存路径
    return cache_path
# 从文件中提取一个去重的文本集合（集合）
# 预期文件格式是每行一个项目
def read_set_from_file(filename):
    collection = set()
    # 使用 utf-8 编码打开文件
    with open(filename, 'r', encoding='utf-8') as file_:
        # 逐行读取文件内容，去除行尾的换行符后添加到集合中
        for line in file_:
            collection.add(line.rstrip())
    # 返回集合
    return collection

# 获取文件扩展名
def get_file_extension(path, dot=True, lower=True):
    # 获取文件路径的扩展名
    ext = os.path.splitext(path)[1]
    # 如果 dot 为 True，则保留扩展名中的点号
    ext = ext if dot else ext[1:]
    # 如果 lower 为 True，则将扩展名转换为小写
    return ext.lower() if lower else ext

# BigGAN 的配置类
class BigGANConfig(object):
    """ Configuration class to store the configuration of a `BigGAN`. 
        Defaults are for the 128x128 model.
        layers tuple are (up-sample in the layer ?, input channels, output channels)
    """
    def __init__(self,
                 output_dim=128,
                 z_dim=128,
                 class_embed_dim=128,
                 channel_width=128,
                 num_classes=1000,
                 layers=[(False, 16, 16),
                         (True, 16, 16),
                         (False, 16, 16),
                         (True, 16, 8),
                         (False, 8, 8),
                         (True, 8, 4),
                         (False, 4, 4),
                         (True, 4, 2),
                         (False, 2, 2),
                         (True, 2, 1)],
                 attention_layer_position=8,
                 eps=1e-4,
                 n_stats=51):
        """Constructs BigGANConfig. """
        # 初始化 BigGAN 的配置参数
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.class_embed_dim = class_embed_dim
        self.channel_width = channel_width
        self.num_classes = num_classes
        self.layers = layers
        self.attention_layer_position = attention_layer_position
        self.eps = eps
        self.n_stats = n_stats

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BigGANConfig` from a Python dictionary of parameters."""
        # 从 Python 字典中构建 BigGANConfig 实例
        config = BigGANConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BigGANConfig` from a json file of parameters."""
        # 从 JSON 文件中构建 BigGANConfig ��例
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        # 将实例序列化为 Python 字典
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        # 将实例序列化为 JSON 字符串
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

# 使用谱范数封装的二维卷积层
def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)

# 使用谱范数封装的线性层
def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)

# 使用谱范数封装的嵌入层
def sn_embedding(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Embedding(**kwargs), eps=eps)

# 自注意力层
class SelfAttn(nn.Module):
    """ Self attention Layer"""
    # 初始化 SelfAttn 类，设置输入通道数和 epsilon 值
    def __init__(self, in_channels, eps=1e-12):
        # 调用父类的初始化方法
        super(SelfAttn, self).__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 创建 theta 路径的 1x1 卷积层，并使用 spectral normalization
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8,
                                        kernel_size=1, bias=False, eps=eps)
        # 创建 phi 路径的 1x1 卷积层，并使用 spectral normalization
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8,
                                      kernel_size=1, bias=False, eps=eps)
        # 创建 g 路径的 1x1 卷积层，并使用 spectral normalization
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2,
                                    kernel_size=1, bias=False, eps=eps)
        # 创建输出卷积层的 1x1 卷积层，并使用 spectral normalization
        self.snconv1x1_o_conv = snconv2d(in_channels=in_channels//2, out_channels=in_channels,
                                         kernel_size=1, bias=False, eps=eps)
        # 创建最大池化层
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        # 创建 Softmax 层
        self.softmax  = nn.Softmax(dim=-1)
        # 创建可学习参数 gamma
        self.gamma = nn.Parameter(torch.zeros(1))

    # 前向传播函数
    def forward(self, x):
        # 获取输入 x 的尺寸信息
        _, ch, h, w = x.size()
        # Theta 路径
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi 路径
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # 注意力图
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g 路径
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # 注意力加权的 g - o_conv
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_o_conv(attn_g)
        # 输出
        out = x + self.gamma*attn_g
        return out
class BigGANBatchNorm(nn.Module):
    """ This is a batch norm module that can handle conditional input and can be provided with pre-computed
        activation means and variances for various truncation parameters.
        We cannot just rely on torch.batch_norm since it cannot handle
        batched weights (pytorch 1.0.1). We computate batch_norm our-self without updating running means and variances.
        If you want to train this model you should add running means and variance computation logic.
    """
    # 初始化函数，定义了 BigGANBatchNorm 类的属性和参数
    def __init__(self, num_features, condition_vector_dim=None, n_stats=51, eps=1e-4, conditional=True):
        super(BigGANBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional

        # 使用预先计算的统计数据来处理不同截断参数的情况
        self.register_buffer('running_means', torch.zeros(n_stats, num_features))
        self.register_buffer('running_vars', torch.ones(n_stats, num_features))
        self.step_size = 1.0 / (n_stats - 1)

        # 如果是有条件的批量归一化
        if conditional:
            assert condition_vector_dim is not None
            self.scale = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
            self.offset = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))

    # 前向传播函数
    def forward(self, x, truncation, condition_vector=None):
        # 获取与此截断相关的预先计算的统计数据
        coef, start_idx = math.modf(truncation / self.step_size)
        start_idx = int(start_idx)
        if coef != 0.0:  # 插值
            running_mean = self.running_means[start_idx] * coef + self.running_means[start_idx + 1] * (1 - coef)
            running_var = self.running_vars[start_idx] * coef + self.running_vars[start_idx + 1] * (1 - coef)
        else:
            running_mean = self.running_means[start_idx]
            running_var = self.running_vars[start_idx]

        if self.conditional:
            running_mean = running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            running_var = running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            weight = 1 + self.scale(condition_vector).unsqueeze(-1).unsqueeze(-1)
            bias = self.offset(condition_vector).unsqueeze(-1).unsqueeze(-1)

            out = (x - running_mean) / torch.sqrt(running_var + self.eps) * weight + bias
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias,
                               training=False, momentum=0.0, eps=self.eps)

        return out

class GenBlock(nn.Module):
    # 初始化生成器块，设置输入大小、输出大小、条件向量维度、缩减因子、是否上采样、统计数、eps值
    def __init__(self, in_size, out_size, condition_vector_dim, reduction_factor=4, up_sample=False,
                 n_stats=51, eps=1e-12):
        # 调用父类的初始化方法
        super(GenBlock, self).__init__()
        # 设置是否上采样
        self.up_sample = up_sample
        # 判断是否需要减少通道数
        self.drop_channels = (in_size != out_size)
        # 计算中间大小
        middle_size = in_size // reduction_factor

        # 初始化批量归一化层
        self.bn_0 = BigGANBatchNorm(in_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        # 初始化卷积层
        self.conv_0 = snconv2d(in_channels=in_size, out_channels=middle_size, kernel_size=1, eps=eps)

        self.bn_1 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_1 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)

        self.bn_2 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_2 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)

        self.bn_3 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_3 = snconv2d(in_channels=middle_size, out_channels=out_size, kernel_size=1, eps=eps)

        # 初始化ReLU激活函数
        self.relu = nn.ReLU()

    # 前向传播函数
    def forward(self, x, cond_vector, truncation):
        # 保存输入x
        x0 = x

        # 执行第一个批量归一化层、ReLU激活函数、卷积层操作
        x = self.bn_0(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_0(x)

        # 执行第二个批量归一化层、ReLU激活函数、上采样（如果需要）、卷积层操作
        x = self.bn_1(x, truncation, cond_vector)
        x = self.relu(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv_1(x)

        # 执行第三个批量归一化层、ReLU激活函数、卷积层操作
        x = self.bn_2(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_2(x)

        # 执行第四个批量归一化层、ReLU激活函数、卷积层操作
        x = self.bn_3(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_3(x)

        # 如���需要减少通道数，进行通道数减半操作
        if self.drop_channels:
            new_channels = x0.shape[1] // 2
            x0 = x0[:, :new_channels, ...]
        # 如果需要上采样，进行上采样操作
        if self.up_sample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest')

        # 将两部分特征相加作为输出
        out = x + x0
        return out
class Generator(nn.Module):
    def __init__(self, config):
        # 初始化生成器类，继承自 nn.Module
        super(Generator, self).__init__()
        # 保存配置信息
        self.config = config
        # 从配置中获取通道宽度
        ch = config.channel_width
        # 计算条件向量的维度
        condition_vector_dim = config.z_dim * 2

        # 生成器的线性层，输入为条件向量的维度，输出为特定维度
        self.gen_z = snlinear(in_features=condition_vector_dim,
                              out_features=4 * 4 * 16 * ch, eps=config.eps)

        layers = []
        # 遍历配置中的层信息
        for i, layer in enumerate(config.layers):
            # 如果当前层是注意力层的位置
            if i == config.attention_layer_position:
                # 添加自注意力层
                layers.append(SelfAttn(ch*layer[1], eps=config.eps))
            # 添加生成块
            layers.append(GenBlock(ch*layer[1],
                                   ch*layer[2],
                                   condition_vector_dim,
                                   up_sample=layer[0],
                                   n_stats=config.n_stats,
                                   eps=config.eps))
        # 将所有层组成模块列表
        self.layers = nn.ModuleList(layers)

        # 生成器的批归一化层
        self.bn = BigGANBatchNorm(ch, n_stats=config.n_stats, eps=config.eps, conditional=False)
        # ReLU 激活函数
        self.relu = nn.ReLU()
        # 生成器的卷积层，将特征图转换为 RGB 图像
        self.conv_to_rgb = snconv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, eps=config.eps)
        # Tanh 激活函数
        self.tanh = nn.Tanh()

    def forward(self, cond_vector, truncation):
        # 生成随机噪声
        z = self.gen_z(cond_vector[0].unsqueeze(0))

        # 调整张量形状以适应 TF 权重格式
        z = z.view(-1, 4, 4, 16 * self.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()

        next_available_latent_index = 1
        # 遍历所有层
        for layer in self.layers:
            # 如果是生成块
            if isinstance(layer, GenBlock):
                # 使用生成块
                z = layer(z, cond_vector[next_available_latent_index].unsqueeze(0), truncation)
                next_available_latent_index += 1
            else:
                z = layer(z)

        # 批归一化
        z = self.bn(z, truncation)
        z = self.relu(z)
        z = self.conv_to_rgb(z)
        z = z[:, :3, ...]
        z = self.tanh(z)
        return z

class BigGAN(nn.Module):
    """BigGAN Generator."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        # 根据预训练模型名称或路径加载模型
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            model_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            model_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)

        try:
            # 解析模型文件和配置文件
            resolved_model_file = cached_path(model_file, cache_dir=cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error("Wrong model name, should be a valid path to a folder containing "
                         "a {} file and a {} file or a model name in {}".format(
                         WEIGHTS_NAME, CONFIG_NAME, PRETRAINED_MODEL_ARCHIVE_MAP.keys()))
            raise

        logger.info("loading model {} from cache at {}".format(pretrained_model_name_or_path, resolved_model_file))

        # 加载配置
        config = BigGANConfig.from_json_file(resolved_config_file)
        logger.info("Model config {}".format(config))

        # 实例化模型
        model = cls(config, *inputs, **kwargs)
        state_dict = torch.load(resolved_model_file, map_location='cpu' if not torch.cuda.is_available() else None)
        model.load_state_dict(state_dict, strict=False)
        return model

    def __init__(self, config):
        # 初始化 BigGAN 类，继承自 nn.Module
        super(BigGAN, self).__init__()
        # 保存配置信息
        self.config = config
        # 线性层，用于生成器的嵌入
        self.embeddings = nn.Linear(config.num_classes, config.z_dim, bias=False)
        # 生成器实例
        self.generator = Generator(config)
    # 定义一个前向传播函数，接受输入 z（随机噪声）、class_label（类别标签）、truncation（截断值）
    def forward(self, z, class_label, truncation):
        # 断言截断值在 (0, 1] 范围内
        assert 0 < truncation <= 1

        # 通过类别标签获取对应的嵌入向量
        embed = self.embeddings(class_label)
        # 将随机噪声 z 和类别嵌入向量拼接在一起，形成条件向量
        cond_vector = torch.cat((z, embed), dim=1)

        # 使用条件向量和截断值作为参数，生成图像数据
        z = self.generator(cond_vector, truncation)
        # 返回生成的图像数据
        return z
```