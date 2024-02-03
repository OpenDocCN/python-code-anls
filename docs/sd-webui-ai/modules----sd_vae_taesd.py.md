# `stable-diffusion-webui\modules\sd_vae_taesd.py`

```py
"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)

https://github.com/madebyollin/taesd
"""
# 导入所需的库
import os
import torch
import torch.nn as nn

# 导入自定义模块
from modules import devices, paths_internal, shared

# 存储 SD-VAE 模型的字典
sd_vae_taesd_models = {}

# 定义卷积层函数
def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

# 定义 Clamp 类，用于对输入进行 tanh 处理
class Clamp(nn.Module):
    @staticmethod
    def forward(x):
        return torch.tanh(x / 3) * 3

# 定义 Block 类，包含卷积、激活函数和跳跃连接
class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

# 定义解码器函数
def decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )

# 定义编码器函数
def encoder():
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 4),
    )

# 定义 TAESD 解码器类
class TAESDDecoder(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5
    # 初始化 TAESD 模型，加载预训练的解码器模型
    def __init__(self, decoder_path="taesd_decoder.pth"):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        # 调用父类的初始化方法
        super().__init__()
        # 创建解码器对象
        self.decoder = decoder()
        # 加载解码器模型的参数
        self.decoder.load_state_dict(
            # 从指定路径加载模型参数，根据设备类型选择在 CPU 或 GPU 上加载
            torch.load(decoder_path, map_location='cpu' if devices.device.type != 'cuda' else None))
class TAESDEncoder(nn.Module):
    # 定义类属性 latent_magnitude 为 3
    latent_magnitude = 3
    # 定义类属性 latent_shift 为 0.5
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth"):
        """初始化预训练的 TAESD 模型，从给定的检查点加载到给定设备上。"""
        super().__init__()
        # 创建 encoder 对象
        self.encoder = encoder()
        # 加载 encoder 模型的状态字典
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location='cpu' if devices.device.type != 'cuda' else None))


def download_model(model_path, model_url):
    # 如果模型路径不存在，则创建目录
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 打印下载模型的信息
        print(f'Downloading TAESD model to: {model_path}')
        # 从给定 URL 下载模型到指定路径
        torch.hub.download_url_to_file(model_url, model_path)


def decoder_model():
    # 根据是否为 sdxl 模型选择不同的模型名称
    model_name = "taesdxl_decoder.pth" if getattr(shared.sd_model, 'is_sdxl', False) else "taesd_decoder.pth"
    # 获取已加载的模型
    loaded_model = sd_vae_taesd_models.get(model_name)

    # 如果模型未加载，则下载模型并加载
    if loaded_model is None:
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        download_model(model_path, 'https://github.com/madebyollin/taesd/raw/main/' + model_name)

        # 如果模型文件存在，则创建 TAESDDecoder 模型并加载
        if os.path.exists(model_path):
            loaded_model = TAESDDecoder(model_path)
            loaded_model.eval()
            loaded_model.to(devices.device, devices.dtype)
            sd_vae_taesd_models[model_name] = loaded_model
        else:
            raise FileNotFoundError('TAESD model not found')

    # 返回加载的模型的 decoder 部分
    return loaded_model.decoder


def encoder_model():
    # 根据是否为 sdxl 模型选择不同的模型名称
    model_name = "taesdxl_encoder.pth" if getattr(shared.sd_model, 'is_sdxl', False) else "taesd_encoder.pth"
    # 获取已加载的模型
    loaded_model = sd_vae_taesd_models.get(model_name)
    # 如果加载的模型为空
    if loaded_model is None:
        # 拼接模型路径
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        # 下载模型
        download_model(model_path, 'https://github.com/madebyollin/taesd/raw/main/' + model_name)

        # 如果模型路径存在
        if os.path.exists(model_path):
            # 加载 TAESDEncoder 模型
            loaded_model = TAESDEncoder(model_path)
            # 设置为评估模式
            loaded_model.eval()
            # 将模型移动到指定设备和数据类型
            loaded_model.to(devices.device, devices.dtype)
            # 将加载的模型存储到字典中
            sd_vae_taesd_models[model_name] = loaded_model
        else:
            # 如果模型路径不存在，则抛出文件未找到异常
            raise FileNotFoundError('TAESD model not found')

    # 返回加载的模型的编码器
    return loaded_model.encoder
```