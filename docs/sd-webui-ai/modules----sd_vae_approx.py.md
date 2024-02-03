# `stable-diffusion-webui\modules\sd_vae_approx.py`

```
# 导入 os 模块
import os

# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn 模块
from torch import nn
# 从 modules 模块中导入 devices, paths, shared 模块
from modules import devices, paths, shared

# 创建空字典 sd_vae_approx_models
sd_vae_approx_models = {}

# 定义 VAEApprox 类，继承自 nn.Module 类
class VAEApprox(nn.Module):
    # 初始化函数
    def __init__(self):
        super(VAEApprox, self).__init__()
        # 定义多个卷积层
        self.conv1 = nn.Conv2d(4, 8, (7, 7))
        self.conv2 = nn.Conv2d(8, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 32, (3, 3))
        self.conv6 = nn.Conv2d(32, 16, (3, 3))
        self.conv7 = nn.Conv2d(16, 8, (3, 3))
        self.conv8 = nn.Conv2d(8, 3, (3, 3))

    # 前向传播函数
    def forward(self, x):
        # 定义额外的填充值
        extra = 11
        # 对输入进行双线性插值，将尺寸扩大两倍
        x = nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        # 对输入进行填充
        x = nn.functional.pad(x, (extra, extra, extra, extra))

        # 遍历多个卷积层，对输入进行卷积和激活操作
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, ]:
            x = layer(x)
            x = nn.functional.leaky_relu(x, 0.1)

        # 返回处理后的结果
        return x

# 定义下载模型的函数
def download_model(model_path, model_url):
    # 如果模型路径不存在，则创建路径
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 打印下载模型的信息
        print(f'Downloading VAEApprox model to: {model_path}')
        # 使用 torch.hub 下载模型到指定路径
        torch.hub.download_url_to_file(model_url, model_path)

# 定义获取模型的函数
def model():
    # 根据共享模型的属性判断模型名称
    model_name = "vaeapprox-sdxl.pt" if getattr(shared.sd_model, 'is_sdxl', False) else "model.pt"
    # 获取指定模型名称的模型
    loaded_model = sd_vae_approx_models.get(model_name)
    # 如果加载的模型为空
    if loaded_model is None:
        # 拼接模型路径
        model_path = os.path.join(paths.models_path, "VAE-approx", model_name)
        # 如果模型路径不存在
        if not os.path.exists(model_path):
            # 重新设置模型路径
            model_path = os.path.join(paths.script_path, "models", "VAE-approx", model_name)

        # 如果模型路径仍不存在
        if not os.path.exists(model_path):
            # 重新设置模型路径
            model_path = os.path.join(paths.models_path, "VAE-approx", model_name)
            # 下载模型
            download_model(model_path, 'https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/' + model_name)

        # 实例化 VAEApprox 模型
        loaded_model = VAEApprox()
        # 加载模型参数
        loaded_model.load_state_dict(torch.load(model_path, map_location='cpu' if devices.device.type != 'cuda' else None))
        # 设置模型为评估模式
        loaded_model.eval()
        # 将模型移动到指定设备和数据类型
        loaded_model.to(devices.device, devices.dtype)
        # 将加载的模型存储到字典中
        sd_vae_approx_models[model_name] = loaded_model

    # 返回加载的模型
    return loaded_model
# 定义一个函数，用于对输入的样本进行简单的近似处理
def cheap_approximation(sample):
    # 根据链接提供的信息，选择不同的系数矩阵
    if shared.sd_model.is_sdxl:
        coeffs = [
            [ 0.3448,  0.4168,  0.4395],
            [-0.1953, -0.0290,  0.0250],
            [ 0.1074,  0.0886, -0.0163],
            [-0.3730, -0.2499, -0.2088],
        ]
    else:
        coeffs = [
            [ 0.298,  0.207,  0.208],
            [ 0.187,  0.286,  0.173],
            [-0.158,  0.189,  0.264],
            [-0.184, -0.271, -0.473],
        ]

    # 将系数矩阵转换为 PyTorch 张量，并将其移动到与输入样本相同的设备上
    coefs = torch.tensor(coeffs).to(sample.device)

    # 使用 Einstein Summation Notation 对输入样本和系数矩阵进行乘积运算
    x_sample = torch.einsum("...lxy,lr -> ...rxy", sample, coefs)

    # 返回处理后的样本
    return x_sample
```