# `stable-diffusion-webui\modules\models\diffusion\uni_pc\sampler.py`

```
"""SAMPLING ONLY."""
# 仅用于采样

import torch
# 导入 torch 库

from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC
# 从 uni_pc 模块中导入 NoiseScheduleVP、model_wrapper、UniPC 类
from modules import shared, devices
# 从 modules 模块中导入 shared、devices 模块

class UniPCSampler(object):
    # 定义 UniPCSampler 类
    def __init__(self, model, **kwargs):
        # 初始化函数，接受模型和其他参数
        super().__init__()
        # 调用父类的初始化函数
        self.model = model
        # 设置模型属性
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        # 定义一个将输入转换为 torch 张量的函数
        self.before_sample = None
        # 初始化 before_sample 属性为 None
        self.after_sample = None
        # 初始化 after_sample 属性为 None
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))
        # 注册缓冲区，将 model.alphas_cumprod 转换为 torch 张量并存储在 alphas_cumprod 属性中

    def register_buffer(self, name, attr):
        # 定义注册缓冲区的方法
        if type(attr) == torch.Tensor:
            # 如果属性是 torch 张量
            if attr.device != devices.device:
                # 如果属性的设备不是指定的设备
                attr = attr.to(devices.device)
                # 将属性转移到指定设备
        setattr(self, name, attr)
        # 设置对象的属性为给定的属性值

    def set_hooks(self, before_sample, after_sample, after_update):
        # 定义设置钩子的方法
        self.before_sample = before_sample
        # 设置 before_sample 属性为给定的 before_sample
        self.after_sample = after_sample
        # 设置 after_sample 属性为给定的 after_sample
        self.after_update = after_update
        # 设置 after_update 属性为给定的 after_update

    @torch.no_grad()
    # 使用 torch.no_grad() 上下文管理器，禁用梯度计算
```