# `.\lucidrains\vit-pytorch\vit_pytorch\extractor.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn

# 检查值是否存在
def exists(val):
    return val is not None

# 返回输入值
def identity(t):
    return t

# 克隆并分离张量
def clone_and_detach(t):
    return t.clone().detach()

# 应用函数到元组或单个值
def apply_tuple_or_single(fn, val):
    if isinstance(val, tuple):
        return tuple(map(fn, val))
    return fn(val)

# 定义 Extractor 类，继承自 nn.Module
class Extractor(nn.Module):
    def __init__(
        self,
        vit,
        device = None,
        layer = None,
        layer_name = 'transformer',
        layer_save_input = False,
        return_embeddings_only = False,
        detach = True
    ):
        super().__init__()
        # 初始化属性
        self.vit = vit

        self.data = None
        self.latents = None
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

        self.layer = layer
        self.layer_name = layer_name
        self.layer_save_input = layer_save_input # 是否保存层的输入或输出
        self.return_embeddings_only = return_embeddings_only

        # 根据 detach 参数选择克隆并分离函数或返回输入值函数
        self.detach_fn = clone_and_detach if detach else identity

    # 钩子函数，用于提取特征
    def _hook(self, _, inputs, output):
        layer_output = inputs if self.layer_save_input else output
        self.latents = apply_tuple_or_single(self.detach_fn, layer_output)

    # 注册钩子函数
    def _register_hook(self):
        if not exists(self.layer):
            assert hasattr(self.vit, self.layer_name), 'layer whose output to take as embedding not found in vision transformer'
            layer = getattr(self.vit, self.layer_name)
        else:
            layer = self.layer

        handle = layer.register_forward_hook(self._hook)
        self.hooks.append(handle)
        self.hook_registered = True

    # 弹出钩子函数
    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    # 清除特征
    def clear(self):
        del self.latents
        self.latents = None

    # 前向传播函数
    def forward(
        self,
        img,
        return_embeddings_only = False
    ):
        assert not self.ejected, 'extractor has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)

        target_device = self.device if exists(self.device) else img.device
        latents = apply_tuple_or_single(lambda t: t.to(target_device), self.latents)

        if return_embeddings_only or self.return_embeddings_only:
            return latents

        return pred, latents
```