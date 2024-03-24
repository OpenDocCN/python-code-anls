# `.\lucidrains\vit-pytorch\vit_pytorch\recorder.py`

```
# 从 functools 模块导入 wraps 装饰器
from functools import wraps
# 导入 torch 模块
import torch
# 从 torch 模块导入 nn 模块
from torch import nn

# 从 vit_pytorch.vit 模块导入 Attention 类
from vit_pytorch.vit import Attention

# 定义一个函数，用于查找指定类型的模块
def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

# 定义一个 Recorder 类，继承自 nn.Module 类
class Recorder(nn.Module):
    # 初始化方法
    def __init__(self, vit, device = None):
        super().__init__()
        self.vit = vit

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    # 私有方法，用于注册钩子函数
    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    # 注册钩子函数的方法
    def _register_hook(self):
        # 查找所有 transformer 模块中的 Attention 模块
        modules = find_modules(self.vit.transformer, Attention)
        # 为每个 Attention 模块注册前向钩子函数
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    # 弹出 Recorder 对象的方法
    def eject(self):
        self.ejected = True
        # 移除所有钩子函数
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    # 清空记录数据的方法
    def clear(self):
        self.recordings.clear()

    # 记录数据的方法
    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    # 前向传播方法
    def forward(self, img):
        # 断言 Recorder 对象未被弹出
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        # 如果钩子函数未注册，则注册钩子函数
        if not self.hook_registered:
            self._register_hook()

        # 对输入图片进行预测
        pred = self.vit(img)

        # 将所有记录数据移动到指定设备上
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings))

        # 如果有记录数据，则在指定维度上堆叠
        attns = torch.stack(recordings, dim = 1) if len(recordings) > 0 else None
        return pred, attns
```