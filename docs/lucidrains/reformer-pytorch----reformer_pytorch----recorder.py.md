# `.\lucidrains\reformer-pytorch\reformer_pytorch\recorder.py`

```py
# 导入需要的模块
from torch import nn
from reformer_pytorch.reformer_pytorch import LSHAttention, LSHSelfAttention
from collections import defaultdict

# 定义 Recorder 类，继承自 nn.Module
class Recorder(nn.Module):
    # 初始化函数
    def __init__(self, net):
        super().__init__()
        self.iter = 0
        self.recordings = defaultdict(list)  # 使用 defaultdict 创建一个空列表的字典
        self.net = net
        self.on = True
        self.ejected = False

    # 弹出函数
    def eject(self):
        self.ejected = True
        self.clear()
        self.unwire()
        return self.net

    # 连接函数
    def wire(self):
        # 遍历网络中的模块，如果是 LSHAttention 类型，则设置 _return_attn 为 True
        for module in self.net.modules():
            if isinstance(module, LSHAttention):
                module._return_attn = True
            # 如果是 LSHSelfAttention 类型，则设置 callback 为 self.record 函数
            if isinstance(module, LSHSelfAttention):
                module.callback = self.record

    # 断开连接函数
    def unwire(self):
        # 遍历网络中的模块，如果是 LSHAttention 类型，则设置 _return_attn 为 False
        for module in self.net.modules():
            if isinstance(module, LSHAttention):
                module._return_attn = False
            # 如果是 LSHSelfAttention 类型，则设置 callback 为 None
            if isinstance(module, LSHSelfAttention):
                module.callback = None

    # 打开记录功能
    def turn_on(self):
        self.on = True

    # 关闭记录功能
    def turn_off(self):
        self.on = False

    # 清空记录
    def clear(self):
        del self.recordings
        self.recordings = defaultdict(list)  # 使用 defaultdict 创建一个空列表的字典
        self.iter = 0

    # 记录函数
    def record(self, attn, buckets):
        if not self.on: return
        data = {'attn': attn.detach().cpu(), 'buckets': buckets.detach().cpu()}
        self.recordings[self.iter].append(data)

    # 前向传播函数
    def forward(self, x, **kwargs):
        assert not self.ejected, 'Recorder has already been ejected and disposed'
        if self.on:
            self.wire()

        out = self.net(x, **kwargs)

        self.iter += 1
        self.unwire()
        return out
```