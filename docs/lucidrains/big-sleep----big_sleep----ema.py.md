# `.\lucidrains\big-sleep\big_sleep\ema.py`

```
# 导入必要的库
from copy import deepcopy
import torch
from torch import nn

# 定义指数移动平均类
class EMA(nn.Module):
    # 初始化函数，接受模型和衰减率作为参数
    def __init__(self, model, decay):
        super().__init__()
        self.model = model
        self.decay = decay
        # 注册缓冲区
        self.register_buffer('accum', torch.tensor(1.))
        self._biased = deepcopy(self.model)
        self.average = deepcopy(self.model)
        # 将偏置参数和平均参数初始化为零
        for param in self._biased.parameters():
            param.detach_().zero_()
        for param in self.average.parameters():
            param.detach_().zero_()
        # 更新参数
        self.update()

    # 更新函数，用于更新指数移动平均
    @torch.no_grad()
    def update(self):
        assert self.training, 'Update should only be called during training'

        # 更新累积值
        self.accum *= self.decay

        # 获取模型参数、偏置参数和平均参数
        model_params = dict(self.model.named_parameters())
        biased_params = dict(self._biased.named_parameters())
        average_params = dict(self.average.named_parameters())
        assert model_params.keys() == biased_params.keys() == average_params.keys(), f'Model parameter keys incompatible with EMA stored parameter keys'

        # 更新参数
        for name, param in model_params.items():
            biased_params[name].mul_(self.decay)
            biased_params[name].add_((1 - self.decay) * param)
            average_params[name].copy_(biased_params[name])
            average_params[name].div_(1 - self.accum)

        # 获取模型缓冲区、偏置缓冲区和平均缓冲区
        model_buffers = dict(self.model.named_buffers())
        biased_buffers = dict(self._biased.named_buffers())
        average_buffers = dict(self.average.named_buffers())
        assert model_buffers.keys() == biased_buffers.keys() == average_buffers.keys()

        # 更新缓冲区
        for name, buffer in model_buffers.items():
            biased_buffers[name].copy_(buffer)
            average_buffers[name].copy_(buffer)

    # 前向传播函数，根据是否处于训练状态返回模型或平均模型的输出
    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.average(*args, **kwargs)
```