# `.\pytorch\test\package\package_a\test_nn_module.py`

```
# Owner(s): ["oncall: package/deploy"]

import torch


class TestNnModule(torch.nn.Module):
    def __init__(self, nz=6, ngf=9, nc=3):
        super().__init__()
        # 定义神经网络的主要结构
        self.main = torch.nn.Sequential(
            # 输入是 Z，经过一层转置卷积
            torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),  # 批量归一化
            torch.nn.ReLU(True),  # ReLU 激活函数
            # 输出尺寸：(ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),  # 批量归一化
            torch.nn.ReLU(True),  # ReLU 激活函数
            # 输出尺寸：(ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),  # 批量归一化
            torch.nn.ReLU(True),  # ReLU 激活函数
            # 输出尺寸：(ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),  # 批量归一化
            torch.nn.ReLU(True),  # ReLU 激活函数
            # 输出尺寸：(ngf) x 32 x 32
            torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()  # Tanh 激活函数，输出范围在[-1, 1]
            # 输出尺寸：(nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```