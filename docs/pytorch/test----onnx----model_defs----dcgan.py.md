# `.\pytorch\test\onnx\model_defs\dcgan.py`

```
import torch
import torch.nn as nn

# 可配置参数
bsz = 64  # 批大小
imgsz = 64  # 图像尺寸
nz = 100  # 噪声向量大小
ngf = 64  # 生成器中特征图数量
ndf = 64  # 判别器中特征图数量
nc = 3  # 图像通道数（彩色图像为3）

# 自定义权重初始化函数，用于初始化生成器和判别器中的权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # 如果模块是卷积层，使用正态分布初始化权重
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        # 如果模块是批归一化层，初始化权重为正态分布，偏置为零
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# 生成器网络模型
class _netG(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是 Z，经过卷积层
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 尺寸变化：(ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 尺寸变化：(ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 尺寸变化：(ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 尺寸变化：(ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 尺寸变化：(nc) x 64 x 64，输出生成的图像
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            # 如果有多个GPU且输入是CUDA张量，则在多个GPU上并行计算
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# 判别器网络模型
class _netD(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是 (nc) x 64 x 64 的图像
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 尺寸变化：(ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 尺寸变化：(ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 尺寸变化：(ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 尺寸变化：(ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # 输出判别结果，尺寸变化：1 x 1 x 1，表示真实与假的概率
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            # 如果有多个GPU且输入是CUDA张量，则在多个GPU上并行计算
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)
```