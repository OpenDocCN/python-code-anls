# `.\lucidrains\vector-quantize-pytorch\examples\autoencoder_lfq.py`

```
# 导入所需的库
from tqdm.auto import trange
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 导入自定义的 LFQ 模块
from vector_quantize_pytorch import LFQ

# 设置训练参数
lr = 3e-4
train_iter = 1000
seed = 1234
codebook_size = 2 ** 8
entropy_loss_weight = 0.02
diversity_gamma = 1.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义 LFQAutoEncoder 类，继承自 nn.Module
class LFQAutoEncoder(nn.Module):
    def __init__(
        self,
        codebook_size,
        **vq_kwargs
    ):
        super().__init__()
        assert log2(codebook_size).is_integer()
        quantize_dim = int(log2(codebook_size))

        # 编码器部分
        self.encode = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GroupNorm(4, 32, affine=False),  # 添加规范化层
            nn.Conv2d(32, quantize_dim, kernel_size=1),
        )

        # LFQ 模块
        self.quantize = LFQ(dim=quantize_dim, **vq_kwargs)

        # 解码器部分
        self.decode = nn.Sequential(
            nn.Conv2d(quantize_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        return

    # 前向传播函数
    def forward(self, x):
        x = self.encode(x)
        x, indices, entropy_aux_loss = self.quantize(x)
        x = self.decode(x)
        return x.clamp(-1, 1), indices, entropy_aux_loss

# 训练函数
def train(model, train_loader, train_iterations=1000):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    # 迭代训练数据集
    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))
        out, indices, entropy_aux_loss = model(x)

        rec_loss = F.l1_loss(out, x)
        (rec_loss + entropy_aux_loss).backward()

        opt.step()
        pbar.set_description(
              f"rec loss: {rec_loss.item():.3f} | "
            + f"entropy aux loss: {entropy_aux_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / codebook_size * 100:.3f}"
        )
    return

# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# 加载 FashionMNIST 数据集
train_dataset = DataLoader(
    datasets.FashionMNIST(
        root="~/data/fashion_mnist", train=True, download=True, transform=transform
    ),
    batch_size=256,
    shuffle=True,
)

# 打印提示信息
print("baseline")

# 设置随机种子
torch.random.manual_seed(seed)

# 创建 LFQAutoEncoder 模型实例
model = LFQAutoEncoder(
    codebook_size = codebook_size,
    entropy_loss_weight = entropy_loss_weight,
    diversity_gamma = diversity_gamma
).to(device)

# 定义优化器
opt = torch.optim.AdamW(model.parameters(), lr=lr)

# 训练模型
train(model, train_dataset, train_iterations=train_iter)
```