# `.\lucidrains\vector-quantize-pytorch\examples\autoencoder.py`

```py
# 导入所需的库
from tqdm.auto import trange
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vector_quantize_pytorch import VectorQuantize

# 设置超参数
lr = 3e-4
train_iter = 1000
num_codes = 256
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义简单的 VQ 自编码器模型
class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, **vq_kwargs):
        super().__init__()
        # 定义模型的层
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                VectorQuantize(dim=32, accept_image_fmap=True, **vq_kwargs),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            ]
        )
        return

    # 前向传播函数
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, VectorQuantize):
                x, indices, commit_loss = layer(x)
            else:
                x = layer(x)

        return x.clamp(-1, 1), indices, commit_loss

# 训练函数
def train(model, train_loader, train_iterations=1000, alpha=10):
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
        out, indices, cmt_loss = model(x)
        rec_loss = (out - x).abs().mean()
        (rec_loss + alpha * cmt_loss).backward()

        opt.step()
        # 更新进度条显示
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"cmt loss: {cmt_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )
    return

# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = DataLoader(
    datasets.FashionMNIST(
        root="~/data/fashion_mnist", train=True, download=True, transform=transform
    ),
    batch_size=256,
    shuffle=True,
)

# 打印信息并开始训练
print("baseline")
torch.random.manual_seed(seed)
model = SimpleVQAutoEncoder(codebook_size=num_codes).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)
```