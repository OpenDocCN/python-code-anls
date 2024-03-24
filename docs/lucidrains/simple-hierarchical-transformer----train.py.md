# `.\lucidrains\simple-hierarchical-transformer\train.py`

```
# 导入必要的库
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# 导入自定义的简单分层Transformer模型
from simple_hierarchical_transformer import HierarchicalTransformer

# 导入加速器库
from accelerate import Accelerator

# 初始化加速器
accelerator = Accelerator()

# 获取设备信息和打印函数
device = accelerator.device
acc_print = accelerator.print

# 定义常量
NUM_BATCHES = int(1e5)
BATCH_SIZE = 2
GRADIENT_ACCUMULATE_EVERY = 8
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
SEQ_LEN = 2048
GENERATE_LENGTH = 1024

# 定义辅助函数
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# 实例化Transformer模型
model = HierarchicalTransformer(
    num_tokens = 256,
    dim = 1024,
    depth = 8,
    seq_len = SEQ_LEN,
    hierarchies = (1, 2),
    window_sizes = (32, 64),
    use_flash_attn = True
).to(device)

# 准备enwik8数据
with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

# 定义数据集类
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

# 创建训练集和验证集
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# 定义优化器
optim = Adam(model.parameters(), lr = LEARNING_RATE)

# 准备模型、优化器和数据加载器
model, optim, train_loader, val_loader = accelerator.prepare(
    model, optim, train_loader, val_loader
)

# 训练模型
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss, (ce_loss, recon_loss, prophet_loss) = model(next(train_loader), return_loss = True)
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

    acc_print(f"training loss: {ce_loss.item()}")
    accelerator.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            _, (ce_loss, *_) = model(next(val_loader), return_loss = True)
            acc_print(f"validation loss: {ce_loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        acc_print(f"%s \n\n %s", (prime, "*" * 100))

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        acc_print(output_str, "\n")
```