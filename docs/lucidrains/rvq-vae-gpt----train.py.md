# `.\lucidrains\rvq-vae-gpt\train.py`

```
# 导入所需的库
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# 导入自定义模块
from rvq_vae_gpt import TextVQVAE

# 定义常量
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
SAVE_EVERY = 1000
SEQ_LEN = 2048

# 定义辅助函数
def cycle(loader):
    # 无限循环生成数据
    while True:
        for data in loader:
            yield data

def first(it):
    # 返回迭代器的第一个元素
    return it[0]

def decode_token(token):
    # 将 token 解码为字符
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    # 将 tokens 解码为字符串
    return "".join(list(map(decode_token, tokens)))

# 实例化 TextVQVAE 模型
model = TextVQVAE(
    num_tokens = 256,    
    dim = (128, 256, 512),
    depth = (2, 2, 4),
    local_attn_window_size = 64,
    num_codebooks = 8,
    strides = (2, 2, 2)
).cuda()

# 准备 enwik8 数据
with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

# 定义 TextSamplerDataset 类
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

# 创建训练集和验证集的 DataLoader
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# 定义优化器
optim = Adam(model.parameters(), lr = LEARNING_RATE)

# 训练模型
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        loss.backward()

    print(f"training loss: {loss.item():.3f}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i == 0:
        continue

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_text = next(val_loader)
            loss, recon = model(valid_text, return_reconstruction = True)

            print(f"validation loss: {loss.item():.3f}")

            print(f"\n\n\n[input text]\n\n {decode_tokens(first(valid_text))}")
            print(f"\n\n[reconstructed text]\n\n {decode_tokens(first(recon))}\n\n")

    if i % SAVE_EVERY == 0:
        model.save('./text-vae.pt')
```