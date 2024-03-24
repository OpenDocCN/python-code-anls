# `.\lucidrains\mixture-of-attention\train.py`

```py
# 导入必要的库
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from mixture_of_attention.transformer import Transformer
from mixture_of_attention.autoregressive_wrapper import AutoregressiveWrapper

# 定义常量
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

# 定义辅助函数

# 从 token 解码为字符
def decode_token(token):
    return str(chr(max(32, token)))

# 从 tokens 解码为字符串
def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# 实例化 Transformer 模型
model = Transformer(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    num_experts = 2,
    seq_len = SEQ_LEN,
    local_attn_window_size = 64,
    num_routed_queries = 32,
    num_routed_key_values = 64,
    cosine_sim_routing = True,
    use_triton = True
)

model = AutoregressiveWrapper(model).cuda()

# 准备 enwik8 数据

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

# 定义自定义数据集类
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
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
        loss.backward(loss / GRADIENT_ACCUMULATE_EVERY)

    print(f"training loss: {loss.item()}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(output_str, "\n")
```