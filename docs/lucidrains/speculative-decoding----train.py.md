# `.\lucidrains\speculative-decoding\train.py`

```py
# 导入所需的库
import gzip
import random
import tqdm
import numpy as np
import time
from functools import wraps, partial
import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.cuda import synchronize, Event
from torch.utils.data import DataLoader, Dataset
from speculative_decoding import (
    Decoder,
    base_decoding,
    speculative_decoding
)

# 定义常量
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512
GAMMA = 5
DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义辅助函数
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

def benchmark(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        start_event = timer()
        end_event = timer()
        start_event.record()

        out = fn(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return out, elapsed_time_ms
    return inner

# 实例化 Transformer 模型
device = torch.device(DEVICE_STR)
model = Decoder(
    num_tokens = 256,
    dim = 512,
    depth = 10
).to(device)

# 实例化小型 Transformer 模型
small_model = Decoder(
    num_tokens = 256,
    dim = 512,
    depth = 2
).to(device)

# 准备 enwik8 数据
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
small_optim = Adam(small_model.parameters(), lr = LEARNING_RATE)

# 训练过程
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()
    small_model.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        loss = model(data, return_loss = True)
        small_loss = small_model(data, return_loss = True)

        (loss / GRAD_ACCUM_EVERY).backward()
        (small_loss / GRAD_ACCUM_EVERY).backward()

    print(f"training loss: {loss.item():.3f}")
    print(f"training small loss: {small_loss.item():.3f}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(small_model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    small_optim.step()
    small_optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss = model(valid_data, return_loss = True)
            print(f"validation loss: {loss.item():.3f}")

            small_loss = small_model(valid_data, return_loss = True)
            print(f"validation small loss: {small_loss.item():.3f}")
    # 检查是否达到生成频率
    if i % GENERATE_EVERY == 0:
        # 将模型设置为评估模式
        model.eval()
        small_model.eval()

        # 从验证数据集中随机选择一个样本作为输入
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        # 将输入解码为文本
        prime = decode_tokens(inp)
        # 打印输入的提示信息
        print(f"%s \n\n %s", (prime, "*" * 100))

        # 将输入转换为张量
        prompt = inp[None, ...]

        # 使用基本解码函数生成文本序列，并记录基本解码时间
        sampled, base_decode_elapsed = benchmark(base_decoding)(model, prompt, GENERATE_LENGTH)

        # 使用推测解码函数生成文本序列，并记录推测解码时间以及接受的标记数量
        (spec_decode_sampled, num_accepted), spec_decode_elapsed = benchmark(speculative_decoding)(model, small_model, prompt, GENERATE_LENGTH, GAMMA)

        # 将基本解码和推测解码的输出解码为文本
        base_decode_output = decode_tokens(sampled[0])
        spec_decode_output = decode_tokens(spec_decode_sampled[0])

        # 打印基本解码的输出
        print("\nbase decoding:\n\n", base_decode_output, "\n")
        # 打印推测解码的输出
        print("\nspec decoding:\n\n", spec_decode_output, "\n")

        # 打印基本解码的时间
        print(f'base decoding in: {base_decode_elapsed:.3f}ms\n')
        # 打印推测解码的时间
        print(f'spec decoding in: {spec_decode_elapsed:.3f}ms\n')
        # 打印平均接受的标记数量
        print(f'average num accepted: {num_accepted:.1f} / {GAMMA}\n')
```