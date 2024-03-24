# `.\lucidrains\product-key-memory\train.py`

```py
# 导入所需的库
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from product_key_memory.transformer import Transformer

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
    seq_len = SEQ_LEN,
    pkm_layers = (4,),
    pkm_kwargs = dict(
        heads = 4,
        num_keys = 128,
        topk = 32,
        dim_head = 128,
        input_dropout = 0.,
        query_dropout = 0.,
        value_dropout = 0.,
        attn_dropout = 0.,
        use_layernorm = True,
        pre_layernorm = True,
        differentiable_topk = False,
        concat_values_and_combine = False
    )
).cuda()

# 准备 enwik8 数据

# 读取并处理 enwik8 数据
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