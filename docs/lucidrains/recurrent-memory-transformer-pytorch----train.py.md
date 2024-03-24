# `.\lucidrains\recurrent-memory-transformer-pytorch\train.py`

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

# 导入自定义的模型和包装器
from recurrent_memory_transformer_pytorch import RecurrentMemoryTransformer, RecurrentMemoryTransformerWrapper

# 定义常量
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 250
GENERATE_LENGTH = 2048
SEQ_LEN = 2048

# 定义辅助函数

# 从 token 解码为字符
def decode_token(token):
    return str(chr(max(32, token)))

# 从 tokens 解码为字符串
def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# 实例化 RecurrentMemoryTransformer 模型
model = RecurrentMemoryTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8,
    seq_len = 512,
    use_flash_attn = True,
    num_memory_tokens = 128,
    use_xl_memories = True,
    xl_mem_len = 256
)

# 使用包装器对模型进行包装
model = RecurrentMemoryTransformerWrapper(model)

# 将模型移至 GPU
model.cuda()

# 准备 enwik8 数据

# 从压缩文件中读取数据
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
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

# 创建训练集和验证集的数据加载器
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# 定义优化器
optim = Adam(model.parameters(), lr = LEARNING_RATE)

# 训练模型
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model.train()

    total_loss = 0.
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(
            next(train_loader),
            memory_replay_backprop = True,
            mrbp_loss_weight = 1. / GRADIENT_ACCUMULATE_EVERY
        )

        total_loss += loss

    print(f"training loss: {total_loss.item()}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss, _ = model(next(val_loader), return_loss = True)
            print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        sample = model.generate(inp[None, :], length = GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(output_str, "\n")
```