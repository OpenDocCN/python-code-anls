# `.\lucidrains\block-recurrent-transformer-pytorch\train.py`

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

# 导入加速库
from accelerate import Accelerator
# 导入自定义的模型和训练器
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer, RecurrentTrainerWrapper

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

# 将 token 解码为字符
def decode_token(token):
    return str(chr(max(32, token)))

# 将 tokens 解码为字符串
def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# 初始化加速器
accelerator = Accelerator()

# 获取设备和打印函数
device = accelerator.device
acc_print = accelerator.print

# 实例化模型
model = BlockRecurrentTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8,
    max_seq_len = 1024,
    block_width = 512,
    num_state_vectors = 512,
    recurrent_layers = (4,),
    use_flash_attn = True
)

# 实例化训练器
train_wrapper = RecurrentTrainerWrapper(
    model,
    xl_memories_dropout = 0.1,
    state_dropout = 0.1,
)

# 将模型移动到设备上
model.to(device)

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

# 创建训练集和验证集的数据加载器
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

# 训练过程
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = train_wrapper(next(train_loader))
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

    acc_print(f"training loss: {loss.item()}")
    accelerator.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = train_wrapper(next(val_loader))
            acc_print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        acc_print(f"%s \n\n %s", (prime, "*" * 100))

        sample = train_wrapper.generate(inp[None, ...], length = GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        acc_print(output_str, "\n")
```