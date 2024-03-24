# `.\lucidrains\all-normalization-transformer\train_enwik8.py`

```py
# 导入所需的模块
from all_normalization_transformer import TransformerLM
from all_normalization_transformer.autoregressive_wrapper import AutoregressiveWrapper
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# 定义常量
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 3e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

# 定义辅助函数

# 从 token 解码为字符
def decode_token(token):
    return str(chr(max(32, token)))

# 从 tokens 解码为字符串
def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# 实例化模型

# 创建 TransformerLM 模型对象
model = TransformerLM(
    num_tokens = 256,
    dim = 512,
    depth = 12,
    max_seq_len = SEQ_LEN,
    heads = 8,
    causal = True,
    only_norm = True,
    shared_kv = True
)

# 将模型包装为 AutoregressiveWrapper
model = AutoregressiveWrapper(model)
# 将模型移动到 GPU 上
model.cuda()

# 准备 enwik8 数据

# 从压缩文件中读取数据
with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

# 定义自定义数据集类
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

# 创建训练集和验证集的数据集对象
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
# 创建训练集和验证集的数据加载器
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# 定义优化器
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练模型
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            print(f'validation loss: {loss.item()}')

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        inp = inp[:SEQ_LEN]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = decode_tokens(sample)
        print(output_str)
```