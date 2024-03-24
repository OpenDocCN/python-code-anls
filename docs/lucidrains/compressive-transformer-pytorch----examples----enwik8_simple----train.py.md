# `.\lucidrains\compressive-transformer-pytorch\examples\enwik8_simple\train.py`

```py
# 导入所需的库
from compressive_transformer_pytorch import CompressiveTransformer
from compressive_transformer_pytorch.autoregressive_wrapper import AutoregressiveWrapper
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
BATCH_SIZE = 16
MAX_BATCH_SIZE = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH    = 512
GENERATE_LENGTH = 1024
SEQ_LEN = 512
NUM_SEGMENTS = 4

# 定义辅助函数
def cycle(loader):
    # 无限循环生成数据
    while True:
        for data in loader:
            yield data

def decode_token(token):
    # 将 token 解码为字符
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    # 将 tokens 解码为字符串
    return ''.join(list(map(decode_token, tokens)))

# 实例化模型
model = CompressiveTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    seq_len = SEQ_LEN,
    mem_len = SEQ_LEN,
    cmem_len = SEQ_LEN // 4,
    heads = 8,
    memory_layers = [6,7,8]
)
model = AutoregressiveWrapper(model)
model.cuda()

# 准备 enwik8 数据
with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

# 定义数据集类
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, segments):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.segments = segments
        self.total_len = seq_len * segments

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.total_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.total_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.total_len

# 创建训练集和验证集
train_dataset = TextSamplerDataset(data_train, SEQ_LEN, NUM_SEGMENTS)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN, NUM_SEGMENTS)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# 定义优化器
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练模型
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    grad_accum_every = BATCH_SIZE / MAX_BATCH_SIZE

    for mlm_loss, aux_loss, is_last in model(next(train_loader), max_batch_size = MAX_BATCH_SIZE, return_loss = True):
        loss = mlm_loss + aux_loss
        (loss / grad_accum_every).backward()

        print(f'training loss: {mlm_loss.item():.4f} | aux_loss: {aux_loss.item():.4f}')

        if is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            for loss, aux_loss, _ in model(next(val_loader), return_loss = True):
                print(f'validation loss: {loss.item():.4f}')

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        inp = inp[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = decode_tokens(sample)
        print(output_str)
```