# `.\lucidrains\speculative-decoding\train_prophet.py`

```
# 导入必要的库
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

# 创建计时器
timer = partial(Event, enable_timing = True)

# 导入自定义模块
from speculative_decoding.speculative_decoding_with_prophet import (
    Decoder,
    ModelWithProphetWrapper,
    base_decoding,
    speculative_decoding_with_prophet_model
)

# 定义常量
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
PRIME_LENGTH = 128
GENERATE_EVERY = 100
GENERATE_LENGTH = 512
SEQ_LEN = 512
GAMMA = 5
TRAIN_PROPHET = True

DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义辅助函数

# 生成数据循环
def cycle(loader):
    while True:
        for data in loader:
            yield data

# 解码单个 token
def decode_token(token):
    return str(chr(max(32, token)))

# 解码一组 tokens
def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# 计时装饰器
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
)

prophet = Decoder(
    num_tokens = 256,
    dim = 512,
    depth = 2
)

model_and_prophet = ModelWithProphetWrapper(
    model,
    prophet,
    prophet_train_length = GAMMA + 2,
    num_leading_start_tokens = 2,
    detach_model_embed_for_prophet = False
).to(device)

# 准备 enwik8 数据

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

# 创建数据集类
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

# 创建训练和验证数据集
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))

# 选择优化器参数
params = model_and_prophet.parameters() if TRAIN_PROPHET else model.parameters()

# 创建优化器
optim = Adam(params, lr = LEARNING_RATE)

# 训练循环
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model_and_prophet.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        total_loss, (loss, prophet_loss) = model_and_prophet(data)

        (total_loss / GRAD_ACCUM_EVERY).backward()

    print(f"training loss: {loss.item():.3f}")
    print(f"training prophet loss: {prophet_loss.item():.3f}")

    torch.nn.utils.clip_grad_norm_(model_and_prophet.parameters(), 0.5)

    optim.step()
    optim.zero_grad()
    # 检查是否达到生成频率
    if i % GENERATE_EVERY == 0:
        # 将模型和prophet评估为当前状态
        model_and_prophet.eval()

        # 从验证数据集中随机选择一个样本作为输入
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        # 将输入解码为文本
        prime = decode_tokens(inp)
        # 打印输入的prime文本和分隔符
        print(f"%s \n\n %s", (prime, "*" * 100))

        # 将输入转换为张量
        prompt = inp[None, ...]

        # 使用基本解码函数对模型进行基本解码
        sampled, base_decode_elapsed = benchmark(base_decoding)(model, prompt, GENERATE_LENGTH)

        # 使用带有prophet模型的推测解码函数对模型进行推测解码
        (spec_decode_sampled, num_accepted), spec_decode_elapsed = benchmark(speculative_decoding_with_prophet_model)(model_and_prophet, prompt, GENERATE_LENGTH, GAMMA)

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
        # 打印平均接受的数量
        print(f'average num accepted: {num_accepted:.1f} / {GAMMA}\n')
```