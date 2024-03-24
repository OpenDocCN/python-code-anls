# `.\lucidrains\linear-attention-transformer\examples\enwik8_deepspeed\train.py`

```
# 导入 deepspeed 库
import deepspeed

# 从 linear_attention_transformer 模块中导入 LinearAttentionTransformerLM 类
from linear_attention_transformer import LinearAttentionTransformerLM
# 从 linear_attention_transformer.autoregressive_wrapper 模块中导入 AutoregressiveWrapper 类
from linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper

# 导入 argparse 库
import argparse
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# 定义函数 add_argument，用于解析命令行参数
def add_argument():
    parser=argparse.ArgumentParser(description='enwik8')

    # 添加命令行参数
    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')

    # 添加 deepspeed 配置参数
    parser = deepspeed.add_config_arguments(parser)
    # 解析命令行参数
    args = parser.parse_args()
    return args

# 定义常量
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 4096

# 定义辅助函数

# 解码单个 token
def decode_token(token):
    return str(chr(max(32, token)))

# 解码一组 tokens
def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# 实例化模型

# 创建 LinearAttentionTransformerLM 模型对象
model = LinearAttentionTransformerLM(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    max_seq_len = SEQ_LEN,
    heads = 8,
    causal = True,
    reversible = True,
    blindspot_size = 2,
    shift_tokens = True,
    n_local_attn_heads = (8, 8, 8, 8, 4, 4, 2, 2)
)

# 将模型包装在 AutoregressiveWrapper 中
model = AutoregressiveWrapper(model)
# 将模型移动到 GPU
model.cuda()

# 准备 enwik8 数据

# 使用 gzip 打开 enwik8 数据文件
with gzip.open('./data/enwik8.gz') as file:
    # 从文件中读取数据并转换为 numpy 数组
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    # 将数据分为训练集和验证集
    trX, vaX = np.split(X, [int(90e6)])
    # 将数据转换为 PyTorch 张量
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

# 定义 TextSamplerDataset 类，用于创建数据集
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq, torch.ones_like(full_seq).bool()

    def __len__(self):
        return self.data.size(0) // self.seq_len

# 创建训练集和验证集数据集对象
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

# 设置 deepspeed

# 解析命令行参数
cmd_args = add_argument()
# 初始化 deepspeed
model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(),  training_data=train_dataset)

# 训练

# 遍历训练数据加载器
for i, (data, mask) in enumerate(trainloader):
    model_engine.train()

    # 将数据移动到指定设备
    data = data.to(model_engine.local_rank)
    # 计算损失
    loss = model_engine(data, return_loss = True, randomly_truncate_sequence = True)
    # 反向传播
    model_engine.backward(loss)
    # 更新参数
    model_engine.step()
    print(loss.item())

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            inp, _ = random.choice(val_dataset)
            loss = model(inp[None, :].cuda(), return_loss = True)
            print(f'validation loss: {loss.item()}')

    if i != 0 and model_engine.local_rank == 0 and i % GENERATE_EVERY == 0:
        model.eval()
        inp, _ = random.choice(val_dataset)
        print(inp.shape, inp)
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp.cuda(), GENERATE_LENGTH)
        output_str = decode_tokens(sample)
        print(output_str)
```