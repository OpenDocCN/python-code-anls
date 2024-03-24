# `.\lucidrains\reformer-pytorch\examples\enwik8_deepspeed\train.py`

```
# 导入 deepspeed 库
import deepspeed

# 从 reformer_pytorch 库中导入 ReformerLM 类
from reformer_pytorch import ReformerLM
# 从 reformer_pytorch 库中导入 TrainingWrapper 类
from reformer_pytorch.generative_tools import TrainingWrapper

# 导入 argparse 库
import argparse
# 导入 random 库
import random
# 导入 tqdm 库
import tqdm
# 导入 gzip 库
import gzip
# 导入 numpy 库
import numpy as np
# 导入 torch 库
import torch
# 从 torch 中导入 optim 模块
import torch.optim as optim
# 从 torch.nn 中导入 functional 模块
from torch.nn import functional as F
# 从 torch.utils.data 中导入 DataLoader 和 Dataset 类
from torch.utils.data import DataLoader, Dataset

# 定义 add_argument 函数
def add_argument():
    # 创建 ArgumentParser 对象，描述为 'enwik8'
    parser=argparse.ArgumentParser(description='enwik8')

    # 添加参数 '--with_cuda'，默认为 False，支持存储为 True
    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    # 添加参数 '--use_ema'，默认为 False，支持存储为 True
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    # 添加参数 '-b' 或 '--batch_size'，默认为 32，类型为整数
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    # 添加参数 '-e' 或 '--epochs'，默认为 30，类型为整数
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    # 添加参数 '--local_rank'，类型为整数，默认为 -1
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')

    # 调用 deepspeed 库的 add_config_arguments 函数，将参数添加到 parser 中
    parser = deepspeed.add_config_arguments(parser)
    # 解析参数并返回结果
    args=parser.parse_args()
    return args

# 定义常量
EPOCHS = 20
GRADIENT_ACCUMULATE_EVERY = 4
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

# 创建 ReformerLM 模型对象
model = ReformerLM(
    dim = 512,
    depth = 6,
    max_seq_len = SEQ_LEN,
    num_tokens = 256,
    heads = 8,
    bucket_size = 64,
    n_hashes = 4,
    ff_chunks = 10,
    lsh_dropout = 0.1,
    weight_tie = True,
    causal = True,
    n_local_attn_heads = 4,
    use_full_attn = False # set this to true for comparison with full attention
)

# 使用 TrainingWrapper 对模型进行包装
model = TrainingWrapper(model)
# 将模型移至 GPU
model.cuda()

# 准备 enwik8 数据

# 使用 gzip 打开 enwik8.gz 文件
with gzip.open('./data/enwik8.gz') as file:
    # 从文件中读取数据并转换为 numpy 数组
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    # 将数据分为训练集和验证集
    trX, vaX = np.split(X, [int(90e6)])
    # 将数据转换为 torch 张量
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

# 定义 TextSamplerDataset 类，继承自 Dataset 类
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        # 随机选择起始位置
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        # 获取完整序列
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

# 创建训练集和验证集的 Dataset 对象
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

# 设置 deepspeed

# 添加参数并获取命令行参数
cmd_args = add_argument()
# 使用 deepspeed 初始化模型引擎、优化器、训练数据加载器
model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(),  training_data=train_dataset)

# 训练

# 循环训练多个 epochs
for _ in range(EPOCHS):
    # 遍历训练数据加载器
    for i, data in enumerate(trainloader):
        # 设置模型为训练模式
        model_engine.train()
        # 将数据移至本地 GPU
        data = data.to(model_engine.local_rank)
        # 计算损失
        loss = model_engine(data, return_loss = True)
        # 反向传播
        model_engine.backward(loss)
        # 更新参数
        model_engine.step()
        # 打印损失值
        print(loss.item() * GRADIENT_ACCUMULATE_EVERY)

        # 每隔一定步数进行验证
        if i % VALIDATE_EVERY == 0:
            # 设置模型为评估模式
            model.eval()
            with torch.no_grad():
                # 从验证集中随机选择一个样本
                inp = random.choice(val_dataset)[:-1]
                # 计算验证集上的损失
                loss = model(inp[None, :].cuda(), return_loss = True)
                print(f'validation loss: {loss.item()}')

        # 每隔一定步数生成文本
        if i % GENERATE_EVERY == 0:
            # 设置模型为评估模式
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            # 生成文本
            sample = model.generate(inp.cuda(), GENERATE_LENGTH)
            output_str = decode_tokens(sample)
            print(output_str)
```