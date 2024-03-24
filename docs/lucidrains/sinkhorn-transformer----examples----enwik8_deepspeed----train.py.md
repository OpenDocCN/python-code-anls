# `.\lucidrains\sinkhorn-transformer\examples\enwik8_deepspeed\train.py`

```py
import deepspeed  # 导入deepspeed库

from sinkhorn_transformer import SinkhornTransformerLM  # 从sinkhorn_transformer库中导入SinkhornTransformerLM类
from sinkhorn_transformer.autoregressive_wrapper import AutoregressiveWrapper  # 从sinkhorn_transformer库中导入AutoregressiveWrapper类

import argparse  # 导入argparse库
import random  # 导入random库
import tqdm  # 导入tqdm库
import gzip  # 导入gzip库
import numpy as np  # 导入numpy库，并重命名为np
import torch  # 导入torch库
import torch.optim as optim  # 从torch库中导入optim模块
from torch.nn import functional as F  # 从torch库中导入functional模块，并重命名为F
from torch.utils.data import DataLoader, Dataset  # 从torch.utils.data库中导入DataLoader和Dataset类

def add_argument():  # 定义函数add_argument
    parser=argparse.ArgumentParser(description='enwik8')  # 创建一个ArgumentParser对象，设置描述信息为'enwik8'

    parser.add_argument('--with_cuda', default=False, action='store_true',  # 添加一个参数'--with_cuda'，默认值为False，如果存在则设置为True
                        help='use CPU in case there\'s no GPU support')  # 设置参数'--with_cuda'的帮助信息
    parser.add_argument('--use_ema', default=False, action='store_true',  # 添加一个参数'--use_ema'，默认值为False，如果存在则设置为True
                        help='whether use exponential moving average')  # 设置参数'--use_ema'的帮助信息
    parser.add_argument('-b', '--batch_size', default=32, type=int,  # 添加一个参数'-b'或'--batch_size'，默认值为32，类型为整数
                        help='mini-batch size (default: 32)')  # 设置参数'-b'或'--batch_size'的帮助信息
    parser.add_argument('-e', '--epochs', default=30, type=int,  # 添加一个参数'-e'或'--epochs'，默认值为30，类型为整数
                        help='number of total epochs (default: 30)')  # 设置参数'-e'或'--epochs'的帮助信息
    parser.add_argument('--local_rank', type=int, default=-1,  # 添加一个参数'--local_rank'，类型为整数，默认值为-1
                       help='local rank passed from distributed launcher')  # 设置参数'--local_rank'的帮助信息

    parser = deepspeed.add_config_arguments(parser)  # 调用deepspeed库中的add_config_arguments函数
    args = parser.parse_args()  # 解析命令行参数
    return args  # 返回参数args

# constants

VALIDATE_EVERY  = 100  # 定义常量VALIDATE_EVERY为100
GENERATE_EVERY  = 500  # 定义常量GENERATE_EVERY为500
GENERATE_LENGTH = 1024  # 定义常量GENERATE_LENGTH为1024
SEQ_LEN = 4096  # 定义常量SEQ_LEN为4096

# helpers

def decode_token(token):  # 定义函数decode_token，接受一个token参数
    return str(chr(max(32, token)))  # 返回ASCII码对应的字符，如果小于32则返回空格

def decode_tokens(tokens):  # 定义函数decode_tokens，接受一个tokens参数
    return ''.join(list(map(decode_token, tokens)))  # 将tokens中的每个token转换为字符，并拼接成字符串

# instantiate model

model = SinkhornTransformerLM(  # 创建SinkhornTransformerLM模型对象
    num_tokens = 256,  # 设置num_tokens参数为256
    emb_dim = 128,  # 设置emb_dim参数为128
    dim = 512,  # 设置dim参数为512
    depth = 8,  # 设置depth参数为8
    max_seq_len = SEQ_LEN,  # 设置max_seq_len参数为SEQ_LEN
    heads = 8,  # 设置heads参数为8
    bucket_size = 128,  # 设置bucket_size参数为128
    ff_chunks = 10,  # 设置ff_chunks参数为10
    causal = True,  # 设置causal参数为True
    reversible = True,  # 设置reversible参数为True
    attn_dropout = 0.1,  # 设置attn_dropout参数为0.1
    n_local_attn_heads = 4  # 设置n_local_attn_heads参数为4
)

model = AutoregressiveWrapper(model)  # 使用AutoregressiveWrapper对模型进行包装
model.cuda()  # 将模型移动到GPU上

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:  # 打开enwik8.gz文件
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)  # 从文件中读取数据，转换为numpy数组
    trX, vaX = np.split(X, [int(90e6)])  # 将数据分割为训练集和验证集
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)  # 将数据转换为PyTorch张量

class TextSamplerDataset(Dataset):  # 定义TextSamplerDataset类，继承自Dataset类
    def __init__(self, data, seq_len):  # 定义初始化方法，接受data和seq_len参数
        super().__init__()  # 调用父类的初始化方法
        self.data = data  # 设置数据属性为传入的data
        self.seq_len = seq_len  # 设置序列长度属性为传入的seq_len

    def __getitem__(self, index):  # 定义获取数据项方法，接受index参数
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))  # 生成随机起始位置
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()  # 获取完整序列
        return full_seq  # 返回完整序列

    def __len__(self):  # 定义长度方法
        return self.data.size(0) // self.seq_len  # 返回数据长度除以序列长度的整数部分

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)  # 创建训练集数据集对象
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)  # 创建验证集数据集对象

# setup deepspeed

cmd_args = add_argument()  # 调用add_argument函数，获取命令行参数
model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(),  training_data=train_dataset)  # 使用deepspeed初始化模型引擎、优化器、训练数据加载器

# training

for i, data in enumerate(trainloader):  # 遍历训练数据加载器
    model_engine.train()  # 设置模型为训练模式
    data = data.to(model_engine.local_rank)  # 将数据移动到指定设备
    loss = model_engine(data, return_loss = True)  # 计算损失
    model_engine.backward(loss)  # 反向传播
    model_engine.step()  # 更新模型参数
    print(loss.item() * 4)  # 打印损失值

    if i % VALIDATE_EVERY == 0:  # 每隔VALIDATE_EVERY次迭代进行验证
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            inp = random.choice(val_dataset)  # 从验证集中随机选择一个样本
            loss = model(inp[None, :].cuda(), return_loss = True)  # 计算验证集上的损失
            print(f'validation loss: {loss.item()}')  # 打印验证损失值

    if model_engine.local_rank == 0 and i % GENERATE_EVERY == 0:  # 如果是主进程且每隔GENERATE_EVERY次迭代生成样本
        model.eval()  # 设置��型为评估模式
        inp = random.choice(val_dataset)[:-1]  # 从验证集中随机选择一个样本，并去掉最后一个字符
        prime = decode_tokens(inp)  # 解码得到的输入序列
        print(f'%s \n\n %s', (prime, '*' * 100))  # 打印输入序列和分隔符

        sample = model.generate(inp.cuda(), GENERATE_LENGTH)  # 生成样本
        output_str = decode_tokens(sample)  # 解码生成的样本
        print(output_str)  # 打印生成的样本
```