# `.\lucidrains\routing-transformer\examples\enwik8_deepspeed\train.py`

```
import deepspeed  # 导入deepspeed库

from routing_transformer import RoutingTransformerLM  # 从routing_transformer库中导入RoutingTransformerLM类
from routing_transformer.autoregressive_wrapper import AutoregressiveWrapper  # 从routing_transformer.autoregressive_wrapper库中导入AutoregressiveWrapper类

import argparse  # 导入argparse库，用于解析命令行参数
import random  # 导入random库，用于生成随机数
import tqdm  # 导入tqdm库，用于显示进度条
import gzip  # 导入gzip库，用于处理gzip文件
import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入torch库，用于构建神经网络
import torch.optim as optim  # 导入torch.optim库，用于优化器
from torch.nn import functional as F  # 从torch.nn库中导入functional模块
from torch.utils.data import DataLoader, Dataset  # 从torch.utils.data库中导入DataLoader和Dataset类

def add_argument():  # 定义函数add_argument，用于添加命令行参数
    parser=argparse.ArgumentParser(description='enwik8')  # 创建ArgumentParser对象，设置描述信息'enwik8'

    parser.add_argument('--with_cuda', default=False, action='store_true',  # 添加--with_cuda参数，默认为False，设置为True时执行store_true操作
                        help='use CPU in case there\'s no GPU support')  # 添加参数帮助信息
    parser.add_argument('--use_ema', default=False, action='store_true',  # 添加--use_ema参数，默认为False，设置为True时执行store_true操作
                        help='whether use exponential moving average')  # 添加参数帮助信息
    parser.add_argument('-b', '--batch_size', default=32, type=int,  # 添加-b或--batch_size参数，默认为32，类型为整数
                        help='mini-batch size (default: 32)')  # 添加参数帮助信息
    parser.add_argument('-e', '--epochs', default=30, type=int,  # 添加-e或--epochs参数，默认为30，类型为整数
                        help='number of total epochs (default: 30)')  # 添加参数帮助信息
    parser.add_argument('--local_rank', type=int, default=-1,  # 添加--local_rank参数，类型为整数，默认值为-1
                       help='local rank passed from distributed launcher')  # 添加参数帮助信息

    parser = deepspeed.add_config_arguments(parser)  # 调用deepspeed库的add_config_arguments函数
    args = parser.parse_args()  # 解析命令行参数并返回
    return args  # 返回参数值

# constants

VALIDATE_EVERY  = 100  # 定义常量VALIDATE_EVERY为100
GENERATE_EVERY  = 500  # 定义常量GENERATE_EVERY为500
GENERATE_LENGTH = 1024  # 定义常量GENERATE_LENGTH为1024
SEQ_LEN = 4096  # 定义常量SEQ_LEN为4096

# helpers

def decode_token(token):  # 定义函数decode_token，用于解码token
    return str(chr(max(32, token)))  # 返回ASCII码对应的字符，如果小于32则返回空格

def decode_tokens(tokens):  # 定义函数decode_tokens，用于解码tokens
    return ''.join(list(map(decode_token, tokens)))  # 将解码后的tokens拼接成字符串

# instantiate model

model = RoutingTransformerLM(  # 创建RoutingTransformerLM模型对象
    num_tokens = 256,  # 设置模型参数num_tokens为256
    dim = 512,  # 设置模型参数dim为512
    depth = 8,  # 设置模型参数depth为8
    max_seq_len = SEQ_LEN,  # 设置模型参数max_seq_len为SEQ_LEN
    heads = 8,  # 设置模型参数heads为8
    causal = True,  # 设置模型参数causal为True
    window_size = 128,  # 设置模型参数window_size为128
    reversible = True,  # 设置模型参数reversible为True
    ff_chunks = 2,  # 设置模型参数ff_chunks为2
    attn_dropout = 0.1,  # 设置模型参数attn_dropout为0.1
    rel_pos_emb = False,  # 设置模型参数rel_pos_emb为False
    n_local_attn_heads = (8, 8, 8, 8, 4, 4, 2, 2)  # 设置模型参数n_local_attn_heads为元组
)

model = AutoregressiveWrapper(model)  # 创建AutoregressiveWrapper对象，包装RoutingTransformerLM模型
model.cuda()  # 将模型移动到GPU上

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:  # 使用gzip打开enwik8.gz文件
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)  # 从文件中读取数据并转换为numpy数组
    trX, vaX = np.split(X, [int(90e6)])  # 将数据分割为训练集和验证集
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)  # 将数据转换为PyTorch张量

class TextSamplerDataset(Dataset):  # 定义TextSamplerDataset类，继承自Dataset类
    def __init__(self, data, seq_len):  # 定义初始化方法，接受数据和序列长度作为参数
        super().__init__()  # 调用父类的初始化方法
        self.data = data  # 设置数据属性
        self.seq_len = seq_len  # 设置序列长度属性

    def __getitem__(self, index):  # 定义获取数据项方法，接受索引作为参数
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))  # 生成随机起始位置
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()  # 获取完整序列
        return full_seq, torch.ones_like(full_seq).bool()  # 返回完整序列和掩码

    def __len__(self):  # 定义长度方法
        return self.data.size(0) // self.seq_len  # 返回数据长度除以序列长度的整数部分

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)  # 创建训练集数据集对象
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)  # 创建验证集数据集对象

# setup deepspeed

cmd_args = add_argument()  # 调用add_argument函数，获取命令行参数
model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(),  training_data=train_dataset)  # 初始化deepspeed

# training

for i, (data, mask) in enumerate(trainloader):  # 遍历训练数据加载器
    model_engine.train()  # 设置模型为训练模式

    data = data.to(model_engine.local_rank)  # 将数据移动到指定设备
    loss = model_engine(data, return_loss = True, randomly_truncate_sequence = True)  # 计算损失
    model_engine.backward(loss)  # 反向传播
    model_engine.step()  # 更新模型参数
    print(loss.item())  # 打印损失值

    if i % VALIDATE_EVERY == 0:  # 每隔VALIDATE_EVERY次迭代进行验证
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            inp, _ = random.choice(val_dataset)  # 从验证集中随机选择一个样本
            loss = model(inp[None, :].cuda(), return_loss = True)  # 计算验证集上的损失
            print(f'validation loss: {loss.item()}')  # 打印验证集损失值

    if i != 0 and model_engine.local_rank == 0 and i % GENERATE_EVERY == 0:  # 每隔GENERATE_EVERY次迭代生成文本
        model.eval()  # 设置模型为评估模式
        inp, _ = random.choice(val_dataset)  # 从验证集中随机选择一个样本
        print(inp.shape, inp)  # 打印输入数据的形状和内容
        prime = decode_tokens(inp)  # 解码输入数据
        print(f'%s \n\n %s', (prime, '*' * 100))  # 打印解码后的输入数据和分隔符

        sample = model.generate(inp.cuda(), GENERATE_LENGTH)  # 生成文本
        output_str = decode_tokens(sample)  # 解码生成的文本
        print(output_str)  # 打印生成的文本
```