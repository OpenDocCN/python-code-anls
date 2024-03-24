# `.\lucidrains\performer-pytorch\examples\enwik8_deepspeed\train.py`

```py
import deepspeed  # 导入deepspeed库

from performer_pytorch import PerformerLM  # 从performer_pytorch库中导入PerformerLM类
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper  # 从performer_pytorch.autoregressive_wrapper库中导入AutoregressiveWrapper类

import argparse  # 导入argparse库，用于解析命令行参数
import random  # 导入random库，用于生成随机数
import tqdm  # 导入tqdm库，用于显示进度条
import gzip  # 导入gzip库，用于处理gzip压缩文件
import numpy as np  # 导入numpy库，用于处理数组
import torch  # 导入torch库，用于构建神经网络
import torch.optim as optim  # 从torch库中导入optim模块，用于定义优化器
from torch.nn import functional as F  # 从torch库中导入functional模块，用于定义神经网络的函数
from torch.utils.data import DataLoader, Dataset  # 从torch.utils.data库中导入DataLoader和Dataset类，用于处理数据集

def add_argument():  # 定义函数add_argument，用于添加命令行参数
    parser=argparse.ArgumentParser(description='enwik8')  # 创建一个ArgumentParser对象，设置描述信息为'enwik8'

    parser.add_argument('--with_cuda', default=False, action='store_true',  # 添加一个名为'--with_cuda'的命令行参数，默认值为False，如果存在则设置为True
                        help='use CPU in case there\'s no GPU support')  # 设置参数的帮助信息
    parser.add_argument('--use_ema', default=False, action='store_true',  # 添加一个名为'--use_ema'的命令行参数，默认值为False，如果存在则设置为True
                        help='whether use exponential moving average')  # 设置参数的帮助信息
    parser.add_argument('-b', '--batch_size', default=32, type=int,  # 添加一个名为'-b'或'--batch_size'的命令行参数，默认值为32，类型为整数
                        help='mini-batch size (default: 32)')  # 设置参数的帮助信息
    parser.add_argument('-e', '--epochs', default=30, type=int,  # 添加一个名为'-e'或'--epochs'的命令行参数，默认值为30，类型为整数
                        help='number of total epochs (default: 30)')  # 设置参数的帮助信息
    parser.add_argument('--local_rank', type=int, default=-1,  # 添加一个名为'--local_rank'的命令行参数，类型为整数，默认值为-1
                       help='local rank passed from distributed launcher')  # 设置参数的帮助信息

    parser = deepspeed.add_config_arguments(parser)  # 调用deepspeed库中的add_config_arguments函数，添加配置参数
    args=parser.parse_args()  # 解析命令行参数并返回结果
    return args  # 返回解析后的参数对象

# constants

EPOCHS = 20  # 定义常量EPOCHS为20，表示训练的总轮数
VALIDATE_EVERY  = 100  # 定义常量VALIDATE_EVERY为100，表示每隔100步进行一次验证
GENERATE_EVERY  = 500  # 定义常量GENERATE_EVERY为500，表示每隔500步生成一次数据
GENERATE_LENGTH = 512  # 定义常量GENERATE_LENGTH为512，表示生成数据的长度
SEQ_LEN = 1024  # 定义常量SEQ_LEN为1024，表示序列的长度

# helpers

def decode_token(token):  # 定义函数decode_token，用于将token解码为字符
    return str(chr(max(32, token)))  # 返回ASCII码对应的字符，如果小于32则返回空格

def decode_tokens(tokens):  # 定义函数decode_tokens，用于将tokens解码为字符串
    return ''.join(list(map(decode_token, tokens)))  # 将tokens中的每个token解码为字符并拼接成字符串

# instantiate model

model = PerformerLM(  # 创建PerformerLM模型对象
    num_tokens = 256,  # 设置模型的token数量为256
    dim = 512,  # 设置模型的维度为512
    depth = 6,  # 设置模型的深度为6
    max_seq_len = SEQ_LEN,  # 设置模型的最大序列长度为SEQ_LEN
    heads = 8,  # 设置模型的头数为8
    causal = True,  # 设置模型为因果模型
    reversible = True,  # 设置模型为可逆模型
    nb_features = 256,  # 设置模型的特征数量为256
    use_scalenorm = True,  # 设置模型使用scalenorm
)

model = AutoregressiveWrapper(model)  # 使用AutoregressiveWrapper对模型进行包装
model.cuda()  # 将模型移动到GPU上

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:  # 打开enwik8.gz文件
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)  # 从文件中读取数据并转换为numpy数组
    trX, vaX = np.split(X, [int(90e6)])  # 将数据分割为训练集和验证集
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)  # 将数据转换为PyTorch张量

class TextSamplerDataset(Dataset):  # 定义TextSamplerDataset类，继承自Dataset类
    def __init__(self, data, seq_len):  # 定义初始化方法，接受数据和序列长度作为参数
        super().__init__()  # 调用父类的初始化方法
        self.data = data  # 设置数据属性
        self.seq_len = seq_len  # 设置序列长度属性

    def __getitem__(self, index):  # 定义获取数据项的方法
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))  # 随机生成起始位置
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()  # 获取完整序列
        return full_seq  # 返回完整序列

    def __len__(self):  # 定义获取数据集长度的方法
        return self.data.size(0) // self.seq_len  # 返回数据集长度除以序列长度的整数部分

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)  # 创建训练数据集对象
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)  # 创建验证数据集对象

# setup deepspeed

cmd_args = add_argument()  # 调用add_argument函数，获取命令行参数
model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(),  training_data=train_dataset)  # 使用deepspeed初始化模型引擎、优化器、数据加载器

# training

for _ in range(EPOCHS):  # 循环训练EPOCHS轮
    for i, data in enumerate(trainloader):  # 遍历训练数据加载器
        model_engine.train()  # 设置模型为训练模式
        data = data.to(model_engine.local_rank)  # 将数据移动到指定设备
        loss = model_engine(data, return_loss = True)  # 计算损失
        model_engine.backward(loss)  # 反向传播计算梯度
        model_engine.step()  # 更新模型参数
        print(loss.item() * GRADIENT_ACCUMULATE_EVERY)  # 打印损失值乘以梯度累积步数

        if model_engine.local_rank != 0:  # 如果不是主进程
            continue  # 继续下一次循环

        if i % VALIDATE_EVERY == 0:  # 每隔VALIDATE_EVERY步进行一次验证
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():  # 禁用梯度计算
                inp = random.choice(val_dataset)[:-1]  # 从验证集中随机选择一个输入序列
                loss = model(inp[None, :].cuda(), return_loss = True)  # 计算验证集上的损失
                print(f'validation loss: {loss.item()}')  # 打印验证损失值

        if i % GENERATE_EVERY == 0:  # 每隔GENERATE_EVERY步生成一次数据
            model.eval()  # 设置模型为评估模式
            inp = random.choice(val_dataset)[:-1]  # 从验证集中随机选择一个输入序列
            prime = decode_tokens(inp)  # 解码输入序列
            print(f'%s \n\n %s', (prime, '*' * 100))  # 打印输入序列和分隔符

            sample = model.generate(inp.cuda(), GENERATE_LENGTH)  # 生成数据
            output_str = decode_tokens(sample)  # 解码生成的数据
            print(output_str)  # 打印生成的数据
```