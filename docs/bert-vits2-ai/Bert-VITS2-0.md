# Bert-VITS2 源码解析 0

# `D:\src\Bert-VITS2\attentions.py`

```python
import math  # 导入数学库
import torch  # 导入PyTorch
from torch import nn  # 从PyTorch中导入神经网络
from torch.nn import functional as F  # 从PyTorch中导入函数式模块

import commons  # 导入自定义的commons模块
import logging  # 导入日志模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class LayerNorm(nn.Module):  # 定义LayerNorm类，继承自nn.Module
    def __init__(self, channels, eps=1e-5):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.channels = channels  # 设置channels属性
        self.eps = eps  # 设置eps属性

        self.gamma = nn.Parameter(torch.ones(channels))  # 初始化gamma参数
        self.beta = nn.Parameter(torch.zeros(channels))  # 初始化beta参数

    def forward(self, x):  # 前向传播函数
        x = x.transpose(1, -1)  # 调整x的维度
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)  # 对x进行Layer Norm
        return x.transpose(1, -1)  # 返回调整维度后的x


@torch.jit.script  # 使用Torch Script装饰器
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):  # 定义fused_add_tanh_sigmoid_multiply函数
    n_channels_int = n_channels[0]  # 获取n_channels的第一个元素
    in_act = input_a + input_b  # 计算input_a和input_b的和
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # 计算tanh激活函数
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # 计算sigmoid激活函数
    acts = t_act * s_act  # 计算t_act和s_act的乘积
    return acts  # 返回结果


class Encoder(nn.Module):  # 定义Encoder类，继承自nn.Module
    def __init__(  # 初始化函数
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        isflow=True,
        **kwargs
    ):
        super().__init__()  # 调用父类的初始化函数
        self.hidden_channels = hidden_channels  # 设置hidden_channels属性
        self.filter_channels = filter_channels  # 设置filter_channels属性
        self.n_heads = n_heads  # 设置n_heads属性
        self.n_layers = n_layers  # 设置n_layers属性
        self.kernel_size = kernel_size  # 设置kernel_size属性
        self.p_dropout = p_dropout  # 设置p_dropout属性
        self.window_size = window_size  # 设置window_size属性
        self.cond_layer_idx = self.n_layers  # 设置cond_layer_idx属性
        if "gin_channels" in kwargs:  # 如果gin_channels在kwargs中
            self.gin_channels = kwargs["gin_channels"]  # 设置gin_channels属性
            if self.gin_channels != 0:  # 如果gin_channels不为0
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)  # 初始化spk_emb_linear
                self.cond_layer_idx = (  # 设置cond_layer_idx属性
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                logging.debug(self.gin_channels, self.cond_layer_idx)  # 记录日志
                assert (  # 断言
                    self.cond_layer_idx < self.n_layers
                ), "cond_layer_idx should be less than n_layers"  # 如果不满足条件，抛出异常
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层
        self.attn_layers = nn.ModuleList()  # 初始化注意力层列表
        self.norm_layers_1 = nn.ModuleList()  # 初始化LayerNorm层列表
        self.ffn_layers = nn.ModuleList()  # 初始化FeedForward层列表
        self.norm_layers_2 = nn.ModuleList()  # 初始化LayerNorm层列表
        for i in range(self.n_layers):  # 遍历n_layers
            self.attn_layers.append(  # 向attn_layers列表中添加元素
                MultiHeadAttention(  # 创建MultiHeadAttention实例
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))  # 向norm_layers_1列表中添加元素
            self.ffn_layers.append(  # 向ffn_layers列表中添加元素
                FFN(  # 创建FFN实例
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))  # 向norm_layers_2列表中添加元素

    def forward(self, x, x_mask, g=None):  # 前向传播函数
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)  # 计算注意力掩码
        x = x * x_mask  # 对x进行掩码
        for i in range(self.n_layers):  # 遍历n_layers
            if i == self.cond_layer_idx and g is not None:  # 如果i等于cond_layer_idx并且g不为None
                g = self.spk_emb_linear(g.transpose(1, 2))  # 计算g的线性变换
                g = g.transpose(1, 2)  # 调整g的维度
                x = x + g  # 更新x
                x = x * x_mask  # 对x进行掩码
            y = self.attn_layers[i](x, x, attn_mask)  # 计算注意力层输出
            y = self.drop(y)  # Dropout
            x = self.norm_layers_1[i](x + y)  # LayerNorm
            y = self.ffn_layers[i](x, x_mask)  # 计算FeedForward层输出
            y = self.drop(y)  # Dropout
            x = self.norm_layers_2[i](x + y)  # LayerNorm
        x = x * x_mask  # 对x进行掩码
        return x  # 返回结果
```

# `D:\src\Bert-VITS2\bert_gen.py`

```python
# 导入torch库
import torch
# 从multiprocessing库中导入Pool类
from multiprocessing import Pool
# 导入commons模块
import commons
# 导入utils模块
import utils
# 从tqdm库中导入tqdm类
from tqdm import tqdm
# 从text模块中导入check_bert_models, cleaned_text_to_sequence, get_bert函数
from text import check_bert_models, cleaned_text_to_sequence, get_bert
# 导入argparse模块
import argparse
# 从torch.multiprocessing库中导入mp模块
import torch.multiprocessing as mp
# 从config模块中导入config对象
from config import config

# 定义process_line函数，参数为x
def process_line(x):
    # 解包x为line和add_blank
    line, add_blank = x
    # 获取设备信息
    device = config.bert_gen_config.device
    # 如果使用多设备
    if config.bert_gen_config.use_multi_device:
        # 获取当前进程的rank
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        # 如果CUDA可用
        if torch.cuda.is_available():
            # 计算GPU ID
            gpu_id = rank % torch.cuda.device_count()
            # 设置设备为cuda
            device = torch.device(f"cuda:{gpu_id}")
        else:
            # 设置设备为cpu
            device = torch.device("cpu")
    # 解析line中的信息
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果add_blank为True
    if add_blank:
        # 在phone、tone、language中插入0
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 更新word2ph
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    # 生成bert_path
    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")

    try:
        # 尝试加载bert
        bert = torch.load(bert_path)
        assert bert.shape[-1] == len(phone)
    except Exception:
        # 获取bert
        bert = get_bert(text, word2ph, language_str, device)
        assert bert.shape[-1] == len(phone)
        # 保存bert
        torch.save(bert, bert_path)

# 获取预处理文本配置
preprocess_text_config = config.preprocess_text_config

# 如果当前模块为主模块
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数-c，类型为字符串，默认值为config.bert_gen_config.config_path
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    # 添加参数--num_processes，类型为整数，默认值为config.bert_gen_config.num_processes
    parser.add_argument(
        "--num_processes", type=int, default=config.bert_gen_config.num_processes
    )
    # 解析命令行参数
    args, _ = parser.parse_known_args()
    # 获取配置路径
    config_path = args.config
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(config_path)
    # 检查bert模型
    check_bert_models()
    # 初始化lines列表
    lines = []
    # 读取训练文件中的行并添加到lines列表中
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    # 读取验证文件中的行并添加到lines列表中
    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    # 创建add_blank列表
    add_blank = [hps.data.add_blank] * len(lines)

    # 如果lines不为空
    if len(lines) != 0:
        # 获取进程数
        num_processes = args.num_processes
        # 创建进程池
        with Pool(processes=num_processes) as pool:
            # 遍历lines并调用process_line函数
            for _ in tqdm(
                pool.imap_unordered(process_line, zip(lines, add_blank)),
                total=len(lines),
            ):
                # 使用pass语句作为占位符
                pass  # 这里是缩进的代码块，表示循环体

    # 打印bert生成完毕的信息
    print(f"bert生成完毕!, 共有{len(lines)}个bert.pt生成!")
```

# `D:\src\Bert-VITS2\commons.py`

```python
import math  # 导入math库
import torch  # 导入torch库
from torch.nn import functional as F  # 从torch.nn库中导入functional模块并重命名为F

# 初始化权重
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__  # 获取类名
    if classname.find("Conv") != -1:  # 判断类名中是否包含"Conv"
        m.weight.data.normal_(mean, std)  # 对权重数据进行正态分布初始化

# 获取填充值
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)  # 计算填充值

# 转换填充形状
def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]  # 反转pad_shape
    pad_shape = [item for sublist in layer for item in sublist]  # 将pad_shape展开为一维列表
    return pad_shape  # 返回转换后的填充形状

# 插入元素
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)  # 创建长度为2*lst长度+1的列表
    result[1::2] = lst  # 将lst中的元素插入到result列表中
    return result  # 返回插入元素后的列表

# KL散度计算
def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5  # 计算KL散度
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )  # 计算KL散度
    return kl  # 返回KL散度

# 从Gumbel分布中采样
def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001  # 生成均匀分布样本
    return -torch.log(-torch.log(uniform_samples))  # 从Gumbel分布中采样

# 从Gumbel分布中采样
def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)  # 从Gumbel分布中采样
    return g  # 返回采样结果

# 切片段
def slice_segments(x, ids_str, segment_size=4):
    gather_indices = ids_str.view(x.size(0), 1, 1).repeat(
        1, x.size(1), 1
    ) + torch.arange(segment_size, device=x.device)  # 生成切片索引
    return torch.gather(x, 2, gather_indices)  # 对输入张量进行切片

# 随机切片段
def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()  # 获取张量的形状
    if x_lengths is None:
        x_lengths = t  # 如果长度未指定，则使用t
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=0)  # 计算最大切片索引
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)  # 生成随机切片索引
    ret = slice_segments(x, ids_str, segment_size)  # 对输入张量进行切片
    return ret, ids_str  # 返回切片结果和切片索引

# 获取时间信号
def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)  # 生成位置张量
    num_timescales = channels // 2  # 计算时间尺度数量
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )  # 计算时间尺度增量
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )  # 计算时间尺度的倒数
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)  # 计算缩放时间
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)  # 拼接正弦和余弦信号
    signal = F.pad(signal, [0, 0, 0, channels % 2])  # 对信号进行填充
    signal = signal.view(1, channels, length)  # 调整信号形状
    return signal  # 返回时间信号

# 添加时间信号
def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()  # 获取张量的形状
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)  # 获取时间信号
    return x + signal.to(dtype=x.dtype, device=x.device)  # 返回添加时间信号后的张量

# 拼接时间信号
def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()  # 获取张量的形状
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)  # 获取时间信号
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)  # 返回拼接时间信号后的张量

# 生成下三角掩码
def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)  # 生成下三角掩码
    return mask  # 返回下三角掩码

# 融合操作
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]  # 获取通道数
    in_act = input_a + input_b  # 输入张量相加
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # 计算tanh激活
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # 计算sigmoid激活
    acts = t_act * s_act  # 融合操作
    return acts  # 返回融合结果

# 转换填充形状
def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]  # 反转pad_shape
    pad_shape = [item for sublist in layer for item in sublist]  # 将pad_shape展开为一维列表
    return pad_shape  # 返回转换后的填充形状

# 1维张量向右移动
def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]  # 对输入张量进行填充和切片
    return x  # 返回移动后的张量

# 生成路径
def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """

    b, _, t_y, t_x = mask.shape  # 获取张量的形状
    cum_duration = torch.cumsum(duration, -1)  # 计算累积持续时间

    cum_duration_flat = cum_duration.view(b * t_x)  # 展平累积持续时间
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)  # 生成路径掩码
    path = path.view(b, t_x, t_y)  # 调整路径形状
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]  # 对路径进行填充和切片
    path = path.unsqueeze(1).transpose(2, 3) * mask  # 路径与掩码相乘
    return path  # 返回生成的路径

# 梯度裁剪
def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))  # 过滤出梯度不为空的参数
    norm_type = float(norm_type)  # 将norm_type转换为浮点数
    if clip_value is not None:
        clip_value = float(clip_value)  # 将clip_value转换为浮点数

    total_norm = 0  # 初始化总梯度范数
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)  # 计算参数梯度范数
        total_norm += param_norm.item() ** norm_type  # 累加参数梯度范数的幂
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)  # 对梯度进行裁剪
    total_norm = total_norm ** (1.0 / norm_type)  # 计算总梯度范数
    return total_norm  # 返回总梯度范数
```

# `D:\src\Bert-VITS2\compress_model.py`

```python
from collections import OrderedDict  # 导入OrderedDict类
from text.symbols import symbols  # 从text.symbols模块中导入symbols变量
import torch  # 导入torch模块

from tools.log import logger  # 从tools.log模块中导入logger变量
import utils  # 导入utils模块
from models import SynthesizerTrn  # 从models模块中导入SynthesizerTrn类
import os  # 导入os模块

# 定义函数copyStateDict，用于复制状态字典
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):  # 如果状态字典的第一个键以"module"开头
        start_idx = 1  # 则将start_idx设为1
    else:
        start_idx = 0  # 否则将start_idx设为0
    new_state_dict = OrderedDict()  # 创建一个有序字典
    for k, v in state_dict.items():  # 遍历状态字典的键值对
        name = ",".join(k.split(".")[start_idx:])  # 将键按"."分割后取start_idx位置之后的部分，然后用","连接
        new_state_dict[name] = v  # 将新的键值对添加到新的状态字典中
    return new_state_dict  # 返回新的状态字典

# 定义函数removeOptimizer，用于移除优化器
def removeOptimizer(config: str, input_model: str, ishalf: bool, output_model: str):
    hps = utils.get_hparams_from_file(config)  # 从配置文件中获取超参数

    net_g = SynthesizerTrn(  # 创建SynthesizerTrn对象
        len(symbols),  # 符号的长度
        hps.data.filter_length // 2 + 1,  # 数据的滤波器长度
        hps.train.segment_size // hps.data.hop_length,  # 训练段的大小除以数据的跳跃长度
        n_speakers=hps.data.n_speakers,  # 说话人数量
        **hps.model,  # 其他模型参数
    )

    optim_g = torch.optim.AdamW(  # 创建AdamW优化器
        net_g.parameters(),  # 优化器的参数
        hps.train.learning_rate,  # 学习率
        betas=hps.train.betas,  # beta值
        eps=hps.train.eps,  # epsilon值
    )

    state_dict_g = torch.load(input_model, map_location="cpu")  # 从文件中加载状态字典
    new_dict_g = copyStateDict(state_dict_g)  # 复制状态字典
    keys = []  # 创建一个空列表
    for k, v in new_dict_g["model"].items():  # 遍历新状态字典中的键值对
        if "enc_q" in k:  # 如果"k"中包含"enc_q"
            continue  # 则跳过当前循环
        keys.append(k)  # 将"k"添加到keys列表中

    new_dict_g = (  # 根据条件选择新的状态字典
        {k: new_dict_g["model"][k].half() for k in keys}  # 如果ishalf为真，则将新状态字典中的值转换为半精度
        if ishalf  # 如果ishalf为真
        else {k: new_dict_g["model"][k] for k in keys}  # 否则保持原状态字典中的值
    )

    torch.save(  # 保存模型
        {
            "model": new_dict_g,  # 模型的状态字典
            "iteration": 0,  # 迭代次数
            "optimizer": optim_g.state_dict(),  # 优化器的状态字典
            "learning_rate": 0.0001,  # 学习率
        },
        output_model,  # 输出模型
    )


if __name__ == "__main__":
    import argparse  # 导入argparse模块

    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    parser.add_argument("-c", "--config", type=str, default="configs/config.json")  # 添加参数--config
    parser.add_argument("-i", "--input", type=str)  # 添加参数--input
    parser.add_argument("-o", "--output", type=str, default=None)  # 添加参数--output
    parser.add_argument(  # 添加参数--half
        "-hf", "--half", action="store_true", default=False, help="Save as FP16"
    )

    args = parser.parse_args()  # 解析参数

    output = args.output  # 获取输出路径

    if output is None:  # 如果输出路径为空
        import os.path  # 导入os.path模块

        filename, ext = os.path.splitext(args.input)  # 获取输入文件的文件名和扩展名
        half = "_half" if args.half else ""  # 如果args.half为真，则half为"_half"，否则为空字符串
        output = filename + "_release" + half + ext  # 设置输出路径

    removeOptimizer(args.config, args.input, args.half, output)  # 移除优化器
    logger.info(f"压缩模型成功, 输出模型: {os.path.abspath(output)}")  # 记录日志信息
```

# `D:\src\Bert-VITS2\config.py`

```python
"""
@Desc: 全局配置文件读取
"""
import argparse  # 导入argparse库，用于解析命令行参数
import yaml  # 导入yaml库，用于读取和写入YAML文件
from typing import Dict, List  # 从typing库中导入Dict和List类型
import os  # 导入os库，用于与操作系统交互
import shutil  # 导入shutil库，用于高级文件操作
import sys  # 导入sys库，用于访问与Python解释器交互的变量和函数
```

# `D:\src\Bert-VITS2\data_utils.py`

```python
import os  # 导入os模块
import random  # 导入random模块
import torch  # 导入torch模块
import torch.utils.data  # 导入torch.utils.data模块
from tqdm import tqdm  # 从tqdm模块中导入tqdm函数
from tools.log import logger  # 从tools.log模块中导入logger对象
import commons  # 导入commons模块
from mel_processing import spectrogram_torch, mel_spectrogram_torch  # 从mel_processing模块中导入spectrogram_torch和mel_spectrogram_torch函数
from utils import load_wav_to_torch, load_filepaths_and_text  # 从utils模块中导入load_wav_to_torch和load_filepaths_and_text函数
from text import cleaned_text_to_sequence  # 从text模块中导入cleaned_text_to_sequence函数
from config import config  # 从config模块中导入config对象
```

# `D:\src\Bert-VITS2\export_onnx.py`

```python
# 导入export_onnx函数
from onnx_modules import export_onnx
# 导入os模块
import os

# 判断是否为主程序
if __name__ == "__main__":
    # 设置导出路径
    export_path = "BertVits2.2PT"
    # 设置模型路径
    model_path = "model\\G_0.pth"
    # 设置配置文件路径
    config_path = "model\\config.json"
    # 设置novq变量为False
    novq = False
    # 设置dev变量为False
    dev = False
    # 如果onnx文件夹不存在，则创建
    if not os.path.exists("onnx"):
        os.makedirs("onnx")
    # 如果导出路径对应的文件夹不存在，则创建
    if not os.path.exists(f"onnx/{export_path}"):
        os.makedirs(f"onnx/{export_path}")
    # 调用export_onnx函数，传入导出路径、模型路径、配置文件路径、novq和dev变量
    export_onnx(export_path, model_path, config_path, novq, dev)
```

# `D:\src\Bert-VITS2\infer.py`

```python
"""
版本管理、兼容推理及模型加载实现。
版本说明：
    1. 版本号与github的release版本号对应，使用哪个release版本训练的模型即对应其版本号
    2. 请在模型的config.json中显示声明版本号，添加一个字段"version" : "你的版本号"
特殊版本说明：
    1.1.1-fix： 1.1.1版本训练的模型，但是在推理时使用dev的日语修复
    2.3：当前版本
"""
import torch
import commons
from text import cleaned_text_to_sequence, get_bert

# from clap_wrapper import get_clap_audio_feature, get_clap_text_feature
from text.cleaner import clean_text
import utils

from models import SynthesizerTrn
from text.symbols import symbols

from oldVersion.V220.models import SynthesizerTrn as V220SynthesizerTrn
from oldVersion.V220.text import symbols as V220symbols
from oldVersion.V210.models import SynthesizerTrn as V210SynthesizerTrn
from oldVersion.V210.text import symbols as V210symbols
from oldVersion.V200.models import SynthesizerTrn as V200SynthesizerTrn
from oldVersion.V200.text import symbols as V200symbols
from oldVersion.V111.models import SynthesizerTrn as V111SynthesizerTrn
from oldVersion.V111.text import symbols as V111symbols
from oldVersion.V110.models import SynthesizerTrn as V110SynthesizerTrn
from oldVersion.V110.text import symbols as V110symbols
from oldVersion.V101.models import SynthesizerTrn as V101SynthesizerTrn
from oldVersion.V101.text import symbols as V101symbols

from oldVersion import V111, V110, V101, V200, V210, V220

# 当前版本信息
latest_version = "2.3"

# 版本兼容
SynthesizerTrnMap = {
    "2.2": V220SynthesizerTrn,
    "2.1": V210SynthesizerTrn,
    "2.0.2-fix": V200SynthesizerTrn,
    "2.0.1": V200SynthesizerTrn,
    "2.0": V200SynthesizerTrn,
    "1.1.1-fix": V111SynthesizerTrn,
    "1.1.1": V111SynthesizerTrn,
    "1.1": V110SynthesizerTrn,
    "1.1.0": V110SynthesizerTrn,
    "1.0.1": V101SynthesizerTrn,
    "1.0": V101SynthesizerTrn,
    "1.0.0": V101SynthesizerTrn,
}

symbolsMap = {
    "2.2": V220symbols,
    "2.1": V210symbols,
    "2.0.2-fix": V200symbols,
    "2.0.1": V200symbols,
    "2.0": V200symbols,
    "1.1.1-fix": V111symbols,
    "1.1.1": V111symbols,
    "1.1": V110symbols,
    "1.1.0": V110symbols,
    "1.0.1": V101symbols,
    "1.0": V101symbols,
    "1.0.0": V101symbols,
}


# def get_emo_(reference_audio, emotion, sid):
#     emo = (
#         torch.from_numpy(get_emo(reference_audio))
#         if reference_audio and emotion == -1
#         else torch.FloatTensor(
#             np.load(f"emo_clustering/{sid}/cluster_center_{emotion}.npy")
#         )
#     )
#     return emo


def get_net_g(model_path: str, version: str, device: str, hps):
    if version != latest_version:
        net_g = SynthesizerTrnMap[version](
            len(symbolsMap[version]),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    else:
        # 当前版本模型 net_g
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    return net_g


def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    style_text = None if style_text == "" else style_text
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = get_bert(
        norm_text, word2ph, language_str, device, style_text, style_weight
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert_ori
        ja_bert = torch.randn(1024, len(phone))
        en_bert = torch.randn(1024, len(phone))
    elif language_str == "JP":
        bert = torch.randn(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.randn(1024, len(phone))
    elif language_str == "EN":
        bert = torch.randn(1024, len(phone))
        ja_bert = torch.randn(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language


def infer(
    text,
    emotion,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
    reference_audio=None,
    skip_start=False,
    skip_end=False,
    style_text=None,
    style_weight=0.7,
):
    # 2.2版本参数位置变了
    inferMap_V4 = {
        "2.2": V220.infer,
    }
    # 2.1 参数新增 emotion reference_audio skip_start skip_end
    inferMap_V3 = {
        "2.1": V210.infer,
    }
    # 支持中日英三语版本
    inferMap_V2 = {
        "2.0.2-fix": V200.infer,
        "2.0.1": V200.infer,
        "2.0": V200.infer,
        "1.1.1-fix": V111.infer_fix,
        "1.1.1": V111.infer,
        "1.1": V110.infer,
        "1.1.0": V110.infer,
    }
    # 仅支持中文版本
    # 在测试中，并未发现两个版本的模型不能互相通用
    inferMap_V1 = {
        "1.0.1": V101.infer,
        "1.0": V101.infer,
        "1.0.0": V101.infer,
    }
    version = hps.version if hasattr(hps, "version") else latest_version
    # 非当前版本，根据版本号选择合适的infer
    if version != latest_version:
        if version in inferMap_V4.keys():
            emotion = ""  # Use empty emotion prompt
            return inferMap_V4[version](
                text,
                emotion,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                language,
                hps,
                net_g,
                device,
                reference_audio,
                skip_start,
                skip_end,
                style_text,
                style_weight,
            )
        if version in inferMap_V3.keys():
            emotion = 0
            return inferMap_V3[version](
                text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                language,
                hps,
                net_g,
                device,
                reference_audio,
                emotion,
                skip_start,
                skip_end,
                style_text,
                style_weight,
            )
        if version in inferMap_V2.keys():
            return inferMap_V2[version](
                text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                language,
                hps,
                net_g,
                device,
            )
        if version in inferMap_V1.keys():
            return inferMap_V1[version](
                text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                hps,
                net_g,
                device,
            )
    # 在此处实现当前版本的推理
    # emo = get_emo_(reference_audio, emotion, sid)
    # if isinstance(reference_audio, np.ndarray):
    #     emo = get_clap_audio_feature(reference_audio, device)
    # else:
    #     emo = get_clap_text_feature(emotion, device)
    # emo = torch.squeeze(emo, dim=1)

    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        style_text=style_text,
        style_weight=style_weight,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # emo = emo.to(device).unsqueeze(0)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            speakers,
            ja_bert,
            en_bert,
        )  # , emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio


def infer_multilang(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
    reference_audio=None,
    emotion=None,
    skip_start=False,
    skip_end=False,
):
    bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []
    # emo = get_emo_(reference_audio, emotion, sid)
    # if isinstance(reference_audio, np.ndarray):
    #     emo = get_clap_audio_feature(reference_audio, device)
    # else:
    #     emo = get_clap_text_feature(emotion, device)
    # emo = torch.squeeze(emo, dim=1)
    for idx, (txt, lang) in enumerate(zip(text, language)):
        _skip_start = (idx != 0) or (skip_start and idx == 0)
        _skip_end = (idx != len(language) - 1) or skip_end
        (
            temp_bert,
            temp_ja_bert,
            temp_en_bert,
            temp_phones,
            temp_tones,
            temp_lang_ids,
        ) = get_text(txt, lang, hps, device)
        if _skip_start:
            temp_bert = temp_bert[:, 3:]
            temp_ja_bert = temp_ja_bert[:, 3:]
            temp_en_bert = temp_en_bert[:, 3:]
            temp_phones = temp_phones[3:]
            temp_tones = temp_tones[3:]
            temp_lang_ids = temp_lang_ids[3:]
        if _skip_end:
            temp_bert = temp_bert[:, :-2]
            temp_ja_bert = temp_ja_bert[:, :-2]
            temp_en_bert = temp_en_bert[:, :-2]
            temp_phones = temp_phones[:-2]
            temp_tones = temp_tones[:-2]
            temp_lang_ids = temp_lang_ids[:-2]
        bert.append(temp_bert)
        ja_bert.append(temp_ja_bert)
        en_bert.append(temp_en_bert)
        phones.append(temp_phones)
        tones.append(temp_tones)
        lang_ids.append(temp_lang_ids)
    bert = torch.concatenate(bert, dim=1)
    ja_bert = torch.concatenate(ja_bert, dim=1)
    en_bert = torch.concatenate(en_bert, dim=1)
    phones = torch.concatenate(phones, dim=0)
    tones = torch.concatenate(tones, dim=0)
    lang_ids = torch.concatenate(lang_ids, dim=0)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        # emo = emo.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            speakers,
            ja_bert,
            en_bert,
        )  # , emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio
```

# `D:\src\Bert-VITS2\losses.py`

```python
import torch  # 导入PyTorch库
import torchaudio  # 导入torchaudio库
from transformers import AutoModel  # 从transformers库中导入AutoModel类


def feature_loss(fmap_r, fmap_g):  # 定义名为feature_loss的函数，接受fmap_r和fmap_g两个参数
    loss = 0  # 初始化loss为0
    for dr, dg in zip(fmap_r, fmap_g):  # 遍历fmap_r和fmap_g
        for rl, gl in zip(dr, dg):  # 遍历dr和dg
            rl = rl.float().detach()  # 将rl转换为float类型并分离梯度
            gl = gl.float()  # 将gl转换为float类型
            loss += torch.mean(torch.abs(rl - gl))  # 计算rl和gl的绝对值差的均值并加到loss上

    return loss * 2  # 返回loss乘以2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):  # 定义名为discriminator_loss的函数，接受disc_real_outputs和disc_generated_outputs两个参数
    loss = 0  # 初始化loss为0
    r_losses = []  # 初始化r_losses为空列表
    g_losses = []  # 初始化g_losses为空列表
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):  # 遍历disc_real_outputs和disc_generated_outputs
        dr = dr.float()  # 将dr转换为float类型
        dg = dg.float()  # 将dg转换为float类型
        r_loss = torch.mean((1 - dr) ** 2)  # 计算r_loss
        g_loss = torch.mean(dg**2)  # 计算g_loss
        loss += r_loss + g_loss  # 将r_loss和g_loss加到loss上
        r_losses.append(r_loss.item())  # 将r_loss的值加到r_losses列表中
        g_losses.append(g_loss.item())  # 将g_loss的值加到g_losses列表中

    return loss, r_losses, g_losses  # 返回loss, r_losses, g_losses


def generator_loss(disc_outputs):  # 定义名为generator_loss的函数，接受disc_outputs一个参数
    loss = 0  # 初始化loss为0
    gen_losses = []  # 初始化gen_losses为空列表
    for dg in disc_outputs:  # 遍历disc_outputs
        dg = dg.float()  # 将dg转换为float类型
        l = torch.mean((1 - dg) ** 2)  # 计算l
        gen_losses.append(l)  # 将l的值加到gen_losses列表中
        loss += l  # 将l加到loss上

    return loss, gen_losses  # 返回loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):  # 定义名为kl_loss的函数，接受z_p, logs_q, m_p, logs_p, z_mask五个参数
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()  # 将z_p转换为float类型
    logs_q = logs_q.float()  # 将logs_q转换为float类型
    m_p = m_p.float()  # 将m_p转换为float类型
    logs_p = logs_p.float()  # 将logs_p转换为float类型
    z_mask = z_mask.float()  # 将z_mask转换为float类型

    kl = logs_p - logs_q - 0.5  # 计算kl
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)  # 计算kl
    kl = torch.sum(kl * z_mask)  # 计算kl
    l = kl / torch.sum(z_mask)  # 计算l
    return l  # 返回l


class WavLMLoss(torch.nn.Module):  # 定义名为WavLMLoss的类，继承自torch.nn.Module
    def __init__(self, model, wd, model_sr, slm_sr=16000):  # 定义初始化方法，接受model, wd, model_sr, slm_sr四个参数
        super(WavLMLoss, self).__init__()  # 调用父类的初始化方法
        self.wavlm = AutoModel.from_pretrained(model)  # 使用model参数初始化wavlm
        self.wd = wd  # 初始化wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)  # 使用model_sr, slm_sr初始化resample
        self.wavlm.eval()  # 设置wavlm为评估模式
        for param in self.wavlm.parameters():  # 遍历wavlm的参数
            param.requires_grad = False  # 设置参数的梯度为False

    def forward(self, wav, y_rec):  # 定义名为forward的方法，接受wav, y_rec两个参数
        with torch.no_grad():  # 禁用梯度
            wav_16 = self.resample(wav)  # 使用resample对wav进行重采样
            wav_embeddings = self.wavlm(  # 使用wavlm对wav_16进行处理
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
        y_rec_16 = self.resample(y_rec)  # 使用resample对y_rec进行重采样
        y_rec_embeddings = self.wavlm(  # 使用wavlm对y_rec_16进行处理
            input_values=y_rec_16.squeeze(), output_hidden_states=True
        ).hidden_states

        floss = 0  # 初始化floss为0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):  # 遍历wav_embeddings和y_rec_embeddings
            floss += torch.mean(torch.abs(er - eg))  # 计算floss

        return floss.mean()  # 返回floss的均值

    def generator(self, y_rec):  # 定义名为generator的方法，接受y_rec一个参数
        y_rec_16 = self.resample(y_rec)  # 使用resample对y_rec进行重采样
        y_rec_embeddings = self.wavlm(  # 使用wavlm对y_rec_16进行处理
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
        y_rec_embeddings = (  # 对y_rec_embeddings进行处理
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings)  # 使用wd对y_rec_embeddings进行处理
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)  # 计算loss_gen

        return loss_gen  # 返回loss_gen

    def discriminator(self, wav, y_rec):  # 定义名为discriminator的方法，接受wav, y_rec两个参数
        with torch.no_grad():  # 禁用梯度
            wav_16 = self.resample(wav)  # 使用resample对wav进行重采样
            wav_embeddings = self.wavlm(  # 使用wavlm对wav_16进行处理
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_rec_16 = self.resample(y_rec)  # 使用resample对y_rec进行重采样
            y_rec_embeddings = self.wavlm(  # 使用wavlm对y_rec_16进行处理
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states

            y_embeddings = (  # 对y_embeddings进行处理
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (  # 对y_rec_embeddings进行处理
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)  # 使用wd对y_embeddings进行处理
        y_d_gs = self.wd(y_rec_embeddings)  # 使用wd对y_rec_embeddings进行处理

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs  # 初始化y_df_hat_r, y_df_hat_g

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)  # 计算r_loss
        g_loss = torch.mean((y_df_hat_g) ** 2)  # 计算g_loss

        loss_disc_f = r_loss + g_loss  # 计算loss_disc_f

        return loss_disc_f.mean()  # 返回loss_disc_f的均值

    def discriminator_forward(self, wav):  # 定义名为discriminator_forward的方法，接受wav一个参数
        with torch.no_grad():  # 禁用梯度
            wav_16 = self.resample(wav)  # 使用resample对wav进行重采样
            wav_embeddings = self.wavlm(  # 使用wavlm对wav_16进行处理
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_embeddings = (  # 对y_embeddings进行处理
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)  # 使用wd对y_embeddings进行处理

        return y_d_rs  # 返回y_d_rs
```

# `D:\src\Bert-VITS2\mel_processing.py`

```python
import torch  # 导入torch库
import torch.utils.data  # 导入torch.utils.data库
from librosa.filters import mel as librosa_mel_fn  # 从librosa.filters库中导入mel函数
import warnings  # 导入warnings库

warnings.filterwarnings(action="ignore")  # 忽略警告
MAX_WAV_VALUE = 32768.0  # 定义最大音频值

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):  # 定义动态范围压缩函数
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)  # 返回动态范围压缩后的值

def dynamic_range_decompression_torch(x, C=1):  # 定义动态范围解压缩函数
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C  # 返回动态范围解压缩后的值

def spectral_normalize_torch(magnitudes):  # 定义频谱归一化函数
    output = dynamic_range_compression_torch(magnitudes)  # 使用动态范围压缩函数
    return output  # 返回归一化后的频谱

def spectral_de_normalize_torch(magnitudes):  # 定义频谱反归一化函数
    output = dynamic_range_decompression_torch(magnitudes)  # 使用动态范围解压缩函数
    return output  # 返回反归一化后的频谱

mel_basis = {}  # 初始化mel基础
hann_window = {}  # 初始化hann窗口

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):  # 定义频谱图函数
    if torch.min(y) < -1.0:  # 如果y的最小值小于-1.0
        print("min value is ", torch.min(y))  # 打印最小值
    if torch.max(y) > 1.0:  # 如果y的最大值大于1.0
        print("max value is ", torch.max(y))  # 打印最大值

    global hann_window  # 使用全局变量hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)  # 获取数据类型和设备
    wnsize_dtype_device = str(win_size) + "_" + dtype_device  # 获取窗口大小和数据类型设备
    if wnsize_dtype_device not in hann_window:  # 如果wnsize_dtype_device不在hann_window中
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(  # 使用hann窗口
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(  # 使用pad函数
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(  # 使用stft函数
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)  # 计算频谱的平方和并开方
    return spec  # 返回频谱

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):  # 定义频谱到mel频谱的转换函数
    global mel_basis  # 使用全局变量mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)  # 获取数据类型和设备
    fmax_dtype_device = str(fmax) + "_" + dtype_device  # 获取最大频率和数据类型设备
    if fmax_dtype_device not in mel_basis:  # 如果fmax_dtype_device不在mel_basis中
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)  # 使用librosa_mel_fn函数
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(  # 转换为torch张量
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)  # 使用matmul函数
    spec = spectral_normalize_torch(spec)  # 使用频谱归一化函数
    return spec  # 返回mel频谱

def mel_spectrogram_torch(  # 定义mel频谱图函数
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:  # 如果y的最小值小于-1.0
        print("min value is ", torch.min(y))  # 打印最小值
    if torch.max(y) > 1.0:  # 如果y的最大值大于1.0
        print("max value is ", torch.max(y))  # 打印最大值

    global mel_basis, hann_window  # 使用全局变量mel_basis和hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)  # 获取数据类型和设备
    fmax_dtype_device = str(fmax) + "_" + dtype_device  # 获取最大频率和数据类型设备
    wnsize_dtype_device = str(win_size) + "_" + dtype_device  # 获取窗口大小和数据类型设备
    if fmax_dtype_device not in mel_basis:  # 如果fmax_dtype_device不在mel_basis中
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)  # 使用librosa_mel_fn函数
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(  # 转换为torch张量
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:  # 如果wnsize_dtype_device不在hann_window中
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(  # 使用hann窗口
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(  # 使用pad函数
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(  # 使用stft函数
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)  # 计算频谱的平方和并开方

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)  # 使用matmul函数
    spec = spectral_normalize_torch(spec)  # 使用频谱归一化函数

    return spec  # 返回mel频谱
```

# `D:\src\Bert-VITS2\models.py`

```py
# 添加注释
```python
import math  # 导入数学库
import torch  # 导入torch
from torch import nn  # 从torch中导入nn
from torch.nn import functional as F  # 从torch.nn中导入functional

import commons  # 导入commons
import modules  # 导入modules
import attentions  # 导入attentions
import monotonic_align  # 导入monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # 从torch.nn中导入Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # 从torch.nn.utils中导入weight_norm, remove_weight_norm, spectral_norm

from commons import init_weights, get_padding  # 从commons中导入init_weights, get_padding
from text import symbols, num_tones, num_languages  # 从text中导入symbols, num_tones, num_languages
```

# `D:\src\Bert-VITS2\modules.py`

```python
import math  # 导入数学库
import torch  # 导入PyTorch
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.nn import functional as F  # 从PyTorch中导入函数模块

from torch.nn import Conv1d  # 从PyTorch中导入一维卷积层
from torch.nn.utils import weight_norm, remove_weight_norm  # 从PyTorch中导入权重归一化和移除权重归一化

import commons  # 导入自定义的commons模块
from commons import init_weights, get_padding  # 从commons模块中导入初始化权重和获取填充的函数
from transforms import piecewise_rational_quadratic_transform  # 从transforms模块中导入分段有理二次变换函数
from attentions import Encoder  # 从attentions模块中导入Encoder类

LRELU_SLOPE = 0.1  # 定义LRELU_SLOPE为0.1


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.channels = channels  # 设置通道数
        self.eps = eps  # 设置eps

        self.gamma = nn.Parameter(torch.ones(channels))  # 设置gamma参数
        self.beta = nn.Parameter(torch.zeros(channels))  # 设置beta参数

    def forward(self, x):  # 前向传播函数
        x = x.transpose(1, -1)  # 转置x
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)  # 对x进行Layer Norm
        return x.transpose(1, -1)  # 再次转置x


class ConvReluNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        n_layers,
        p_dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()  # 初始化卷积层列表
        self.norm_layers = nn.ModuleList()  # 初始化规范化层列表
        self.conv_layers.append(
            nn.Conv1d(
                in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
        )  # 添加第一个卷积层
        self.norm_layers.append(LayerNorm(hidden_channels))  # 添加第一个规范化层
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))  # 定义ReLU和Dropout
        for _ in range(n_layers - 1):  # 循环添加剩余的卷积层和规范化层
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)  # 添加投影层
        self.proj.weight.data.zero_()  # 初始化投影层的权重
        self.proj.bias.data.zero_()  # 初始化投影层的偏置

    def forward(self, x, x_mask):  # 前向传播函数
        x_org = x  # 保存原始输入
        for i in range(self.n_layers):  # 循环处理每一层
            x = self.conv_layers[i](x * x_mask)  # 卷积
            x = self.norm_layers[i](x)  # 规范化
            x = self.relu_drop(x)  # ReLU和Dropout
        x = x_org + self.proj(x)  # 添加投影层
        return x * x_mask  # 返回结果乘以掩码


class DDSConv(nn.Module):
    """
    Dialted and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)  # 定义Dropout
        self.convs_sep = nn.ModuleList()  # 初始化分离卷积层列表
        self.convs_1x1 = nn.ModuleList()  # 初始化1x1卷积层列表
        self.norms_1 = nn.ModuleList()  # 初始化规范化层列表
        self.norms_2 = nn.ModuleList()  # 初始化规范化层列表
        for i in range(n_layers):  # 循环添加分离卷积层、1x1卷积层和规范化层
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):  # 前向传播函数
        if g is not None:
            x = x + g
        for i in range(self.n_layers):  # 循环处理每一层
            y = self.convs_sep[i](x * x_mask)  # 分离卷积
            y = self.norms_1[i](y)  # 规范化
            y = F.gelu(y)  # GELU激活函数
            y = self.convs_1x1[i](y)  # 1x1卷积
            y = self.norms_2[i](y)  # 规范化
            y = F.gelu(y)  # GELU激活函数
            y = self.drop(y)  # Dropout
            x = x + y  # 残差连接
        return x * x_mask  # 返回结果乘以掩码
```

# `D:\src\Bert-VITS2\onnx_infer.py`

```python
# 导入OnnxInferenceSession类
from onnx_modules.V220_OnnxInference import OnnxInferenceSession
# 导入numpy库并重命名为np
import numpy as np

# 创建OnnxInferenceSession对象Session，传入模型路径和执行提供者
Session = OnnxInferenceSession(
    {
        "enc": "onnx/BertVits2.2PT/BertVits2.2PT_enc_p.onnx",
        "emb_g": "onnx/BertVits2.2PT/BertVits2.2PT_emb.onnx",
        "dp": "onnx/BertVits2.2PT/BertVits2.2PT_dp.onnx",
        "sdp": "onnx/BertVits2.2PT/BertVits2.2PT_sdp.onnx",
        "flow": "onnx/BertVits2.2PT/BertVits2.2PT_flow.onnx",
        "dec": "onnx/BertVits2.2PT/BertVits2.2PT_dec.onnx",
    },
    Providers=["CPUExecutionProvider"],
)

# 创建输入数组x
x = np.array(
    [
        0,
        97,
        0,
        8,
        0,
        78,
        0,
        8,
        0,
        76,
        0,
        37,
        0,
        40,
        0,
        97,
        0,
        8,
        0,
        23,
        0,
        8,
        0,
        74,
        0,
        26,
        0,
        104,
        0,
    ]
)
# 创建与x相同形状的全零数组tone
tone = np.zeros_like(x)
# 创建与x相同形状的全零数组language
language = np.zeros_like(x)
# 创建长度为1的数组sid，值为0
sid = np.array([0])
# 创建形状为(x.shape[0], 1024)的随机数组bert
bert = np.random.randn(x.shape[0], 1024)
# 创建形状为(x.shape[0], 1024)的随机数组ja_bert
ja_bert = np.random.randn(x.shape[0], 1024)
# 创建形状为(x.shape[0], 1024)的随机数组en_bert
en_bert = np.random.randn(x.shape[0], 1024)
# 创建形状为(512, 1)的随机数组emo
emo = np.random.randn(512, 1)

# 调用Session对象进行推理，传入输入数组和其他参数
audio = Session(x, tone, language, bert, ja_bert, en_bert, emo, sid)

# 打印推理结果audio
print(audio)
```