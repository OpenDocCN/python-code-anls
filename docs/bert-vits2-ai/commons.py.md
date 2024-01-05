# `d:/src/tocomm/Bert-VITS2\commons.py`

```
import math  # 导入 math 模块
import torch  # 导入 torch 模块
from torch.nn import functional as F  # 从 torch.nn 模块中导入 functional，并重命名为 F


def init_weights(m, mean=0.0, std=0.01):
    # 获取传入参数 m 的类名
    classname = m.__class__.__name__
    # 如果类名中包含 "Conv"，则对权重进行正态分布初始化
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    # 计算 padding 大小
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    # 将 pad_shape 反转，并展开成一维数组
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape
def intersperse(lst, item):
    # 创建一个长度为原列表长度两倍加一的新列表
    result = [item] * (len(lst) * 2 + 1)
    # 将原列表的元素插入到新列表的奇数索引位置
    result[1::2] = lst
    # 返回结果列表
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    # 计算KL散度
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    # 返回KL散度
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    # 从Gumbel分布中抽样，避免溢出
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))
def rand_gumbel_like(x):
    # 生成一个与输入张量相同大小的 Gumbel 分布随机数张量
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    # 根据 ids_str 中的索引值，从输入张量 x 中切片出指定大小的段
    gather_indices = ids_str.view(x.size(0), 1, 1).repeat(
        1, x.size(1), 1
    ) + torch.arange(segment_size, device=x.device)
    return torch.gather(x, 2, gather_indices)


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    # 计算每个样本可以切片的起始位置的最大索引
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=0)
    # 生成每个样本的随机起始位置索引
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)  # 调用slice_segments函数，对输入的x进行分段处理，返回处理后的结果
    return ret, ids_str  # 返回处理后的结果和ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)  # 创建一个长度为length的浮点数张量，表示位置信息
    num_timescales = channels // 2  # 计算时间尺度的数量
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )  # 计算时间尺度的对数增量
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )  # 计算时间尺度的倒数
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)  # 计算缩放后的时间
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)  # 将sin和cos的结果拼接成信号
    signal = F.pad(signal, [0, 0, 0, channels % 2])  # 对信号进行填充，使其长度为channels
    signal = signal.view(1, channels, length)  # 调整信号的形状
    return signal  # 返回时间信号
# 为输入的一维张量添加时间信号
def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    # 获取输入张量的维度信息
    b, channels, length = x.size()
    # 调用函数获取时间信号
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    # 将时间信号添加到输入张量上，并返回结果
    return x + signal.to(dtype=x.dtype, device=x.device)

# 在输入的一维张量上拼接时间信号
def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    # 获取输入张量的维度信息
    b, channels, length = x.size()
    # 调用函数获取时间信号
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    # 将时间信号拼接到输入张量上，并返回结果
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)

# 生成一个下三角形状的掩码张量
def subsequent_mask(length):
    # 生成一个下三角形状的张量，并在最外层添加两个维度
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    # 返回生成的掩码张量
    return mask

# 使用 TorchScript 将函数编译成脚本
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    # 将 n_channels 转换为整数类型
    n_channels_int = n_channels[0]
    in_act = input_a + input_b  # 计算两个输入的和
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # 对输入的部分数据进行双曲正切函数处理
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # 对输入的部分数据进行Sigmoid函数处理
    acts = t_act * s_act  # 将处理后的数据相乘
    return acts  # 返回处理后的数据


def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]  # 将输入的pad_shape列表倒序排列
    pad_shape = [item for sublist in layer for item in sublist]  # 将倒序排列后的列表展开成一维列表
    return pad_shape  # 返回处理后的一维列表


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]  # 对输入的x进行填充操作，并去除最后一列
    return x  # 返回处理后的x


def sequence_mask(length, max_length=None):
    if max_length is None:  # 如果没有指定最大长度
        max_length = length.max()  # 找到输入长度中的最大值
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)  # 创建一个从0到最大长度的张量
    return x.unsqueeze(0) < length.unsqueeze(1)  # 返回一个布尔类型的张量，表示每个位置是否小于对应的长度


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]  # 输入持续时间的张量形状
    mask: [b, 1, t_y, t_x]  # 输入掩码的张量形状
    """

    b, _, t_y, t_x = mask.shape  # 获取掩码张量的形状信息
    cum_duration = torch.cumsum(duration, -1)  # 沿着最后一个维度计算输入持续时间的累积和

    cum_duration_flat = cum_duration.view(b * t_x)  # 将累积和张量展平为一维
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)  # 根据累积和生成路径掩码，并转换为与输入掩码相同的数据类型
    path = path.view(b, t_x, t_y)  # 将路径掩码张量重新变形为与输入掩码相同的形状
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]  # 对路径掩码进行填充操作
    path = path.unsqueeze(1).transpose(2, 3) * mask  # 对路径掩码进行维度扩展和转置，并与输入掩码相乘
    return path  # 返回生成的路径掩码
def clip_grad_value_(parameters, clip_value, norm_type=2):
    # 如果参数是单个张量，则转换为列表
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    # 过滤出梯度不为None的参数
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    # 将norm_type转换为浮点数
    norm_type = float(norm_type)
    # 如果clip_value不为None，则将其转换为浮点数
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    # 计算参数的梯度的范数
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        # 如果clip_value不为None，则对梯度进行截断
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    # 返回参数的梯度的总体范数
    return total_norm
```