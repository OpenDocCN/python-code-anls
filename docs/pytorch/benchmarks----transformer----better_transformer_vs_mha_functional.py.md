# `.\pytorch\benchmarks\transformer\better_transformer_vs_mha_functional.py`

```
"""
Tests the performance of torch.nn.MultiheadAttention's fast path (BetterTransformer)
vs the slow path (torch.nn.functional.multi_head_attention)

To run this script install these dependencies:

pip install tqdm
pip install prettytable
"""

import argparse
import itertools
import json
import random

import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np

from prettytable import PrettyTable
from tqdm import tqdm

import torch

warnings.filterwarnings("ignore")

# 创建一个默认值为整数的字典，用于记录错误次数
error_dict = defaultdict(int)


def benchmark_torch_function(iters, f, *args, **kwargs):
    # 执行函数两次，预热 GPU
    f(*args, **kwargs)
    f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        # 执行函数进行性能测试
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    # 计算平均每次执行的时间，单位为毫秒
    # elapsed_time 的分辨率为 0.5 微秒，因此乘以 1000 以增加分辨率
    return start_event.elapsed_time(end_event) * 1000 / iters, *f(*args, **kwargs)


def run(
    a: int,
    b: int,
    iters: int,
    batch_size: int,
    sequence_length: int,
    embed_dim: int,
    num_heads: int,
    device: str,
    dtype: str,
    block_size: int,
    seed,
):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    from scipy.stats import beta

    # 生成批次中每个元素的长度，确保长度的倍数为 block_size
    lengths = (
        beta.rvs(a, b, size=batch_size)
        * (sequence_length + block_size - 1)
        // block_size
    )
    lengths = list(map(int, list(lengths)))
    lengths = [l * block_size for l in lengths]
    lengths = [max(l, block_size) for l in lengths]

    # Used to enforce no padding
    # 用于确保没有填充，每个批次中的元素具有相同的序列长度
    # lengths = [sequence_length] * batch_size

    # 确保批次中至少有一个元素具有最大的序列长度
    lengths[random.randint(0, batch_size - 1)] = sequence_length

    # 创建查询张量列表 q，每个元素表示一个查询张量
    q = [torch.randn(l, embed_dim, device=device, dtype=dtype) for l in lengths]
    # 将查询张量列表转换为嵌套张量
    q = torch.nested.nested_tensor(q, device=device, dtype=dtype)
    # 使用相同的查询张量作为键 k 和值 v
    k, v = q, q

    # 创建线性层，用于输入张量 q 的线性映射
    qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)
    # 创建线性层，用于输出张量的线性映射
    proj = torch.nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    # 创建 MultiheadAttention 实例，用于执行多头注意力机制
    native_mha = torch.nn.MultiheadAttention(
        embed_dim, num_heads, batch_first=True, device=device, dtype=dtype
    ).eval()
    # 设置输入投影的权重和偏置
    native_mha.in_proj_weight = qkv.weight
    native_mha.in_proj_bias = qkv.bias
    # 设置输出投影的权重和偏置
    native_mha.out_proj.weight = proj.weight
    native_mha.out_proj.bias = proj.bias

    # 创建查询掩码 q_mask
    q_mask = torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(
            [torch.tensor([True] * length, dtype=torch.bool) for length in lengths]
        ),
        0,
    )
    q_mask = q_mask.cuda()

    # 如果查询掩码的第二个维度为零，则返回 None
    if q_mask.size(1) == 0:
        return None

    # 对 native MHA 在核心中的性能进行基准测试
    # 使用 torch.backends.cuda.sdp_kernel() 配置 CUDA，禁用数学计算（enable_math=False），启用 Flash 功能（enable_flash=True）
    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True):
        # 进入推断模式
        with torch.inference_mode():
            # 对 native_mha 函数进行基准测试，记录时间、输出和其它信息
            time_native_mha_fast, y_native_mha_fast, _ = benchmark_torch_function(
                iters, native_mha, q, k, v, need_weights=False
            )
    
    # 将输入张量 q 转换为填充张量，填充值为 0
    q = q.to_padded_tensor(0)
    # 将 k 和 v 张量设置为与 q 相同
    k = q
    v = q
    
    # 内部 Flash Attention
    # 对 native_mha 函数进行基准测试，记录时间、输出和其它信息，使用 key_padding_mask 进行掩码操作
    time_native_mha_slow, y_native_mha_slow, _ = benchmark_torch_function(
        iters, native_mha, q, k, v, key_padding_mask=~q_mask, need_weights=False
    )
    
    # 将 y_native_mha_fast 张量转换为填充张量，填充值为 0，如果其为嵌套张量
    if y_native_mha_fast.is_nested:
        y_native_mha_fast = torch.nested.to_padded_tensor(y_native_mha_fast, 0)
    # 将 y_native_mha_fast 乘以 q_mask 张量的 unsqueeze(-1) 结果
    y_native_mha_fast = y_native_mha_fast * q_mask.unsqueeze(-1)
    
    # 将 y_native_mha_slow 张量转换为填充张量，填充值为 0，如果其为嵌套张量
    if y_native_mha_slow.is_nested:
        y_native_mha_slow = torch.nested.to_padded_tensor(y_native_mha_slow, 0)
    # 将 y_native_mha_slow 乘以 q_mask 张量的 unsqueeze(-1) 结果
    y_native_mha_slow = y_native_mha_slow * q_mask.unsqueeze(-1)
    
    # 执行正确性检查
    # 构建错误字典的条目名称，包含批处理大小、序列长度、头数和嵌入维度
    entry_name = f"batch:{batch_size}_seq_len:{sequence_length}_n_heads:{num_heads}_embed_dim:{embed_dim}"
    try:
        # 使用 torch.testing.assert_close() 检查两个张量的接近程度，指定绝对误差和相对误差的阈值
        torch.testing.assert_close(
            y_native_mha_fast, y_native_mha_slow, atol=1e-3, rtol=1e-3
        )
    except AssertionError as e:
        # 如果断言失败，将错误计数添加到 error_dict，并输出错误字典的内容
        error_dict[entry_name] += 1
        pprint(error_dict)
    
    # 计算填充的比例，即未使用部分的比例
    padding = 1 - q_mask.float().mean().item()
    
    # 计算 Flash Attention 的加速比
    speedup_fast_internal = time_native_mha_slow / time_native_mha_fast
    
    # 构建有序字典的条目，记录数据类型、批处理大小、序列长度、头数、嵌入维度、时间信息以及计算得到的指标
    result_entry = OrderedDict()
    result_entry["dtype"] = dtype
    result_entry["batch_size"] = batch_size
    result_entry["sequence_length"] = sequence_length
    result_entry["n_heads"] = num_heads
    result_entry["embed_dim"] = embed_dim
    result_entry["time_native_mha_slow(\u00B5s)"] = f"{time_native_mha_slow:.3f}"
    result_entry["time_native_mha_fast (\u00B5s)"] = f"{time_native_mha_fast:.3f}"
    result_entry["speedup flash_mha v native_mha"] = f"{speedup_fast_internal:.3f}"
    result_entry["padding"] = f"{padding:.3f}"
    
    # 返回构建的结果条目
    return result_entry
# 定义主函数，接受两个可选参数：保存路径和错误路径
def main(save_path: Optional[Path], error_path: Optional[Path]):
    # 创建一个漂亮的表格对象
    table = PrettyTable()
    # 创建一个默认字典来存储条目
    entries = defaultdict(list)

    # 打印当前使用的 CUDA 设备名称
    print("CUDA device: ", torch.cuda.get_device_name(0))
    # 迭代次数设置为100
    iters = 100
    # 初始化表头为None
    header = None
    # 定义不同的批次大小、序列长度、嵌入维度、头数、块大小和 beta 值的组合
    batch_sizes = [16, 32, 64, 128, 256]
    sequence_lengths = [64, 128, 256, 512]
    embed_dims = [512, 1024]
    num_heads_list = [8, 16]
    betas = range(1, 64, 4)

    # 使用 tqdm 显示进度条，遍历所有组合
    for batch_size, sequence_length, embed_dim, num_heads, block_size, b in tqdm(
        list(
            itertools.product(
                batch_sizes, sequence_lengths, embed_dims, num_heads_list, [2], betas
            )
        )
    ):
        # 设置随机种子
        seed = 26214  # Magic number that works well for higher b values
        # 调用 run 函数运行实验，获取实验结果条目
        entry = run(
            1,
            b * 0.05,
            iters,
            batch_size,
            sequence_length,
            embed_dim,
            num_heads,
            "cuda",
            torch.float16,
            block_size,
            seed,
        )
        # 如果实验结果为空，跳过当前循环
        if entry is None:
            continue
        # 如果表头为空，设置表头为 entry 的键列表
        if header is None:
            table.field_names = list(entry.keys())
            header = list(entry.keys())
        # 创建空列表 row，存储 entry 的值，并将其添加到表格中
        row = []
        for k, v in entry.items():
            row.append(v)
            entries[k].append(v)
        table.add_row(row)

    # 将完整的表格打印到控制台
    print(table)
    # 打印错误字典的详细信息
    pprint(error_dict)

    # 获取表格的 CSV 字符串
    csv_string = table.get_csv_string()
    # 如果指定了保存路径，则将表格保存为 CSV 文件
    if save_path is not None:
        with open(save_path, "w") as csvfile:
            csvfile.write(csv_string)

    # 打印总错误数
    print(f"Total errors: {sum(error_dict.values())}")
    # 如果指定了错误路径，则将错误字典保存为 JSON 文件
    if error_path is not None:
        with open(error_path, "w") as file:
            file.write(json.dumps(error_dict))


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加保存结果的路径参数
    parser.add_argument(
        "--save-path", "--save_path", type=str, help="Path to save the results"
    )
    # 添加保存错误的路径参数
    parser.add_argument(
        "--error-save-path",
        "--error_save_path",
        type=str,
        help="Path to save the errors",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 将保存路径和错误路径转换为 Path 对象，如果未指定则为 None
    save_path = Path(args.save_path) if args.save_path else None
    error_path = Path(args.error_save_path) if args.error_save_path else None

    # 调用主函数，传入保存路径和错误路径
    main(save_path, error_path)
```