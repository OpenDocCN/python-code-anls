# `.\pytorch\benchmarks\transformer\attention_bias_benchmarks.py`

```py
import itertools
from dataclasses import asdict, dataclass
from functools import partial
from typing import Callable, List, Union

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch.nn.attention.bias import CausalBias, CausalVariant
from torch.nn.parameter import Parameter


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # 对函数进行预热，运行5次
    for _ in range(5):
        func(*args, **kwargs)
    # 创建一个基准计时器对象
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    # 返回自适应自动范围运行时间的中位数（转换为微秒）
    return t0.adaptive_autorange(min_run_time=0.1).median * 1e6


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    num_heads: int
    q_seq_len: int
    k_seq_len: int
    embed_dim: int
    dtype: torch.dtype

    @property
    def head_dim(self) -> int:
        # 计算头部维度，即嵌入维度除以头部数目
        return self.embed_dim // self.num_heads

    def asdict(self):
        # 将数据类转换为字典形式
        dict_obj = asdict(self)
        dict_obj["head_dim"] = self.head_dim
        return dict_obj


@dataclass(frozen=True)
class ExperimentResults:
    materialized_mask_time: float
    attn_mask_subclass_time: float

    def get_entries(self) -> List:
        # 返回实验结果的条目列表，格式化时间为浮点数字符串
        return [
            f"{self.materialized_mask_time:2f}",
            f"{self.attn_mask_subclass_time:2f}",
        ]


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def get_entries(self) -> List:
        # 返回实验配置和结果的条目列表
        return self.config.get_entries() + self.results.get_entries()


def generate_inputs(
    batch_size, q_sequence_length, kv_sequence_length, embed_dim, dtype, device
):
    # 创建查询和键值的形状
    q_shape = (batch_size, q_sequence_length, embed_dim)
    kv_shape = (batch_size, kv_sequence_length, embed_dim)

    # 部分应用函数，生成随机张量
    make_q = partial(torch.rand, q_shape, device=device, dtype=dtype)
    make_kv = partial(torch.rand, kv_shape, device=device, dtype=dtype)
    return make_q(), make_kv(), make_kv()


class CompositeMHA(torch.nn.Module):
    def __init__(self, num_heads, embed_dim, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # 初始化模型参数和属性
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # 定义查询、键和值的投影权重参数
        self.q_proj_weight = Parameter(
            torch.empty((embed_dim, embed_dim), **factory_kwargs)
        )
        self.k_proj_weight = Parameter(
            torch.empty((embed_dim, embed_dim), **factory_kwargs)
        )
        self.v_proj_weight = Parameter(
            torch.empty((embed_dim, embed_dim), **factory_kwargs)
        )
        # 输出投影权重参数
        self.out_proj = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.num_heads = num_heads
    # 定义一个前向传播方法，用于执行多头注意力机制的操作
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Union[torch.Tensor, CausalBias],
    ):
        # 对查询向量进行线性变换，使用权重 self.q_proj_weight
        query_projected = F.linear(query, self.q_proj_weight)
        # 对键向量进行线性变换，使用权重 self.k_proj_weight
        key_projected = F.linear(key, self.k_proj_weight)
        # 对值向量进行线性变换，使用权重 self.v_proj_weight
        value_projected = F.linear(value, self.v_proj_weight)

        # 将查询向量按照指定维度重新排列，并进行转置，以便进行多头机制的并行计算
        query = query.view(
            query_projected.size(0), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # 将键向量按照指定维度重新排列，并进行转置，以便进行多头机制的并行计算
        key = key.view(
            key_projected.size(0), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # 将值向量按照指定维度重新排列，并进行转置，以便进行多头机制的并行计算
        value = value.view(
            value_projected.size(0), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # 使用函数库中的缩放点积注意力函数计算注意力权重和加权后的值
        attn = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
        )

        # 将多头注意力的输出重新排列，以匹配 nn.MHA 的返回签名
        attn = attn.transpose(1, 2).reshape(query.size(0), -1, self.embed_dim)
        # 对注意力输出进行线性变换，使用权重 self.out_proj
        return F.linear(attn, self.out_proj)

    # 重置模型参数的方法
    def reset_parameters(self):
        # 使用 Xavier 均匀分布初始化查询向量的线性变换权重 self.q_proj_weight
        nn.init.xavier_uniform_(self.q_proj_weight)
        # 使用 Xavier 均匀分布初始化键向量的线性变换权重 self.k_proj_weight
        nn.init.xavier_uniform_(self.k_proj_weight)
        # 使用 Xavier 均匀分布初始化值向量的线性变换权重 self.v_proj_weight
        nn.init.xavier_uniform_(self.v_proj_weight)
        # 将输出向量的线性变换权重 self.out_proj 初始化为常数 0
        nn.init.constant_(self.out_proj, 0.0)
# 定义一个函数，运行单个实验并返回实验结果对象
def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    # 将设备设置为 CUDA
    device = torch.device("cuda")
    # 创建一个 CompositeMHA 对象，使用给定的头数、嵌入维度、设备和数据类型
    composite_mha = CompositeMHA(
        config.num_heads, config.embed_dim, device, config.dtype
    )
    # 重置 CompositeMHA 对象的参数
    composite_mha.reset_parameters()
    # 生成查询、键、值张量作为输入，使用给定的批量大小、查询序列长度、键序列长度、嵌入维度和设备
    query, key, value = generate_inputs(
        config.batch_size,
        config.q_seq_len,
        config.k_seq_len,
        config.embed_dim,
        config.dtype,
        device,
    )
    # 创建一个因果偏置对象作为注意力掩码，使用 LOWER_RIGHT 变体，给定的查询序列长度和键序列长度
    attn_mask = CausalBias(
        CausalVariant.LOWER_RIGHT, config.q_seq_len, config.k_seq_len
    )
    # 在指定设备上实例化注意力掩码张量
    attn_mask_tensor = attn_mask._materialize(device)

    # 使用 benchmark_torch_function_in_microseconds 函数基准测试 CompositeMHA 对象在不同掩码下的运行时间
    materialized_mask_time = benchmark_torch_function_in_microseconds(
        composite_mha, query, key, value, attn_mask_tensor
    )
    # 使用子类化的注意力掩码对象进行相同基准测试
    attn_mask_subclass_time = benchmark_torch_function_in_microseconds(
        composite_mha, query, key, value, attn_mask
    )
    # 断言两种掩码方式下的 CompositeMHA 对象输出的结果接近
    torch.testing.assert_close(
        composite_mha(query, key, value, attn_mask_tensor),
        composite_mha(query, key, value, attn_mask),
    )

    # 返回实验结果对象，包括两种掩码方式下的运行时间
    return ExperimentResults(
        materialized_mask_time=materialized_mask_time,
        attn_mask_subclass_time=attn_mask_subclass_time,
    )


# 生成一组实验配置列表
def generate_experiment_configs() -> List[ExperimentConfig]:
    # 定义实验的批量大小、头数、查询和键值序列长度、嵌入维度和数据类型的组合
    batch_sizes = [1, 8, 16, 128]
    num_heads = [16, 32]
    q_kv_seq_lens = [(128, 256), (256, 416), (512, 4097), (1024, 2048), (1, 2048)]
    embed_dims = [2048, 4096]
    dtypes = [
        torch.bfloat16,
    ]
    all_configs = []
    # 生成所有可能的实验配置组合
    for bsz, heads, (q_seq_len, kv_seq_len), embed_dim, dtype in itertools.product(
        batch_sizes, num_heads, q_kv_seq_lens, embed_dims, dtypes
    ):
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                q_seq_len=q_seq_len,
                k_seq_len=kv_seq_len,
                embed_dim=embed_dim,
                dtype=dtype,
            )
        )

    return all_configs


# 计算两种实验结果之间的速度提升比例
def calculate_speedup(results: ExperimentResults) -> float:
    return results.materialized_mask_time / results.attn_mask_subclass_time


# 打印实验结果的表格
def print_results(results: List[Experiment]):
    # 计算所有实验结果的速度提升比例
    speedups = [calculate_speedup(r.results) for r in results]

    # 找到速度提升比例的最大值和最小值的索引
    max_speedup_index = np.argmax(speedups)
    min_speedup_index = np.argmin(speedups)

    # 获取速度提升比例最大和最小的实验配置字典
    max_config_dict = results[max_speedup_index].config.asdict()
    min_config_dict = results[min_speedup_index].config.asdict()

    # 创建表格数据
    table_data = [
        {
            "Type": "Average",
            "Speedup": np.mean(speedups),
            **dict.fromkeys(max_config_dict),
        },
        {"Type": "Max", "Speedup": speedups[max_speedup_index], **max_config_dict},
        {"Type": "Min", "Speedup": speedups[min_speedup_index], **min_config_dict},
    ]

    # 打印表格
    print(tabulate(table_data, headers="keys", tablefmt="pretty"))


# 主函数，入口点
def main():
    seed = 123
    # 设置 NumPy 和 PyTorch 的随机种子，以确保实验的可重复性
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 初始化一个空列表，用于存储实验结果
    results = []
    
    # 对每个生成的实验配置运行一个计时实验，比较 nn_mha 和 composite_mha 的性能
    for config in tqdm(generate_experiment_configs()):
        # 创建一个实验对象，并运行单个实验
        results.append(Experiment(config, run_single_experiment(config)))
    
    # 打印实验结果
    print_results(results)
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行（而不是被导入为模块），则执行以下代码块
    main()
```