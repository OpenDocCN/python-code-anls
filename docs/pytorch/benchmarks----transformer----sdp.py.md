# `.\pytorch\benchmarks\transformer\sdp.py`

```py
# 导入需要的库和模块
import argparse  # 解析命令行参数的模块
import itertools  # 提供用于创建迭代器的函数的模块
import random  # 生成伪随机数的模块

import warnings  # 控制警告行为的模块
from dataclasses import dataclass  # 创建数据类的装饰器
from pathlib import Path  # 操作文件路径的模块
from pprint import pprint  # 漂亮打印数据结构的模块
from typing import List, Optional  # 提供类型提示的模块

import numpy as np  # 多维数组和矩阵运算的模块
from prettytable import PrettyTable  # 创建漂亮的表格的模块
from tqdm import tqdm  # 创建进度条的模块

import torch  # PyTorch深度学习库
import torch.utils.benchmark as benchmark  # 运行性能基准测试的模块
from torch.backends.cuda import sdp_kernel  # CUDA加速计算的模块

warnings.filterwarnings("ignore")  # 忽略警告消息


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int  # 批量大小
    num_heads: int  # 注意力头的数量
    max_sequence_len: int  # 最大序列长度
    embed_dimension: int  # 嵌入维度
    dtype: torch.dtype  # 数据类型
    pad_percentage: Optional[float]  # 填充百分比（可选）
    enable_math: bool  # 启用数学运算
    enable_flash: bool  # 启用闪存
    enable_mem_efficient: bool  # 启用内存效率
    enable_cudnn: bool  # 启用cuDNN加速

    def get_entries(self) -> List:
        return [
            self.batch_size,
            self.num_heads,
            self.max_sequence_len,
            self.embed_dimension,
            self.dtype,
            self.pad_percentage,
            self.enable_math,
            self.enable_flash,
            self.enable_mem_efficient,
            self.enable_cudnn,
        ]

    @classmethod
    def get_entry_names(cls) -> List[str]:
        return [
            "batch_size",
            "num_heads",
            "max_sequence_len",
            "embed_dimension",
            "dtype",
            "pad_percentage",
            "enable_math",
            "enable_flash",
            "enable_mem_efficient",
            "enable_cudnn",
        ]


@dataclass(frozen=True)
class ExperimentResults:
    nn_mha_time: float  # 非编译MHA时间
    compiled_nn_mha_time: Optional[float]  # 编译非编译MHA时间（可选）
    composite_mha_time: float  # 复合MHA时间
    compiled_composite_mha_time: Optional[float]  # 编译复合MHA时间（可选）

    def get_entries(self) -> List:
        return [
            f"{self.nn_mha_time:2f}",
            f"{self.compiled_nn_mha_time:2f}" if self.compiled_nn_mha_time else None,
            f"{self.composite_mha_time:2f}",
            f"{self.compiled_composite_mha_time:2f}" if self.compiled_composite_mha_time else None,
        ]

    @classmethod
    def get_entry_names(cls) -> List[str]:
        return [
            "nn_mha_time (\u00B5s)",  # 非编译MHA时间（微秒）
            "compiled_nn_mha_time (\u00B5s)",  # 编译非编译MHA时间（微秒）
            "composite_mha_time (\u00B5s)",  # 复合MHA时间（微秒）
            "compiled_composite_mha_time (\u00B5s)",  # 编译复合MHA时间（微秒）
        ]


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig  # 实验配置对象
    results: ExperimentResults  # 实验结果对象

    def get_entries(self) -> List:
        return self.config.get_entries() + self.results.get_entries()


class CompositeMHA(torch.nn.Module):
    def __init__(self, num_heads, in_proj_weight, in_proj_bias, out_proj):
        super().__init__()
        self.in_proj_weight = in_proj_weight  # 输入投影权重
        self.in_proj_bias = in_proj_bias  # 输入投影偏置
        self.out_proj = out_proj  # 输出投影
        self.num_heads = num_heads  # 注意力头的数量
    # 定义一个前向传播方法，用于多头注意力机制
    def forward(self, query, key, value, mask):
        # 检查 query、key、value 是否是同一个张量，若不是则抛出异常
        if not (query is key and key is value):
            raise NotImplementedError(
                "query, key and value must be the same Tensor for now."
            )
        # 检查是否支持 mask，目前未实现支持 mask 的功能，若有则抛出异常
        if mask is not None:
            raise NotImplementedError("mask is currently not supported.")

        # 对 query 应用线性变换，使用输入的权重和偏置
        query_projected = torch.nn.functional.linear(
            query, self.in_proj_weight, self.in_proj_bias
        )

        # 获取 batch 大小、嵌入维度和头部维度
        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        # 将 query_projected 分割成 query、key、value 三部分
        query, key, value = query_projected.chunk(3, -1)

        # 将每个部分重塑为 (batch_size, seq_len, num_heads, head_dim)，并转置最后两个维度
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        # 使用缩放点积注意力计算注意力矩阵 attn，attn 的形状为 (batch_size, num_heads, seq_len, head_dim)
        attn = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,  # 注意力掩码，默认为 None
            dropout_p=0.0,   # dropout 概率，默认为 0.0，即不应用 dropout
            is_causal=False, # 是否是因果注意力，默认为 False
        )

        # 将注意力矩阵 attn 转置并重塑为 (batch_size, seq_len, num_heads * head_dim)
        attn = attn.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)

        # 使用输出投影层 out_proj 处理注意力矩阵，并返回结果，第二个返回值为 None
        return self.out_proj(attn), None
# 确保 pt 对象中的 _qkv_same_embed_dim 属性为真，否则触发断言错误
assert pt._qkv_same_embed_dim
# 获取 pt 对象的 in_proj_weight 属性，确保其不为 None
in_proj_weight = pt.in_proj_weight
assert in_proj_weight is not None  # 再次确保 in_proj_weight 不为 None
# 确保 pt 对象的 batch_first 属性为真
assert pt.batch_first
# 返回一个 CompositeMHA 对象，使用 pt 对象的 num_heads, in_proj_weight, in_proj_bias, out_proj 属性作为参数
return CompositeMHA(pt.num_heads, pt.in_proj_weight, pt.in_proj_bias, pt.out_proj)


# 生成一个随机的批次数据
def generate_rand_batch(
    batch_size,
    max_sequence_len,
    embed_dimension,
    pad_percentage=None,
    dtype=torch.float16,
    device="cuda",
):
    if not pad_percentage:
        # 如果 pad_percentage 为 None，则返回一个随机生成的张量，以及 None
        return (
            torch.randn(
                batch_size,
                max_sequence_len,
                embed_dimension,
                dtype=dtype,
                device=device,
            ),
            None,
        )
    # 使用高斯分布生成每个序列的长度列表
    seq_len_list = [
        int(max_sequence_len * (1 - random.gauss(pad_percentage, 0.01)))
        for _ in range(batch_size)
    ]
    # 将随机选择的序列长度设置为 max_sequence_len，以确保至少有一个序列长度等于 max_sequence_len
    seq_len_list[random.randint(0, batch_size - 1)] = max_sequence_len
    # 返回一个 nested_tensor，包含根据 seq_len_list 中的长度生成的随机张量序列，以及 seq_len_list
    return (
        torch.nested.nested_tensor(
            [
                torch.randn(seq_len, embed_dimension, dtype=dtype, device=device)
                for seq_len in seq_len_list
            ]
        ),
        seq_len_list,
    )


# 以微秒为单位对 torch 函数进行基准测试
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    # 创建一个 Timer 对象，用于测量函数 f 的执行时间
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    # 返回函数 f 执行时间的均值（以微秒为单位）
    return t0.blocked_autorange().mean * 1e6


# 对两个张量进行近似相等的断言检查，适用于 nested_tensor
def assert_close_tensors(tensor_a, tensor_b):
    # 如果 tensor_a 和 tensor_b 都是 nested_tensor，则逐个检查它们的元素张量是否在给定的误差范围内相等
    if tensor_a.is_nested and tensor_b.is_nested:
        for a, b in zip(tensor_a.unbind(), tensor_b.unbind()):
            assert torch.allclose(a, b, atol=1e-2, rtol=1e-2)
    else:
        # 否则，直接检查两个张量是否在给定的误差范围内相等
        assert torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3)


# 运行单个实验的函数，使用给定的 ExperimentConfig 参数，返回 ExperimentResults
def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    # 使用 sdp_kernel 上下文管理器来运行实验
    with sdp_kernel(
        enable_math=config.enable_math,
        enable_flash=config.enable_flash,
        enable_mem_efficient=config.enable_mem_efficient,
        enable_cudnn=config.enable_cudnn,
    ):
        # 在此处应该继续添加实验的具体代码，但未提供
        pass
    ) as kernel_choice, torch.inference_mode() as inference_mode:
        # 设置 dropout 概率为 0，初始化 mask 为 None
        dropout_p = 0.0
        mask = None

        # 创建一个多头注意力机制的 PyTorch 模型 nn_mha
        nn_mha = torch.nn.MultiheadAttention(
            embed_dim=config.embed_dimension,
            num_heads=config.num_heads,
            batch_first=True,
            dropout=dropout_p,
        )
        # 将 nn_mha 设为评估模式，并移动到 CUDA 设备上，使用指定的数据类型
        nn_mha = nn_mha.eval().to("cuda", config.dtype)
        
        # 根据 nn_mha 构建一个复合的多头注意力机制 composite_mha
        composite_mha = build_composite_mha_from_nn_mha(nn_mha)
        
        # 生成随机的批量数据 qkv 和对应的长度信息 lengths
        qkv, lengths = generate_rand_batch(
            config.batch_size,
            config.max_sequence_len,
            config.embed_dimension,
            config.pad_percentage,
            config.dtype,
        )
        
        # 使用 nn_mha 处理 qkv 数据，得到 nn_mha_output
        nn_mha_output, _ = nn_mha(qkv, qkv, qkv, mask)
        
        # 使用 composite_mha 处理 qkv 数据，得到 composite_mha_output
        composite_mha_output, _ = composite_mha(qkv, qkv, qkv, mask)

        # 第一次顺序性检查，确保 nn_mha_output 和 composite_mha_output 接近
        assert_close_tensors(nn_mha_output, composite_mha_output)

        # 对 nn_mha 和 composite_mha 分别进行性能基准测试，并记录时间
        nn_mha_time = benchmark_torch_function_in_microseconds(
            nn_mha, qkv, qkv, qkv, mask
        )
        composite_mha_time = benchmark_torch_function_in_microseconds(
            composite_mha, qkv, qkv, qkv, mask
        )

        # 如果 pad_percentage 为 None，则使用 TorchDynamo 编译 nn_mha 和 composite_mha
        if config.pad_percentage is None:
            compiled_nn_mha = torch.compile(nn_mha)
            compiled_composite_mha = torch.compile(composite_mha)

            # 对编译后的 nn_mha 进行性能基准测试，并记录时间
            compiled_nn_mha_time = benchmark_torch_function_in_microseconds(
                compiled_nn_mha, qkv, qkv, qkv, mask
            )

            # 对编译后的 composite_mha 进行性能基准测试，并记录时间
            compiled_composite_mha_time = benchmark_torch_function_in_microseconds(
                compiled_composite_mha,
                qkv,
                qkv,
                qkv,
                mask,
            )
        else:
            # 如果 pad_percentage 不为 None，则将编译后的时间设为 None
            compiled_nn_mha_time = None
            compiled_composite_mha_time = None

        # 将所有结果汇总到 ExperimentResults 中，并返回 Experiment 对象
        results = ExperimentResults(
            nn_mha_time,
            compiled_nn_mha_time,
            composite_mha_time,
            compiled_composite_mha_time,
        )
        return Experiment(config, results)
# 定义一个生成实验配置的函数，可能返回一个配置生成器
def generate_experiments(
    batch_sizes, num_heads, max_seq_lens, embed_dims, dtypes, pad_percentages
) -> List[ExperimentConfig]:
    # 初始化空的配置列表
    configs = []
    # 使用 itertools.product 生成所有可能的参数组合
    for bsz, n_heads, seq_len, embed_dim, dtype, padding in itertools.product(
        batch_sizes, num_heads, max_seq_lens, embed_dims, dtypes, pad_percentages
    ):
        # 创建一个 ExperimentConfig 对象并添加到 configs 列表中
        configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=n_heads,
                max_sequence_len=seq_len,
                embed_dimension=embed_dim,
                dtype=dtype,
                pad_percentage=padding,
                enable_math=False,
                enable_flash=True,
                enable_mem_efficient=True,
                enable_cudnn=True,
            )
        )
    # 返回生成的配置列表
    return configs


def main(save_path: Optional[Path]):
    # 设置随机种子
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 创建一个 ExperimentConfig 对象用于单个实验
    config = ExperimentConfig(
        batch_size=128,
        num_heads=8,
        max_sequence_len=512,
        embed_dimension=512,
        dtype=torch.float16,
        pad_percentage=None,
        enable_math=False,
        enable_flash=True,
        enable_mem_efficient=True,
        enable_cudnn=True,
    )

    # 运行单个实验，并打印实验结果
    experiment = run_single_experiment(config)
    pprint(experiment)

    # 创建一个 PrettyTable 对象用于展示实验结果的表格
    table = PrettyTable()
    table.float_format = ".3"
    # 设置表格的列名，包括 ExperimentConfig 和 ExperimentResults 的条目
    table.field_names = (
        ExperimentConfig.get_entry_names() + ExperimentResults.get_entry_names()
    )

    # 准备多个实验的参数组合
    batch_sizes = [256]
    num_heads = [32]
    max_seq_lens = [256]
    embed_dims = [512]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    pad_percentages = [None, 0.9]

    # 生成多个实验配置对象
    experiment_configs = generate_experiments(
        batch_sizes, num_heads, max_seq_lens, embed_dims, dtypes, pad_percentages
    )

    # 执行多个实验，并将实验结果添加到 table 中
    experiments: List[Experiment] = []
    for experiment_config in tqdm(experiment_configs):
        experiment = run_single_experiment(experiment_config)
        experiments.append(experiment)
        table.add_row(experiment.get_entries())

    # 打印表格
    print(table)

    # 获取表格的 CSV 字符串
    csv_string = table.get_csv_string()
    # 如果指定了保存路径，则将表格保存为 CSV 文件
    if save_path is not None:
        with open(save_path, "w") as csvfile:
            csvfile.write(csv_string)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", "--save_path", type=str, help="Path to save the results"
    )

    args = parser.parse_args()
    save_path = Path(args.save_path) if args.save_path else None
    # 执行主函数，并传入保存路径参数
    main(save_path)
```