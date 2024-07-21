# `.\pytorch\benchmarks\transformer\score_mod.py`

```
import argparse
import itertools
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn.attention._flex_attention import _flex_attention

# 禁用自动动态形状，避免在函数参数更改时重新编译
torch._dynamo.config.automatic_dynamic_shapes = False
# 由于更改函数参数导致重新编译，设置缓存大小限制
torch._dynamo.config.cache_size_limit = 1000

# 导入性能测试函数
from triton.testing import do_bench


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # 预热阶段，运行函数5次
    for _ in range(5):
        func(*args, **kwargs)
    # 测量函数执行时间并将结果转换为毫秒
    return do_bench(lambda: func(*args, **kwargs)) * 1e3


@dataclass(frozen=True)
class ExperimentConfig:
    shape: Tuple[int]               # 实验数据形状
    score_mod: Callable             # 分数修改函数
    dtype: torch.dtype             # 数据类型
    calculate_bwd_time: bool       # 是否计算反向传播时间

    def __post_init__(self):
        assert len(self.shape) == 4, "Shape must be of length 4"  # 确保形状长度为4

    def asdict(self):
        # 将数据类实例转换为字典
        d = asdict(self)
        # 移除 'calculate_bwd_time' 键
        d.pop("calculate_bwd_time", None)
        return d


@dataclass(frozen=True)
class Times:
    eager_time: float          # 急切执行时间
    compiled_time: float       # 编译执行时间


@dataclass(frozen=True)
class ExperimentResults:
    fwd_times: Times           # 前向传播时间
    bwd_times: Optional[Times] # 可选的反向传播时间


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig   # 实验配置
    results: ExperimentResults # 实验结果

    def asdict(self):
        dict1 = self.config.asdict()  # 转换配置为字典
        dict2 = asdict(self.results)  # 转换结果为字典
        return {**dict1, **dict2}


def generate_inputs(
    batch_size: int,
    num_heads: int,
    q_sequence_length: int,
    kv_sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
):
    q_shape = (batch_size, q_sequence_length, num_heads * head_dim)  # 查询张量形状
    kv_shape = (batch_size, kv_sequence_length, num_heads * head_dim)  # 键值对张量形状

    # 创建部分应用函数以生成随机张量
    make_q = partial(
        torch.rand, q_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )
    make_kv = partial(
        torch.rand, kv_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )
    # 生成查询、键和值张量，并对维度进行转置
    query = (
        make_q()
        .view(batch_size, q_sequence_length, num_heads, head_dim)
        .transpose(1, 2)
    )
    key = (
        make_kv()
        .view(batch_size, kv_sequence_length, num_heads, head_dim)
        .transpose(1, 2)
    )
    value = (
        make_kv()
        .view(batch_size, kv_sequence_length, num_heads, head_dim)
        .transpose(1, 2)
    )
    return query, key, value


def run_single_experiment(
    config: ExperimentConfig, dynamic=False, max_autotune=False
) -> ExperimentResults:
    device = torch.device("cuda")  # 设置计算设备为GPU
    batch_size, num_heads, q_seq_len, head_dim = config.shape  # 从配置中获取实验参数
    # 使用 generate_inputs 函数生成输入数据，包括查询、键、值
    query, key, value = generate_inputs(
        batch_size,
        num_heads,
        q_seq_len,
        q_seq_len,
        head_dim,
        config.dtype,
        device,
        requires_grad=config.calculate_bwd_time,
    )

    # 定义 eager_sdpa 函数，调用 F.scaled_dot_product_attention 函数执行自注意力机制
    def eager_sdpa(query, key, value, _):
        return F.scaled_dot_product_attention(query, key, value)

    # 根据 max_autotune 参数选择编译方式，生成 compiled_sdpa 函数
    if max_autotune:
        compiled_sdpa = torch.compile(
            _flex_attention, dynamic=dynamic, mode="max-autotune-no-cudagraphs"
        )
    else:
        compiled_sdpa = torch.compile(_flex_attention, dynamic=dynamic)

    # 获取配置中的 score_mod 参数
    score_mod = config.score_mod

    # 使用 benchmark_torch_function_in_microseconds 函数计算 eager_sdpa 函数的前向时间
    forward_eager_time = benchmark_torch_function_in_microseconds(
        eager_sdpa, query, key, value, score_mod
    )

    # 使用 benchmark_torch_function_in_microseconds 函数计算 compiled_sdpa 函数的前向时间
    forward_compiled_time = benchmark_torch_function_in_microseconds(
        compiled_sdpa, query, key, value, score_mod
    )

    # 如果需要计算反向传播时间
    if config.calculate_bwd_time:
        # 执行 eager_sdpa 函数获取输出
        out_eager = eager_sdpa(query, key, value, score_mod)
        # 生成随机梯度作为反向传播输入
        dOut = torch.randn_like(out_eager)
        # 使用 benchmark_torch_function_in_microseconds 函数计算 eager_sdpa 函数的反向传播时间
        backward_eager_time = benchmark_torch_function_in_microseconds(
            out_eager.backward, dOut, retain_graph=True
        )

        # 执行 compiled_sdpa 函数获取输出
        out_compile = compiled_sdpa(query, key, value, score_mod)
        # 生成随机梯度作为反向传播输入
        dOut = torch.randn_like(out_eager)
        # 使用 benchmark_torch_function_in_microseconds 函数计算 compiled_sdpa 函数的反向传播时间
        backward_compile_time = benchmark_torch_function_in_microseconds(
            out_compile.backward, dOut, retain_graph=True
        )

        # 返回实验结果，包括前向和反向传播时间
        return ExperimentResults(
            fwd_times=Times(forward_eager_time, forward_compiled_time),
            bwd_times=Times(backward_eager_time, backward_compile_time),
        )
    else:
        # 返回实验结果，仅包括前向传播时间
        return ExperimentResults(
            fwd_times=Times(forward_eager_time, forward_compiled_time),
            bwd_times=None,
        )
# 计算给定实验结果中的加速比
def calculate_speedup(results: ExperimentResults, type: str) -> float:
    # 如果类型为 "fwd"，返回前向传播时间的比值
    if type == "fwd":
        return results.fwd_times.eager_time / results.fwd_times.compiled_time
    # 如果类型为 "bwd"，返回后向传播时间的比值，确保结果不为 None
    elif type == "bwd":
        assert results.bwd_times is not None
        return results.bwd_times.eager_time / results.bwd_times.compiled_time
    # 如果类型不是 "fwd" 或 "bwd"，抛出值错误异常
    else:
        raise ValueError(f"Invalid type {type}")


# 从函数对象中获取函数名称，去除局部函数标识和位置信息
def get_func_name(func):
    return func.__name__.split("<locals>.")[-1].split(" at ")[0]


# 计算给定实验结果列表中指定类型的平均加速比和相关配置信息
def get_average_speedups(results: List[Experiment], type: str):
    # 计算加速比列表
    speedups = [calculate_speedup(r.results, type) for r in results]

    # 查找最大和最小加速比的索引
    max_speedup_index = np.argmax(speedups)
    min_speedup_index = np.argmin(speedups)

    # 获取最大和最小加速比对应的配置字典
    max_config_dict = results[max_speedup_index].config.asdict()
    min_config_dict = results[min_speedup_index].config.asdict()

    # 从 score_mod 字符串中提取函数名称
    max_config_dict["score_mod"] = (
        max_config_dict["score_mod"].__name__.split("<locals>.")[-1].split(" at ")[0]
    )
    min_config_dict["score_mod"] = (
        min_config_dict["score_mod"].__name__.split("<locals>.")[-1].split(" at ")[0]
    )

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

    return table_data


# 打印给定实验结果列表的汇总信息
def print_results(results: List[Experiment]):
    # 初始化表格数据为 defaultdict(list)
    table_data = defaultdict(list)
    for experiment in results:
        for key, value in experiment.asdict().items():
            if key == "fwd_times":
                # 将前向传播时间数据添加到表格数据中
                for name, time in value.items():
                    table_data[f"fwd_{name}"].append(float(time))
            elif key == "bwd_times":
                # 如果配置指定计算后向传播时间，则将数据添加到表格数据中
                if experiment.config.calculate_bwd_time:
                    for name, time in value.items():
                        table_data[f"bwd_{name}"].append(float(time))
            else:
                # 将其他键值对直接添加到表格数据中
                table_data[key].append(value)

    # 计算前向传播加速比列表
    fwd_speedups = [calculate_speedup(r.results, type="fwd") for r in results]
    table_data["fwd_speedup"] = fwd_speedups

    # 如果配置指定计算后向传播时间，则计算后向传播加速比列表并添加到表格数据中
    if results[0].config.calculate_bwd_time:
        bwd_speedups = [calculate_speedup(r.results, type="bwd") for r in results]
        table_data["bwd_speedup"] = bwd_speedups

    # 从 score_mod 函数对象中提取函数名称，并添加到表格数据中
    table_data["score_mod"] = [get_func_name(func) for func in table_data["score_mod"]]

    # 打印表格数据，使用 GitHub 格式，小数点精确到三位
    print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    # 打印分隔线和前向传播加速比汇总信息
    print("\n")
    print("FWD Speedups".center(125, "="))
    print("\n")

    # 获取前向传播加速比的平均信息并打印
    average_data = get_average_speedups(results, type="fwd")
    print(tabulate(average_data, headers="keys", tablefmt="github", floatfmt=".3f"))
    # 如果结果列表中第一个元素的配置要求计算反向传播时间
    if results[0].config.calculate_bwd_time:
        # 打印空行
        print("\n")
        # 打印标题 "BWD Speedups" 居中显示，宽度为 125，用 "=" 包围
        print("BWD Speedups".center(125, "="))
        # 打印空行
        print("\n")
        # 获取结果列表中所有结果的反向传播速度提升的平均值数据
        average_data = get_average_speedups(results, type="bwd")
        # 将平均值数据以 GitHub 风格的表格格式打印，保留三位小数
        print(tabulate(average_data, headers="keys", tablefmt="github", floatfmt=".3f"))
def generate_score_mods(score_mods: List[str]) -> List[Callable]:
    # 定义一个无操作函数，接受参数 score, b, h, m, n，并直接返回 score
    def noop(score, b, h, m, n):
        return score

    # 定义一个因果掩码函数，接受参数 score, b, h, token_q, token_kv，
    # 使用 torch.where 函数根据 token_q 和 token_kv 的关系修改 score
    def causal_mask(score, b, h, token_q, token_kv):
        return torch.where(token_q >= token_kv, score, float("-inf"))

    # 定义一个相对偏差函数，接受参数 score, b, h, m, n，返回 score 加上 m 减去 n 的结果
    def relative_bias(score, b, h, m, n):
        return score + (m - n)

    # 定义一个头部偏差函数，接受参数 score, b, h, m, n，返回 score 加上 2 倍的 h
    def head_bias(score, b, h, m, n):
        return score + 2 * h

    # 将各个函数放入字典中，以便根据名称动态获取函数对象
    function_dict = {
        "noop": noop,
        "causal": causal_mask,
        "rel": relative_bias,
        "head_bias": head_bias,
    }
    # 根据 score_mods 中的名称列表，返回对应的函数列表
    return [function_dict[name] for name in score_mods]


def generate_experiment_configs(
    calculate_bwd: bool,
    dtype: torch.dtype,
    batch_sizes: List[int],
    num_heads: List[int],
    seq_lens: List[int],
    head_dims: List[int],
    score_mods: List[str],
) -> List[ExperimentConfig]:
    # 生成 q_kv_seq_lens 列表，其中每个元素是 (i, i)，i 是 seq_lens 中的元素，用于测试 q_len == kv_len
    q_kv_seq_lens = [(i, i) for i in seq_lens]
    dtypes = [dtype]
    # 通过调用 generate_score_mods 函数得到 score_mods 对应的函数列表
    score_mods = generate_score_mods(score_mods)
    all_configs = []
    # 使用 itertools.product 生成所有可能的参数组合
    for (
        bsz,
        n_heads,
        (q_seq_len, kv_seq_len),
        head_dim,
        score_mod,
        dtype,
    ) in itertools.product(
        batch_sizes, num_heads, q_kv_seq_lens, head_dims, score_mods, dtypes
    ):
        # 断言 q_seq_len 等于 kv_seq_len，目前只支持相等长度的输入
        assert q_seq_len == kv_seq_len, "Only equal length inputs supported for now."
        # 创建 ExperimentConfig 对象，并添加到 all_configs 列表中
        all_configs.append(
            ExperimentConfig(
                shape=(bsz, n_heads, q_seq_len, head_dim),
                score_mod=score_mod,
                dtype=dtype,
                calculate_bwd_time=calculate_bwd,
            )
        )

    return all_configs


def main(args):
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = []
    # 遍历 generate_experiment_configs 函数返回的配置列表，执行单个实验并将结果添加到 results 中
    for config in tqdm(
        generate_experiment_configs(
            args.calculate_bwd, args.dtype, args.b, args.nh, args.s, args.d, args.mods
        )
    ):
        results.append(
            Experiment(
                config,
                run_single_experiment(
                    config, dynamic=args.dynamic, max_autotune=args.max_autotune
                ),
            )
        )

    # 打印实验结果
    print_results(results)


if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser(
        description="Run sweep over sizes and score mods for flex attention"
    )
    # 添加参数选项
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Runs a dynamic shapes version of compiled flex attention.",
    )
    parser.add_argument(
        "--calculate-bwd", action="store_true", help="Calculate backward pass times"
    )
    parser.add_argument("-dtype", type=str, help="dtype", default="bfloat16")
    parser.add_argument(
        "-b", type=int, nargs="+", help="batch sizes", default=[2, 8, 16]
    )
    parser.add_argument("-nh", type=int, nargs="+", help="# of heads", default=[16])
    parser.add_argument(
        "-s", type=int, nargs="+", help="sequence lengths", default=[512, 1024, 4096]
    )
    # 添加一个接受整数类型参数的命令行选项 "-d"，可以接受一个或多个值
    parser.add_argument("-d", type=int, nargs="+", help="head dims", default=[64, 128])
    # 添加一个接受字符串类型参数的命令行选项 "-mods"，可以接受一个或多个值
    parser.add_argument(
        "-mods",
        type=str,
        nargs="+",
        help="score mods",
        default=["noop", "causal", "rel", "head_bias"],
    )
    # 添加一个开关类型的命令行选项 "--max-autotune"
    # 当开启时，用于启用最大自动调整
    parser.add_argument(
        "--max-autotune", action="store_true", help="Turn on max-autotune"
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 将命令行参数中指定的数据类型名称转换为对应的 torch 模块的数据类型
    args.dtype = getattr(torch, args.dtype)

    # 调用主函数，并传递解析后的参数对象
    main(args)
```