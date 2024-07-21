# `.\pytorch\benchmarks\transformer\sdpa.py`

```py
```python`
import itertools
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, List, Tuple

from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.utils.benchmark as benchmark
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # 进行预热，运行 func 函数 5 次以确保 JIT 编译完成
    for _ in range(5):
        func(*args, **kwargs)
    # 创建一个 benchmark.Timer 对象，用于测量 func 函数的执行时间
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    # 返回 func 函数执行时间的中位数，单位是微秒
    return t0.adaptive_autorange(min_run_time=0.1).median * 1e6


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int
    embed_dim: int
    is_causal: bool
    dtype: torch.dtype
    backend: SDPBackend
    device: torch.device = torch.device("cuda")

    @property
    def head_dim(self) -> int:
        # 返回头部维度，即 embed_dim 除以 num_heads 的结果
        return self.embed_dim // self.num_heads

    def asdict(self):
        # 将 ExperimentConfig 对象转换为字典形式
        dict_obj = asdict(self)
        dict_obj["head_dim"] = self.head_dim
        return dict_obj


@dataclass(frozen=True)
class ExperimentResults:
    forward_time: float
    backward_time: float

    def asdict(self):
        # 将 ExperimentResults 对象转换为字典形式
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def asdict(self):
        # 将 Experiment 对象转换为包含 config 和 results 字典的大字典
        dict1 = asdict(self.config)
        dict2 = asdict(self.results)
        return {**dict1, **dict2}


def get_input(
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 创建随机输入张量 q, k, v
    q = torch.randn(
        (config.batch_size, config.num_heads, config.q_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    k = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    v = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    return q, k, v


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    # 获取输入 q, k, v
    q, k, v = get_input(config)
    is_causal = config.is_causal
    # 创建适当的上下文环境，使用 sdpa_kernel 函数或 nullcontext 函数
    context = (
        sdpa_kernel(config.backend) if config.backend is not None else nullcontext()
    )
    # 使用上下文管理器 `context`，执行以下代码块
    with context:
        # 使用 `benchmark_torch_function_in_microseconds` 函数测试 `scaled_dot_product_attention` 函数的执行时间
        forward_time = benchmark_torch_function_in_microseconds(
            scaled_dot_product_attention,  # 使用的函数
            q,  # 参数 q
            k,  # 参数 k
            v,  # 参数 v
            is_causal=is_causal,  # 是否因果
            attn_mask=None,  # 注意力掩码，这里为空
        )
        # 直接调用 `scaled_dot_product_attention` 函数，生成 `out_torch` 结果
        out_torch = scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, attn_mask=None
        )
        # 生成一个与 `out_torch` 相同大小的随机张量 `dOut`
        dOut = torch.randn_like(out_torch)
        # 使用 `benchmark_torch_function_in_microseconds` 测试 `out_torch.backward` 方法的执行时间，同时保留计算图
        backward_time = benchmark_torch_function_in_microseconds(
            out_torch.backward, dOut, retain_graph=True
        )

    # 返回一个 `ExperimentResults` 对象，包含前向和反向传播的时间
    return ExperimentResults(
        forward_time=forward_time,
        backward_time=backward_time,
    )
# 定义一个函数生成实验配置列表，返回一个 ExperimentConfig 对象的列表
def generate_experiment_configs() -> List[ExperimentConfig]:
    # 定义不同的批量大小
    batch_sizes = [
        1,  # 批量大小为1
        8,  # 批量大小为8
    ]
    # 定义头的数量为16
    num_heads = [16]
    # 定义查询和键/值序列长度的组合
    q_kv_seq_lens = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    # 定义嵌入维度为2048
    embed_dims = [2048]
    # 定义后端，如果为 None，则所有后端都启用
    backends = [None]
    # 定义数据类型为 torch 的 bfloat16 类型
    dtypes = [
        torch.bfloat16,
    ]
    # 定义是否为因果关系的标志，True 或 False
    is_causal = [True, False]
    # 初始化一个空的配置列表
    all_configs = []
    # 使用 itertools.product 迭代所有参数的组合
    for (
        bsz,         # 批量大小
        heads,       # 头数量
        (q_seq_len, kv_seq_len),  # 查询和键/值序列长度
        embed_dim,   # 嵌入维度
        causal,      # 是否因果关系
        dtype,       # 数据类型
        backend,     # 后端
    ) in itertools.product(
        batch_sizes, num_heads, q_kv_seq_lens, embed_dims, is_causal, dtypes, backends
    ):
        # 创建 ExperimentConfig 对象并加入到 all_configs 列表中
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                embed_dim=embed_dim,
                is_causal=causal,
                dtype=dtype,
                backend=backend,
            )
        )

    # 返回所有生成的配置列表
    return all_configs


# 定义一个打印实验结果的函数，接受一个 Experiment 对象的列表作为参数
def print_results(experiments: List[Experiment]):
    # 使用 defaultdict 创建一个空的列表字典 table_data
    table_data = defaultdict(list)
    # 遍历每个实验对象
    for experiment in experiments:
        # 将实验对象转换为字典，并将键值对添加到 table_data 中对应的列表中
        for key, value in experiment.asdict().items():
            table_data[key].append(value)
    # 删除 table_data 中的 "device" 键
    del table_data["device"]
    # 如果 "backend" 列表中第一个元素为 None，则删除整个 "backend" 键
    if table_data["backend"][0] is None:
        del table_data["backend"]
    # 使用 tabulate 函数打印 table_data 的内容，设置表头和表格格式
    print(tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".3f"))


# 主函数入口
def main():
    # 设置随机种子
    seed = 123
    torch.manual_seed(seed)
    # 初始化一个结果列表
    results = []
    # 使用 tqdm 显示进度条，遍历生成的所有实验配置
    for config in tqdm(generate_experiment_configs()):
        # 运行单个实验，并将实验结果加入 results 列表中
        results.append(Experiment(config, run_single_experiment(config)))

    # 打印所有实验结果
    print_results(results)


# 如果该脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```