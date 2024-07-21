# `.\pytorch\benchmarks\gpt_fast\benchmark.py`

```
# 导入必要的模块
import argparse  # 用于解析命令行参数
import csv  # 用于CSV文件的读写操作
import dataclasses  # 用于数据类的定义
import os  # 提供与操作系统交互的功能

from generate import run_llama2_7b_bf16, run_llama2_7b_int8, run_mixtral_8x7b_int8  # 导入本地的模块和函数
from triton.testing import do_bench  # 导入本地模块中的特定函数

import torch  # 导入PyTorch深度学习库
import torch.nn as nn  # 导入PyTorch中的神经网络模块
from torch.utils.flop_counter import FlopCounterMode  # 导入PyTorch中用于浮点操作统计的模块

# A100 GPU的BF16混合精度峰值浮点运算能力
WARMUP_ITER = 5  # 预热迭代次数

A100_40G_BF16_TFLOPS = 312  # A100 GPU的BF16混合精度峰值浮点运算能力（以TFLOPS为单位）

# 数据类，用于记录实验结果
@dataclasses.dataclass
class Experiment:
    name: str  # 实验名称
    metric: str  # 度量标准
    target: float  # 目标数值
    actual: float  # 实际数值
    dtype: str  # 数据类型
    device: str  # 设备类型
    is_model: bool = False  # 是否为模型

# 简单的多层感知机（MLP）模型定义
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype):
        super().__init__()
        # 定义包含线性层和层归一化的神经网络模块列表
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim, dtype=dtype),
                nn.LayerNorm(hidden_dim, dtype=dtype),
                nn.Linear(hidden_dim, output_dim, dtype=dtype),
                nn.LayerNorm(output_dim, dtype=dtype),
            ]
        )

    def forward(self, x):
        # 前向传播函数，遍历每一层并依次计算输出
        for layer in self.layers:
            x = layer(x)
        return x

# 运行MLP模型的层归一化和GELU激活函数的实验
def run_mlp_layer_norm_gelu(device: str = "cuda"):
    # 定义不同数据类型下的预期FLOPs利用率
    dtype_flops_utilization_map = {
        torch.bfloat16: "0.8",
    }
    input_shapes = [1024, 4096, 8192, 16384]  # 输入张量的形状列表
    intermediate_size = 14336  # 中间层的尺寸
    results = []  # 存储实验结果的列表

    # 遍历每种数据类型和对应的预期FLOPs利用率
    for dtype, expected_flops_utilization in dtype_flops_utilization_map.items():
        flops_utilization = 0
        for D in input_shapes:
            # 创建MLP模型实例并移至指定设备
            mod = SimpleMLP(
                input_dim=D, hidden_dim=intermediate_size, output_dim=D, dtype=dtype
            ).to(device)

            x = torch.randn(D, device=device, dtype=torch.bfloat16)  # 生成随机输入张量

            with FlopCounterMode(display=False) as mode:
                mod(x)  # 执行一次前向传播以统计FLOPs

            flops = mode.get_total_flops()  # 获取总的FLOPs数量

            compiled_mod = torch.compile(mod, dynamic=False)  # 编译模型，关闭动态图

            for _ in range(WARMUP_ITER):
                compiled_mod(x)  # 执行预热迭代

            # 测量模型运行时间，单位为微秒
            us_per_iter = do_bench(lambda: compiled_mod(x)) * 1000
            # 计算并累加FLOPs利用率
            flops_utilization += us_per_iter * flops / 1e9 / A100_40G_BF16_TFLOPS

        # 计算平均FLOPs利用率
        flops_utilization = flops_utilization / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")  # 获取数据类型的字符串表示
        # 将实验结果存入列表
        results.append(
            Experiment(
                "mlp_layer_norm_gelu",
                "flops_utilization",
                expected_flops_utilization,
                f"{flops_utilization:.02f}",
                dtype_str,
                device,
            )
        )
    return results  # 返回所有实验结果的列表

# 运行层归一化实验
def run_layer_norm(device: str = "cuda"):
    # 定义不同数据类型下的内存带宽映射
    dtype_memory_bandwidth_map = {
        torch.bfloat16: "950",
    }
    input_shapes = [1024, 4096, 8192, 16384]  # 输入张量的形状列表
    BS = 4096  # 批量大小
    results = []  # 存储实验结果的列表
    # 遍历 dtype_memory_bandwidth_map 字典中的每一对键值对，其中键是数据类型，值是期望的内存带宽
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        # 初始化内存带宽
        memory_bandwidth = 0
        
        # 遍历 input_shapes 列表中的每个形状 D
        for D in input_shapes:
            # 创建一个具有 D 维度的 LayerNorm 模块，并将其移动到指定的 device 上
            mod = nn.LayerNorm(D).to(device)
            
            # 生成一个具有 BS 大小和 D 维度的随机张量 x，并将其移动到指定的 device 上，并指定数据类型为 dtype
            x = torch.randn(BS, D, device=device, dtype=dtype)
            
            # 使用 torch.compile 函数对模块进行静态编译，dynamic=False 表示禁用动态图功能
            compiled_mod = torch.compile(mod, dynamic=False)
            
            # 对编译后的模块进行预热迭代，以提前将其加载到 GPU 等加速设备中
            for _ in range(WARMUP_ITER):
                compiled_mod(x)
            
            # 测量每次迭代的执行时间，并将其转换为每秒执行的微秒数（us_per_iter）
            us_per_iter = do_bench(lambda: compiled_mod(x)) * 1000
            
            # 计算每秒内存带宽，单位为 GB/s
            # 1e6/us_per_iter 是每秒执行的迭代次数，乘以 2*BS*D*dtype.itemsize 是每次迭代传输的字节数
            memory_bandwidth += (1e6 / us_per_iter) * 2 * BS * D * dtype.itemsize / 1e9
        
        # 计算平均内存带宽，以所有输入形状的数量 input_shapes 为分母
        memory_bandwidth = memory_bandwidth / len(input_shapes)
        
        # 将 dtype 转换为字符串形式，并去除其中的 "torch." 前缀
        dtype_str = str(dtype).replace("torch.", "")
        
        # 将本次实验的结果添加到 results 列表中，以 Experiment 对象的形式存储
        results.append(
            Experiment(
                "layer_norm",  # 实验名称为 "layer_norm"
                "memory_bandwidth(GB/s)",  # 测量指标为 "memory_bandwidth(GB/s)"
                expected_memory_bandwidth,  # 期望的内存带宽，作为对照值
                f"{memory_bandwidth:.02f}",  # 计算得到的内存带宽，保留两位小数
                dtype_str,  # 数据类型的字符串表示
                device,  # 指定的设备
            )
        )
    
    # 返回所有实验结果的列表
    return results
# 使用装饰器为函数配置补丁，设置 coordinate_descent_tuning 为 True
@torch._inductor.config.patch(coordinate_descent_tuning=True)
# 定义函数 run_gather_gemv，接受一个设备参数，默认为 "cuda"
def run_gather_gemv(device: str = "cuda"):
    # 设定 E 的值为 8
    E = 8
    # 创建数据类型到内存带宽映射的字典
    dtype_memory_bandwidth_map = {
        torch.int8: "990",
        torch.bfloat16: "1060",
    }
    # 定义输入形状的列表
    input_shapes = [1024, 4096, 8192, 16384]
    # 创建结果列表
    results = []
    # 遍历数据类型到内存带宽映射的字典
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        # 初始化内存带宽
        memory_bandwidth = 0
        # 遍历输入形状列表
        for D in input_shapes:

            # 定义 gather_gemv 函数，用于执行 gemv 操作
            def gather_gemv(W, score_idxs, x):
                return W[score_idxs].to(x.dtype) @ x

            # 生成随机张量 W，形状为 (E, D, D)，设备为给定设备，数据类型为 dtype
            W = torch.randn(E, D, D, device=device).to(dtype=dtype)
            # 生成随机张量 x，形状为 (D)，设备为给定设备，数据类型为 torch.bfloat16
            x = torch.randn(D, device=device, dtype=torch.bfloat16)
            # 生成张量 score_idxs，内容为 [3, 5]，设备为给定设备
            score_idxs = torch.tensor([3, 5], device=device)

            # 编译 gather_gemv 函数，dynamic=False 表示静态编译
            compiled_fn = torch.compile(gather_gemv, dynamic=False)

            # 进行预热迭代
            for _ in range(WARMUP_ITER):
                compiled_fn(W, score_idxs, x)

            # 测量执行时间
            us_per_iter = do_bench(lambda: compiled_fn(W, score_idxs, x)) * 1000
            # 计算内存带宽
            memory_bandwidth += (1e6 / us_per_iter) * 2 * D * D * dtype.itemsize / 1e9

        # 计算平均内存带宽
        memory_bandwidth = memory_bandwidth / len(input_shapes)
        # 获取数据类型的字符串表示，去除 torch. 前缀
        dtype_str = str(dtype).replace("torch.", "")
        # 将实验结果添加到结果列表
        results.append(
            Experiment(
                "gather_gemv",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
                dtype_str,
                device,
            )
        )
    # 返回结果列表
    return results


# 使用装饰器为函数配置补丁，设置 coordinate_descent_tuning 为 True
@torch._inductor.config.patch(coordinate_descent_tuning=True)
# 定义函数 run_gemv，接受一个设备参数，默认为 "cuda"
def run_gemv(device: str = "cuda"):
    # 创建数据类型到内存带宽映射的字典
    dtype_memory_bandwidth_map = {
        torch.int8: "870",
        torch.bfloat16: "990",
    }
    # 定义输入形状的列表
    input_shapes = [1024, 4096, 8192, 16384]
    # 创建结果列表
    results = []
    # 遍历数据类型到内存带宽映射的字典
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        # 初始化内存带宽
        memory_bandwidth = 0
        # 遍历输入形状列表
        for D in input_shapes:

            # 定义 gemv 函数，用于执行 gemv 操作
            def gemv(W, x):
                return W.to(x.dtype) @ x

            # 生成随机张量 W，形状为 (D, D)，设备为 "cuda"，数据类型为 dtype
            W = torch.randn(D, D, device="cuda").to(dtype=dtype)
            # 生成随机张量 x，形状为 (D)，设备为 "cuda"，数据类型为 torch.bfloat16
            x = torch.randn(D, device="cuda", dtype=torch.bfloat16)

            # 编译 gemv 函数，dynamic=False 表示静态编译
            compiled_fn = torch.compile(gemv, dynamic=False)

            # 进行预热迭代
            for _ in range(WARMUP_ITER):
                compiled_fn(W, x)

            # 测量执行时间
            us_per_iter = do_bench(lambda: compiled_fn(W, x)) * 1000
            # 计算内存带宽
            memory_bandwidth += (1e6 / us_per_iter) * D * D * dtype.itemsize / 1e9

        # 计算平均内存带宽
        memory_bandwidth = memory_bandwidth / len(input_shapes)
        # 获取数据类型的字符串表示，去除 torch. 前缀
        dtype_str = str(dtype).replace("torch.", "")
        # 将实验结果添加到结果列表
        results.append(
            Experiment(
                "gemv",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
                dtype_str,
                device,
            )
        )
    # 返回结果列表
    return results


# 定义函数 output_csv，接受输出文件、标题和行作为参数
def output_csv(output_file, headers, row):
    # 检查输出文件是否存在
    if os.path.exists(output_file):
        # 如果文件存在，打开并读取文件内容为二维列表，如果文件为空则默认为一个空列表
        with open(output_file) as fd:
            lines = list(csv.reader(fd)) or [[]]
            # 如果提供了新的表头并且比文件中的表头长，则更新文件中的表头
            if headers and len(headers) > len(lines[0]):
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        # 如果文件不存在，则初始化行数据为提供的表头
        lines = [headers]

    # 如果输出文件名不是默认文件名，创建输出文件所在目录（如果目录不存在）
    if output_file != DEFAULT_OUTPUT_FILE:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 将当前行数据转换为字符串列表，保留小数点后六位（如果是浮点数）
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
    
    # 将更新后的二维列表写入到输出文件中，每行数据以换行符结束
    with open(output_file, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        # 遍历二维列表的每一行，确保每行数据长度与表头长度相同，不足的部分用 "0" 填充
        for line in lines:
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))
# 默认输出文件名
DEFAULT_OUTPUT_FILE = "gpt_fast_benchmark.csv"

# 所有实验的函数列表，包括不同的 GPT 模型和微基准测试
all_experiments = {
    run_llama2_7b_bf16,    # 运行 LLaMa 2.7B 模型的 BF16 版本
    run_llama2_7b_int8,    # 运行 LLaMa 2.7B 模型的 INT8 版本
    run_mixtral_8x7b_int8, # 运行 Mixtral 8x7B 模型的 INT8 版本
    run_mlp_layer_norm_gelu,  # 运行 MLP Layer Norm Gelu 微基准测试
    run_layer_norm,        # 运行 Layer Norm 微基准测试
    run_gather_gemv,       # 运行 Gather Gemv 微基准测试
    run_gemv               # 运行 Gemv 微基准测试
}


def main(output_file=DEFAULT_OUTPUT_FILE):
    # 存储所有实验结果的列表
    results = []

    # 遍历所有实验函数
    for func in all_experiments:
        # 调用实验函数获取结果列表
        lst = func()
        # 将结果列表中的每个实验结果转换为元组，并加入到结果列表中
        for x in lst:
            results.append(dataclasses.astuple(x))

    # 获取 Experiment 类的所有字段名作为 CSV 文件的表头
    headers = [field.name for field in dataclasses.fields(Experiment)]

    # 将每行结果写入输出 CSV 文件
    for row in results:
        output_csv(output_file, headers, row)


if __name__ == "__main__":
    # 解析命令行参数，设置输出文件名
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Set the output CSV file to save the benchmark results",
    )
    args = parser.parse_args()

    # 执行主函数，传入命令行参数指定的输出文件名
    main(output_file=args.output)
```