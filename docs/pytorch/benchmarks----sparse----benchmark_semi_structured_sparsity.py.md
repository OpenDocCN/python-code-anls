# `.\pytorch\benchmarks\sparse\benchmark_semi_structured_sparsity.py`

```py
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import random  # 用于生成随机数

import pandas as pd  # 导入 pandas 库，用于数据处理
from tqdm import tqdm  # 导入 tqdm，用于显示进度条

import torch  # 导入 PyTorch 库
import torch.utils.benchmark as benchmark  # 导入 benchmark 模块，用于性能评估
from torch import nn  # 导入神经网络模块
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured  # 导入稀疏张量相关函数


# 设置 PyTorch 的打印选项
torch.set_printoptions(
    precision=2,  # 打印浮点数的精度为两位小数
    threshold=None,  # 打印所有元素，不设限制
    edgeitems=16,  # 最多显示16个元素，超出部分以省略号表示
    linewidth=480,  # 每行输出的字符数限制为480
    profile=None,  # 不进行性能分析
    sci_mode=False,  # 禁用科学计数法显示
)


# 用于剪枝器的辅助模型定义
class Model(nn.Module):
    def __init__(self, m, k, dtype=None):
        super().__init__()
        # 线性层，输入维度为k，输出维度为m
        self.linear = nn.Linear(k, m)

    def forward(self, x):
        return self.linear(x)


def rand_sparse_semi_structured_mask(
    r, c, dtype=torch.float16, device="cuda", choice=None
):
    """
    This function returns a 1:2 sparse matrix of size (r, c).
    Note that this means this matrix will also be 2:4 and 4:8 sparse as well.
    """
    choices = [[0, 1], [1, 0]]
    # 生成稀疏矩阵的随机模板
    mask_entries = [choice or random.choice(choices) for i in range(r * c // 2)]

    return (
        torch.tensor(mask_entries, dtype=dtype, device=device)
        .reshape(r, c)  # 转换为r行c列的形状
        .contiguous()  # 保证存储是连续的
    )


def test_linear(m, k, n, dtype, contiguous, backend):
    SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
    # 生成稀疏权重矩阵
    mask = rand_sparse_semi_structured_mask(m, k, dtype=dtype)
    sparse_weight = torch.rand(m, k).to(dtype).cuda() * mask
    input_tensor = torch.zeros(n, k).to(dtype).cuda()
    # 创建并加载模型到 GPU 上
    model = Model(m, k).to(dtype).cuda().eval()

    dense_measurement = benchmark.Timer(
        stmt="model(input_tensor)",  # 执行模型推理的语句
        globals=locals(),  # 传递局部变量作为全局变量
    ).blocked_autorange()  # 自动测量多次，并按块进行计时

    dense_output = model(input_tensor)  # 密集张量的输出
    print(dense_output.shape)

    # 稀疏化权重
    model.linear.weight = nn.Parameter(
        to_sparse_semi_structured(
            sparse_weight,
        )
    )

    sparse_output = model(input_tensor)  # 稀疏张量的输出
    print(sparse_output.shape)

    sparse_measurement = benchmark.Timer(
        stmt="model(input_tensor)",  # 执行模型推理的语句
        globals=locals(),  # 传递局部变量作为全局变量
    ).blocked_autorange()  # 自动测量多次，并按块进行计时

    correct = torch.allclose(dense_output, sparse_output, rtol=1e-3, atol=1e-3)  # 检查密集输出和稀疏输出是否接近

    return {
        "test_function": "linear",
        "m": m,
        "k": k,
        "n": n,
        "dtype": str(dtype),
        "backend": backend,
        "sparse_latency (ms)": sparse_measurement.median * 1000,  # 稀疏张量推理的中位数延迟，转换为毫秒
        "dense_latency (ms)": dense_measurement.median * 1000,  # 密集张量推理的中位数延迟，转换为毫秒
        "speedup (d/s)": dense_measurement.median / sparse_measurement.median,  # 稀疏加速比
        "correct": correct,  # 推理结果是否正确
        "contiguous": sparse_output.is_contiguous(),  # 稀疏输出是否是连续的
    }


def test_tensor(m, k, n, dtype, contiguous, backend):
    A = rand_sparse_semi_structured_mask(m, k, dtype=dtype)
    B = torch.zeros(k, n).to(dtype).cuda()
    bias = torch.rand(n).to(dtype).cuda()

    sA = to_sparse_semi_structured(A)

    # torch.mm calculation
    # 如果数据类型不是 torch.int8，则进行稠密矩阵乘法运算
    if dtype is not torch.int8:
        # 计算稠密矩阵乘法的输出结果
        dense_output = torch.mm(A, B)

        # 使用 benchmark.Timer 进行稠密矩阵乘法的性能测试
        dense_measurement = benchmark.Timer(
            stmt="torch.mm(A, B)",
            globals=locals(),
        ).blocked_autorange()

    else:
        # 如果数据类型是 torch.int8，则打印不支持的信息
        print("int8 baseline not supported")
        # 使用稀疏矩阵 sA 进行矩阵乘法运算
        dense_output = torch.mm(sA, B)

        # 使用 benchmark.Timer 进行稀疏矩阵乘法的性能测试
        dense_measurement = benchmark.Timer(
            stmt="torch.mm(sA, B)",
            globals=locals(),
        ).blocked_autorange()

    # 计算稀疏矩阵乘法的输出结果
    sparse_output = torch.mm(sA, B)
    # 使用 benchmark.Timer 进行稀疏矩阵乘法的性能测试
    sparse_measurement = benchmark.Timer(
        stmt="torch.mm(sA, B)",
        globals=locals(),
    ).blocked_autorange()

    # 检查稠密输出和稀疏输出的接近程度
    correct = torch.allclose(dense_output, sparse_output, rtol=1e-3, atol=1e-3)

    # 返回测试结果的字典
    return {
        "test_function": "tensor",
        "m": m,
        "k": k,
        "n": n,
        "dtype": str(dtype),
        "backend": backend,
        "sparse_latency (ms)": sparse_measurement.median * 1000,
        "dense_latency (ms)": dense_measurement.median * 1000,
        "speedup (d/s)": dense_measurement.median / sparse_measurement.median,
        "correct": correct,
        "contiguous": sparse_output.is_contiguous(),
    }
if __name__ == "__main__":
    # 定义数据类型的查找字典，将字符串映射为对应的 torch 数据类型
    dtype_lookup = {
        "int8": torch.int8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    # 创建参数解析器对象，用于解析命令行参数并提供描述信息
    parser = argparse.ArgumentParser(description="Semi-Structured Sparsity Benchmarks")
    
    # 添加命令行参数 --mode，指定运行模式，只能是给定的几种选择之一
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "nvidia-bert",
            "nvidia-fixed-k",
            "nvidia-fixed-mn",
        ],
    )
    
    # 添加命令行参数 --dtype，指定数据类型，可以从预定义的 dtype_lookup 中选择，默认为 "fp16"
    parser.add_argument(
        "--dtype",
        type=str,
        choices=dtype_lookup.keys(),
        default="fp16",
    )
    
    # 添加命令行参数 --backend，指定后端，可以是 "cutlass" 或 "cusparselt"，默认为 "cusparselt"
    parser.add_argument("-contiguous", action="store_true")  # 添加可选参数 -contiguous，标志是否连续存储
    parser.add_argument("-e2e", action="store_true")  # 添加可选参数 -e2e，标志是否进行端到端测试
    parser.add_argument("-save", action="store_true")  # 添加可选参数 -save，标志是否保存结果
    
    # 解析命令行参数
    args = parser.parse_args()

    # 根据参数设置 eval_fn 函数，如果 -e2e 被指定，则为 test_linear，否则为 test_tensor
    if args.e2e:
        eval_fn = test_linear
    else:
        eval_fn = test_tensor

    # 打印开始的基准测试信息，包括 mode 和 dtype 的值
    print(f"Started benchmark: {args.mode} | dtype: {args.dtype}")
    
    # 根据指定的 dtype 获取对应的 torch 数据类型
    dtype = dtype_lookup[args.dtype]

    # 根据不同的 mode 运行不同的基准测试
    if args.mode == "nvidia-bert":
        # 针对 nvidia-bert 模式，定义不同的 bert_shapes，对每个形状运行 eval_fn
        bert_shapes = [
            (3072, 1024, 16384),
            (4096, 1024, 16384),
            (1024, 1024, 16384),
            (1024, 4096, 16384),
        ]
        results = (
            eval_fn(m, k, n, dtype, args.contiguous, args.backend)
            for (m, k, n) in tqdm(bert_shapes)  # 使用 tqdm 迭代 bert_shapes，并显示进度条
        )

    elif args.mode == "nvidia-fixed-k":
        # 针对 nvidia-fixed-k 模式，定义不同的 mn_vals，对每个 mn 值运行 eval_fn
        mn_vals = [
            3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240,
            11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480,
        ]
        results = (
            eval_fn(mn, 10240, mn, dtype, args.contiguous, args.backend)
            for mn in tqdm(mn_vals)  # 使用 tqdm 迭代 mn_vals，并显示进度条
        )

    elif args.mode == "nvidia-fixed-mn":
        # 针对 nvidia-fixed-mn 模式，定义不同的 k_vals，对每个 k 值运行 eval_fn
        k_vals = [
            2560, 3840, 5120, 6400, 7680, 8960, 10240,
            11520, 12800, 14080, 15360, 16640, 17920, 19200, 20480,
        ]
        results = (
            eval_fn(10240, k, 10240, dtype, args.contiguous, args.backend)
            for k in tqdm(k_vals)  # 使用 tqdm 迭代 k_vals，并显示进度条
        )

    # 将结果生成为 pandas 的 DataFrame 对象
    df = pd.DataFrame.from_records(results)
    
    # 如果指定了 -save 参数，则将 DataFrame 结果保存为 CSV 文件
    if args.save:
        save_file = f"{args.mode}_{args.dtype}_{args.backend}.csv"
        df.to_csv(save_file)
        print(f"Finished benchmark: {args.mode} saved results to {save_file}")
    
    # 打印生成的 DataFrame 对象
    print(df)
```