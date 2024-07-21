# `.\pytorch\benchmarks\sparse\dlmc\matmul_bench.py`

```
# 稀疏矩阵乘法性能基准测试

# 该基准测试用于测试稀疏矩阵乘法的性能。
# 它用于比较稀疏矩阵例程的性能，包括 `稀疏 @ 向量`、`稀疏 @ 稀疏` 和 `稀疏 @ 密集`，
# 使用不同的后端（CPU/CUDA）以及与其他框架（如 scipy）进行比较。

import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统相关的模块
import sys  # 导入系统相关的模块

from scipy.sparse import isspmatrix  # 导入 scipy 中用于检测稀疏矩阵的函数

import torch  # 导入 PyTorch 深度学习框架
import torch.utils.benchmark as benchmark_utils  # 导入 PyTorch 的性能基准测试工具

from .utils import load_dlmc_dataset  # 从当前目录下的 utils 模块导入 load_dlmc_dataset 函数


def scipy_matmul(mat1, mat2):
    # 如果 mat1 和 mat2 都是稀疏矩阵，则执行稀疏矩阵乘法并转换为 COO 格式返回
    if isspmatrix(mat1) and isspmatrix(mat2):
        return mat1.dot(mat2).tocoo()
    # 否则执行普通的稠密矩阵乘法
    return mat1.dot(mat2)


def matmul_backward(a_dense, b_dense, grad_output):
    # 使用 a_dense 和 b_dense 进行矩阵乘法，计算梯度传播到 grad_output
    r1 = a_dense.matmul(b_dense)
    r1.backward(grad_output)


def sparse_matmul_backward(a, b, grad_output):
    # 使用稀疏张量 a 和 b 进行稀疏矩阵乘法，计算梯度传播到 grad_output
    c = torch.sparse.mm(a, b)
    c.backward(grad_output)


OPS_MAP = {
    "sparse@sparse": "torch.sparse.mm",
    "sparse@dense": "torch.matmul",
    "sparse@vector": "torch.matmul",
}


# 从用户输入中获取参数，使用 `argparse` 模块
def parse_args():
    parser = argparse.ArgumentParser(description="matmul benchmark")  # 创建参数解析器
    parser.add_argument("--path", type=str, help="DLMC dataset path")  # 添加路径参数
    parser.add_argument("--dataset", type=str, default="magnitude_pruning")  # 添加数据集参数，默认为 magnitude_pruning
    parser.add_argument("--hidden-size", "--hidden_size", default=2048, type=int)  # 添加隐藏层大小参数，默认为 2048
    parser.add_argument("--backward-test", "--backward_test", action="store_true")  # 添加是否进行反向测试的标志
    parser.add_argument(
        "--operation",
        type=str,
        help="|".join(OPS_MAP.keys()),  # 操作类型参数，可选值为 OPS_MAP 的键列表
        default=next(iter(OPS_MAP)),  # 默认选择 OPS_MAP 中的第一个键
    )
    parser.add_argument("--with-cuda", "--with_cuda", action="store_true")  # 添加是否使用 CUDA 的标志
    parser.add_argument(
        "--timer-min-run-time", "--timer_min_run_time", default=1, type=float  # 计时器最小运行时间
    )
    return parser


def get_tasks(op, backward_test, device):
    # 定义一个函数filter_ops，根据给定的operation参数返回不同的操作列表
    def filter_ops(operation):
        # 如果backward_test为真，则设置test_name为device + ":matmul-backward"
        if backward_test:
            test_name = device + ":matmul-backward"
            # 返回两个元组组成的列表，每个元组表示一种操作的描述
            return [
                (
                    test_name,
                    device,
                    "torch:" + operation.replace("sparse", "dense"),
                    "matmul_backward(dx, dy, grad_output)",
                ),
                (
                    test_name,
                    device,
                    "torch:" + operation,
                    "sparse_matmul_backward(x, y, sparse_grad_output)",
                ),
            ]
        else:
            # 如果backward_test为假，则设置test_name为device + ":matmul-forward"
            test_name = device + ":matmul-forward"
            # 使用filter函数过滤None值，返回一个包含有效元组的列表
            return list(
                filter(
                    None,
                    [
                        (
                            test_name,
                            device,
                            "torch:" + operation.replace("sparse", "dense"),
                            f"{OPS_MAP[operation]}(dx, dy)",
                        ),
                        (
                            test_name,
                            device,
                            "torch:" + operation,
                            f"{OPS_MAP[operation]}(x, y)",
                        ),
                        (
                            test_name,
                            device,
                            "scipy:" + operation,
                            "scipy_matmul(sx, sy)",
                        )
                        if device == "cpu"  # 如果device为cpu，则包含scipy的操作
                        else None,  # 否则返回None
                    ],
                )
            )

    # 创建一个字典all_operations，包含不同操作的测试数据
    all_operations = {
        "sparse@sparse": filter_ops("sparse@sparse"),  # 使用filter_ops函数处理sparse@sparse操作
        "sparse@dense": filter_ops("sparse@dense"),    # 使用filter_ops函数处理sparse@dense操作
        "sparse@vector": filter_ops("sparse@vector"),  # 使用filter_ops函数处理sparse@vector操作
    }
    # 返回指定操作op对应的all_operations中的数据
    return all_operations[op]
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 解析命令行参数
    parser = parse_args()
    # 解析具体的命令行参数
    args = parser.parse_args()

    # 如果需要使用 CUDA 但 CUDA 不可用，则抛出运行时错误
    if args.with_cuda and not torch.cuda.is_available():
        raise RuntimeError("No CUDA available")

    # 获取数据集路径和名称
    dataset_path = args.path
    dataset_name = args.dataset
    # 拼接数据集完整路径
    dataset_path = os.path.join(dataset_path, dataset_name)
    # 根据命令行参数确定设备是 CUDA 还是 CPU
    device = "cuda" if args.with_cuda else "cpu"

    # 根据命令行参数获取任务列表
    tasks = get_tasks(args.operation, args.backward_test, device)
    # 设置重复次数
    repeats = 3
    # 创建定时器列表
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,  # 执行的语句
            globals={
                "scipy_matmul": scipy_matmul,  # 全局变量：scipy_matmul
                "matmul_backward": matmul_backward,  # 全局变量：matmul_backward
                "sparse_matmul_backward": sparse_matmul_backward,  # 全局变量：sparse_matmul_backward
                **variables,  # 其他变量
            },
            label=label,  # 标签
            sub_label=sub_label,  # 子标签
            description=f"{sparsity}",  # 描述
            env=device,  # 环境变量设备
        )
        for sparsity in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]  # 不同稀疏度的循环
        for label, device, sub_label, stmt in tasks  # 任务的循环
        for variables in load_dlmc_dataset(  # 加载数据集的循环
            dataset_path,
            args.operation,
            args.hidden_size,
            sparsity,
            device,
            args.backward_test,
        )
    ]

    measurements = []  # 测量结果列表

    # 对于每个定时器，重复多次测量
    for i, timer in enumerate(timers * repeats):
        # 执行测量，并记录元数据（设备信息）
        m = timer.blocked_autorange(min_run_time=args.timer_min_run_time)
        m.metadata = {"device": "cuda" if m.task_spec.env.find("cuda") >= 0 else "cpu"}
        measurements.append(m)  # 将测量结果添加到列表中
        # 打印进度条
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()  # 打印换行

    # 进行性能比较
    comparison = benchmark_utils.Compare(measurements)

    # 打印结果标题
    print("== Results " + "=" * 80 + "\n" + "/" * 95 + "\n")
    comparison.print()  # 输出性能比较结果
```