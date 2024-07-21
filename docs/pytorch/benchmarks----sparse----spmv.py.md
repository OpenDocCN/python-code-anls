# `.\pytorch\benchmarks\sparse\spmv.py`

```
import argparse  # 导入处理命令行参数的模块
import sys  # 导入系统相关的模块

import torch  # 导入PyTorch库，用于科学计算

from .utils import Event, gen_sparse_coo, gen_sparse_coo_and_csr, gen_sparse_csr  # 导入自定义模块中的事件和稀疏矩阵生成函数


def test_sparse_csr(m, nnz, test_count):
    start_timer = Event(enable_timing=True)  # 创建一个启用计时的事件对象
    stop_timer = Event(enable_timing=True)  # 创建另一个启用计时的事件对象

    csr = gen_sparse_csr((m, m), nnz)  # 生成稀疏的压缩行格式（CSR）矩阵
    vector = torch.randn(m, dtype=torch.double)  # 生成一个双精度随机张量

    times = []
    for _ in range(test_count):
        start_timer.record()  # 记录计时开始
        csr.matmul(vector)  # 对CSR矩阵进行向量乘法
        stop_timer.record()  # 记录计时结束
        times.append(start_timer.elapsed_time(stop_timer))  # 计算并记录时间差

    return sum(times) / len(times)  # 返回平均时间


def test_sparse_coo(m, nnz, test_count):
    start_timer = Event(enable_timing=True)  # 创建一个启用计时的事件对象
    stop_timer = Event(enable_timing=True)  # 创建另一个启用计时的事件对象

    coo = gen_sparse_coo((m, m), nnz)  # 生成稀疏的坐标格式（COO）矩阵
    vector = torch.randn(m, dtype=torch.double)  # 生成一个双精度随机张量

    times = []
    for _ in range(test_count):
        start_timer.record()  # 记录计时开始
        coo.matmul(vector)  # 对COO矩阵进行向量乘法
        stop_timer.record()  # 记录计时结束
        times.append(start_timer.elapsed_time(stop_timer))  # 计算并记录时间差

    return sum(times) / len(times)  # 返回平均时间


def test_sparse_coo_and_csr(m, nnz, test_count):
    start = Event(enable_timing=True)  # 创建一个启用计时的事件对象
    stop = Event(enable_timing=True)  # 创建另一个启用计时的事件对象

    coo, csr = gen_sparse_coo_and_csr((m, m), nnz)  # 生成稀疏的COO和CSR格式的矩阵
    vector = torch.randn(m, dtype=torch.double)  # 生成一个双精度随机张量

    times = []
    for _ in range(test_count):
        start.record()  # 记录计时开始
        coo.matmul(vector)  # 对COO矩阵进行向量乘法
        stop.record()  # 记录计时结束
        times.append(start.elapsed_time(stop))  # 计算并记录时间差

    coo_mean_time = sum(times) / len(times)  # 计算COO平均时间

    times = []
    for _ in range(test_count):
        start.record()  # 记录计时开始
        csr.matmul(vector)  # 对CSR矩阵进行向量乘法
        stop.record()  # 记录计时结束
        times.append(start.elapsed_time(stop))  # 计算并记录时间差

    csr_mean_time = sum(times) / len(times)  # 计算CSR平均时间

    return coo_mean_time, csr_mean_time  # 返回COO和CSR的平均时间


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpMV")  # 创建命令行参数解析器

    parser.add_argument("--format", default="csr", type=str)  # 添加--format参数，指定稀疏矩阵格式
    parser.add_argument("--m", default="1000", type=int)  # 添加--m参数，指定矩阵维度
    parser.add_argument("--nnz-ratio", "--nnz_ratio", default="0.1", type=float)  # 添加--nnz-ratio参数，指定非零元素占比
    parser.add_argument("--outfile", default="stdout", type=str)  # 添加--outfile参数，指定输出文件
    parser.add_argument("--test-count", "--test_count", default="10", type=int)  # 添加--test-count参数，指定测试次数

    args = parser.parse_args()  # 解析命令行参数

    if args.outfile == "stdout":
        outfile = sys.stdout  # 如果输出文件是stdout，则输出到标准输出流
    elif args.outfile == "stderr":
        outfile = sys.stderr  # 如果输出文件是stderr，则输出到标准错误流
    else:
        outfile = open(args.outfile, "a")  # 否则打开指定文件进行追加写入

    test_count = args.test_count  # 获取测试次数
    m = args.m  # 获取矩阵维度
    nnz_ratio = args.nnz_ratio  # 获取非零元素占比

    nnz = int(nnz_ratio * m * m)  # 计算非零元素的数量
    if args.format == "csr":
        time = test_sparse_csr(m, nnz, test_count)  # 如果指定使用CSR格式，则进行CSR格式的测试
    elif args.format == "coo":
        time = test_sparse_coo(m, nnz, test_count)  # 如果指定使用COO格式，则进行COO格式的测试
    elif args.format == "both":
        time_coo, time_csr = test_sparse_coo_and_csr(m, nnz, test_count)  # 如果同时测试COO和CSR格式，则进行同时测试

    if args.format != "both":
        print(
            "format=",
            args.format,
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " time=",
            time,
            file=outfile,
        )  # 输出测试结果到指定文件
    else:
        # 在输出文件中打印格式为COO的稀疏矩阵信息
        print(
            "format=coo",
            " nnz_ratio=",
            nnz_ratio,   # 打印非零元素比例
            " m=",
            m,           # 打印矩阵的维度m
            " time=",
            time_coo,    # 打印COO格式转换时间
            file=outfile,  # 输出到指定文件
        )
        # 在输出文件中打印格式为CSR的稀疏矩阵信息
        print(
            "format=csr",
            " nnz_ratio=",
            nnz_ratio,   # 打印非零元素比例
            " m=",
            m,           # 打印矩阵的维度m
            " time=",
            time_csr,    # 打印CSR格式转换时间
            file=outfile,  # 输出到指定文件
        )
```