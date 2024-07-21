# `.\pytorch\benchmarks\sparse\spmm.py`

```
import argparse  # 导入命令行参数解析模块
import sys  # 导入系统相关模块

from utils import Event, gen_sparse_coo, gen_sparse_coo_and_csr, gen_sparse_csr  # 导入自定义模块和函数

import torch  # 导入PyTorch深度学习库


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpMM")  # 创建参数解析器对象，描述为"SpMM"

    parser.add_argument("--format", default="csr", type=str)  # 添加命令行参数--format，默认值为"csr"，类型为字符串
    parser.add_argument("--m", default="1000", type=int)  # 添加命令行参数--m，默认值为1000，类型为整数
    parser.add_argument("--n", default="1000", type=int)  # 添加命令行参数--n，默认值为1000，类型为整数
    parser.add_argument("--k", default="1000", type=int)  # 添加命令行参数--k，默认值为1000，类型为整数
    parser.add_argument("--nnz-ratio", "--nnz_ratio", default="0.1", type=float)  # 添加命令行参数--nnz-ratio（或--nnz_ratio），默认值为0.1，类型为浮点数
    parser.add_argument("--outfile", default="stdout", type=str)  # 添加命令行参数--outfile，默认值为"stdout"，类型为字符串
    parser.add_argument("--test-count", "--test_count", default="10", type=int)  # 添加命令行参数--test-count（或--test_count），默认值为10，类型为整数

    args = parser.parse_args()  # 解析命令行参数并存储到args对象中

    if args.outfile == "stdout":  # 如果outfile参数值为"stdout"
        outfile = sys.stdout  # 将输出重定向到标准输出
    elif args.outfile == "stderr":  # 如果outfile参数值为"stderr"
        outfile = sys.stderr  # 将输出重定向到标准错误输出
    else:  # 否则
        outfile = open(args.outfile, "a")  # 打开指定文件以追加方式写入，并将文件对象赋给outfile变量

    test_count = args.test_count  # 将命令行参数中的test_count值赋给test_count变量
    m = args.m  # 将命令行参数中的m值赋给m变量
    n = args.n  # 将命令行参数中的n值赋给n变量
    k = args.k  # 将命令行参数中的k值赋给k变量
    nnz_ratio = args.nnz_ratio  # 将命令行参数中的nnz_ratio值赋给nnz_ratio变量

    nnz = int(nnz_ratio * m * k)  # 计算稀疏矩阵的非零元素数量

    if args.format == "csr":  # 如果命令行参数中的format值为"csr"
        time = test_sparse_csr(m, n, k, nnz, test_count)  # 调用test_sparse_csr函数进行测试
    elif args.format == "coo":  # 如果命令行参数中的format值为"coo"
        time = test_sparse_coo(m, n, k, nnz, test_count)  # 调用test_sparse_coo函数进行测试
    elif args.format == "both":  # 如果命令行参数中的format值为"both"
        time_coo, time_csr = test_sparse_coo_and_csr(m, n, k, nnz, test_count)  # 调用test_sparse_coo_and_csr函数进行测试
    # 如果命令行参数中的格式为 "both"，则输出 COO 格式和 CSR 格式的性能数据到指定文件
    if args.format == "both":
        # 输出 COO 格式性能数据到文件，包括非零元素比率、矩阵维度和计算时间
        print(
            "format=coo",
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " n=",
            n,
            " k=",
            k,
            " time=",
            time_coo,
            file=outfile,
        )
        # 输出 CSR 格式性能数据到文件，包括非零元素比率、矩阵维度和计算时间
        print(
            "format=csr",
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " n=",
            n,
            " k=",
            k,
            " time=",
            time_csr,
            file=outfile,
        )
    else:
        # 否则，根据命令行参数中指定的格式输出相应格式的性能数据到指定文件
        print(
            "format=",
            args.format,
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " n=",
            n,
            " k=",
            k,
            " time=",
            time,
            file=outfile,
        )
```