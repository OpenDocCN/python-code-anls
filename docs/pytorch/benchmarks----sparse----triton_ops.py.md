# `.\pytorch\benchmarks\sparse\triton_ops.py`

```
# 导入 torch 库，用于创建和操作张量
import torch


# 创建一个被分块的稀疏张量
def create_blocked_tensor(B, M, N, blocksize, sparsity, dtype, device):
    # 断言确保稀疏度 sparsity 在 0 到 1 之间
    assert (
        sparsity <= 1.0 and sparsity >= 0.0
    ), "sparsity should be a value between 0 and 1"
    # 断言确保 M 和 N 可以被块大小 blocksize 整除
    assert M % blocksize[0] == 0
    assert N % blocksize[1] == 0
    # 计算张量的形状，如果 B 为 0，则在结果元组中省略 B 的维度
    shape = (B, M // blocksize[0], N // blocksize[1])[int(B == 0) :]
    # 创建一个稀疏张量 A，元素值为 0 或 1，1 的概率为 1 - sparsity
    A = torch.bernoulli(torch.full(shape, 1 - sparsity, dtype=dtype, device=device))
    # 预期的非零元素个数
    expected_nnz = int((1 - sparsity) * M * N / (blocksize[0] * blocksize[1]))
    # 找到所有非零元素的索引
    nonzero_indices = A.flatten().nonzero()
    # 实际的非零元素个数
    actual_nnz = nonzero_indices.shape[0]
    # 如果实际非零元素个数超过预期，则随机清除多余的非零元素
    if actual_nnz > expected_nnz:
        selected_nonzeros = torch.randperm(actual_nnz)[: actual_nnz - expected_nnz]
        A.flatten()[nonzero_indices[selected_nonzeros]] = 0
    # 如果实际非零元素个数少于预期，则随机添加足够数量的非零元素
    elif actual_nnz < expected_nnz:
        zero_indices = (A == 0).flatten().nonzero()
        selected_zeros = torch.randperm(zero_indices.shape[0])[
            : expected_nnz - actual_nnz
        ]
        A.flatten()[zero_indices[selected_zeros]] = 1
    # 将 A 沿着最后两个维度重复块大小次数，形成分块的稀疏张量
    A = torch.repeat_interleave(A, blocksize[0], dim=-2)
    A = torch.repeat_interleave(A, blocksize[1], dim=-1)
    # 返回创建的稀疏张量 A
    return A


# 测试函数，用于评估测试函数的性能
def _test_worker(test_func):
    # 导入 triton 库
    import triton
    # 运行测试函数，并返回运行时间统计信息
    ms, ms_min, ms_max = triton.testing.do_bench(
        test_func, warmup=500, rep=100, fast_flush=False
    )
    # 计算每秒的浮点运算次数 TFLOPS
    tflops = 2 * m * k * n * 1e-12 / (ms * 1e-3)
    # 返回运行时间和 TFLOPS
    return ms, tflops


# 测试稠密矩阵乘法函数
def test_dense_dense_mm(x, y, **meta):
    # 定义测试函数，执行稠密矩阵乘法
    def test_func(x=x.to_dense(), y=y):
        return torch.matmul(x, y)
    # 调用测试函数的性能评估并返回结果
    return _test_worker(test_func)


# 测试 torch.matmul 函数的性能
def test_torch_matmul(x, y, **meta):
    # 定义测试函数，执行 torch.matmul
    def test_func(x=x, y=y):
        return torch.matmul(x, y)
    # 调用测试函数的性能评估并返回结果
    return _test_worker(test_func)


# 测试稀疏块矩阵乘法函数
def test_bsr_dense_mm(x, y, **meta):
    # 导入 torch 稀疏模块中的 bsr_dense_mm 函数
    from torch.sparse._triton_ops import bsr_dense_mm
    # 定义测试函数，执行 bsr_dense_mm 函数
    def test_func(x=x, y=y):
        return bsr_dense_mm(
            x, y, meta=dict(GROUP_SIZE_ROW=4, num_stages=1, num_warps=4)
        )
    # 调用测试函数的性能评估并返回结果
    return _test_worker(test_func)


# 测试带元数据的稀疏块矩阵乘法函数
def test_bsr_dense_mm_with_meta(x, y, **meta):
    # 导入 torch 稀疏模块中的 bsr_dense_mm 函数
    from torch.sparse._triton_ops import bsr_dense_mm
    # 定义测试函数，执行带元数据的 bsr_dense_mm 函数
    def test_func(x=x, y=y, meta=meta):
        return bsr_dense_mm(x, y, meta=meta)
    # 调用测试函数的性能评估并返回结果
    return _test_worker(test_func)


# 测试稀疏块散列矩阵乘法函数
def test_bsr_scatter_mm2(x, y, **meta):
    # 导入 torch 稀疏模块中的 bsr_scatter_mm 和 bsr_scatter_mm_indices_data 函数
    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data
    # 使用 bsr_scatter_mm_indices_data 函数生成散列矩阵的索引数据
    indices_data = bsr_scatter_mm_indices_data(
        x, y, indices_format="scatter_mm", **meta
    )
    # 定义测试函数，执行 bsr_scatter_mm 函数
    def test_func(x=x, y=y):
        return bsr_scatter_mm(x, y, indices_data=indices_data)
    # 调用测试函数的性能评估并返回结果
    return _test_worker(test_func)


# 测试压缩的稀疏块散列矩阵乘法函数
def test_bsr_scatter_mm6(x, y, **meta):
    # 导入 torch 稀疏模块中的 bsr_scatter_mm 和 bsr_scatter_mm_indices_data 函数
    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data
    # 使用 bsr_scatter_mm_indices_data 函数生成压缩的散列矩阵的索引数据
    indices_data = bsr_scatter_mm_indices_data(
        x, y, indices_format="bsr_strided_mm_compressed", **meta
    )
    # 定义测试函数，执行 bsr_scatter_mm 函数
    def test_func(x=x, y=y):
        return bsr_scatter_mm(x, y, indices_data=indices_data)
    # 调用测试函数的性能评估并返回结果
    return _test_worker(test_func)


# 测试稀疏块散列矩阵乘法函数
def test_bsr_scatter_mm(x, y, **meta):
    # 待实现的测试函数
    pass
    # 导入来自 torch.sparse._triton_ops 的 bsr_scatter_mm 和 bsr_scatter_mm_indices_data 函数
    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data
    
    # 定义名为 test_func 的函数，参数为 x 和 y，默认值分别为之前定义的 x 和 y
    def test_func(x=x, y=y):
        # 调用 bsr_scatter_mm_indices_data 函数，传入 x, y 和额外参数 indices_format="bsr_strided_mm_compressed"，返回稀疏矩阵的索引数据
        indices_data = bsr_scatter_mm_indices_data(
            x, y, indices_format="bsr_strided_mm_compressed", **meta
        )
        # 调用 bsr_scatter_mm 函数，传入 x, y 和索引数据 indices_data，返回稀疏矩阵的结果
        return bsr_scatter_mm(x, y, indices_data=indices_data)
    
    # 返回使用 _test_worker 函数对 test_func 的结果进行处理的结果
    return _test_worker(test_func)
# 导入PyTorch的函数库torch.nn.functional，用于实现神经网络的各种功能
import torch.nn.functional as F

# 定义一个函数test_linear，用于测试线性函数
def test_linear(x, y, **meta):
    # 在函数内部导入torch.nn.functional，确保可以使用其功能

    # 定义内部函数test_func，接受参数x和将y在最后两个维度进行转置后的结果
    def test_func(x=x, y=y.transpose(-2, -1)):
        # 调用torch.nn.functional中的linear函数，对输入的y和x进行线性变换
        return F.linear(y, x)

    # 调用测试工作函数_test_worker，传入内部函数test_func的引用并返回结果
    return _test_worker(test_func)

# 当作为主程序运行时执行以下代码
if __name__ == "__main__":
    # 导入命令行参数解析模块argparse，用于处理命令行输入
    import argparse
    # 导入atexit模块，用于注册程序退出时的清理函数
    import atexit
    # 导入itertools模块，提供了用于操作迭代器的函数
    import itertools
    # 导入sys模块，提供了与Python解释器交互的函数
    import sys

    # 导入triton模块，用于特定领域的高性能计算
    import triton

    # 从torch.testing模块中导入make_tensor函数，用于创建测试用的张量
    from torch.testing import make_tensor

    # 设置随机种子为0，保证随机数的可重复性
    torch.manual_seed(0)

    # 定义一个函数integer_list，接受一个字符串a，返回其逗号分隔的整数列表
    def integer_list(a):
        return list(map(int, a.split(",")))

    # 定义一个函数float_list，接受一个字符串a，返回其逗号分隔的浮点数列表
    def float_list(a):
        return list(map(float, a.split(",")))

    # 定义一个函数integer_or_float_list，接受一个字符串a，根据内容返回整数或浮点数列表
    def integer_or_float_list(a):
        lst = []
        for n in a.split(","):
            if n.count(":") == 1:
                start, end = map(int, n.split(":"))
                lst.extend(range(start, end))
            elif n.count(":") == 2:
                start, end, step = map(int, n.split(":"))
                lst.extend(range(start, end, step))
            elif "." in n:
                lst.append(float(n))
            else:
                lst.append(int(n))
        return lst

    # 创建一个argparse.ArgumentParser对象，用于解析命令行参数并生成帮助信息
    parser = argparse.ArgumentParser(description="SpTritonOps")

    # 添加命令行参数--ops，设置默认值和类型为字符串，表示要执行的操作列表
    parser.add_argument(
        "--ops",
        default="dense_dense_mm,bsr_dense_mm,bsr_scatter_mm6",
        type=str,
    )
    # 添加命令行参数--b，设置默认值为0，类型为整数，表示某个参数b的值
    parser.add_argument("--b", default="0", type=int)

    # 添加命令行参数--m，设置默认值为"1024"，类型为整数列表，表示尺寸参数m的值
    parser.add_argument("--m", default="1024", type=integer_list)
    # 添加命令行参数--k，设置默认值为None，类型为整数列表，表示尺寸参数k的值
    parser.add_argument("--k", default=None, type=integer_list)
    # 添加命令行参数--n，设置默认值为None，类型为整数列表，表示尺寸参数n的值
    parser.add_argument("--n", default=None, type=integer_list)
    # 添加命令行参数--bm，设置默认值为"16"，类型为整数列表，表示块尺寸参数bm的值
    parser.add_argument("--bm", default="16", type=integer_list)
    # 添加命令行参数--bk，设置默认值为None，类型为整数列表，表示块尺寸参数bk的值
    parser.add_argument("--bk", default=None, type=integer_list)
    # 添加命令行参数--tile_m，设置默认值为None，类型为整数列表，表示矩阵分块参数tile_m的值
    parser.add_argument("--tile_m", default=None, type=integer_list)
    # 添加命令行参数--tile_n，设置默认值为None，类型为整数列表，表示矩阵分块参数tile_n的值
    parser.add_argument("--tile_n", default=None, type=integer_list)
    # 添加命令行参数--split_n，设置默认值为None，类型为整数列表，表示矩阵分块参数split_n的值
    parser.add_argument("--split_n", default=None, type=integer_list)
    # 添加命令行参数--group_size，设置默认值为None，类型为整数列表，表示分组大小参数group_size的值
    parser.add_argument("--group_size", default=None, type=integer_list)
    # 添加命令行参数--num_warps，设置默认值为None，类型为整数列表，表示线程束数参数num_warps的值
    parser.add_argument("--num_warps", default=None, type=integer_list)
    # 添加命令行参数--num_stages，设置默认值为None，类型为整数列表，表示阶段数参数num_stages的值
    parser.add_argument("--num_stages", default=None, type=integer_list)
    # 添加命令行参数--sparsity，设置默认值为"0.5"，类型为整数或浮点数列表，表示稀疏度参数sparsity的值
    parser.add_argument("--sparsity", default="0.5", type=integer_or_float_list)
    # 添加命令行参数--dtype，设置默认值为"float16"，类型为字符串，表示数据类型参数dtype的值
    parser.add_argument("--dtype", default="float16", type=str)
    # 添加命令行参数--device，设置默认值为"cuda"，类型为字符串，表示设备类型参数device的值
    parser.add_argument("--device", default="cuda", type=str)
    # 添加命令行参数--repeat，设置默认值为1，类型为整数，表示重复次数参数repeat的值
    parser.add_argument("--repeat", default="1", type=int)
    # 添加命令行参数--outfile，设置默认值为"stdout"，类型为字符串，表示输出文件路径参数outfile的值
    parser.add_argument("--outfile", default="stdout", type=str)
    # 添加命令行参数--star，设置默认值为False，类型为布尔值，表示是否启用星形模式参数star的值
    parser.add_argument("--star", default=False, action="store_true")

    # 解析命令行参数，并将其存储在args对象中
    args = parser.parse_args()

    # 根据参数--outfile的值确定输出文件的目标，可以是标准输出、标准错误或指定文件
    if args.outfile == "stdout":
        outfile = sys.stdout
    elif args.outfile == "stderr":
        outfile = sys.stderr
    else:
        outfile = open(args.outfile, "a")

    # 将参数--ops的值按逗号分割成列表，存储在ops变量中
    ops = args.ops.split(",")

    # 将参数--b的值赋给变量b
    b = args.b

    # 将参数--m的值赋给变量m_list，如果为None则默认为[1024]
    m_list = args.m or [1024]
    # 将参数--n的值赋给变量n_list，如果为None则默认为[None]
    n_list = args.n or [None]
    # 将参数--k的值赋给变量k_list，如果为None则默认为[None]
    k_list = args.k or [None]
    # 将参数--bm的值赋给变量bm_list，如果为None则默认为[16]
    bm_list = args.bm or [16]
    # 将参数--bk的值赋给变量bk_list，如果为None则默认为[None]
    bk_list = args.bk or [None]
    # 将参数--split_n的值赋
    # 如果 args.num_warps 为 None，则使用 [None]，否则使用 args.num_warps 自身的值
    num_warps_list = args.num_warps or [None]
    # 如果 args.num_stages 为 None，则使用 [None]，否则使用 args.num_stages 自身的值
    num_stages_list = args.num_stages or [None]
    # 如果 args.sparsity 为 None，则使用 [0.5]，否则使用 args.sparsity 自身的值
    sparsity_list = args.sparsity or [0.5]
    # 从 torch 模块中获取 args.dtype 对应的数据类型，并赋值给 dtype 变量
    dtype = getattr(torch, args.dtype)
    # 如果参数 args.star 大于 0，则执行以下操作
    if args.star > 0:
        # 导入 torch.sparse._triton_ops 模块
        import torch.sparse._triton_ops

        # 断言以下集合中的长度都为 1
        assert {len(m_list), len(n_list), len(k_list), len(bm_list), len(bk_list)} == {1}
        
        # 从列表中获取 m, n, k, bm, bk 的值，如果 n_list 或 bk_list 中的值为 None，则使用默认值
        m = m_list[0]
        n = n_list[0] or m
        k = k_list[0] or m
        bm = bm_list[0]
        bk = bk_list[0] or bm
        
        # 根据不同的操作名字在 triton_ops 模块中调用相应的函数，并获取返回的 meta 数据
        if "bsr_scatter_mm6" in ops:
            meta = torch.sparse._triton_ops.scatter_mm_meta(m, k, n, bm, bk)
        elif "bsr_dense_mm_with_meta" in ops:
            meta = torch.sparse._triton_ops.bsr_dense_mm_meta(m, k, n, bm, bk)
        else:
            # 如果操作名字不在支持的列表中，则抛出未实现错误
            raise NotImplementedError(f"--star not implemented for operations in {ops}")
        
        # 如果操作名字包含 "bsr_scatter_mm6"
        if "bsr_scatter_mm6" in ops:
            # 如果 split_n_list 的第一个元素为 None，则根据 meta 数据设置默认值
            if split_n_list[0] is None:
                split_n_list = [
                    meta["SPLIT_N"] // 2,
                    meta["SPLIT_N"],
                    meta["SPLIT_N"] * 2,
                ][int(meta["SPLIT_N"] == 1) :]
            elif split_n_list[0] == 0:
                split_n_list = [meta["SPLIT_N"]]
            
            # 如果 tile_m_list 的第一个元素为 None，则根据 meta 数据设置默认值
            if tile_m_list[0] is None:
                tile_m_list = [meta["TILE_M"] // 2, meta["TILE_M"], meta["TILE_M"] * 2][
                    int(meta["TILE_M"] == 16) :
                ]
            elif tile_m_list[0] == 0:
                tile_m_list = [meta["TILE_M"]]
            
            # 如果 tile_n_list 的第一个元素为 None，则根据 meta 数据设置默认值
            if tile_n_list[0] is None:
                tile_n_list = [meta["TILE_N"] // 2, meta["TILE_N"], meta["TILE_N"] * 2][
                    int(meta["TILE_N"] == 16) :
                ]
            elif tile_n_list[0] == 0:
                tile_n_list = [meta["TILE_N"]]
            
            # 如果 group_size_list 的第一个元素为 None，则根据 meta 数据设置默认值
            if group_size_list[0] is None:
                group_size_list = [
                    meta["GROUP_SIZE"] - 1,
                    meta["GROUP_SIZE"],
                    meta["GROUP_SIZE"] + 1,
                ][int(meta["GROUP_SIZE"] == 1) :]
            elif group_size_list[0] == 0:
                group_size_list = [meta["GROUP_SIZE"]]
        
        # 如果操作名字包含 "bsr_dense_mm_with_meta"
        if "bsr_dense_mm_with_meta" in ops:
            # 如果 group_size_list 的第一个元素为 None，则根据 meta 数据设置默认值
            if group_size_list[0] is None:
                group_size_list = [
                    meta["GROUP_SIZE_ROW"] - 1,
                    meta["GROUP_SIZE_ROW"],
                    meta["GROUP_SIZE_ROW"] + 1,
                ][int(meta["GROUP_SIZE_ROW"] == 1) :]
            elif group_size_list[0] == 0:
                group_size_list = [meta["GROUP_SIZE_ROW"]]
        
        # 如果 num_warps_list 的第一个元素为 None，则根据 meta 数据设置默认值
        if num_warps_list[0] is None:
            num_warps_list = [
                meta["num_warps"] // 2,
                meta["num_warps"],
                meta["num_warps"] * 2,
            ][int(meta["num_warps"] == 1) :]
        elif num_warps_list[0] == 0:
            num_warps_list = [meta["num_warps"]]
        
        # 如果 num_stages_list 的第一个元素为 None，则根据 meta 数据设置默认值
        if num_stages_list[0] is None:
            num_stages_list = [
                meta["num_stages"] - 1,
                meta["num_stages"],
                meta["num_stages"] + 1,
            ][int(meta["num_stages"] == 1) :]
        elif num_stages_list[0] == 0:
            num_stages_list = [meta["num_stages"]]
    # 从参数中获取设备信息
    device = args.device
    # 创建一个空集合，用于存储 dense_dense_mm_sizes
    dense_dense_mm_sizes = set()
    # 初始化目标性能为空
    target_performance = None
    # 定义性能相对误差容差
    performance_rtol = 1e-2

    # 初始化一个空列表，用于存储最佳消息
    best_messages = []

    # 注册一个退出时的回调函数，用于显示最佳消息
    @atexit.register
    def show_best_messages(best_messages=best_messages):
        print("TOP 10:")
        # 打印最后10条最佳消息
        for m in best_messages[-10:]:
            print(m)
        # 刷新标准输出缓冲区
        sys.stdout.flush()

    # 使用 itertools.product 生成器生成所有可能的参数组合
    for m, k, n, bm, bk, sparsity in itertools.product(
        m_list, k_list, n_list, bm_list, bk_list, sparsity_list
```