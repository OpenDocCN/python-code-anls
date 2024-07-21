# `.\pytorch\torch\utils\benchmark\examples\sparse\compare.py`

```
# mypy: allow-untyped-defs
"""Example of Timer and Compare APIs:

$ python -m examples.sparse.compare
"""

import pickle   # 导入pickle模块，用于序列化和反序列化Python对象
import sys      # 导入sys模块，用于访问系统相关的变量和函数
import time     # 导入time模块，用于时间相关操作

import torch    # 导入PyTorch库
import torch.utils.benchmark as benchmark_utils   # 导入PyTorch的benchmark工具

class FauxTorch:
    """Emulate different versions of pytorch.

    In normal circumstances this would be done with multiple processes
    writing serialized measurements, but this simplifies that model to
    make the example clearer.
    """
    def __init__(self, real_torch, extra_ns_per_element):
        self._real_torch = real_torch   # 初始化真实的torch对象
        self._extra_ns_per_element = extra_ns_per_element   # 初始化每个元素的额外纳秒开销

    @property
    def sparse(self):
        return self.Sparse(self._real_torch, self._extra_ns_per_element)

    class Sparse:
        def __init__(self, real_torch, extra_ns_per_element):
            self._real_torch = real_torch   # 初始化真实的torch对象
            self._extra_ns_per_element = extra_ns_per_element   # 初始化每个元素的额外纳秒开销

        def extra_overhead(self, result):
            # time.sleep has a ~65 us overhead, so only fake a
            # per-element overhead if numel is large enough.
            size = sum(result.size())   # 计算结果张量的元素总数
            if size > 5000:   # 如果元素总数大于5000
                time.sleep(size * self._extra_ns_per_element * 1e-9)   # 根据元素数目和每个元素的额外纳秒开销进行休眠
            return result

        def mm(self, *args, **kwargs):
            return self.extra_overhead(self._real_torch.sparse.mm(*args, **kwargs))   # 执行稀疏矩阵乘法并添加额外开销

def generate_coo_data(size, sparse_dim, nnz, dtype, device):
    """
    Parameters
    ----------
    size : tuple
    sparse_dim : int
    nnz : int
    dtype : torch.dtype
    device : str
    Returns
    -------
    indices : torch.tensor
    values : torch.tensor
    """
    if dtype is None:
        dtype = 'float32'

    indices = torch.rand(sparse_dim, nnz, device=device)   # 生成稀疏张量的随机索引
    indices.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(indices))   # 对索引进行尺寸调整和乘法操作
    indices = indices.to(torch.long)   # 将索引转换为长整型
    values = torch.rand([nnz, ], dtype=dtype, device=device)   # 生成稀疏张量的随机值
    return indices, values

def gen_sparse(size, density, dtype, device='cpu'):
    sparse_dim = len(size)   # 获取稀疏张量的维度
    nnz = int(size[0] * size[1] * density)   # 计算稀疏张量的非零元素数量
    indices, values = generate_coo_data(size, sparse_dim, nnz, dtype, device)   # 生成稀疏张量的索引和值
    return torch.sparse_coo_tensor(indices, values, size, dtype=dtype, device=device)   # 返回稀疏COO张量对象

def main():
    tasks = [
        ("matmul", "x @ y", "torch.sparse.mm(x, y)"),   # 定义任务列表，每个任务包含描述和相应的表达式
        ("matmul", "x @ y + 0", "torch.sparse.mm(x, y) + zero"),
    ]

    serialized_results = []   # 初始化序列化结果列表
    repeats = 2   # 定义重复次数
    # 创建一个 timers 列表，其中每个元素是 benchmark_utils.Timer 对象，用于执行性能基准测试
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,  # 待执行的语句
            globals={  # 执行语句时使用的全局变量字典
                "torch": torch if branch == "master" else FauxTorch(torch, overhead_ns),  # 根据分支选择真实或模拟的 torch 对象
                "x": gen_sparse(size=size, density=density, dtype=torch.float32),  # 生成稀疏张量 x
                "y": torch.rand(size, dtype=torch.float32),  # 生成随机张量 y
                "zero": torch.zeros(()),  # 生成一个标量零张量 zero
            },
            label=label,  # 性能测试的标签
            sub_label=sub_label,  # 性能测试的子标签
            description=f"size: {size}",  # 描述性能测试的描述字符串
            env=branch,  # 执行环境的分支名称
            num_threads=num_threads,  # 使用的线程数
        )
        for branch, overhead_ns in [("master", None), ("my_branch", 1), ("severe_regression", 10)]  # 遍历不同分支及其开销
        for label, sub_label, stmt in tasks  # 遍历不同任务及其标签、子标签、语句
        for density in [0.05, 0.1]  # 遍历不同的稀疏度
        for size in [(8, 8), (32, 32), (64, 64), (128, 128)]  # 遍历不同的大小
        for num_threads in [1, 4]  # 遍历不同的线程数
    ]

    # 对 timers 列表中的每个 timer 进行重复执行并将结果序列化后存入 serialized_results 列表
    for i, timer in enumerate(timers * repeats):
        serialized_results.append(pickle.dumps(
            timer.blocked_autorange(min_run_time=0.05)  # 执行计时器的性能测试，并进行序列化
        ))
        # 打印当前进度条信息，显示已完成的测试数量和总测试数量
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()

    # 创建一个 benchmark_utils.Compare 对象，用于比较已序列化的测试结果
    comparison = benchmark_utils.Compare([
        pickle.loads(i) for i in serialized_results  # 反序列化每个序列化结果并组成比较对象的列表
    ])

    # 打印未格式化的比较结果的分隔线和标题
    print("== Unformatted " + "=" * 80 + "\n" + "/" * 95 + "\n")
    comparison.print()

    # 对比较结果进行格式化处理，调整有效数字位数并添加颜色标记
    print("== Formatted " + "=" * 80 + "\n" + "/" * 93 + "\n")
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()
# 如果当前脚本作为主程序执行（而不是被导入），则执行 main() 函数
if __name__ == "__main__":
    main()
```