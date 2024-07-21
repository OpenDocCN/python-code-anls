# `.\pytorch\torch\utils\benchmark\examples\sparse\fuzzer.py`

```
# 设置 mypy 工具允许未定义的函数
"""Timer 和 Sparse Fuzzer API 的示例：

$ python -m examples.sparse.fuzzer
"""

# 导入系统相关的库
import sys

# 导入 PyTorch 的性能测试工具模块
import torch.utils.benchmark as benchmark_utils

# 主函数入口
def main():
    # 创建一个 Fuzzer 对象 add_fuzzer，用于生成不同配置的稀疏张量和相关参数
    add_fuzzer = benchmark_utils.Fuzzer(
        parameters=[
            # 定义一组参数，每个参数包含多个被模糊化的参数对象
            [
                benchmark_utils.FuzzedParameter(
                    name=f"k{i}",  # 参数名称
                    minval=16,  # 最小值
                    maxval=16 * 1024,  # 最大值
                    distribution="loguniform",  # 分布类型
                ) for i in range(3)  # 生成三个参数对象
            ],
            benchmark_utils.FuzzedParameter(
                name="dim_parameter",  # 参数名称
                distribution={2: 0.6, 3: 0.4},  # 分布字典
            ),
            benchmark_utils.FuzzedParameter(
                name="sparse_dim",  # 参数名称
                distribution={1: 0.3, 2: 0.4, 3: 0.3},  # 分布字典
            ),
            benchmark_utils.FuzzedParameter(
                name="density",  # 参数名称
                distribution={0.1: 0.4, 0.05: 0.3, 0.01: 0.3},  # 分布字典
            ),
            benchmark_utils.FuzzedParameter(
                name="coalesced",  # 参数名称
                distribution={True: 0.7, False: 0.3},  # 分布字典
            )
        ],
        tensors=[
            # 定义稀疏张量的列表，每个张量包含多个被模糊化的稀疏张量对象
            [
                benchmark_utils.FuzzedSparseTensor(
                    name=name,  # 张量名称
                    size=tuple([f"k{i}" for i in range(3)]),  # 大小
                    min_elements=64 * 1024,  # 最小元素数
                    max_elements=128 * 1024,  # 最大元素数
                    sparse_dim="sparse_dim",  # 稀疏维度参数名称
                    density="density",  # 密度参数名称
                    dim_parameter="dim_parameter",  # 维度参数名称
                    coalesced="coalesced"  # 合并标志参数名称
                ) for name in ("x", "y")  # 生成稀疏张量对象，名称为"x"和"y"
            ],
        ],
        seed=0,  # 设置随机种子
    )

    n = 100  # 循环迭代次数
    measurements = []  # 存储性能测量结果的列表

    # 开始循环迭代，生成稀疏张量配置和相关属性，进行性能测试
    for i, (tensors, tensor_properties, _) in enumerate(add_fuzzer.take(n=n)):
        x = tensors["x"]  # 获取张量"x"
        y = tensors["y"]  # 获取张量"y"
        shape = ", ".join(tuple(f'{i:>4}' for i in x.shape))  # 获取张量"x"的形状字符串表示
        x_tensor_properties = tensor_properties["x"]  # 获取张量"x"的属性
        description = "".join([
            f"| {shape:<20} | ",
            f"{x_tensor_properties['sparsity']:>9.2f} | ",
            f"{x_tensor_properties['sparse_dim']:>9d} | ",
            f"{x_tensor_properties['dense_dim']:>9d} | ",
            f"{('True' if x_tensor_properties['is_hybrid'] else 'False'):>9} | ",
            f"{('True' if x.is_coalesced() else 'False'):>9} | "
        ])  # 构建描述性字符串，包含张量形状、稀疏度等信息

        # 创建一个 Timer 对象，用于测量执行时间
        timer = benchmark_utils.Timer(
            stmt="torch.sparse.sum(x) + torch.sparse.sum(y)",  # 执行的语句
            globals=tensors,  # 全局变量，传入张量
            description=description,  # 测试描述
        )
        measurements.append(timer.blocked_autorange(min_run_time=0.1))  # 执行性能测试并将结果添加到列表中
        measurements[-1].metadata = {"nnz": x._nnz()}  # 添加元数据，记录非零元素的数量
        print(f"\r{i + 1} / {n}", end="")  # 打印进度信息
        sys.stdout.flush()  # 刷新输出

    print()  # 输出换行，结束进度信息显示

    # 继续字符串处理以生成美观的输出
    print(f"Average attempts per valid config: {1. / (1. - add_fuzzer.rejection_rate):.1f}")

    # 定义一个函数，用于基于稀疏张量非零元素的数量进行时间计算
    def time_fn(m):
        return m.mean / m.metadata["nnz"]

    # 根据时间函数对测量结果进行排序
    measurements.sort(key=time_fn)
    # 定义格式化字符串模板，用于输出测量结果的表头
    template = f"{{:>6}}{' ' * 16} Shape{' ' * 17}    sparsity{' ' * 4}sparse_dim{' ' * 4}dense_dim{' ' * 4}hybrid{' ' * 4}coalesced\n{'-' * 108}"
    
    # 打印最佳测量结果的表头
    print(template.format("Best:"))
    
    # 遍历前10个测量结果
    for m in measurements[:10]:
        # 打印每个测量结果的执行时间，并格式化为纳秒每元素
        print(f"{time_fn(m) * 1e9:>5.2f} ns / element     {m.description}")
    
    # 打印换行符后的最差测量结果的表头
    print("\n" + template.format("Worst:"))
    
    # 遍历最后10个测量结果
    for m in measurements[-10:]:
        # 打印每个测量结果的执行时间，并格式化为纳秒每元素
        print(f"{time_fn(m) * 1e9:>5.2f} ns / element     {m.description}")
# 如果当前脚本作为主程序运行，则执行 main() 函数
if __name__ == "__main__":
    main()
```