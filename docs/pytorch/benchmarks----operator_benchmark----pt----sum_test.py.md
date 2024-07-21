# `.\pytorch\benchmarks\operator_benchmark\pt\sum_test.py`

```
# 导入名为 operator_benchmark 的模块，并重命名为 op_bench
import operator_benchmark as op_bench
# 导入 torch 模块
import torch

"""Microbenchmarks for sum reduction operator."""
# 这段注释指明了以下代码是关于求和操作符的微基准测试。

# PT add 操作符的配置
sum_configs = op_bench.cross_product_configs(
    R=[64, 256],  # 被减少维度的长度
    V=[32, 512],  # 其它维度的长度
    dim=[0, 1],  # 求和的维度
    contiguous=[True, False],  # 是否连续存储
    device=["cpu", "cuda"],  # 运行设备
    tags=["short"],  # 测试标签
) + op_bench.cross_product_configs(
    R=[1024, 8192],  # 被减少维度的长度
    V=[512, 1024],  # 其它维度的长度
    dim=[0, 1],  # 求和的维度
    contiguous=[True, False],  # 是否连续存储
    device=["cpu", "cuda"],  # 运行设备
    tags=["long"],  # 测试标签
)

# 定义一个名为 SumBenchmark 的类，继承自 op_bench.TorchBenchmarkBase
class SumBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, R, V, dim, contiguous, device):
        shape = (R, V) if dim == 0 else (V, R)  # 根据 dim 的值确定张量的形状
        tensor = torch.rand(shape, device=device)  # 创建一个随机张量，放在指定设备上

        if not contiguous:
            storage = torch.empty([s * 2 for s in shape], device=device)  # 创建一个非连续存储的张量
            storage[::2, ::2] = tensor  # 将随机张量的值赋给非连续存储张量的部分位置
            self.input_tensor = storage[::2, ::2]  # 使用非连续存储张量的一部分作为输入张量
        else:
            self.input_tensor = tensor  # 使用连续存储的随机张量作为输入张量

        self.inputs = {"input_tensor": self.input_tensor, "dim": dim}  # 设置输入字典
        self.set_module_name("sum")  # 设置模块名为 sum

    def forward(self, input_tensor, dim: int):
        return input_tensor.sum(dim=dim)  # 执行张量的求和操作，并返回结果

# 生成 PyTorch 测试的代码，使用之前定义的 sum_configs 和 SumBenchmark 类
op_bench.generate_pt_test(sum_configs, SumBenchmark)

# 如果当前脚本作为主程序运行，则执行 operator benchmark 的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```