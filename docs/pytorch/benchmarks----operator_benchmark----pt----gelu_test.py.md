# `.\pytorch\benchmarks\operator_benchmark\pt\gelu_test.py`

```
# 导入 operator_benchmark 库并重命名为 op_bench
import operator_benchmark as op_bench
# 导入 PyTorch 库
import torch

"""
Microbenchmarks for the gelu operators.
"""

# 定义 gelu 运行配置的组合，包括不同的尺寸和设备选项
gelu_configs_long = op_bench.cross_product_configs(
    N=[1, 4], C=[3], H=[16, 256], W=[16, 256], device=["cpu"], tags=["long"]
)

# 定义 GeluBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class GeluBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置输入数据的随机张量
    def init(self, N, C, H, W, device):
        self.inputs = {"input": torch.rand(N, C, H, W, device=device)}

    # 前向方法，计算输入张量的 gelu 函数结果
    def forward(self, input):
        return torch.nn.functional.gelu(input)

# 使用 op_bench.generate_pt_test 方法生成 PyTorch 的性能测试用例
op_bench.generate_pt_test(gelu_configs_long, GeluBenchmark)

# 如果该脚本作为主程序运行，则执行 operator_benchmark 库的主程序入口
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```