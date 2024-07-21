# `.\pytorch\benchmarks\operator_benchmark\pt\matmul_test.py`

```
# 导入 operator_benchmark 模块并命名为 op_bench
import operator_benchmark as op_bench
# 导入 torch 模块
import torch

"""Microbenchmarks for MatMul operator"""

# PT 矩阵乘法运算符的短配置列表
mm_short_configs = op_bench.config_list(
    # 定义属性名称和属性值列表
    attr_names=["M", "N", "K", "trans_a", "trans_b"],
    attrs=[
        [1, 1, 1, True, False],     # 示例配置 1
        [128, 128, 128, True, False],   # 示例配置 2
        [256, 256, 256, False, True],   # 示例配置 3
    ],
    # 交叉组合的配置项
    cross_product_configs={
        "device": ["cpu", "cuda"],   # 设备选项：CPU 和 CUDA
    },
    tags=["short"],   # 标记为短测试
)

# PT 矩阵乘法运算符的长配置列表
mm_long_configs = op_bench.cross_product_configs(
    # 定义各个参数的取值列表
    M=[32],
    N=[512, 128],
    K=[64],
    trans_a=[False, True],
    trans_b=[True, False],
    device=["cpu", "cuda"],   # 设备选项：CPU 和 CUDA
    tags=["long"],   # 标记为长测试
)

# MatMulBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class MatMulBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，接受 M, N, K, trans_a, trans_b, device 参数
    def init(self, M, N, K, trans_a, trans_b, device):
        # 初始化输入字典
        self.inputs = {
            "input_one": torch.rand(M, N, device=device)   # 根据 trans_a 参数生成不同形状的随机张量
            if trans_a
            else torch.rand(N, M, device=device).t(),
            "input_two": torch.rand(N, K, device=device)   # 根据 trans_b 参数生成不同形状的随机张量
            if trans_b
            else torch.rand(K, N, device=device).t(),
        }
        self.set_module_name("matmul")   # 设置模块名称为 "matmul"

    # 前向方法，接受 input_one 和 input_two 两个输入参数，并返回它们的矩阵乘法结果
    def forward(self, input_one, input_two):
        return torch.matmul(input_one, input_two)

# 生成 PT 测试用例，组合长配置和短配置，并使用 MatMulBenchmark 类
op_bench.generate_pt_test(mm_long_configs + mm_short_configs, MatMulBenchmark)

# 如果作为主程序运行，则执行 operator_benchmark 的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```