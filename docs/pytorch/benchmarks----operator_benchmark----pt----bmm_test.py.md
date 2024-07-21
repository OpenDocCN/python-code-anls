# `.\pytorch\benchmarks\operator_benchmark\pt\bmm_test.py`

```
# 导入名为operator_benchmark的模块，并使用别名op_bench
import operator_benchmark as op_bench
# 导入PyTorch库
import torch

"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""

# 定义一个继承自TorchBenchmarkBase的类，用于执行bmm运算的基准测试
class BmmBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置输入参数和运算符类型
    def init(self, B, M, N, K, device, op):
        # 构造输入数据字典，包括两个张量batch1和batch2
        self.inputs = {
            "batch1": torch.rand(
                (B, M, K), device=device, requires_grad=self.auto_set()
            ),
            "batch2": torch.rand(
                (
                    B,
                    K,
                    N,
                ),
                device=device,
                requires_grad=self.auto_set(),
            ),
        }
        # 设置模块名称，指定实际使用的运算符类型
        self.set_module_name(f"bmm (actual op={op}")
        # 根据指定的op类型选择torch.bmm或torch.matmul作为运算符
        self.op = torch.bmm if op == "bmm" else torch.matmul

    # 前向方法，执行矩阵乘法运算
    def forward(self, batch1, batch2):
        return self.op(batch1, batch2)

# 使用op_bench提供的函数生成bmm运算的配置组合
bmm_configs = op_bench.cross_product_configs(
    B=[2, 100],
    M=[8, 256],
    N=[256, 16],
    K=[16, 32],
    device=["cpu"],
    tags=["short"],
    op=["bmm", "matmul"],
)

# 生成PyTorch测试代码，基于给定的配置
op_bench.generate_pt_test(bmm_configs, BmmBenchmark)

# 如果当前脚本作为主程序执行，调用op_bench提供的基准测试运行器
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```