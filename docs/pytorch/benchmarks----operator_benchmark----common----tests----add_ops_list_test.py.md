# `.\pytorch\benchmarks\operator_benchmark\common\tests\add_ops_list_test.py`

```py
# 导入operator_benchmark库，并重命名为op_bench
import operator_benchmark as op_bench

# 导入torch库
import torch

# 配置用于逐点一元操作的参数列表
unary_ops_configs = op_bench.config_list(
    attrs=[
        [128, 128],  # 定义一个参数组合 [M, N]
    ],
    attr_names=["M", "N"],  # 定义参数的名称
    tags=["short"],  # 添加标签 "short"
)

# 定义逐点一元操作的列表
unary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],  # 定义操作名称和操作函数
    attrs=[
        ["abs", torch.abs],   # 添加操作 "abs" 和 torch.abs 函数
        ["acos", torch.acos],  # 添加操作 "acos" 和 torch.acos 函数
    ],
)

# 定义一个基于Torch的逐点一元操作基准类
class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, op_func):
        self.input_one = torch.rand(M, N)  # 生成一个大小为 MxN 的随机张量
        self.op_func = op_func  # 设置操作函数为传入的 op_func

    def forward(self):
        return self.op_func(self.input_one)  # 执行操作函数 op_func 对输入张量进行操作

# 从操作列表 unary_ops_list 和配置列表 unary_ops_configs 生成PyTorch测试
op_bench.generate_pt_tests_from_op_list(
    unary_ops_list, unary_ops_configs, UnaryOpBenchmark
)

# 如果当前文件作为主程序运行，执行operator_benchmark库的主函数入口
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```