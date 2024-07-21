# `.\pytorch\benchmarks\operator_benchmark\common\tests\random_sample_test.py`

```py
# 导入名为 operator_benchmark 的模块并重命名为 op_bench
import operator_benchmark as op_bench

# 导入 PyTorch 库
import torch

# 使用 op_bench 模块的 random_sample_configs 函数生成配置参数字典
configs = op_bench.random_sample_configs(
    # M 取值范围为 1 到 6
    M=[1, 2, 3, 4, 5, 6],
    # N 取值范围为 7 到 12
    N=[7, 8, 9, 10, 11, 12],
    # K 取值范围为 13 到 18
    K=[13, 14, 15, 16, 17, 18],
    # probs 保存每个值的权重
    probs=op_bench.attr_probs(
        # M 对应的权重列表
        M=[0.5, 0.2, 0.1, 0.05, 0.03, 0.1],
        # N 对应的权重列表
        N=[0.1, 0.3, 0.4, 0.02, 0.03, 0.04],
        # K 对应的权重列表
        K=[0.03, 0.6, 0.04, 0.02, 0.03, 0.01],
    ),
    # total_samples 指定返回的输入数量为 10
    total_samples=10,
    # 给生成的测试样例添加标签 "short"
    tags=["short"],
)

# 定义一个名为 AddBenchmark 的类，继承自 op_bench.TorchBenchmarkBase 类
class AddBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，接收 M、N、K 三个参数
    def init(self, M, N, K):
        # 初始化 input_one 和 input_two 为随机生成的 M x N x K 的张量
        self.input_one = torch.rand(M, N, K)
        self.input_two = torch.rand(M, N, K)
        # 设置当前模块的名称为 "add"
        self.set_module_name("add")

    # 前向方法，执行张量的加法操作
    def forward(self):
        return torch.add(self.input_one, self.input_two)

# 使用 op_bench 模块的 generate_pt_test 函数生成 PyTorch 的测试
op_bench.generate_pt_test(configs, AddBenchmark)

# 如果当前脚本作为主程序运行，则执行 operator_benchmark 的 benchmark_runner.main 函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```