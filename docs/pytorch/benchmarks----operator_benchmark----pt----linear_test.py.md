# `.\pytorch\benchmarks\operator_benchmark\pt\linear_test.py`

```py
# 从 pt 模块导入 configs 模块
from pt import configs

# 导入 operator_benchmark 并重命名为 op_bench
import operator_benchmark as op_bench

# 导入 torch 库
import torch

# 导入 torch 的神经网络模块 nn
import torch.nn as nn

# 定义一个用于 Linear 操作的微基准类
class LinearBenchmark(op_bench.TorchBenchmarkBase):

    # 初始化方法，设置输入参数 N（样本数）、IN（输入特征数）、OUT（输出特征数）、device（设备类型）
    def init(self, N, IN, OUT, device):
        # 创建一个名为 input_one 的输入字典，其中包含一个形状为 (N, IN) 的随机张量，使用指定设备
        self.inputs = {"input_one": torch.rand(N, IN, device=device)}
        
        # 创建一个 nn.Linear 模块，输入特征数为 IN，输出特征数为 OUT，同时移至指定设备
        self.linear = nn.Linear(IN, OUT).to(device=device)
        
        # 设置模块名为 "linear"
        self.set_module_name("linear")

    # 前向传播方法，接受输入 input_one，并返回经过线性层处理后的结果
    def forward(self, input_one):
        return self.linear(input_one)

# 使用 operator_benchmark 提供的函数生成基准测试，使用 configs 模块中的线性配置来进行测试
op_bench.generate_pt_test(
    configs.linear_configs_short + configs.linear_configs_long, LinearBenchmark
)

# 如果当前脚本为主程序执行
if __name__ == "__main__":
    # 运行 operator_benchmark 中的基准测试运行器
    op_bench.benchmark_runner.main()
```