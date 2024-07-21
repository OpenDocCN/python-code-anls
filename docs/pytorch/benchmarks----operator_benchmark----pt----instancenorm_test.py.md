# `.\pytorch\benchmarks\operator_benchmark\pt\instancenorm_test.py`

```
# 导入名为 operator_benchmark 的模块作为 op_bench 别名
import operator_benchmark as op_bench
# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的函数模块 nn.functional，并使用 F 别名
import torch.nn.functional as F

# 定义一个名为 instancenorm_configs_short 的变量，用于存储微基准测试的配置
instancenorm_configs_short = op_bench.cross_product_configs(
    # 两组不同的维度配置
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    # 标记为 "short"
    tags=["short"],
)

# 定义一个名为 InstanceNormBenchmark 的类，继承自 op_bench.TorchBenchmarkBase
class InstanceNormBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，接收 dims 参数
    def init(self, dims):
        # 提取维度 dims 中的通道数
        num_channels = dims[1]
        # 初始化输入数据字典
        self.inputs = {
            # 随机生成 dims 大小的张量，数值范围为 [-128, 128)
            "input": (torch.rand(*dims) - 0.5) * 256,
            # 随机生成 num_channels 大小的张量，数据类型为 float
            "weight": torch.rand(num_channels, dtype=torch.float),
            # 随机生成 num_channels 大小的张量，数据类型为 float
            "bias": torch.rand(num_channels, dtype=torch.float),
            # 设定一个很小的 epsilon 值
            "eps": 1e-5,
        }

    # 前向传播方法，接收 input, weight, bias, eps 四个参数
    def forward(self, input, weight, bias, eps: float):
        # 调用 PyTorch 中的 instance_norm 函数进行实例归一化操作
        return F.instance_norm(input, weight=weight, bias=bias, eps=eps)

# 生成 PyTorch 测试用例，传入微基准测试配置和 InstanceNormBenchmark 类
op_bench.generate_pt_test(instancenorm_configs_short, InstanceNormBenchmark)

# 如果当前脚本作为主程序运行，则执行 operator_benchmark 的主程序入口
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```