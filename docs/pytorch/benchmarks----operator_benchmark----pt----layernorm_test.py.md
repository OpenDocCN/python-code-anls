# `.\pytorch\benchmarks\operator_benchmark\pt\layernorm_test.py`

```
# 导入 operator_benchmark 库和 torch 库，用于性能测试和神经网络操作
import operator_benchmark as op_bench
import torch
import torch.nn.functional as F

# 定义多维度的 layernorm 微基准测试配置
layernorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (1, 8, 16),      # 第一种输入维度
        (8, 8, 16),      # 第二种输入维度
        (32, 8, 16),     # 第三种输入维度
        (64, 128, 56, 56),  # 第四种输入维度
    ),
    tags=["short"],       # 设置测试标签为 "short"
)

# 定义 LayerNormBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class LayerNormBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，接受 dims 参数作为输入维度
    def init(self, dims):
        # 生成随机输入，并对其进行归一化处理
        input = (torch.rand(*dims) - 0.5) * 256
        self.inputs = {
            "input": input,   # 输入数据
            "weight": torch.rand(*input.size()[1:], dtype=torch.float),  # 归一化权重
            "bias": torch.rand(*input.size()[1:], dtype=torch.float),    # 归一化偏差
            "eps": 1e-5,      # 归一化过程中的小值防止除零
        }

    # 前向方法，接受输入、权重、偏差和 eps 参数，执行 layernorm 操作并返回结果
    def forward(self, input, weight, bias, eps: float):
        return F.layer_norm(input, input.size()[1:], weight=weight, bias=bias, eps=eps)

# 生成 PyTorch 的性能测试
op_bench.generate_pt_test(layernorm_configs_short, LayerNormBenchmark)

# 如果脚本被直接运行，则执行性能基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```