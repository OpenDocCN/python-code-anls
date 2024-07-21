# `.\pytorch\benchmarks\operator_benchmark\pt\qtensor_method_test.py`

```
# 导入 operator_benchmark 模块，简称为 op_bench
import operator_benchmark as op_bench
# 导入 torch 模块
import torch

# 定义用于点对点操作和约简单目操作的配置
qmethods_configs_short = op_bench.config_list(
    attr_names=["M", "N"],  # 定义属性名称列表
    attrs=[
        [32, 32],  # 属性取值为 [32, 32]
    ],
    cross_product_configs={
        "dtype": [torch.quint8],  # 数据类型为 torch.quint8
        "contig": [False, True],   # 是否连续存储的布尔值列表
    },
    tags=["short"],  # 标签为 "short"
)

# 定义用于长操作的交叉产品配置
qmethods_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],                 # M 取值为 [256, 1024]
    N=[256, 1024],                 # N 取值为 [256, 1024]
    dtype=[torch.qint8, torch.qint32],  # 数据类型为 torch.qint8 或 torch.qint32
    contig=[False, True],          # 是否连续存储的布尔值列表
    tags=["long"],                 # 标签为 "long"
)


# 定义基础量化方法基准类 _QMethodBenchmarkBase，继承自 op_bench.TorchBenchmarkBase
class _QMethodBenchmarkBase(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dtype, contig):
        # 创建随机的浮点输入张量 f_input
        f_input = torch.rand(M, N)
        scale = 1.0
        zero_point = 0
        # 对 f_input 进行量化得到 q_input
        self.q_input = torch.quantize_per_tensor(
            f_input, scale=scale, zero_point=zero_point, dtype=dtype
        )
        # 如果 contig 是 False，则对 q_input 进行维度置换
        if not contig:
            permute_dims = list(range(self.q_input.ndim))[::-1]
            self.q_input = self.q_input.permute(permute_dims)

        # 初始化 inputs 字典，包含量化输入 q_input
        self.inputs = {
            "q_input": self.q_input,
        }


# 定义基于量化输入复制操作的基准类 QMethodTensorInputCopyBenchmark，继承自 _QMethodBenchmarkBase
class QMethodTensorInputCopyBenchmark(_QMethodBenchmarkBase):
    def forward(self, q_input):
        # 返回 q_input 的就地复制结果
        return q_input.copy_(q_input)


# 生成基于 PyTorch 的运算测试，结合短和长配置，应用于 QMethodTensorInputCopyBenchmark 类
op_bench.generate_pt_test(
    qmethods_configs_short + qmethods_configs_long, QMethodTensorInputCopyBenchmark
)

# 如果该脚本作为主程序运行，则执行 operator_benchmark 的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```