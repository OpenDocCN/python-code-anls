# `.\pytorch\benchmarks\operator_benchmark\pt\qarithmetic_test.py`

```py
# 导入 operator_benchmark 库并重命名为 op_bench
import operator_benchmark as op_bench
# 导入 torch 库
import torch
# 从 torch._ops 模块中导入 ops
from torch._ops import ops

# 定义用于量化算术运算的配置列表，包括不同的 N 值、数据类型、是否连续存储和标签信息
qarithmetic_binary_configs = op_bench.cross_product_configs(
    N=(2, 8, 64, 512),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    contig=(False, True),
    tags=("short",),
)

# 定义量化二元算术运算的操作列表，每个操作包括操作名称和对应的函数
qarithmetic_binary_ops = op_bench.op_list(
    attrs=(
        ("add", ops.quantized.add),
        ("add_relu", ops.quantized.add_relu),
        ("mul", ops.quantized.mul),
    ),
    attr_names=("op_name", "op_func"),
)

# 定义量化二元标量算术运算的操作列表，每个操作包括操作名称和对应的函数
qarithmetic_binary_scalar_ops = op_bench.op_list(
    attrs=(
        ("add_scalar", ops.quantized.add_scalar),
        ("mul_scalar", ops.quantized.mul_scalar),
    ),
    attr_names=("op_name", "op_func"),
)

# 定义 _QFunctionalBinaryArithmeticBenchmarkBase 类，继承自 op_bench.TorchBenchmarkBase
class _QFunctionalBinaryArithmeticBenchmarkBase(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置量化功能函数 QFunctional，生成输入数据 q_input_a
    def setup(self, N, dtype, contig):
        self.qfunctional = torch.ao.nn.quantized.QFunctional()

        # TODO: 考虑更多多样的形状
        f_input = (torch.rand(N, N) - 0.5) * 256
        self.scale = 1.0
        self.zero_point = 0
        self.q_input_a = torch.quantize_per_tensor(
            f_input, scale=self.scale, zero_point=self.zero_point, dtype=dtype
        )

        # 如果不是连续存储，则对输入数据进行维度置换
        if not contig:
            permute_dims = list(range(f_input.ndim))[::-1]
            self.q_input_a = self.q_input_a.permute(permute_dims)

# 定义 QFunctionalBenchmark 类，继承自 _QFunctionalBinaryArithmeticBenchmarkBase 类
class QFunctionalBenchmark(_QFunctionalBinaryArithmeticBenchmarkBase):
    # 初始化方法，调用父类的 setup 方法设置数据和量化功能函数
    def init(self, N, dtype, contig, op_func):
        super().setup(N, dtype, contig)
        # 设置输入字典，包括 q_input_a、q_input_b、scale 和 zero_point
        self.inputs = {
            "q_input_a": self.q_input_a,
            "q_input_b": self.q_input_a,
            "scale": self.scale,
            "zero_point": self.zero_point,
        }
        self.op_func = op_func

    # 前向方法，执行具体的量化算术运算操作 op_func
    def forward(self, q_input_a, q_input_b, scale: float, zero_point: int):
        return self.op_func(q_input_a, q_input_b, scale=scale, zero_point=zero_point)

# 使用 op_bench.generate_pt_tests_from_op_list 函数生成基于操作列表的 PyTorch 测试用例，
# 包括 qarithmetic_binary_ops 列表、qarithmetic_binary_configs 配置和 QFunctionalBenchmark 类
op_bench.generate_pt_tests_from_op_list(
    qarithmetic_binary_ops, qarithmetic_binary_configs, QFunctionalBenchmark
)

# 定义 QFunctionalScalarBenchmark 类，继承自 _QFunctionalBinaryArithmeticBenchmarkBase 类
class QFunctionalScalarBenchmark(_QFunctionalBinaryArithmeticBenchmarkBase):
    # 初始化方法，调用父类的 setup 方法设置数据和量化功能函数
    def init(self, N, dtype, contig, op_func):
        super().setup(N, dtype, contig)
        # 设置输入字典，包括 q_input 和 scalar_input
        self.inputs = {"q_input": self.q_input_a, "scalar_input": 42}
        self.op_func = op_func

    # 前向方法，执行具体的量化标量算术运算操作 op_func
    def forward(self, q_input, scalar_input: int):
        return self.op_func(q_input, scalar_input)

# 使用 op_bench.generate_pt_tests_from_op_list 函数生成基于操作列表的 PyTorch 测试用例，
# 包括 qarithmetic_binary_scalar_ops 列表、qarithmetic_binary_configs 配置和 QFunctionalScalarBenchmark 类
op_bench.generate_pt_tests_from_op_list(
    qarithmetic_binary_scalar_ops,
    qarithmetic_binary_configs,
    QFunctionalScalarBenchmark,
)

# 如果该脚本作为主程序运行，则调用 op_bench.benchmark_runner.main() 方法
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```