# `.\pytorch\benchmarks\operator_benchmark\pt\qunary_test.py`

```
# 导入名为 operator_benchmark 的模块，并重命名为 op_bench
import operator_benchmark as op_bench
# 导入 PyTorch 库
import torch

"""Microbenchmarks for quantized unary operators (point-wise and reduction)."""
# 用于量化一元操作（逐点和归约）的微基准

# Configs for pointwise and reduction unary ops
# 逐点和归约一元操作的配置

# 短基准配置，包括属性名称为 M 和 N，属性值为 [512, 512]，数据类型为 torch.quint8，标签为 "short"
qunary_ops_configs_short = op_bench.config_list(
    attr_names=["M", "N"],
    attrs=[
        [512, 512],
    ],
    cross_product_configs={
        "dtype": [torch.quint8],
    },
    tags=["short"],
)

# 长基准配置，包括 M 和 N 属性为 [256, 1024]，数据类型为 torch.quint8, torch.qint8, torch.qint32，标签为 "long"
qunary_ops_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],
    N=[256, 1024],
    dtype=[torch.quint8, torch.qint8, torch.qint32],
    tags=["long"],
)


class QUnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    # 量化一元操作基准类，继承自 op_bench.TorchBenchmarkBase
    def init(self, M, N, dtype, op_func):
        # 初始化函数，接收 M、N、dtype 和 op_func 参数
        f_input = torch.rand(M, N)  # 生成大小为 MxN 的随机张量 f_input
        scale = 1.0  # 设置量化参数 scale
        zero_point = 0  # 设置量化参数 zero_point
        self.inputs = {
            "q_input": torch.quantize_per_tensor(
                f_input, scale=scale, zero_point=zero_point, dtype=dtype
            )  # 对 f_input 进行量化，生成名为 q_input 的量化张量
        }
        self.op_func = op_func  # 将 op_func 参数赋值给实例变量 self.op_func

    def forward(self, q_input):
        # 前向传播函数，接收量化输入 q_input
        return self.op_func(q_input)  # 调用 self.op_func 对 q_input 进行操作


# TODO: Uncomment the ops whenever they are implemented for quantized tensor.
# 当量化张量的操作实现后取消注释这些操作

# 量化一元操作列表，包括属性名称为 op_name 和 op_func
qunary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        # 属性列表，每个元素是一个二元组，包含操作名和对应的 Torch 函数
        # ['q_argsort', torch.argsort],  # 返回张量元素的排序索引
        # ['q_clone', torch.clone],      # 克隆张量
        # ['q_mean', torch.mean],        # 计算张量的均值
        ["q_relu", torch.relu],          # 对张量进行 ReLU 激活函数操作
        ["q_relu_", torch.relu_],        # 在张量上原地应用 ReLU 激活函数
        ["q_sort", torch.sort],          # 对张量进行排序
    ],
# 从 op_bench 模块中导入生成基于操作列表的 PyTorch 测试的函数
op_bench.generate_pt_tests_from_op_list(
    qunary_ops_list,
    qunary_ops_configs_short + qunary_ops_configs_long,
    QUnaryOpBenchmark,
)

# === Other unary ops (i.e. the ones that need parameters as args) ===

# 创建用于点积和降维一元操作的短配置列表
qunary_ops_topk_configs_short = op_bench.config_list(
    attr_names=["M", "N", "k"],
    attrs=[
        [512, 512, 5],
    ],
    cross_product_configs={
        "dtype": [torch.quint8],
    },
    tags=["short"],
)

# 创建用于点积和降维一元操作的长配置列表
qunary_ops_topk_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],
    N=[256, 1024],
    k=[1, 3, 5],
    dtype=[torch.quint8, torch.qint8, torch.qint32],
    tags=["long"],
)

# 定义 QTopkOpBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class QTopkOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dtype, k):
        # 创建随机输入张量 f_input
        f_input = torch.rand(M, N)
        scale = 1.0
        zero_point = 0
        # 使用 quantize_per_tensor 方法对输入张量进行量化
        self.inputs = {
            "q_input": torch.quantize_per_tensor(
                f_input, scale=scale, zero_point=zero_point, dtype=dtype
            ),
            "k": k,
        }
        self.set_module_name("qtopk")

    def forward(self, q_input, k: int):
        # 返回输入张量 q_input 的前 k 个最大值
        return torch.topk(q_input, k)

# 生成基于 qunary_ops_topk_configs_short 和 qunary_ops_topk_configs_long 的 PyTorch 测试
op_bench.generate_pt_test(
    qunary_ops_topk_configs_short + qunary_ops_topk_configs_long, QTopkOpBenchmark
)

# 如果当前脚本作为主程序运行，则执行 op_bench.benchmark_runner.main() 函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```