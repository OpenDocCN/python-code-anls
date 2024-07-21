# `.\pytorch\benchmarks\operator_benchmark\pt\qcat_test.py`

```
# 从 typing 模块中导入 List 类型
from typing import List

# 导入 operator_benchmark 库，并重命名为 op_bench
import operator_benchmark as op_bench

# 导入 PyTorch 库
import torch
import torch.ao.nn.quantized as nnq

# 定义用于量化 Cat 操作符的微基准测试

# 短配置项列表，用于 PT Cat 操作符
qcat_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "L", "dim"],
    attrs=[
        [256, 512, 1, 2, 0],
        [512, 512, 2, 1, 1],
    ],
    cross_product_configs={
        "contig": ("all", "one", "none"),
        "dtype": (torch.quint8, torch.qint8, torch.qint32),
    },
    tags=["short"],
)

# 长配置项列表，用于 PT Cat 操作符
qcat_configs_long = op_bench.cross_product_configs(
    M=[128, 1024],
    N=[128, 1024],
    K=[1, 2],
    L=[5, 7],
    dim=[0, 1, 2],
    contig=["all", "one", "none"],
    dtype=[torch.quint8],
    tags=["long"],
)

# 定义 QCatBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class QCatBenchmark(op_bench.TorchBenchmarkBase):

    # 初始化方法，接受 M, N, K, L, dim, contig, dtype 参数
    def init(self, M, N, K, L, dim, contig, dtype):
        # 创建随机的浮点输入数据
        f_input = (torch.rand(M, N, K) - 0.5) * 256
        # 创建量化函数对象
        self.qf = nnq.QFunctional()
        scale = 1.0
        zero_point = 0
        # 设置量化函数的 scale 和 zero_point 属性
        self.qf.scale = scale
        self.qf.zero_point = zero_point

        # 确保 contig 参数值为 "none", "one", "all" 中的一个
        assert contig in ("none", "one", "all")
        # 对输入数据进行量化处理，创建量化张量 q_input
        q_input = torch.quantize_per_tensor(f_input, scale, zero_point, dtype)
        # 创建维度倒置后的量化张量 q_input_non_contig，并确保其是连续的
        permute_dims = tuple(range(q_input.ndim - 1, -1, -1))
        q_input_non_contig = q_input.permute(permute_dims).contiguous()
        q_input_non_contig = q_input_non_contig.permute(permute_dims)
        # 根据 contig 参数值选择输入数据
        if contig == "all":
            self.input = (q_input, q_input)
        elif contig == "one":
            self.input = (q_input, q_input_non_contig)
        elif contig == "none":
            self.input = (q_input_non_contig, q_input_non_contig)

        # 设置测试用例的输入参数和维度参数
        self.inputs = {"input": self.input, "dim": dim}
        # 设置模块名为 "qcat"
        self.set_module_name("qcat")

    # 前向方法，接受 input（List[torch.Tensor]）和 dim（int）参数
    def forward(self, input: List[torch.Tensor], dim: int):
        # 调用量化函数对象的 cat 方法进行拼接操作
        return self.qf.cat(input, dim=dim)

# 生成 PyTorch 测试用例，包括短和长配置项
op_bench.generate_pt_test(qcat_configs_short + qcat_configs_long, QCatBenchmark)

# 如果当前脚本作为主程序运行，则执行基准测试运行
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```