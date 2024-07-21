# `.\pytorch\benchmarks\operator_benchmark\pt\qcomparators_test.py`

```
# 导入 operator_benchmark 模块，用于性能基准测试
# 导入 torch 模块，用于深度学习框架中的张量操作
import operator_benchmark as op_bench
import torch

# 定义各种配置组合，用于性能基准测试的参数设置
qcomparators_configs = op_bench.cross_product_configs(
    N=(8, 64),  # 张量维度大小，分别为 8 和 64
    dtype=(torch.quint8, torch.qint8, torch.qint32),  # 张量的数据类型，包括量化整数类型
    contig=(False, True),  # 是否是连续内存布局
    other_scalar=(False, True),  # 是否涉及其他标量值
    out_variant=(False, True),  # 是否使用输出变体
    tags=("short",),  # 性能测试标签
)

# 定义操作列表，包括操作名和对应的 torch 函数
qcomparators_ops = op_bench.op_list(
    attrs=(
        ("eq", torch.eq),  # 等于操作
        ("ne", torch.ne),  # 不等于操作
        ("lt", torch.lt),  # 小于操作
        ("gt", torch.gt),  # 大于操作
        ("le", torch.le),  # 小于等于操作
        ("ge", torch.ge),  # 大于等于操作
    ),
    attr_names=("op_name", "op_func"),  # 属性名称
)


class QComparatorBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, dtype, contig, other_scalar, out_variant, op_func):
        # 初始化输入张量 f_input，形状为 (N, N)，数值范围在 [-128, 128)
        f_input = (torch.rand(N, N) - 0.5) * 256
        scale = 1.0  # 量化参数：比例因子
        zero_point = 0  # 量化参数：零点

        # 对输入张量进行量化为 q_input_a 和 q_input_b
        q_input_a = torch.quantize_per_tensor(
            f_input, scale=scale, zero_point=zero_point, dtype=dtype
        )
        q_input_b = q_input_a.clone()

        # 如果 contig=False，则对 q_input_a 进行维度重新排列
        if not contig:
            permute_dims = list(range(f_input.ndim))[::-1]
            q_input_a = q_input_a.permute(permute_dims)

        # 设置基准测试所需的属性和输入
        self.qop = op_func
        self.inputs = {
            "q_input_a": q_input_a,
            "q_input_b": q_input_b,
            "out_variant": out_variant,
            "other_scalar": other_scalar,
        }

    def forward(self, q_input_a, q_input_b, out_variant: bool, other_scalar: bool):
        # 根据 out_variant 和 other_scalar 参数执行不同的操作
        if out_variant:
            if other_scalar:
                return self.qop(q_input_a, 42, out=torch.tensor(True, dtype=torch.bool))
            else:
                return self.qop(
                    q_input_a, q_input_b, out=torch.tensor(True, dtype=torch.bool)
                )
        else:
            if other_scalar:
                return self.qop(q_input_a, 42)
            else:
                return self.qop(q_input_a, q_input_b)


# 根据操作列表、配置和基准测试类生成 PyTorch 操作的性能测试
op_bench.generate_pt_tests_from_op_list(
    qcomparators_ops, qcomparators_configs, QComparatorBenchmark
)

# 如果脚本作为主程序运行，则执行性能基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```