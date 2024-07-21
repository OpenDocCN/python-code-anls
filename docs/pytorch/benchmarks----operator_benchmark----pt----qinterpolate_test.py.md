# `.\pytorch\benchmarks\operator_benchmark\pt\qinterpolate_test.py`

```
import operator_benchmark as op_bench  # 导入 operator_benchmark 库，重命名为 op_bench
import torch  # 导入 PyTorch 库

"""Microbenchmarks for the quantized interpolate op.

Note: We are not benchmarking `upsample` as it is being deprecated, and calls
the `interpolate` anyway.
"""

# 定义长配置列表，用于量化插值操作的微基准测试
qinterpolate_long_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],  # 定义属性名称列表
    attrs=[
        [512, 512, 512],  # 配置属性值，包含单个配置 [512, 512, 512]
    ],
    cross_product_configs={
        "dtype": [torch.quint8, torch.qint8, torch.qint32],  # 数据类型的交叉组合
        "mode": ["nearest", "bilinear"],  # 插值模式的选择
        "scale": [0.5, 1.0, 2.0],  # 缩放因子的选择
        "contig": [True],  # 是否连续的选择，目前仅 True，未来考虑增加 False
    },
    tags=["long"],  # 标记为长基准测试
)

# 定义短配置列表，用于量化插值操作的微基准测试
qinterpolate_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K", "dtype", "mode", "scale", "contig"],  # 定义属性名称列表
    attrs=[
        [32, 32, 32, torch.quint8, "nearest", 0.5, True],     # 下采样
        [32, 32, 32, torch.quint8, "bilinear", 0.5, True],   # 下采样
        [32, 32, 32, torch.quint8, "nearest", 2.0, True],    # 上采样
        [32, 32, 32, torch.quint8, "bilinear", 2.0, True],   # 上采样
        [3, 720, 1280, torch.quint8, "bilinear", 0.83333, True],  # 下采样
    ],
    tags=["short"],  # 标记为短基准测试
)

# 定义 QInterpolateBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class QInterpolateBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, dtype, mode, scale, contig):
        f_input = (torch.rand(1, M, N, K) - 0.5) * 256  # 生成随机的浮点数输入张量
        scale = 0.1  # 设置缩放因子为 0.1
        zero_point = 42  # 设置零点为 42
        self.q_input = torch.quantize_per_tensor(
            f_input, scale=scale, zero_point=zero_point, dtype=dtype  # 对输入张量进行量化
        )
        if not contig:  # 如果 contig 不连续
            permute_dims = list(range(self.q_input.ndim))[::-1]  # 获取反转的维度排列
            self.q_input = self.q_input.permute(permute_dims)  # 对输入张量进行维度置换

        # 设置输入参数字典
        self.inputs = {"q_input": self.q_input, "scale_factor": scale, "mode": mode}
        self.set_module_name("q_interpolate")  # 设置模块名称为 "q_interpolate"

    def forward(self, q_input, scale_factor: float, mode: str):
        return torch.nn.functional.interpolate(
            q_input, scale_factor=scale_factor, mode=mode  # 执行插值操作
        )

# 生成 PyTorch 测试
op_bench.generate_pt_test(
    qinterpolate_short_configs + qinterpolate_long_configs, QInterpolateBenchmark
)

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    op_bench.benchmark_runner.main()  # 运行基准测试的主函数
```