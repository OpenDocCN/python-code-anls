# `.\pytorch\benchmarks\operator_benchmark\pt\batchnorm_test.py`

```
import operator_benchmark as op_bench  # 导入 operator_benchmark 库，用于性能基准测试
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的 functional 模块

"""Microbenchmarks for batchnorm operator."""
# 批量归一化运算符的微基准测试

# Benchmark cudnn if available
# 如果 cudnn 可用，则进行基准测试
if torch.backends.cudnn.is_available:

    def cudnn_benchmark_configs(configs):
        result = []
        for config in configs:
            is_cuda = any("cuda" in attr.values() for attr in config)
            # 检查配置中是否包含 cuda 设备
            if is_cuda:
                result.append((*config, dict(cudnn=True)))
            result.append((*config, dict(cudnn=False)))
        return result

else:

    def cudnn_benchmark_configs(configs):
        # 如果 cudnn 不可用，则返回不启用 cudnn 的配置列表
        return [(*config, dict(cudnn=False)) for config in configs]


batchnorm_configs_short = cudnn_benchmark_configs(  # 使用 cudnn_benchmark_configs 函数获取短配置列表
    op_bench.config_list(
        attr_names=["M", "N", "K"],
        attrs=[
            [1, 256, 3136],
        ],
        cross_product_configs={
            "device": ["cpu", "cuda"],  # 设备选择为 cpu 或 cuda
            "training": [True, False],  # 训练模式为 True 或 False
        },
        tags=["short"],  # 标记为短测试
    )
)

batchnorm_configs_long = cudnn_benchmark_configs(  # 使用 cudnn_benchmark_configs 函数获取长配置列表
    op_bench.cross_product_configs(
        M=[2, 128],
        N=[8192, 2048],
        K=[1],
        device=["cpu", "cuda"],  # 设备选择为 cpu 或 cuda
        training=[True, False],  # 训练模式为 True 或 False
        tags=["long"],  # 标记为长测试
    )
)


class BatchNormBenchmark(op_bench.TorchBenchmarkBase):
    # 批量归一化基准类，继承自 op_bench.TorchBenchmarkBase

    def init(self, M, N, K, device, training, cudnn):
        # 初始化方法，设置输入参数
        self.inputs = {
            "input_one": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
            ),
            "mean": torch.rand(N, device=device),
            "var": torch.rand(N, device=device),
            "weight": torch.rand(N, device=device),
            "bias": torch.rand(N, device=device),
            "training": training,
            "cudnn": cudnn,
        }
        self.set_module_name("batchnorm")  # 设置模块名称为 batchnorm

    def forward(self, input_one, mean, var, weight, bias, training, cudnn):
        # 前向方法，执行批量归一化操作
        with torch.backends.cudnn.flags(enabled=cudnn):
            return F.batch_norm(input_one, mean, var, weight, bias, training)


op_bench.generate_pt_test(  # 生成 PyTorch 基准测试
    batchnorm_configs_short + batchnorm_configs_long, BatchNormBenchmark
)
op_bench.generate_pt_gradient_test(  # 生成 PyTorch 梯度基准测试
    batchnorm_configs_short + batchnorm_configs_long, BatchNormBenchmark
)


batchnorm1d_configs_short = cudnn_benchmark_configs(  # 使用 cudnn_benchmark_configs 函数获取短配置列表
    op_bench.config_list(
        attr_names=["N", "C"],
        attrs=[
            [3136, 256],
        ],
        cross_product_configs={
            "device": ["cpu", "cuda"],  # 设备选择为 cpu 或 cuda
            "training": [True, False],  # 训练模式为 True 或 False
        },
        tags=["short"],  # 标记为短测试
    )
)

batchnorm1d_configs_long = cudnn_benchmark_configs(  # 使用 cudnn_benchmark_configs 函数获取长配置列表
    op_bench.cross_product_configs(
        N=[2, 128],
        C=[8192, 2048],
        device=["cpu", "cuda"],  # 设备选择为 cpu 或 cuda
        training=[True, False],  # 训练模式为 True 或 False
        tags=["long"],  # 标记为长测试
    )
)


class BatchNorm1dBenchmark(op_bench.TorchBenchmarkBase):
    # 一维批量归一化基准类，继承自 op_bench.TorchBenchmarkBase
    # 初始化方法，用于设置 BatchNorm 模块的初始参数和状态
    def init(self, N, C, device, training, cudnn):
        # 初始化输入参数字典，包括输入数据、均值、方差、权重、偏置、训练模式和 cudnn 标志
        self.inputs = {
            "input_one": torch.rand(N, C, device=device, requires_grad=self.auto_set()),
            "mean": torch.rand(C, device=device),
            "var": torch.rand(C, device=device),
            "weight": torch.rand(C, device=device),
            "bias": torch.rand(C, device=device),
            "training": training,
            "cudnn": cudnn,
        }
        # 设置模块名称为 "batchnorm"
        self.set_module_name("batchnorm")

    # 前向传播方法，执行 Batch Normalization 操作
    def forward(self, input_one, mean, var, weight, bias, training, cudnn):
        # 使用 torch.backends.cudnn.flags 设置 cudnn 标志的启用状态
        with torch.backends.cudnn.flags(enabled=cudnn):
            # 调用 F.batch_norm 执行 Batch Normalization 操作，返回处理后的结果
            return F.batch_norm(input_one, mean, var, weight, bias, training)
# 生成批归一化操作的性能测试，包括短和长配置的 BatchNorm1d
op_bench.generate_pt_test(
    batchnorm1d_configs_short + batchnorm1d_configs_long, BatchNorm1dBenchmark
)

# 生成批归一化操作的梯度测试，包括短和长配置的 BatchNorm1d
op_bench.generate_pt_gradient_test(
    batchnorm1d_configs_short + batchnorm1d_configs_long, BatchNorm1dBenchmark
)

# 如果脚本被直接执行，而不是被作为模块导入，则执行性能基准测试
if __name__ == "__main__":
    # 运行性能基准测试的主函数
    op_bench.benchmark_runner.main()
```