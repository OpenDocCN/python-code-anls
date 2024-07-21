# `.\pytorch\benchmarks\operator_benchmark\pt\pool_test.py`

```
# 引入operator_benchmark作为op_bench的别名，用于性能基准测试
import operator_benchmark as op_bench
# 引入PyTorch库
import torch
import torch.nn as nn

"""
MaxPool1d和AvgPool1d运算符的微基准测试。
"""

# 针对pool-1d运算的配置
pool_1d_configs_short = op_bench.config_list(
    attr_names=["kernel", "stride", "N", "C", "L"],
    attrs=[
        [3, 1, 8, 256, 256],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

# 长时间运行的pool-1d配置
pool_1d_configs_long = op_bench.cross_product_configs(
    kernel=[3],
    stride=[1, 2],
    N=[8, 16],
    C=[3],
    L=[128, 256],
    device=["cpu", "cuda"],
    tags=["long"],
)

# pool-1d操作列表
pool_1d_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["MaxPool1d", nn.MaxPool1d],
        ["AvgPool1d", nn.AvgPool1d],
    ],
)


class Pool1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, kernel, stride, N, C, L, device, op_func):
        # 初始化输入数据为随机张量
        self.inputs = {"input": torch.rand(N, C, L, device=device)}
        # 使用给定的操作函数初始化池化操作
        self.op_func = op_func(kernel, stride=stride)

    def forward(self, input):
        # 执行池化操作
        return self.op_func(input)


# 生成来自操作列表的PyTorch测试用例
op_bench.generate_pt_tests_from_op_list(
    pool_1d_ops_list, pool_1d_configs_short + pool_1d_configs_long, Pool1dBenchmark
)


"""
MaxPool2d和AvgPool2d运算符的微基准测试。
"""


# 针对pool-2d运算的配置
pool_2d_configs_short = op_bench.config_list(
    attr_names=["kernel", "stride", "N", "C", "H", "W"],
    attrs=[
        [[3, 1], [2, 1], 1, 16, 32, 32],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

# 长时间运行的pool-2d配置
pool_2d_configs_long = op_bench.cross_product_configs(
    kernel=[[3, 2], [3, 3]],
    stride=[[2, 2]],
    N=[8, 16],
    C=[32],
    H=[32, 64],
    W=[32, 64],
    device=["cpu", "cuda"],
    tags=["long"],
)

# pool-2d操作列表
pool_2d_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["MaxPool2d", nn.MaxPool2d],
        ["AvgPool2d", nn.AvgPool2d],
        ["AdaptiveMaxPool2d", lambda kernel, stride: nn.AdaptiveMaxPool2d(kernel)],
        [
            "FractionalMaxPool2d",
            lambda kernel, stride: nn.FractionalMaxPool2d(kernel, output_size=2),
        ],
    ],
)


class Pool2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, kernel, stride, N, C, H, W, device, op_func):
        # 初始化输入数据为随机张量
        self.inputs = {"input": torch.rand(N, C, H, W, device=device)}
        # 使用给定的操作函数初始化池化操作
        self.op_func = op_func(kernel, stride=stride)

    def forward(self, input):
        # 执行池化操作
        return self.op_func(input)


# 生成来自操作列表的PyTorch测试用例
op_bench.generate_pt_tests_from_op_list(
    pool_2d_ops_list, pool_2d_configs_short + pool_2d_configs_long, Pool2dBenchmark
)


"""
MaxPool3d和AvgPool3d运算符的微基准测试。
"""


# 针对pool-3d运算的配置
pool_3d_configs_short = op_bench.config_list(
    attr_names=["kernel", "stride", "N", "C", "D", "H", "W"],
    attrs=[
        [[3, 1, 3], [2, 1, 2], 1, 16, 16, 32, 32],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)
# 创建一个包含长配置的池化操作的配置列表，用于性能基准测试
pool_3d_configs_long = op_bench.cross_product_configs(
    kernel=[[3, 2, 3], [3, 3, 3]],  # 池化核的大小，以列表形式指定三维空间的长宽高
    stride=[[2, 2, 2]],            # 池化操作的步幅，以列表形式指定三维空间的长宽高
    N=[8, 16],                     # 输入数据的批量大小
    C=[32],                        # 输入数据的通道数
    D=[32],                        # 输入数据的深度（第一维度）
    H=[32, 64],                    # 输入数据的高度（第二维度）
    W=[32, 64],                    # 输入数据的宽度（第三维度）
    device=["cpu", "cuda"],        # 指定运行设备，可以是 CPU 或 CUDA
    tags=["long"],                 # 配置标签，用于标识此配置为长期执行的基准测试
)


# 创建池化操作的函数列表，包括不同类型的池化操作及其对应的函数或类构造器
pool_3d_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],  # 属性名称，指定为"操作名称"和"操作函数"
    attrs=[
        ["MaxPool3d", nn.MaxPool3d],  # 最大池化操作
        ["AvgPool3d", nn.AvgPool3d],  # 平均池化操作
        ["AdaptiveMaxPool3d", lambda kernel, stride: nn.AdaptiveMaxPool3d(kernel)],  # 自适应最大池化操作
        [
            "FractionalMaxPool3d",
            lambda kernel, stride: nn.FractionalMaxPool3d(kernel, output_size=2),
        ],  # 分数最大池化操作
    ],
)


# 定义池化操作的基准测试类，继承自 TorchBenchmarkBase 类
class Pool3dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, kernel, stride, N, C, D, H, W, device, op_func):
        # 初始化函数，设置输入数据和池化操作函数
        self.inputs = {"input": torch.rand(N, C, D, H, W, device=device)}  # 创建随机输入数据张量
        self.op_func = op_func(kernel, stride=stride)  # 根据参数创建池化操作对象

    def forward(self, input):
        # 前向传播函数，执行池化操作并返回结果
        return self.op_func(input)


# 从操作列表和配置列表中生成基准测试的测试用例
op_bench.generate_pt_tests_from_op_list(
    pool_3d_ops_list,  # 使用的池化操作列表
    pool_3d_configs_short + pool_3d_configs_long,  # 包含短配置和长配置的池化操作配置列表
    Pool3dBenchmark  # 使用的基准测试类
)


# 如果当前脚本作为主程序运行，则执行基准测试的运行器
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```