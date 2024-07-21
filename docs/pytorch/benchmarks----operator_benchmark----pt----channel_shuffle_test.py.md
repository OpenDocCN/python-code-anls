# `.\pytorch\benchmarks\operator_benchmark\pt\channel_shuffle_test.py`

```
# 导入 operator_benchmark 库和 PyTorch 库
import operator_benchmark as op_bench
import torch

"""Microbenchmarks for channel_shuffle operator."""

# 长配置项用于 PT channel_shuffle 操作符的微基准测试
channel_shuffle_long_configs = op_bench.cross_product_configs(
    batch_size=[4, 8],  # 批量大小的列表
    channels_per_group=[32, 64],  # 每组的通道数列表
    height=[32, 64],  # 图像高度列表
    width=[32, 64],  # 图像宽度列表
    groups=[4, 8],  # 分组数列表
    channel_last=[True, False],  # 是否为通道最后格式的布尔值列表
    tags=["long"],  # 长标记的列表
)

# 短配置项用于 PT channel_shuffle 操作符的微基准测试
channel_shuffle_short_configs = op_bench.config_list(
    attr_names=["batch_size", "channels_per_group", "height", "width", "groups"],  # 属性名称列表
    attrs=[
        [2, 16, 16, 16, 2],  # 属性值列表
        [2, 32, 32, 32, 2],
        [4, 32, 32, 32, 4],
        [4, 64, 64, 64, 4],
        [8, 64, 64, 64, 8],
        [16, 64, 64, 64, 16],
    ],
    cross_product_configs={
        "channel_last": [True, False],  # 通道最后格式的布尔值列表
    },
    tags=["short"],  # 短标记的列表
)

# ChannelSHuffleBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class ChannelSHuffleBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, batch_size, channels_per_group, height, width, groups, channel_last):
        # 计算总的通道数
        channels = channels_per_group * groups
        # 定义数据形状
        data_shape = (batch_size, channels, height, width)
        # 创建随机数据张量
        input_data = torch.rand(data_shape)
        # 如果是通道最后格式，将数据转换为连续内存格式的通道最后格式
        if channel_last:
            input_data = input_data.contiguous(memory_format=torch.channels_last)
        # 设置输入数据和分组数
        self.inputs = {"input_data": input_data, "groups": groups}
        # 设置模块名称为 "channel_shuffle"
        self.set_module_name("channel_shuffle")

    def forward(self, input_data, groups: int):
        # 调用 torch 库中的 channel_shuffle 函数进行前向传播
        return torch.channel_shuffle(input_data, groups)

# 生成基于 PyTorch 的测试用例
op_bench.generate_pt_test(
    channel_shuffle_short_configs + channel_shuffle_long_configs,  # 使用短和长配置项生成测试用例
    ChannelSHuffleBenchmark,  # 使用 ChannelSHuffleBenchmark 类进行测试
)

# 如果当前脚本作为主程序运行，调用 operator_benchmark 库中的 benchmark_runner.main() 函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```