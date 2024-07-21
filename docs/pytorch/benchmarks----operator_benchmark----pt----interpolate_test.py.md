# `.\pytorch\benchmarks\operator_benchmark\pt\interpolate_test.py`

```py
# 导入名为 operator_benchmark 的别名 op_bench 的模块
import operator_benchmark as op_bench
# 导入 torch 模块
import torch

"""Microbenchmarks for interpolate operator."""

# 定义 InterpolateBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class InterpolateBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法
    def init(
        self,
        input_size,
        output_size,
        channels_last=False,
        mode="linear",
        dtype=torch.float,
    ):
        # 生成指定大小的随机整数张量 input_image，设备为 CPU
        input_image = torch.randint(
            0,
            256,
            size=input_size,
            dtype=dtype,
            device="cpu",
            requires_grad=self.auto_set(),
        )
        # 如果 channels_last 为 True，则根据输入张量的维数设置 channels_last 内存格式
        if channels_last:
            if input_image.ndim == 4:
                input_image = input_image.contiguous(memory_format=torch.channels_last)
            elif input_image.ndim == 5:
                input_image = input_image.contiguous(
                    memory_format=torch.channels_last_3d
                )
            else:
                # 如果维数不在支持的范围内，抛出 ValueError 异常
                raise ValueError(
                    f"Can not set channels_last to the input of {input_image.ndim} dims"
                )

        # 如果 mode 为 "nearest"，则 align_corners 设为 None；否则设为 False
        align_corners = None if mode == "nearest" else False

        # 如果 mode 为 "linear"，根据 input_image 的维数选择相应的插值模式
        if mode == "linear":
            mode = {
                3: "linear",
                4: "bilinear",
                5: "trilinear",
            }[input_image.ndim]

        # 设置 inputs 字典，包含 input_image、output_size、mode 和 align_corners 参数
        self.inputs = {
            "input_image": input_image,
            "output_size": output_size,
            "mode": mode,
            "align_corners": align_corners,
        }

        # 设置模块名称为 "interpolate"
        self.set_module_name("interpolate")

    # 前向方法，接收 input_image、output_size、mode 和 align_corners 参数
    def forward(self, input_image, output_size, mode, align_corners):
        # 调用 torch.nn.functional.interpolate 执行插值操作
        return torch.nn.functional.interpolate(
            input_image, size=output_size, mode=mode, align_corners=align_corners
        )


# 创建 config_short 变量，包含多个配置的列表，用于性能测试
config_short = op_bench.config_list(
    attr_names=["input_size", "output_size"],
    attrs=[
        [(1, 3, 60, 40), (24, 24)],
        [(1, 3, 600, 400), (240, 240)],
        [(1, 3, 320, 320), (256, 256)],
        [(1, 1, 60, 40), (24, 24)],
        [(1, 1, 600, 400), (240, 240)],
        [(1, 1, 320, 320), (256, 256)],
    ],
    cross_product_configs={
        "channels_last": [True, False],
        "mode": ["nearest", "linear", "bicubic"],
    },
    tags=["short"],
)

# 将额外的配置添加到 config_short 中，用于测试不同的 dtype
config_short += op_bench.config_list(
    attr_names=["input_size", "output_size"],
    attrs=[
        [(1, 3, 60, 40), (24, 24)],
        [(1, 3, 600, 400), (240, 240)],
        [(1, 3, 320, 320), (256, 256)],
        [(1, 1, 60, 40), (24, 24)],
        [(1, 1, 600, 400), (240, 240)],
        [(1, 1, 320, 320), (256, 256)],
    ],
    cross_product_configs={
        "channels_last": [True, False],
        "mode": [
            "nearest",
        ],
        "dtype": [
            torch.uint8,
        ],
    },
    tags=["short"],
)

# 创建 config_long 变量，包含另一组用于性能测试的配置列表
config_long = op_bench.config_list(
    attr_names=["input_size", "output_size"],
    attrs=[
        # List of attribute configurations, each containing two tuples specifying size and stride
        [(1, 3, 320, 320), (512, 512)],   # Configuration 1: Size (1, 3, 320, 320), Stride (512, 512)
        [(1, 3, 500, 500), (256, 256)],   # Configuration 2: Size (1, 3, 500, 500), Stride (256, 256)
        [(1, 3, 500, 500), (800, 800)],   # Configuration 3: Size (1, 3, 500, 500), Stride (800, 800)
        [(1, 1, 320, 320), (512, 512)],   # Configuration 4: Size (1, 1, 320, 320), Stride (512, 512)
        [(1, 1, 500, 500), (256, 256)],   # Configuration 5: Size (1, 1, 500, 500), Stride (256, 256)
        [(1, 1, 500, 500), (800, 800)],   # Configuration 6: Size (1, 1, 500, 500), Stride (800, 800)
        # vectorization test-case
        [(2, 128, 64, 46), (128, 128)],   # Vectorization test-case: Size (2, 128, 64, 46), Stride (128, 128)
        [(2, 128, 64, 46), (32, 24)],     # Vectorization test-case: Size (2, 128, 64, 46), Stride (32, 24)
    ],
    cross_product_configs={
        # Dictionary defining configurations for cross product operation
        "channels_last": [True, False],   # Two configurations for 'channels_last': True and False
        "mode": ["nearest", "linear", "bicubic"],   # Three modes for interpolation: nearest, linear, bicubic
    },
    tags=["long"],   # List of tags associated with these configurations (e.g., 'long')
# 定义用于短配置的基准配置列表
config_short = op_bench.config_list(
    # 定义属性名称列表为输入大小和输出大小
    attr_names=["input_size", "output_size"],
    # 定义属性组合列表
    attrs=[
        # 第一个配置：输入大小 (4, 512, 320)，输出大小 (256,)
        [(4, 512, 320), (256,)],
        # 第二个配置：输入大小 (4, 512, 320)，输出大小 (512,)
        [(4, 512, 320), (512,)],
    ],
    # 定义交叉产品配置，这里只有 mode 属性的值是 ["nearest", "linear"]
    cross_product_configs={
        "mode": ["nearest", "linear"],
    },
    # 定义标签为 ["long"]
    tags=["long"],
)

# 定义用于三维配置的基准配置列表
config_3d = op_bench.config_list(
    # 定义属性名称列表为输入大小和输出大小
    attr_names=["input_size", "output_size"],
    # 定义属性组合列表
    attrs=[
        # 第一个配置：输入大小 (1, 3, 16, 320, 320)，输出大小 (8, 256, 256)
        [(1, 3, 16, 320, 320), (8, 256, 256)],
        # 第二个配置：输入大小 (1, 3, 16, 320, 320)，输出大小 (32, 512, 512)
        [(1, 3, 16, 320, 320), (32, 512, 512)],
        # 第三个配置：输入大小 (1, 16, 32, 64, 64)，输出大小 (16, 32, 32)
        # 这是一个矢量化测试案例
        [(1, 16, 32, 64, 64), (16, 32, 32)],
        # 第四个配置：输入大小 (1, 16, 32, 64, 64)，输出大小 (64, 128, 128)
        [(1, 16, 32, 64, 64), (64, 128, 128)],
    ],
    # 定义交叉产品配置，包括 channels_last 属性值为 [True, False] 和 mode 属性的值为 ["nearest", "linear"]
    cross_product_configs={
        "channels_last": [True, False],
        "mode": ["nearest", "linear"],
    },
    # 定义标签为 ["long"]
    tags=["long"],
)

# 对于每个配置（config_short, config_long, config_3d, config_5d），生成对应的 PyTorch 测试用例，并使用 InterpolateBenchmark 类
for config in (config_short, config_long, config_3d, config_5d):
    op_bench.generate_pt_test(config, InterpolateBenchmark)

# 如果当前脚本被直接运行，则执行 op_bench 的基准测试运行器
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```