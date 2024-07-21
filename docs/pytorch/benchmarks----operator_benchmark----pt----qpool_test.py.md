# `.\pytorch\benchmarks\operator_benchmark\pt\qpool_test.py`

```py
# 导入operator_benchmark库和torch库
import operator_benchmark as op_bench
import torch

# 创建长配置列表用于2D池化基准测试
qpool2d_long_configs = op_bench.config_list(
    attrs=(
        # 输入通道数C，输入高度H，输入宽度W，池化核大小k，步长s，填充p
        (1, 3, 3, (3, 3), (1, 1), (0, 0)),  # 虚拟数据配置 # noqa: E201,E241
        (3, 64, 64, (3, 3), (2, 2), (1, 1)),  # 虚拟数据配置 # noqa: E201,E241
        # 使用VGG16原始输入形状进行池化：(-1, 3, 224, 224)
        (64, 224, 224, (2, 2), (2, 2), (0, 0)),  # MaxPool2d-4  # noqa: E201
        (256, 56, 56, (2, 2), (2, 2), (0, 0)),  # MaxPool2d-16 # noqa: E241
    ),
    attr_names=("C", "H", "W", "k", "s", "p"),  # 输入布局 # 池化参数
    cross_product_configs={
        "N": (1, 4),  # 批量大小N
        "contig": (False, True),  # 是否连续内存
        "dtype": (torch.quint8,),  # 数据类型
    },
    tags=("long",),  # 标签为长时间运行
)

# 创建短配置列表用于2D池化基准测试
qpool2d_short_configs = op_bench.config_list(
    attrs=((1, 3, 3, (3, 3), (1, 1), (0, 0)),),  # 虚拟数据配置
    attr_names=("C", "H", "W", "k", "s", "p"),  # 输入布局 # 池化参数
    cross_product_configs={
        "N": (2,),  # 批量大小N
        "contig": (True,),  # 是否连续内存
        "dtype": (torch.qint32, torch.qint8, torch.quint8),  # 数据类型
    },
    tags=("short",),  # 标签为短时间运行
)

# 创建长配置列表用于自适应平均池化基准测试
qadaptive_avgpool2d_long_configs = op_bench.cross_product_configs(
    input_size=(
        (112, 112),  # MaxPool2d-9
    ),
    output_size=(
        (448, 448),
        (224, 224),  # MaxPool2d-4
        (112, 112),  # MaxPool2d-9
        (56, 56),  # MaxPool2d-16 # noqa: E201,E241
        (14, 14),  # MaxPool2d-30 # noqa: E201,E241
    ),
    N=(1, 4),  # 批量大小N
    C=(1, 3, 64, 128),  # 通道数C
    contig=(False, True),  # 是否连续内存
    dtype=(torch.quint8,),  # 数据类型
    tags=("long",),  # 标签为长时间运行
)

# 创建短配置列表用于自适应平均池化基准测试
qadaptive_avgpool2d_short_configs = op_bench.config_list(
    attrs=((4, 3, (224, 224), (112, 112), True),),
    attr_names=("N", "C", "input_size", "output_size", "contig"),
    cross_product_configs={
        "dtype": (torch.qint32, torch.qint8, torch.quint8),  # 数据类型
    },
    tags=("short",),  # 标签为短时间运行
)

# 定义_QPool2dBenchmarkBase类，继承自op_bench.TorchBenchmarkBase类
class _QPool2dBenchmarkBase(op_bench.TorchBenchmarkBase):
    def setup(self, N, C, H, W, dtype, contig):
        # 输入
        if N == 0:
            f_input = (torch.rand(C, H, W) - 0.5) * 256
        else:
            f_input = (torch.rand(N, C, H, W) - 0.5) * 256

        scale = 1.0
        zero_point = 0

        # 对张量进行量化
        self.q_input = torch.quantize_per_tensor(
            f_input, scale=scale, zero_point=zero_point, dtype=dtype
        )
        if not contig:
            # 将张量排列为NHWC然后再排列回来，使其变为非连续内存
            if N == 0:
                self.q_input = self.q_input.permute(1, 2, 0).contiguous()
                self.q_input = self.q_input.permute(2, 0, 1)
            else:
                self.q_input = self.q_input.permute(0, 2, 3, 1).contiguous()
                self.q_input = self.q_input.permute(0, 3, 1, 2)

        self.inputs = {"q_input": self.q_input}
    # 定义一个方法 `forward`，用于前向传播模型
    def forward(self, q_input):
        # 调用类中的 `pool_op` 方法，传入参数 `q_input`，并返回其结果
        return self.pool_op(q_input)
# 定义 QMaxPool2dBenchmark 类，继承自 _QPool2dBenchmarkBase 类
class QMaxPool2dBenchmark(_QPool2dBenchmarkBase):
    # 初始化方法，接受 N, C, H, W, k, s, p, contig, dtype 参数
    def init(self, N, C, H, W, k, s, p, contig, dtype):
        # 使用 torch.nn.MaxPool2d 创建最大池化操作对象，配置如下参数：
        #   kernel_size: 池化核大小为 k
        #   stride: 步长为 s
        #   padding: 填充大小为 p
        #   dilation: 膨胀系数为 (1, 1)
        #   ceil_mode: 不启用上取整模式
        #   return_indices: 不返回索引
        self.pool_op = torch.nn.MaxPool2d(
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=(1, 1),
            ceil_mode=False,
            return_indices=False,
        )
        # 调用父类的 setup 方法，设置 N, C, H, W, dtype, contig 参数
        super().setup(N, C, H, W, dtype, contig)


# 定义 QAvgPool2dBenchmark 类，继承自 _QPool2dBenchmarkBase 类
class QAvgPool2dBenchmark(_QPool2dBenchmarkBase):
    # 初始化方法，接受 N, C, H, W, k, s, p, contig, dtype 参数
    def init(self, N, C, H, W, k, s, p, contig, dtype):
        # 使用 torch.nn.AvgPool2d 创建平均池化操作对象，配置如下参数：
        #   kernel_size: 池化核大小为 k
        #   stride: 步长为 s
        #   padding: 填充大小为 p
        #   ceil_mode: 不启用上取整模式
        self.pool_op = torch.nn.AvgPool2d(
            kernel_size=k, stride=s, padding=p, ceil_mode=False
        )
        # 调用父类的 setup 方法，设置 N, C, H, W, dtype, contig 参数
        super().setup(N, C, H, W, dtype, contig)


# 定义 QAdaptiveAvgPool2dBenchmark 类，继承自 _QPool2dBenchmarkBase 类
class QAdaptiveAvgPool2dBenchmark(_QPool2dBenchmarkBase):
    # 初始化方法，接受 N, C, input_size, output_size, contig, dtype 参数
    def init(self, N, C, input_size, output_size, contig, dtype):
        # 使用 torch.nn.AdaptiveAvgPool2d 创建自适应平均池化操作对象，配置如下参数：
        #   output_size: 输出大小为 output_size
        self.pool_op = torch.nn.AdaptiveAvgPool2d(output_size=output_size)
        # 调用父类的 setup 方法，设置 N, C, *input_size, dtype, contig 参数
        super().setup(N, C, *input_size, dtype=dtype, contig=contig)


# 使用 op_bench.generate_pt_test 方法生成基于 qadaptive_avgpool2d_short_configs 和
# qadaptive_avgpool2d_long_configs 的测试，并绑定到 QAdaptiveAvgPool2dBenchmark 类
op_bench.generate_pt_test(
    qadaptive_avgpool2d_short_configs + qadaptive_avgpool2d_long_configs,
    QAdaptiveAvgPool2dBenchmark,
)

# 使用 op_bench.generate_pt_test 方法生成基于 qpool2d_short_configs 和 qpool2d_long_configs
# 的测试，并绑定到 QAvgPool2dBenchmark 类
op_bench.generate_pt_test(
    qpool2d_short_configs + qpool2d_long_configs, QAvgPool2dBenchmark
)

# 使用 op_bench.generate_pt_test 方法生成基于 qpool2d_short_configs 和 qpool2d_long_configs
# 的测试，并绑定到 QMaxPool2dBenchmark 类
op_bench.generate_pt_test(
    qpool2d_short_configs + qpool2d_long_configs, QMaxPool2dBenchmark
)

# 如果当前脚本作为主程序运行，调用 op_bench.benchmark_runner.main() 方法执行基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```