# `.\pytorch\benchmarks\operator_benchmark\pt\qconv_test.py`

```
# 从pt模块中导入configs模块
from pt import configs

# 导入operator_benchmark模块并重命名为op_bench
import operator_benchmark as op_bench
# 导入torch模块
import torch
# 导入torch中用于量化神经网络的模块
import torch.ao.nn.quantized as nnq

"""
qConv操作的微基准测试。
"""


class QConv1dBenchmark(op_bench.TorchBenchmarkBase):
    # QConv1dBenchmark类，继承自op_bench.TorchBenchmarkBase类

    # 初始化方法，设置各种参数
    # def init(self, N, IC, OC, L, G, kernel, stride, pad):
    def init(self, IC, OC, kernel, stride, N, L, device):
        G = 1  # 组数设为1
        pad = 0  # 填充设为0
        self.scale = 1.0 / 255  # 设置量化的缩放因子
        self.zero_point = 0  # 设置量化的零点
        # 创建随机输入张量X，大小为N x IC x L，数据类型为float32
        X = torch.randn(N, IC, L, dtype=torch.float32)
        # 对输入张量X进行量化，使用指定的缩放因子和零点，数据类型为quint8
        qX = torch.quantize_per_tensor(
            X, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8
        )
        # 将张量转换为NHWC格式

        # 创建随机权重张量W，大小为OC x (IC // G) x kernel，数据类型为float32
        W = torch.randn(OC, IC // G, kernel, dtype=torch.float32)
        # 对权重张量W进行量化，缩放因子为self.scale，零点设为0，数据类型为qint8
        self.qW = torch.quantize_per_tensor(
            W, scale=self.scale, zero_point=0, dtype=torch.qint8
        )

        # 设置输入字典，包含量化后的输入张量qX
        self.inputs = {"input": qX}

        # 创建QConv1d对象，设定输入通道数IC，输出通道数OC，卷积核大小kernel，步长stride，填充pad，分组数G
        self.qconv1d = nnq.Conv1d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        # 设置QConv1d对象的权重和偏置，这里偏置为None
        self.qconv1d.set_weight_bias(self.qW, None)
        # 设置QConv1d对象的缩放因子为self.scale，数据类型为double
        self.qconv1d.scale = torch.tensor(self.scale, dtype=torch.double)
        # 设置QConv1d对象的零点为self.zero_point，数据类型为int
        self.qconv1d.zero_point = torch.tensor(self.zero_point, dtype=torch.int)
        # 设置模块名称为"QConv1d"
        self.set_module_name("QConv1d")

    # 前向传播方法，接收输入input，返回QConv1d对象对输入的处理结果
    def forward(self, input):
        return self.qconv1d(input)


class QConv2dBenchmark(op_bench.TorchBenchmarkBase):
    # QConv2dBenchmark类，继承自op_bench.TorchBenchmarkBase类

    # 初始化方法，设置各种参数
    # def init(self, N, IC, OC, H, W, G, kernel, stride, pad):
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        # 调用父类的init方法，设置N, IC, OC, (H, W), G, (kernel, kernel), stride, pad等参数

        self.scale = 1.0 / 255  # 设置量化的缩放因子
        self.zero_point = 0  # 设置量化的零点
        # 创建随机输入张量X，大小为N x IC x H x W，数据类型为float32
        X = torch.randn(N, IC, H, W, dtype=torch.float32)
        # 对输入张量X进行量化，使用指定的缩放因子和零点，数据类型为quint8
        qX = torch.quantize_per_tensor(
            X, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8
        )
        # 将张量转换为NHWC格式

        # 创建随机权重张量W，大小为OC x (IC // G) x kernel x kernel，数据类型为float32
        W = torch.randn(OC, IC // G, kernel, kernel, dtype=torch.float32)
        # 对权重张量W进行量化，缩放因子为self.scale，零点设为0，数据类型为qint8
        self.qW = torch.quantize_per_tensor(
            W, scale=self.scale, zero_point=0, dtype=torch.qint8
        )

        # 设置输入字典，包含量化后的输入张量qX
        self.inputs = {"input": qX}

        # 创建QConv2d对象，设定输入通道数IC，输出通道数OC，卷积核大小kernel，步长stride，填充pad，分组数G
        self.qconv2d = nnq.Conv2d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        # 设置QConv2d对象的权重和偏置，这里偏置为None
        self.qconv2d.set_weight_bias(self.qW, None)
        # 设置QConv2d对象的缩放因子为self.scale，数据类型为double
        self.qconv2d.scale = torch.tensor(self.scale, dtype=torch.double)
        # 设置QConv2d对象的零点为self.zero_point，数据类型为int
        self.qconv2d.zero_point = torch.tensor(self.zero_point, dtype=torch.int)
        # 设置模块名称为"QConv2d"
        self.set_module_name("QConv2d")

    # 前向传播方法，接收输入input，返回QConv2d对象对输入的处理结果
    def forward(self, input):
        return self.qconv2d(input)


# 使用op_bench.generate_pt_test生成QConv1dBenchmark的测试用例，传入一系列配置
op_bench.generate_pt_test(
    configs.remove_cuda(configs.conv_1d_configs_short + configs.conv_1d_configs_long),
    QConv1dBenchmark,
)
# 使用op_bench.generate_pt_test生成QConv2dBenchmark的测试用例，传入一系列配置
op_bench.generate_pt_test(
    configs.remove_cuda(configs.conv_2d_configs_short + configs.conv_2d_configs_long),
    QConv2dBenchmark,
)

# 如果当前脚本作为主程序执行，则调用op_bench.benchmark_runner.main()方法开始运行基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```