# `.\pytorch\benchmarks\operator_benchmark\pt\qbatchnorm_test.py`

```
# 导入名为 operator_benchmark 的模块，并重命名为 op_bench
import operator_benchmark as op_bench
# 导入 torch 库
import torch

# 定义一个包含量化批标准化操作微基准的文件

# 定义一个名为 batchnorm_configs_short 的配置列表，包含以下属性组合
batchnorm_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K"],  # 属性名称列表为 M, N, K
    attrs=[
        [1, 256, 3136],  # 具体属性值的列表
    ],
    cross_product_configs={
        "device": ["cpu"],  # 设备为 CPU
        "dtype": (torch.qint8,),  # 数据类型为 torch 的 qint8 类型
    },
    tags=["short"],  # 标签为 "short"
)

# 定义一个 QBatchNormBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class QBatchNormBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，接受 M, N, K, device, dtype 参数
    def init(self, M, N, K, device, dtype):
        self._init(M, N, K, device)
        x_scale = 0.1
        x_zero_point = 0
        # 初始化 self.inputs 字典
        self.inputs = {
            "q_input_one": torch.quantize_per_tensor(
                self.input_one, scale=x_scale, zero_point=x_zero_point, dtype=dtype
            ),
            "mean": torch.rand(N),  # 均值为 N 长度的随机张量
            "var": torch.rand(N),   # 方差为 N 长度的随机张量
            "weight": torch.rand(N),  # 权重为 N 长度的随机张量
            "bias": torch.rand(N),    # 偏置为 N 长度的随机张量
            "eps": 1e-5,               # 微小值 eps
            "Y_scale": 0.1,            # Y 的缩放因子
            "Y_zero_point": 0,         # Y 的零点
        }

    # 私有方法 _init，接受 M, N, K, device 参数
    def _init(self, M, N, K, device):
        pass

    # 前向传播方法，未实现具体逻辑
    def forward(self):
        pass

# 定义一个 QBatchNorm1dBenchmark 类，继承自 QBatchNormBenchmark 类
class QBatchNorm1dBenchmark(QBatchNormBenchmark):
    # 初始化方法，接受 M, N, K, device 参数
    def _init(self, M, N, K, device):
        self.set_module_name("QBatchNorm1d")  # 设置模块名称为 "QBatchNorm1d"
        self.input_one = torch.rand(
            M, N, K, device=device, requires_grad=self.auto_set()
        )  # 初始化 input_one 张量，大小为 M x N x K，设备为 device，是否需要梯度自动设置

    # 前向传播方法，接受多个参数，实现量化一维批标准化操作
    def forward(
        self,
        q_input_one,
        weight,
        bias,
        mean,
        var,
        eps: float,
        Y_scale: float,
        Y_zero_point: int,
    ):
        return torch.ops.quantized.batch_norm1d(
            q_input_one, weight, bias, mean, var, eps, Y_scale, Y_zero_point
        )

# 定义一个 QBatchNorm2dBenchmark 类，继承自 QBatchNormBenchmark 类
class QBatchNorm2dBenchmark(QBatchNormBenchmark):
    # 初始化方法，接受 M, N, K, device 参数
    def _init(self, M, N, K, device):
        self.set_module_name("QBatchNorm2d")  # 设置模块名称为 "QBatchNorm2d"
        # 初始化 input_one 张量，大小为 M x N x K x 1，设备为 device，最后一维为 1，需要梯度自动设置
        self.input_one = torch.rand(
            M, N, K, 1, device=device, requires_grad=self.auto_set()
        )

    # 前向传播方法，接受多个参数，实现量化二维批标准化操作
    def forward(
        self,
        q_input_one,
        weight,
        bias,
        mean,
        var,
        eps: float,
        Y_scale: float,
        Y_zero_point: int,
    ):
        return torch.ops.quantized.batch_norm2d(
            q_input_one, weight, bias, mean, var, eps, Y_scale, Y_zero_point
        )

# 生成 QBatchNorm1dBenchmark 的 PyTorch 测试用例，使用 batchnorm_configs_short 配置
op_bench.generate_pt_test(batchnorm_configs_short, QBatchNorm1dBenchmark)

# 生成 QBatchNorm2dBenchmark 的 PyTorch 测试用例，使用 batchnorm_configs_short 配置
op_bench.generate_pt_test(batchnorm_configs_short, QBatchNorm2dBenchmark)

# 如果运行的是主程序
if __name__ == "__main__":
    # 运行 operator_benchmark 模块中的 benchmark_runner 的主函数
    op_bench.benchmark_runner.main()
```