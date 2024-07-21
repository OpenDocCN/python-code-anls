# `.\pytorch\benchmarks\operator_benchmark\pt\qobserver_test.py`

```py
# 导入自定义的性能基准库 `operator_benchmark` 和 PyTorch 库
import operator_benchmark as op_bench
import torch
import torch.ao.quantization.observer as obs

# 定义用于量化观察器（quantization observer）短配置的字典
qobserver_short_configs_dict = {
    "attr_names": ("C", "M", "N", "dtype", "device"),  # 属性名元组
    "attrs": (
        (3, 512, 512, torch.quint8, "cpu"),   # 具体属性值元组1
        (3, 512, 512, torch.quint8, "cuda"),  # 具体属性值元组2
    ),
    "tags": ("short",),  # 标签元组
}

# 定义用于量化直方图观察器短配置的字典
q_hist_observer_short_configs_dict = {
    "attr_names": ("C", "M", "N", "dtype", "device"),  # 属性名元组
    "attrs": ((3, 512, 512, torch.quint8, "cpu"),),  # 具体属性值元组
    "tags": ("short",),  # 标签元组
}

# 定义用于量化观察器长配置的字典
qobserver_long_configs_dict = {
    "C": (32, 64),  # 通道数范围
    "M": (256, 1024),  # M 维度范围
    "N": (256, 1024),  # N 维度范围
    "device": ("cpu", "cuda"),  # 设备类型元组
    "dtype": (torch.quint8,),  # 数据类型固定为 quint8
    "tags": ("long",),  # 标签元组
}

# 定义用于量化直方图观察器长配置的字典
q_hist_observer_long_configs_dict = {
    "C": (1, 3, 8),  # 通道数范围
    "M": (256, 1024),  # M 维度范围
    "N": (256, 1024),  # N 维度范围
    "device": ("cpu",),  # 只在 CPU 上运行
    "dtype": (torch.quint8,),  # 数据类型固定为 quint8
    "tags": ("long",),  # 标签元组
}

# 创建用于量化观察器的每个张量配置的短列表
qobserver_per_tensor_configs_short = op_bench.config_list(
    cross_product_configs={
        "qscheme": (torch.per_tensor_affine, torch.per_tensor_symmetric)  # 量化方案选择
    },
    **qobserver_short_configs_dict,  # 使用短配置字典
)

# 创建用于量化观察器的每个张量配置的长列表
qobserver_per_tensor_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_tensor_affine, torch.per_tensor_symmetric),  # 量化方案选择
    **qobserver_long_configs_dict,  # 使用长配置字典
)

# 创建用于量化观察器的每个通道配置的短列表
qobserver_per_channel_configs_short = op_bench.config_list(
    cross_product_configs={
        "qscheme": (torch.per_channel_affine, torch.per_channel_symmetric)  # 量化方案选择
    },
    **qobserver_short_configs_dict,  # 使用短配置字典
)

# 创建用于量化观察器的每个通道配置的长列表
qobserver_per_channel_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_channel_affine, torch.per_channel_symmetric),  # 量化方案选择
    **qobserver_long_configs_dict,  # 使用长配置字典
)

# 创建用于量化直方图观察器的每个张量配置的短列表
q_hist_observer_per_tensor_configs_short = op_bench.config_list(
    cross_product_configs={
        "qscheme": (torch.per_tensor_affine, torch.per_tensor_symmetric)  # 量化方案选择
    },
    **q_hist_observer_short_configs_dict,  # 使用短配置字典
)

# 创建用于量化直方图观察器的每个张量配置的长列表
q_hist_observer_per_tensor_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_tensor_affine, torch.per_tensor_symmetric),  # 量化方案选择
    **q_hist_observer_long_configs_dict,  # 使用长配置字典
)

# 创建量化观察器的每个张量操作列表
qobserver_per_tensor_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],  # 属性名列表
    attrs=[
        ["MinMaxObserver", obs.MinMaxObserver],  # 最小-最大观察器
        ["MovingAverageMinMaxObserver", obs.MovingAverageMinMaxObserver],  # 移动平均最小-最大观察器
    ],
)

# 创建量化观察器的每个通道操作列表
qobserver_per_channel_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],  # 属性名列表
    attrs=[
        ["PerChannelMinMaxObserver", obs.PerChannelMinMaxObserver],  # 每通道最小-最大观察器
        [
            "MovingAveragePerChannelMinMaxObserver",
            obs.MovingAveragePerChannelMinMaxObserver,  # 移动平均每通道最小-最大观察器
        ],
    ],
)

# 创建量化直方图观察器的操作列表
q_hist_observer_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],  # 属性名列表
    attrs=[
        ["HistogramObserver", obs.HistogramObserver],  # 直方图观察器
        ["HistogramObserverCalculateQparams", obs.HistogramObserver],  # 计算量化参数的直方图观察器
    ],
)

# 定义量化观察器基准类，继承自 `op_bench.TorchBenchmarkBase`
class QObserverBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，用于设置模型的输入和操作函数
    def init(self, C, M, N, dtype, qscheme, op_func, device):
        # 初始化输入字典，包含一个名为"f_input"的键，值为形状为(C, M, N)的随机张量，存储在给定设备上
        self.inputs = {"f_input": torch.rand(C, M, N, device=device)}
        # 设置操作函数，根据给定的dtype和qscheme创建实例，并将其移到指定设备上
        self.op_func = op_func(dtype=dtype, qscheme=qscheme).to(device)

    # 前向传播方法，接收输入f_input，执行操作函数并返回结果
    def forward(self, f_input):
        # 使用设置的操作函数处理输入f_input
        self.op_func(f_input)
        # 调用操作函数的calculate_qparams方法，返回其结果
        return self.op_func.calculate_qparams()
class QObserverBenchmarkCalculateQparams(op_bench.TorchBenchmarkBase):
    # 定义 QObserverBenchmarkCalculateQparams 类，继承自 op_bench.TorchBenchmarkBase
    def init(self, C, M, N, dtype, qscheme, op_func, device):
        # 初始化方法，接受参数 C, M, N, dtype, qscheme, op_func, device
        self.f_input = torch.rand(C, M, N, device=device)
        # 创建一个 C x M x N 的随机张量 self.f_input，放置在指定的设备上
        self.q_observer = op_func(dtype=dtype, qscheme=qscheme).to(device)
        # 使用 op_func 创建一个量化观察器 self.q_observer，设置数据类型和量化方案，然后将其移动到指定设备
        self.q_observer(self.f_input)
        # 对 self.f_input 运行 self.q_observer，进行量化观察
        self.inputs = {}
        # 初始化一个空字典 self.inputs

    def forward(self):
        # 前向方法
        return self.q_observer.calculate_qparams()
        # 调用 self.q_observer 的 calculate_qparams 方法，返回结果


op_bench.generate_pt_tests_from_op_list(
    qobserver_per_tensor_list,
    qobserver_per_tensor_configs_short + qobserver_per_tensor_configs_long,
    QObserverBenchmark,
)
# 使用 qobserver_per_tensor_list 和 qobserver_per_tensor_configs_short + qobserver_per_tensor_configs_long 生成基于 QObserverBenchmark 的 PyTorch 测试


op_bench.generate_pt_tests_from_op_list(
    qobserver_per_channel_list,
    qobserver_per_channel_configs_short + qobserver_per_channel_configs_long,
    QObserverBenchmark,
)
# 使用 qobserver_per_channel_list 和 qobserver_per_channel_configs_short + qobserver_per_channel_configs_long 生成基于 QObserverBenchmark 的 PyTorch 测试


op_bench.generate_pt_tests_from_op_list(
    q_hist_observer_list,
    q_hist_observer_per_tensor_configs_short + q_hist_observer_per_tensor_configs_long,
    QObserverBenchmarkCalculateQparams,
)
# 使用 q_hist_observer_list 和 q_hist_observer_per_tensor_configs_short + q_hist_observer_per_tensor_configs_long 生成基于 QObserverBenchmarkCalculateQparams 的 PyTorch 测试


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
    # 如果作为主程序运行，则调用 op_bench.benchmark_runner.main() 来执行基准测试
```