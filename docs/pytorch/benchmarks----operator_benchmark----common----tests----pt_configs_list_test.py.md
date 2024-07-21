# `.\pytorch\benchmarks\operator_benchmark\common\tests\pt_configs_list_test.py`

```
# 导入名为operator_benchmark的库，并将其命名为op_bench
import operator_benchmark as op_bench

# 导入torch库
import torch

# 定义了关于元素级加法操作的微基准测试。支持Caffe2和PyTorch两种框架。
"""Microbenchmarks for element-wise Add operator. Supports both Caffe2/PyTorch."""

# 创建了三组配置列表，每组包含不同的M、N、K数值组合
add_short_configs = op_bench.config_list(
    # 属性名称列表为["M", "N", "K"]
    attr_names=["M", "N", "K"],
    # 具体的属性值组合
    attrs=[
        [8, 16, 32],
        [16, 16, 64],
        [64, 64, 128],
    ],
    # 使用交叉配置的方式，定义设备和数据类型的组合
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.float, torch.float64],
    },
    # 标记为"short"
    tags=["short"],
)

# 定义了一个继承自op_bench.TorchBenchmarkBase的AddBenchmark类
class AddBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，接受M、N、K、device、dtype等参数
    def init(self, M, N, K, device, dtype):
        # 初始化input_one张量，形状为(M, N, K)，随机初始化在指定设备上，并需要梯度
        self.input_one = torch.rand(
            M, N, K, device=device, dtype=dtype, requires_grad=True
        )
        # 初始化input_two张量，形状与input_one相同，随机初始化在指定设备上
        self.input_two = torch.rand(M, N, K, device=device, dtype=dtype)
        # 设置当前模块名称为"add"
        self.set_module_name("add")

    # 前向计算方法
    def forward(self):
        # 返回input_one和input_two张量的元素级加法结果
        return torch.add(self.input_one, self.input_two)

# 生成PyTorch的性能测试用例，使用之前定义的add_short_configs和AddBenchmark类
op_bench.generate_pt_test(add_short_configs, AddBenchmark)

# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 运行operator_benchmark库中的benchmark_runner的主函数
    op_bench.benchmark_runner.main()
```