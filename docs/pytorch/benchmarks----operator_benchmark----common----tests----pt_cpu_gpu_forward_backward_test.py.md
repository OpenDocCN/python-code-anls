# `.\pytorch\benchmarks\operator_benchmark\common\tests\pt_cpu_gpu_forward_backward_test.py`

```py
# 导入名为 operator_benchmark 的模块，并将其重命名为 op_bench
import operator_benchmark as op_bench

# 导入名为 torch 的模块
import torch

# 定义一个名为 add_configs 的变量，用于存储操作的配置列表
add_configs = op_bench.cross_product_configs(
    # 操作矩阵的维度 M, N, K 设置为 8
    M=[8], N=[8], K=[8],
    # 设备选择包括 "cuda" 和 "cpu"
    device=["cuda", "cpu"],
    # 添加标签 "short"，用于描述配置
    tags=["short"]
)

# 定义一个名为 AddBenchmark 的类，继承自 op_bench.TorchBenchmarkBase
class AddBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，接受 M, N, K, device 四个参数
    def init(self, M, N, K, device):
        # 创建一个 MxNxK 大小的随机张量 input_one，设备为指定的 device，并设置 requires_grad=True
        self.input_one = torch.rand(M, N, K, device=device, requires_grad=True)
        # 创建一个 MxNxK 大小的随机张量 input_two，设备为指定的 device，并设置 requires_grad=True
        self.input_two = torch.rand(M, N, K, device=device, requires_grad=True)
        # 设置当前测试模块的名称为 "add"
        self.set_module_name("add")

    # 前向传播方法，执行张量 input_one 和 input_two 的加法操作
    def forward(self):
        return torch.add(self.input_one, self.input_two)

# 使用 op_bench 模块生成基准测试用例，参数为 add_configs 和 AddBenchmark 类
op_bench.generate_pt_test(add_configs, AddBenchmark)

# 使用 op_bench 模块生成梯度测试用例，参数为 add_configs 和 AddBenchmark 类
op_bench.generate_pt_gradient_test(add_configs, AddBenchmark)

# 当脚本作为主程序运行时，调用 op_bench 模块的 benchmark_runner.main() 方法启动基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```