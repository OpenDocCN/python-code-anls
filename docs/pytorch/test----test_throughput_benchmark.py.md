# `.\pytorch\test\test_throughput_benchmark.py`

```
```python`
# 导入必要的库和模块
import torch

# 从内部测试工具中导入运行测试、临时文件名、测试用例
from torch.testing._internal.common_utils import run_tests, TemporaryFileName, TestCase
# 从torch.utils中导入性能评估工具ThroughputBenchmark
from torch.utils import ThroughputBenchmark

# 定义一个继承自torch.jit.ScriptModule的两层神经网络模型
class TwoLayerNet(torch.jit.ScriptModule):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        # 第一层线性层，输入维度为D_in，输出维度为H
        self.linear1 = torch.nn.Linear(D_in, H)
        # 第二层线性层，输入维度为2 * H（两个H的叠加），输出维度为D_out
        self.linear2 = torch.nn.Linear(2 * H, D_out)

    @torch.jit.script_method
    def forward(self, x1, x2):
        # 计算第一个输入x1经过第一层线性层后的ReLU激活结果
        h1_relu = self.linear1(x1).clamp(min=0)
        # 计算第二个输入x2经过第一层线性层后的ReLU激活结果
        h2_relu = self.linear1(x2).clamp(min=0)
        # 将两个ReLU激活结果拼接在一起
        cat = torch.cat((h1_relu, h2_relu), 1)
        # 计算拼接后的结果经过第二层线性层的输出
        y_pred = self.linear2(cat)
        # 返回预测结果
        return y_pred

# 定义一个继承自torch.nn.Module的两层神经网络模型
class TwoLayerNetModule(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        # 第一层线性层，输入维度为D_in，输出维度为H
        self.linear1 = torch.nn.Linear(D_in, H)
        # 第二层线性层，输入维度为2 * H（两个H的叠加），输出维度为D_out
        self.linear2 = torch.nn.Linear(2 * H, D_out)

    def forward(self, x1, x2):
        # 计算第一个输入x1经过第一层线性层后的ReLU激活结果
        h1_relu = self.linear1(x1).clamp(min=0)
        # 计算第二个输入x2经过第一层线性层后的ReLU激活结果
        h2_relu = self.linear1(x2).clamp(min=0)
        # 将两个ReLU激活结果拼接在一起
        cat = torch.cat((h1_relu, h2_relu), 1)
        # 计算拼接后的结果经过第二层线性层的输出
        y_pred = self.linear2(cat)
        # 返回预测结果
        return y_pred

# 定义一个测试类，继承自TestCase
class TestThroughputBenchmark(TestCase):
    # 线性测试方法
    def linear_test(self, Module, profiler_output_path=""):
        # 定义输入数据的维度和大小
        D_in = 10
        H = 5
        D_out = 15
        B = 8
        NUM_INPUTS = 2

        # 创建指定模型类型的实例
        module = Module(D_in, H, D_out)

        inputs = []

        # 生成多组输入数据
        for i in range(NUM_INPUTS):
            inputs.append([torch.randn(B, D_in), torch.randn(B, D_in)])

        # 创建性能评估实例
        bench = ThroughputBenchmark(module)

        # 将所有输入数据添加到性能评估实例中
        for input in inputs:
            # 可以同时传递位置参数和关键字参数
            bench.add_input(input[0], x2=input[1])

        # 分别对每组输入数据进行测试
        fornum_warmup_iters=100,  # 预热迭代次数
            num_iters=1000,  # 性能测试迭代次数
            profiler_output_path=profiler_output_path,  # 性能分析输出路径
        )

        print(stats)  # 打印性能统计信息

    def test_script_module(self):
        self.linear_test(TwoLayerNet)  # 测试脚本模块

    def test_module(self):
        self.linear_test(TwoLayerNetModule)  # 测试普通模块

    def test_profiling(self):
        with TemporaryFileName() as fname:
            self.linear_test(TwoLayerNetModule, profiler_output_path=fname)  # 测试性能分析


if __name__ == "__main__":
    run_tests()  # 运行所有测试
```