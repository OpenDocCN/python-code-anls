# `.\pytorch\test\fx\test_common_passes.py`

```py
# Owner(s): ["oncall: fx"]

# 导入必要的库和模块
import itertools

import torch
from torch.fx.experimental.proxy_tensor import make_fx  # 导入make_fx函数
from torch.fx.graph_module import GraphModule  # 导入GraphModule类
from torch.fx.passes.dialect.common.cse_pass import CSEPass  # 导入CSEPass类

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入instantiate_parametrized_tests函数
    parametrize,  # 导入parametrize函数
    run_tests,  # 导入run_tests函数
    TestCase,  # 导入TestCase类
)


# 定义工厂函数，创建并返回一个张量
def FactoryFunctionCall(x, device):
    y = torch.full(x.shape, 3, device=device)  # 用3填充形状与x相同的张量y
    z = torch.add(y, x)  # 将张量y与x相加，结果存储在z中
    return z


# 定义一个函数，创建一个张量y，并返回x与y的和
def TorchTensorCall(x):
    y = torch.tensor(3)  # 创建一个张量y，其值为3
    return x + y  # 返回x与y的和


# 定义一个函数，将输入的张量x连接两次
def TakeList(x):
    z = torch.cat([x, x])  # 将张量x连接两次，结果存储在z中
    return z


# 定义一个函数，对一个张量a执行分割操作，返回分割后的结果
def ReturnList(x):
    a = torch.arange(10).reshape(5, 2)  # 创建一个张量a，形状为(5, 2)，值为0到9
    z = torch.split(a, [1, 4])  # 对张量a进行分割，分割点为[1, 4]，返回分割后的结果z
    return z


# 定义一个函数，对输入的张量x执行变异操作，返回结果
def Mutation(x):
    y = x + 2  # 将张量x加2，结果存储在y中
    y.add_(1)  # 将y自身加1，实现原地操作
    return x + y  # 返回x与y的和


# 定义一个函数，对输入的张量x执行两次变异操作，并返回结果
def MutationInput(x):
    x.add_(1)  # 将张量x自身加1，实现原地操作
    y = x + 2  # 将张量x加2，结果存储在y中
    return x + y  # 返回x与y的和


# 定义一个函数，对输入的张量x执行变异操作，并返回结果
def MutationFactory(x, device):
    y = torch.full(x.shape, 3, device=device)  # 用3填充形状与x相同的张量y，设备为指定的device
    y.add_(1)  # 将张量y自身加1，实现原地操作
    return x + y  # 返回x与y的和


# 定义一个函数，对输入的张量x执行变异操作，并返回结果
def MutationTorchTensorCall(x):
    y = torch.tensor(3)  # 创建一个张量y，其值为3
    y.add_(1)  # 将张量y自身加1，实现原地操作
    return x + y  # 返回x与y的和


# 定义一个函数，对输入的张量x执行变异操作，并返回结果
def MutationMetadata(x):
    x.resize_(2)  # 修改张量x的大小为(2, -1)，实现原地操作
    return x  # 返回修改后的张量x


Passes = [CSEPass]  # 创建Passes列表，包含CSEPass类的实例
Test_Cases = [
    TakeList,  # 添加TakeList函数到测试用例列表
    ReturnList,  # 添加ReturnList函数到测试用例列表
    Mutation,  # 添加Mutation函数到测试用例列表
    MutationInput,  # 添加MutationInput函数到测试用例列表
    MutationMetadata,  # 添加MutationMetadata函数到测试用例列表
    MutationTorchTensorCall,  # 添加MutationTorchTensorCall函数到测试用例列表
]
Factory_Test_Cases = [FactoryFunctionCall, MutationFactory]  # 创建Factory_Test_Cases列表，包含FactoryFunctionCall和MutationFactory函数
Devices = ["cpu"]  # 创建Devices列表，包含"cpu"
if torch.cuda.is_available():
    Devices.append("cuda")  # 如果CUDA可用，将"cuda"添加到Devices列表中


# 定义用于生成测试名称的函数
def name_fn(common_pass, f, device):
    """Names parameterized test cases."""
    return f"{type(common_pass()).__name__}_{f.__name__}_{device}"


# 定义测试类TestCommonPass，继承自TestCase类，用于测试常用的操作
@instantiate_parametrized_tests
class TestCommonPass(TestCase):
    # 参数化测试函数test_correctness，测试操作的正确性
    @parametrize(
        "common_pass,f,device", itertools.product(Passes, Test_Cases, Devices), name_fn
    )
    def test_correctness(self, common_pass, f, device):
        inp = torch.randn(10, device=device)  # 创建一个形状为(10,)的张量inp，设备为指定的device

        traced_m = make_fx(f)(inp)  # 对函数f使用make_fx进行追踪，inp作为输入
        P = common_pass()  # 创建common_pass对象

        res = P(traced_m)  # 对追踪后的模块traced_m应用common_pass
        modified_m = res.graph_module  # 获取应用common_pass后的图模块
        assert isinstance(modified_m, GraphModule)  # 断言modified_m是GraphModule的实例

        inp_copy = inp.clone()  # 克隆输入张量inp，以便测试
        expected = f(inp)  # 计算期望的输出结果
        result = modified_m(inp_copy)  # 使用修改后的模块modified_m计算结果

        self.assertEqual(result, expected)  # 断言计算结果与期望结果相等

    # 参数化测试函数test_correctness_factory，测试工厂函数的正确性
    @parametrize(
        "common_pass,f,device",
        itertools.product(Passes, Factory_Test_Cases, Devices),
        name_fn,
    )
    def test_correctness_factory(self, common_pass, f, device):
        inp = torch.randn(10, device=device)  # 创建一个形状为(10,)的张量inp，设备为指定的device
        traced_m = make_fx(f)(inp, device)  # 对工厂函数f使用make_fx进行追踪，inp和device作为输入
        P = common_pass()  # 创建common_pass对象

        res = P(traced_m)  # 对追踪后的模块traced_m应用common_pass
        modified_m = res.graph_module  # 获取应用common_pass后的图模块
        assert isinstance(modified_m, GraphModule)  # 断言modified_m是GraphModule的实例

        inp_copy = inp.clone()  # 克隆输入张量inp，以便测试
        expected = f(inp, device)  # 计算期望的输出结果
        result = modified_m(inp_copy, device)  # 使用修改后的模块modified_m计算结果

        self.assertEqual(result, expected)  # 断言计算结果与期望结果相等


# 如果作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```