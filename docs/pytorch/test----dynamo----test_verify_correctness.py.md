# `.\pytorch\test\dynamo\test_verify_correctness.py`

```py
# Owner(s): ["module: dynamo"]
# 引入operator模块，用于操作符重载和函数调用
import operator

# 引入PyTorch库
import torch

# 引入torch._dynamo子模块及其config子模块
import torch._dynamo
import torch._dynamo.config as config

# 引入torch._dynamo.test_case中的same函数
from torch._dynamo.testing import same

# 引入torch.fx._lazy_graph_module中的_force_skip_lazy_graph_module函数
from torch.fx._lazy_graph_module import _force_skip_lazy_graph_module


# 定义一个名为Seq的神经网络模块
class Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个序列容器，包含两个线性层和两个激活函数
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    # 前向传播函数，将输入x传递给self.layers的序列容器
    def forward(self, x):
        return self.layers(x)


# 定义一个名为Conv_Bn_Relu的神经网络模块
class Conv_Bn_Relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # 创建一个卷积层，使用无偏置的二维卷积
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # 创建一个批归一化层，设置eps为0.001
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        # 创建一个ReLU激活函数层
        self.relu = torch.nn.ReLU()

    # 前向传播函数，依次执行卷积、批归一化和ReLU激活操作
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# 定义一个名为toy_example的函数，实现给定的数学运算
def toy_example(a, b):
    # 计算x的值
    x = a / (torch.abs(a) + 1)
    # 如果b的元素之和小于0，则将b取反
    if b.sum() < 0:
        b = b * -1
    return x * b


# 定义一个名为transform的函数，接受torch.fx.GraphModule类型的参数并返回相同类型的对象
def transform(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # 遍历图中的每个节点
    for node in gm.graph.nodes:
        # 检查当前节点是否为函数调用类型
        if node.op == "call_function":
            # 如果调用的函数是operator.mul，则将其替换为operator.add
            if node.target == operator.mul:
                node.target = operator.add

    # 对图进行一些检查以确保其良好形式
    gm.graph.lint()

    # 重新编译修改后的图
    gm.recompile()
    
    # 返回经过变换后的图模块对象
    return gm


# 使用config.patch修饰器设置"verify_correctness"属性为True
@config.patch("verify_correctness", True)
# 定义一个名为TestVerifyCorrectness的测试类，继承自torch._dynamo.test_case.TestCase
class TestVerifyCorrectness(torch._dynamo.test_case.TestCase):
    # 定义一个测试方法test_example_inputs
    def test_example_inputs(self):
        # 定义一个名为fn的函数，接受a、bc和d作为参数
        def fn(a, bc, d):
            # 解包bc，得到b和c
            b, c = bc
            # 返回计算结果
            return a / d - b / c

        # 定义一个名为compiler_fn的函数，接受graph和example_inputs作为参数
        def compiler_fn(graph, example_inputs):
            # 使用example_inputs作为参数调用graph，记录返回值到r1
            nonlocal r1
            r1 = graph(*example_inputs)[0]
            # 返回graph的forward方法
            return graph.forward

        # 创建长度为2的空张量a，并填充为1
        a = torch.empty(2).fill_(1)
        # 创建长度为2的空张量b，并填充为2
        b = torch.empty(2).fill_(2)
        # 创建长度为2的空张量c，并填充为3
        c = torch.empty(2).fill_(3)
        # 设置标量d为4
        d = 4
        # 初始化变量r1为None
        r1 = None
        # 调用fn函数，计算r2的值
        r2 = fn(a, (b, c), d)
        # 使用torch._dynamo.optimize_assert优化compiler_fn函数，得到opt_fn
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        # 使用opt_fn计算r3的值
        r3 = opt_fn(a, (b, c), d)

        # 断言r1不为None
        self.assertIsNotNone(r1)
        # 断言r1、r2和r3的形状相同
        self.assertEqual(r1.shape, r2.shape)
        self.assertEqual(r1.shape, r3.shape)
        # 断言r1、r2和r3的设备类型相同
        self.assertEqual(r1.device, r2.device)
        self.assertEqual(r1.device, r3.device)

    # 使用_force_skip_lazy_graph_module修饰器定义一个名为test_torchscript的测试方法
    @_force_skip_lazy_graph_module()
    def test_torchscript(self):
        # 创建一个Seq对象s
        s = Seq()
        # 创建一个长度为10的随机张量i
        i = torch.randn(10)
        # 使用Seq对象s调用i，计算r1
        r1 = s(i)
        # 使用torch._dynamo.optimize("ts")优化Seq对象s，得到opt_s
        opt_s = torch._dynamo.optimize("ts")(s)
        # 使用opt_s调用i，计算r2
        r2 = opt_s(i)
        # 断言r1和r2在值上相似
        self.assertTrue(same(r1, r2))
    def test_incorrect_verify_true(self):
        """
        如果一个糟糕的优化返回一个与原始图不功能等价的图；
        当 config.verify_correctness=True 时，它将检查输出的正确性并引发错误
        """
        # 创建两个大小为 10 的随机张量 i1 和 i2
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        # 定义一个错误的编译函数 incorrect_compile_fn，返回变换后的图的前向方法
        def incorrect_compile_fn(gm, example_inputs):
            return transform(gm).forward

        # 运行 toy_example 函数，这里没有捕获返回值
        toy_example(i1, i2)

        try:
            # 对 toy_example 应用不正确的编译函数进行优化
            opt_toy_example = torch._dynamo.optimize(incorrect_compile_fn)(toy_example)
            # 运行优化后的 toy_example，并传入相同的输入 i1 和 i2
            opt_toy_example(i1, i2)
        except RuntimeError:
            pass
        else:
            # 如果没有引发 RuntimeError，则测试失败
            self.fail("expected failure")

    @config.patch("verify_correctness", False)
    def test_incorrect_verify_false(self):
        """
        错误的优化返回一个与原始图不功能等价的图；
        当 config.verify_correctness=False 时，会返回错误的输出
        """
        # 创建两个大小为 10 的随机张量 i1 和 i2
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        # 定义一个错误的编译函数 incorrect_compile_fn，返回变换后的图的前向方法
        def incorrect_compile_fn(gm, example_inputs):
            return transform(gm).forward

        # 运行 toy_example 函数并保存结果到 r1
        r1 = toy_example(i1, i2)

        # 对 toy_example 应用不正确的编译函数进行优化
        opt_toy_example = torch._dynamo.optimize(incorrect_compile_fn)(toy_example)
        # 运行优化后的 toy_example，并传入相同的输入 i1 和 i2，保存结果到 r2
        r2 = opt_toy_example(i1, i2)

        # 断言 r1 和 r2 不相同
        self.assertTrue(not same(r1, r2))
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests
    # 运行测试函数 run_tests()
    run_tests()
```