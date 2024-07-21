# `.\pytorch\test\functorch\test_minifier.py`

```
# Owner(s): ["module: functorch"]

import torch  # 导入PyTorch库

from functorch import make_fx  # 导入functorch库中的make_fx函数
from functorch.compile import minifier  # 导入functorch库中的minifier函数
from torch._functorch.compile_utils import get_outputs, get_placeholders  # 导入torch库中的get_outputs和get_placeholders函数
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入torch库中的run_tests和TestCase类


class TestMinifier(TestCase):  # 定义测试类TestMinifier，继承自TestCase类

    def test_has_mul_minifier(self):
        def failing_f(x, y):
            y = y / 3  # 除以3
            x = x + 3  # 加3
            x = x * y  # 乘以y
            return x + y  # 返回x加y的结果

        inps = [torch.randn(3), torch.randn(3)]  # 创建包含两个随机张量的列表
        failing_f = make_fx(failing_f)(*inps)  # 使用make_fx函数对failing_f进行功能化处理

        def has_mul(fx_g, inps):
            return torch.ops.aten.mul.Tensor in (i.target for i in fx_g.graph.nodes)  # 检查图中是否存在乘法操作

        min_f, inps = minifier(failing_f, inps, has_mul)  # 使用minifier函数进行最小化处理
        self.assertEqual(len(min_f.graph.nodes), 4)  # 断言最小化后的图节点数为4
        self.assertEqual(len(inps), 2)  # 断言输入张量的数量为2

    def test_has_add_mul(self):
        def failing_f(x):
            x = x * 3  # 乘以3
            x = x + 5  # 加5
            x = x.cos()  # 对x进行余弦计算
            zero = x - x  # 计算零张量
            result = zero / zero  # 计算零张量除以零张量
            result = result + 3  # 加3
            return (result * 2,)  # 返回结果乘以2的元组

        inps = [torch.randn(3)]  # 创建一个包含一个随机张量的列表
        failing_f = make_fx(failing_f)(*inps)  # 使用make_fx函数对failing_f进行功能化处理

        def has_nans(fx_g, inps):
            # 检查图中是否有计算出NaN的节点
            for i in inps:
                if torch.isnan(i).any():
                    return False
            return torch.isnan(fx_g(*inps)[0]).any()

        min_f, inps = minifier(failing_f, inps, has_nans)  # 使用minifier函数进行最小化处理
        self.assertEqual(len(min_f.graph.nodes), 3)  # 断言最小化后的图节点数为3
        self.assertEqual(len(inps), 1)  # 断言输入张量的数量为1

    def test_input_returned(self):
        def f(a, b, c):
            a = a.sin()  # 对a进行正弦计算
            c = c.cos()  # 对c进行余弦计算
            d = a * c  # 计算a与c的乘积
            return (a, b, c, d)  # 返回包含a、b、c、d的元组

        inps = [torch.randn(3) for _ in range(3)]  # 创建包含三个随机张量的列表

        def inputs_returned(fx_g, inps):
            inps = set(get_placeholders(fx_g.graph))  # 获取图中的占位符集合
            outs = set(get_outputs(fx_g.graph))  # 获取图中的输出集合
            return len(inps & outs) > 0  # 检查输入和输出的交集是否大于0

        failing_f = make_fx(f)(*inps)  # 使用make_fx函数对f进行功能化处理
        min_f, inps = minifier(failing_f, inps, inputs_returned)  # 使用minifier函数进行最小化处理
        self.assertEqual(len(min_f.graph.nodes), 2)  # 断言最小化后的图节点数为2
        self.assertEqual(len(inps), 1)  # 断言输入张量的数量为1

    def test_tup_use(self):
        def f(a, b):
            tup = torch.std_mean(a)  # 计算a的标准差和均值
            return (tup[0] + b * tup[1],)  # 返回元组，包含tup[0]加b乘以tup[1]的结果

        inps = [torch.randn(3), torch.randn(3)]  # 创建包含两个随机张量的列表

        def has_add(fx_g, inps):
            return torch.ops.aten.add.Tensor in (i.target for i in fx_g.graph.nodes)  # 检查图中是否存在加法操作

        failing_f = make_fx(f)(*inps)  # 使用make_fx函数对f进行功能化处理
        min_f, inps = minifier(failing_f, inps, has_add)  # 使用minifier函数进行最小化处理

        self.assertEqual(len(min_f.graph.nodes), 4)  # 断言最小化后的图节点数为4
        self.assertEqual(len(inps), 2)  # 断言输入张量的数量为2
    def test_module(self):
        # 定义一个模拟的 PyTorch 模块用于测试
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # 前向传播函数，应用 ReLU 激活函数
                y = self.relu(x)
                # 创建一个全零张量
                zero = y - y
                # 执行零除以零的操作，得到 NaN
                result = zero / zero
                # 加上常数 3
                result = result + 3
                return result

        # 创建 MockModule 的实例
        mod = MockModule()
        # 对模型进行符号化跟踪
        failing_f = torch.fx.symbolic_trace(mod)

        # 准备输入
        inps = [torch.randn(3)]

        # 定义检查器函数，用于验证符号化跟踪后的函数是否有 NaN 值
        def pass_checker(fx_g, inps):
            # 检查输入中是否有任何一个包含 NaN
            for i in inps:
                if torch.isnan(i).any():
                    return False
            # 检查跟踪后的函数的输出是否包含 NaN
            return torch.isnan(fx_g(*inps)[0]).any()

        # 调用最小化函数 minifier，并获取返回的最小化函数和输入
        min_f, inps = minifier(failing_f, inps, pass_checker)
        # 断言最小化后的函数图中节点数为 3
        assert len(min_f.graph.nodes) == 3
        # 断言输入列表长度为 1
        assert len(inps) == 1
# 如果这个脚本作为主程序运行（而不是被导入到其他脚本中执行），则执行run_tests函数
if __name__ == "__main__":
    run_tests()
```