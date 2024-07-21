# `.\pytorch\test\autograd\test_complex.py`

```
# Owner(s): ["module: autograd"]
# 导入 PyTorch 库
import torch
# 导入测试相关的实用函数和类
from torch.testing._internal.common_utils import gradcheck, run_tests, TestCase

# 定义一个测试类 TestAutogradComplex，继承自 TestCase 类
class TestAutogradComplex(TestCase):

    # 定义测试方法 test_view_func_for_complex_views
    def test_view_func_for_complex_views(self):
        # case 1: both parent and child have view_func
        # 创建一个形状为 (2, 2, 2) 的双精度浮点型张量，并要求计算梯度
        x = torch.randn(2, 2, 2, dtype=torch.double, requires_grad=True)
        # 对 x 进行分离并且设置 requires_grad=True
        y = x.detach().requires_grad_(True)

        # 复制张量 x 到 x0
        x0 = x.clone()
        # 将 x0 转换为复数视图
        x1 = torch.view_as_complex(x0)
        # 将 x1 转换为实部视图
        x2 = torch.view_as_real(x1)
        # 将 x2 中的元素乘以 2
        x2.mul_(2)
        # 对 x2 求和后取绝对值，并进行反向传播
        x2.sum().abs().backward()

        # 复制张量 y 到 y0
        y0 = y.clone()
        # 将 y0 中的元素乘以 2
        y0.mul_(2)
        # 对 y0 求和后取绝对值，并进行反向传播
        y0.sum().abs().backward()

        # 断言 x 和 y 的梯度是否相等
        self.assertEqual(x.grad, y.grad)

        # case 2: parent has view_func but child does not
        # 创建一个形状为 (2, 2, 2) 的双精度浮点型张量，并要求计算梯度
        x = torch.randn(2, 2, 2, dtype=torch.double, requires_grad=True)
        # 对 x 进行分离并且设置 requires_grad=True
        y = x.detach().requires_grad_(True)

        # 定义一个函数 fn，接受参数 a
        def fn(a):
            # 复制张量 a 到 b
            b = a.clone()
            # 将 b 转换为复数视图
            b1 = torch.view_as_complex(b)
            # 将 b1 重新调整形状为一维张量
            b2 = b1.reshape(b1.numel())
            return b2

        # 调用 fn 函数，并将结果保存到 x0
        x0 = fn(x)
        # 将 x0 中的元素乘以 2
        x0.mul_(2)
        # 对 x0 求和后取绝对值，并进行反向传播
        x0.sum().abs().backward()

        # 调用 fn 函数，并将结果保存到 y0
        y0 = fn(y)
        # 将 y0 中的元素乘以 2
        y1 = y0.mul(2)
        # 对 y1 求和后取绝对值，并进行反向传播
        y1.sum().abs().backward()

        # 断言 x 和 y 的梯度是否相等
        self.assertEqual(x.grad, y.grad)

        # case 3: parent does not have a view_func but child does
        # 创建一个形状为 (10,) 的复双精度浮点型张量，并要求计算梯度
        x = torch.randn(10, dtype=torch.cdouble, requires_grad=True)
        # 对 x 进行分离并且设置 requires_grad=True
        y = x.detach().requires_grad_(True)

        # 定义一个函数 fn，接受参数 a 和 dim0_size，默认为 5
        def fn(a, dim0_size=5):
            # 复制张量 a 到 b
            b = a.clone()
            # 将 b 调整形状为 (dim0_size, 2)
            b1 = b.reshape(dim0_size, 2)
            # 将 b1 转换为实数视图
            b2 = torch.view_as_real(b1)
            return b2

        # 调用 fn 函数，并将结果保存到 x0
        x0 = fn(x)
        # 将 x0 中的元素乘以 2
        x0.mul_(2)
        # 对 x0 求和后取绝对值，并进行反向传播
        x0.sum().abs().backward()

        # 调用 fn 函数，并将结果保存到 y0
        y0 = fn(y)
        # 将 y0 中的元素乘以 2
        y1 = y0.mul(2)
        # 对 y1 求和后取绝对值，并进行反向传播
        y1.sum().abs().backward()

        # 断言 x 和 y 的梯度是否相等
        self.assertEqual(x.grad, y.grad)

    def test_view_with_multi_output(self):
        # 创建一个形状为 (2, 2, 2) 的双精度浮点型张量 x
        x = torch.randn(2, 2, 2, dtype=torch.double)

        # 将 x 转换为复数视图 x1
        x1 = torch.view_as_complex(x)
        # 对不合法的视图操作总是允许，只要不是就地修改
        res = x1.unbind(0)

        # 断言捕获 RuntimeError，并检查错误消息是否包含特定文本
        with self.assertRaisesRegex(
            RuntimeError, "output of a function that returns multiple views"
        ):
            # 尝试在 res[0] 上执行就地修改操作
            res[0] += torch.rand(2, requires_grad=True)

        # 将 x 设置为要求计算梯度
        x.requires_grad_(True)
        # 将 x 转换为复数视图 x1
        x1 = torch.view_as_complex(x)
        # 对不合法的视图操作总是允许，只要不是就地修改
        res = x1.unbind(0)

        # 断言捕获 RuntimeError，并检查错误消息是否包含特定文本
        with self.assertRaisesRegex(
            RuntimeError, "output of a function that returns multiple views"
        ):
            # 尝试在 res[0] 上执行就地修改操作
            res[0] += torch.rand(2, requires_grad=True)
    def as_identity(self):
        # 定义一个函数，用于测试 view_as_real 和 view_as_complex 的行为，期望其表现像一个恒等映射
        def func(z):
            # 将张量 z 转换为复数形式
            z_ = torch.view_as_complex(z)
            # 选择复数张量 z_ 的最后一个维度的第一个元素
            z_select = torch.select(z_, z_.dim() - 1, 0)
            # 将选择的张量 z_select 转换回实数形式
            z_select_real = torch.view_as_real(z_select)
            # 返回实数形式张量的总和
            return z_select_real.sum()

        # 创建一个形状为 (10, 2, 2) 的双精度随机张量 z，并要求计算梯度
        z = torch.randn(10, 2, 2, dtype=torch.double, requires_grad=True)
        # 使用梯度检查函数 gradcheck 来验证 func 函数的梯度计算是否正确
        gradcheck(func, [z])
        # 对 func(z) 进行反向传播
        func(z).backward()

        # 克隆张量 z，并分离计算图，同时要求计算梯度
        z1 = z.clone().detach().requires_grad_(True)
        # 对 z1 的倒数第二个维度进行选择，并计算其总和后进行反向传播
        torch.select(z1, z1.dim() - 2, 0).sum().backward()

        # 断言张量 z 和 z1 的梯度是否相等
        self.assertEqual(z.grad, z1.grad)
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用函数 run_tests()，用于执行测试代码或者程序的单元测试
    run_tests()
```