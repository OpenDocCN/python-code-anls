# `.\pytorch\test\jit\test_autodiff.py`

```py
# Owner(s): ["oncall: jit"]

# 引入必要的类型注解 List
from typing import List

# 导入 PyTorch 库
import torch

# 导入测试相关的函数和类
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.jit_utils import JitTestCase

# 装饰器，跳过 Torch Dynamo 引擎
@skipIfTorchDynamo()
# 测试类继承自 JitTestCase
class TestAutodiffJit(JitTestCase):
    
    # 测试函数：测试未定义的张量列表
    def test_undefined_tensor_lists(self):
        
        # 定义一个函数 fn，接受一个张量列表和一个额外的张量作为输入
        def fn(tensor_list: List[torch.Tensor], add_tensor):
            # 将张量列表在第一个维度上拼接
            cat = torch.cat(tensor_list, dim=1)
            # 对拼接后的张量应用 sin 函数，并加上 add_tensor
            r = torch.sin(cat + add_tensor)
            # 返回结果张量
            return r
        
        # 对 fn 函数进行脚本化
        fn_s = torch.jit.script(fn)

        # 创建三个随机张量 a, b, y，要求梯度
        a = torch.rand((3, 6), requires_grad=True)
        b = torch.rand((3, 10), requires_grad=True)
        x = [a, b]
        y = torch.rand((3, 16), requires_grad=True)

        # 使用 fn_s 对 x, y 调用，计算结果张量
        ret = fn_s(x, y)
        # 对结果张量求和并执行反向传播
        ret.sum().backward()
        
        # 再次使用 fn_s 对 x, y 调用，计算结果张量
        ret = fn_s(x, y)
        # 对结果张量求和并执行反向传播
        ret.sum().backward()

        # 第三次使用 fn_s 对 x, y 调用，计算结果张量
        ret = fn_s(x, y)
        # 对结果张量求和
        s = ret.sum()

        # 获取 s 的梯度函数，期望有 2 个输入：(grad_output, current_grad_r)
        backward_fn = s.grad_fn.next_functions[0][0]

        # 使用已定义的梯度输出 grad_out 来调用 backward_fn
        grad_out = torch.rand((3, 16))
        grad_inputs = backward_fn(grad_out, None)

        # 期望返回 3 个张量：grad_y, grad_a, grad_b
        self.assertEqual(3, len(grad_inputs))
        for x in grad_inputs:
            self.assertTrue(isinstance(x, torch.Tensor))

        # 现在测试未定义 grad_out 的情况
        grad_inputs = backward_fn(None, None)

        # 期望所有的输出都为 None
        self.assertEqual(3, len(grad_inputs))
        for x in grad_inputs:
            if x is not None:
                # 如果不为 None，期望张量元素的最大值为 0
                self.assertEqual(0, torch.max(torch.abs(x)).item())

    # 测试函数：测试 requires_grad 的输出
    def test_requires_grad_outputs(self):
        # 输出张量应该仅在 eager 模式下需要梯度时才需要梯度
        def fn(a, b, c):
            return a.relu() + b.relu(), c.relu()

        # 创建三个随机张量 a, b, c，其中 c 要求梯度
        a = torch.rand((10, 10), requires_grad=False)
        b = torch.rand((10, 10), requires_grad=False)
        c = torch.rand((10, 10), requires_grad=True)

        # 对 fn 函数进行脚本化
        fn_s = torch.jit.script(fn)

        # 迭代 4 次
        for i in range(4):
            # 调用 fn_s 函数，并获取返回值 x, y
            x, y = fn_s(a, b, c)
            # 第一个输出 x 不需要梯度
            self.assertFalse(x.requires_grad)
            # 第二个输出 y 需要梯度
            self.assertTrue(y.requires_grad)
    def test_requires_grad_outputs_profiled_twice(self):
        # 定义一个测试函数，验证在特定条件下变量的梯度要求
        # "r" 在 gammaln 和 entr 函数中被使用两次，因此被两次进行性能分析。
        # 因此在自动微分图形形成过程中，由于它们是别名，性能分析节点未合并。
        # 然后 DifferentiableGraph 在输出上没有性能分析节点。
        # 需要将 requires_grad 信息添加到输出值上（否则自动微分会使输出需要梯度）。
        # 注意：这依赖于 gammaln 和 entr 没有自动微分实现。
        def fn(a, b, c):
            # 计算 a 的 relu 两次，并将结果存储在 r 中
            r = a.relu().relu()
            # 返回 torch.special.gammaln(r), torch.special.entr(r), c.cos().relu() 的结果
            return torch.special.gammaln(r), torch.special.entr(r), c.cos().relu()

        # 对函数 fn 进行脚本化
        fn_s = torch.jit.script(fn)

        # 创建三个张量 a, b, c
        a = torch.rand((10, 10), requires_grad=False)
        b = torch.rand((10, 10), requires_grad=False)
        c = torch.rand((10, 10), requires_grad=True)

        # 进行四次迭代
        for i in range(4):
            # 在脚本化的函数上调用 fn_s(a, b, c)，并分别将结果存储在 x_s, y_s, z_s 中
            x_s, y_s, z_s = fn_s(a, b, c)
            # 直接调用 fn(a, b, c)，并将结果存储在 x, y, z 中
            x, y, z = fn(a, b, c)

            # 断言脚本化函数和原始函数的 requires_grad 属性相等
            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)

    def test_requires_grad_outputs_side_effects(self):
        # 与上述测试相同，但在中间添加了一个 CallFunction
        @torch.jit.ignore
        def python_fn(x):
            return x.relu()

        def fn(a, b, c):
            # 计算 a 的 sin，并对结果进行 relu
            x = a.sin().relu()
            # 使用 python_fn 计算 b 的 relu
            z = python_fn(x)
            # 返回 torch.relu(x), torch.nn.functional.gelu(x), c.cos().relu() 的结果
            return torch.relu(x), torch.nn.functional.gelu(x), c.cos().relu()

        # 对函数 fn 进行脚本化
        fn_s = torch.jit.script(fn)

        # 创建三个张量 a, b, c
        a = torch.rand((10, 10), requires_grad=False)
        b = torch.rand((10, 10), requires_grad=False)
        c = torch.rand((10, 10), requires_grad=True)

        # 进行四次迭代
        for i in range(4):
            # 在脚本化的函数上调用 fn_s(a, b, c)，并分别将结果存储在 x_s, y_s, z_s 中
            x_s, y_s, z_s = fn_s(a, b, c)
            # 直接调用 fn(a, b, c)，并将结果存储在 x, y, z 中
            x, y, z = fn(a, b, c)

            # 断言脚本化函数和原始函数的 requires_grad 属性相等
            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)

    def test_autodiff_requires_grad_nograd(self):
        @torch.jit.ignore
        def python_fn(x):
            return x.relu()

        def fn(a, b, c):
            # 计算 a 的 sin，并对结果进行 relu
            x = a.sin().relu()
            # 调用 python_fn 计算 b 的 relu
            y = python_fn(b)
            # 将 x 与 c 相加，使用 torch.no_grad() 禁用梯度
            with torch.no_grad():
                z = x + c
            # 返回 x, y, z 的值
            return x, y, z

        # 对函数 fn 进行脚本化
        fn_s = torch.jit.script(fn)

        # 创建三个张量 a, b, c
        a = torch.rand((10, 10), requires_grad=True)
        b = torch.rand((10, 10), requires_grad=True)
        c = torch.rand((10, 10), requires_grad=True)

        # 进行四次迭代
        for i in range(4):
            # 在脚本化的函数上调用 fn_s(a, b, c)，并分别将结果存储在 x_s, y_s, z_s 中
            x_s, y_s, z_s = fn_s(a, b, c)
            # 直接调用 fn(a, b, c)，并将结果存储在 x, y, z 中
            x, y, z = fn(a, b, c)

            # 断言脚本化函数和原始函数的 requires_grad 属性相等
            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)
```