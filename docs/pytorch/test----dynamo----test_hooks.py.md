# `.\pytorch\test\dynamo\test_hooks.py`

```
# Owner(s): ["module: dynamo"]

# 引入上下文管理器和函数装饰器
import contextlib
import functools
# 引入单元测试模块
import unittest

# 引入PyTorch相关模块
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing

# 引入特定功能模块
from functorch.compile import nop
from torch._dynamo import compiled_autograd
from torch._functorch.aot_autograd import aot_module_simplified
from torch.utils.hooks import RemovableHandle

# 定义一个编译函数，使用动态优化和无Python优化
def compiler_fn(gm):
    return torch._dynamo.optimize("inductor", nopython=True, dynamic=True)(gm)

# 全局钩子函数，用于修改梯度
def global_hook_0(grad):
    return grad * 4

def global_hook_1(grad):
    return grad / 2

def global_hook_2(grad):
    return grad * 3

# 初始化一个全局变量
h0 = None

# 定义一个带有值的类
class ClassWithVal:
    def __init__(self, val):
        self.val = val

# 测试用例类，继承自torch._dynamo.test_case.TestCase
class HooksTests(torch._dynamo.test_case.TestCase):
    # 测试：仅在图中Lambda函数注册钩子
    def test_tensor_only_register_hook_in_graph_lambda(self):
        def fn(x):
            # 注册一个Lambda函数作为钩子，对梯度乘以2
            x.register_hook(lambda grad: grad * 2)
            return x

        # 编译计数器
        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)
        # 创建一个需要梯度的张量
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        # 反向传播
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # 断言梯度是否正确
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        # 断言帧计数是否为0
        self.assertEqual(cnts.frame_count, 0)

    # 测试：在图中Lambda函数注册钩子
    def test_tensor_register_hook_in_graph_lambda(self):
        def fn(x, y, z):
            x.register_hook(lambda grad: grad * 2)
            return x, y * y, z * z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    # 测试：在图中Lambda函数注册钩子，并且中途移除钩子
    def test_tensor_register_hook_in_graph_break_handle_lambda(self):
        def fn(x, y, z):
            # 注册一个Lambda函数作为钩子，对梯度乘以2，并返回一个可移除的句柄
            handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            # 移除之前的钩子句柄
            handle.remove()
            # 注册另一个Lambda函数作为钩子，对梯度乘以3
            x.register_hook(lambda grad: grad * 3)
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))
        self.assertEqual(cnts.frame_count, 1)
    def test_tensor_register_hook_multi_handle_return(self):
        # 定义一个函数 fn，接受三个参数 x, y, z
        def fn(x, y, z):
            # 在张量 x 上注册一个 hook，该 hook 对梯度 grad 乘以 2
            handle = x.register_hook(lambda grad: grad * 2)
            # 复制 handle 到 h2
            h2 = handle
            # 计算 z 的平方
            z = z * z
            # 返回 x, y 的平方，z 的平方，handle 和 h2
            return x, y * y, z, handle, h2

        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn，并赋值给 fn
        fn = torch._dynamo.optimize(cnts)(fn)
        # 创建一个 requires_grad 为 True 的张量 v
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        # 调用 fn 函数，将结果解包到 v, y, z, h, h2 中
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        # 根据指定梯度进行反向传播
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # 断言 v 的梯度值是否符合预期
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        # 断言 frame_count 是否为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言 handle 和 h2 非空
        self.assertNotEqual(h, None)
        self.assertNotEqual(h2, None)
        # 断言 h 和 h2 相等
        self.assertEqual(h2, h)

    def test_tensor_register_hook_repeated_handle_return(self):
        # 定义一个函数 fn，接受三个参数 x, y, z
        def fn(x, y, z):
            # 在张量 x 上注册一个 hook，该 hook 对梯度 grad 乘以 2
            handle = x.register_hook(lambda grad: grad * 2)
            # 复制 handle 到 h2
            h2 = handle
            # 计算 z 的平方
            z = z * z
            # 返回 x, y 的平方，z 的平方，handle 和 handle
            return x, y * y, z, handle, handle

        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn，并赋值给 fn
        fn = torch._dynamo.optimize(cnts)(fn)
        # 创建一个 requires_grad 为 True 的张量 v
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        # 调用 fn 函数，将结果解包到 v, y, z, h, h2 中
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        # 根据指定梯度进行反向传播
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # 断言 v 的梯度值是否符合预期
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        # 断言 frame_count 是否为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言 h 类型为 RemovableHandle 类型
        self.assertIsInstance(h, RemovableHandle)
        # 断言 h2 和 h 是同一个对象
        self.assertIs(h2, h)

    def test_removed_handle_return(self):
        # 创建一个 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()

        # 使用 @torch.compile 进行函数编译
        @torch.compile(backend=cnt, fullgraph=True)
        # 定义一个函数 fn，接受三个参数 x, y, z
        def fn(x, y, z):
            # 在张量 x 上注册一个 hook，该 hook 对梯度 grad 乘以 2
            handle = x.register_hook(lambda grad: grad * 2)
            # 计算 z 的平方
            z = z * z
            # 移除 handle，即使重复移除
            handle.remove()
            handle.remove()
            # 返回 x, y 的平方，z 的平方，handle 和 handle
            return x, y * y, z, handle, handle

        # 创建一个 requires_grad 为 True 的张量 v
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        # 调用 fn 函数，将结果解包到 v, y, z, h, h2 中
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        # 根据指定梯度进行反向传播
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # 断言 v 的梯度值是否符合预期
        self.assertEqual(v.grad, torch.tensor([1.0, 2.0, 3.0]))
        # 断言 frame_count 是否为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言 h 类型为 RemovableHandle 类型
        self.assertIsInstance(h, RemovableHandle)
        # 断言 h2 和 h 是同一个对象
        self.assertIs(h2, h)

    def test_tensor_register_hook_repeated_handle_not_local(self):
        # 定义一个函数 fn，接受四个参数 x, y, z, mod
        def fn(x, y, z, mod):
            # 在 mod 对象上注册一个 hook，该 hook 对梯度 grad 乘以 2
            mod.handle = x.register_hook(lambda grad: grad * 2)
            # 计算 z 的平方
            z = z * z
            # 返回 x, y 的平方，z 的平方
            return x, y * y, z

        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn，并赋值给 fn
        fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 创建一个 requires_grad 为 True 的张量 v
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

        # 创建一个空的 nn.Module 对象 mod，并在其上定义一个 handle 属性为 None
        mod = torch.nn.Module()
        mod.handle = None

        # 调用 fn 函数，将结果解包到 v, y, z 中，并传入 mod 对象
        v, y, z = fn(v, torch.randn([2, 2]), torch.randn([2, 2]), mod)
        # 根据指定梯度进行反向传播
        v.backward(torch.tensor([1.0, 2.0, 3.0]))

        # 断言 v 的梯度值是否符合预期
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        # 断言 frame_count 是否为 1
        self.assertEqual(cnts.frame_count, 1)

        # 断言 mod.handle 非空
        self.assertNotEqual(mod.handle, None)
    # 定义一个测试函数，用于验证在图中仅注册局部钩子的情况
    def test_tensor_only_register_hook_in_graph_local(self):
        # 定义局部钩子函数，它将梯度乘以2
        def local_hook(grad):
            return grad * 2

        # 定义一个函数fn，它注册了局部钩子到输入张量x，并返回x本身
        def fn(x):
            x.register_hook(local_hook)
            return x

        # 创建一个编译计数器实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数fn应用优化并重新赋值给fn
        fn = torch._dynamo.optimize(cnts)(fn)
        # 创建一个需要梯度的张量v
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        # 对张量v应用函数fn
        v = fn(v)
        # 对结果张量进行反向传播
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # 断言结果张量的梯度与预期的结果张量相等
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        # 断言编译计数器的帧数为0
        self.assertEqual(cnts.frame_count, 0)

    # 定义一个测试函数，用于验证在图中仅注册内部局部钩子的情况
    def test_tensor_only_register_hook_in_graph_local_inner(self):
        # 定义一个函数fn，它在内部定义了局部钩子函数local_hook，它将梯度乘以2
        def fn(x):
            def local_hook(grad):
                return grad * 2

            # 对输入张量x进行操作，并分别向x和操作结果z注册局部钩子
            z = x * x
            x.register_hook(local_hook)
            z.register_hook(local_hook)
            return x, z

        # 创建一个编译计数器实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数fn应用优化并重新赋值给fn
        fn = torch._dynamo.optimize(cnts)(fn)
        # 创建一个需要梯度的张量v
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        # 对张量v应用函数fn
        v = fn(v)
        # 对结果张量的第一个元素进行反向传播
        v[0].backward(torch.tensor([1.0, 2.0, 3.0]))
        # 断言结果张量的第一个元素的梯度与预期的结果张量相等
        self.assertEqual(v[0].grad, torch.tensor([2.0, 4.0, 6.0]))
        # 断言编译计数器的帧数为1
        self.assertEqual(cnts.frame_count, 1)

    # 定义一个测试函数，用于验证在图中注册局部钩子的情况
    def test_tensor_register_hook_in_graph_local(self):
        # 定义局部钩子函数，它将梯度乘以2
        def local_hook(grad):
            return grad * 2

        # 定义一个函数fn，它注册了局部钩子到输入张量x，并返回x、y*y和z*z
        def fn(x, y, z):
            x.register_hook(local_hook)
            return x, y * y, z * z

        # 创建一个编译计数器实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数fn应用优化并重新赋值给fn
        fn = torch._dynamo.optimize(cnts)(fn)
        # 创建一个需要梯度的张量v
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        # 对张量v以及随机张量y和z应用函数fn，并仅保留结果张量的第一个元素
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        # 对结果张量进行反向传播
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # 断言结果张量的梯度与预期的结果张量相等
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        # 断言编译计数器的帧数为1
        self.assertEqual(cnts.frame_count, 1)

    # 定义一个测试函数，用于验证在图中注册局部钩子并中断处理的情况
    def test_tensor_register_hook_in_graph_break_handle_local(self):
        # 定义局部钩子函数local_hook，它将梯度乘以2
        def local_hook(grad):
            return grad * 2

        # 定义另一个局部钩子函数local_hook2，它将梯度乘以3
        def local_hook2(grad):
            return grad * 3

        # 定义一个函数fn，它注册了局部钩子到输入张量x，并在处理中断后注册了另一个局部钩子到x
        def fn(x, y, z):
            handle = x.register_hook(local_hook)
            z = z * z
            handle.remove()
            x.register_hook(local_hook2)
            return x, y * y, z

        # 创建一个编译计数器实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数fn应用优化并重新赋值给fn
        fn = torch._dynamo.optimize(cnts)(fn)
        # 创建一个需要梯度的张量v
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        # 对张量v以及随机张量y和z应用函数fn，并仅保留结果张量的第一个元素
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        # 对结果张量进行反向传播
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # 断言结果张量的梯度与预期的结果张量相等
        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))

    # 定义一个测试函数，用于验证在图中注册全局钩子的情况
    def test_tensor_register_global_hook(self):
        # 定义一个函数fn，它注册了全局钩子global_hook_0到输入张量x，并返回x和x*x
        def fn(x):
            x.register_hook(global_hook_0)
            return x, x * x

        # 创建一个编译计数器实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 对函数fn应用优化并重新赋值给fn
        fn = torch._dynamo.optimize(cnts)(fn)
        # 创建一个需要梯度的张量v
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        # 对张量v应用函数fn，并仅保留结果张量的第一个元素
        v = fn(v)[0]
        # 对结果张量进行反向传播
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # 断言结果张量的梯度与预期的结果张量相等
        self.assertEqual(v.grad, torch.tensor([4.0, 8.0, 12.0]))
        # 断言编译计数器的帧数为1
        self.assertEqual(cnts.frame_count, 1)
    def test_tensor_register_multiple_hooks(self):
        def fn(x):
            x.register_hook(global_hook_0)  # 注册全局钩子函数 global_hook_0，对输入张量进行操作，乘以4
            x.register_hook(global_hook_1)  # 注册全局钩子函数 global_hook_1，对输入张量进行操作，除以2
            x.register_hook(global_hook_2)  # 注册全局钩子函数 global_hook_2，对输入张量进行操作，乘以3
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)  # 对函数 fn 进行优化编译
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)  # 创建一个张量 v，需要计算梯度
        v = fn(v)[0]  # 调用优化后的 fn 函数，得到结果张量 v
        v.backward(torch.tensor([1.0, 2.0, 3.0]))  # 对结果张量 v 进行反向传播，传入梯度张量
        self.assertEqual(v.grad, torch.tensor([6.0, 12.0, 18.0]))  # 断言梯度计算结果是否正确
        self.assertEqual(cnts.frame_count, 1)  # 断言编译计数是否为1

    def test_tensor_register_multiple_hooks_handles_in_list(self):
        def fn(x):
            h0 = x.register_hook(global_hook_0)  # 注册全局钩子函数 global_hook_0，对输入张量进行操作，乘以4，并保存句柄
            h1 = x.register_hook(global_hook_1)  # 注册全局钩子函数 global_hook_1，对输入张量进行操作，除以2，并保存句柄
            h2 = x.register_hook(global_hook_2)  # 注册全局钩子函数 global_hook_2，对输入张量进行操作，乘以3，并保存句柄
            return x, x * x, h0, h1, h2  # 返回张量 x，张量 x 的平方，以及三个钩子函数的句柄

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)  # 对函数 fn 进行优化编译
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)  # 创建一个张量 v，需要计算梯度
        v, r, handle_0, handle_1, handle_2 = fn(v)  # 调用优化后的 fn 函数，获取结果张量 v 和其它返回值
        v.backward(torch.tensor([1.0, 2.0, 3.0]))  # 对结果张量 v 进行反向传播，传入梯度张量
        self.assertEqual(v.grad, torch.tensor([6.0, 12.0, 18.0]))  # 断言梯度计算结果是否正确
        handle_0.remove()  # 移除钩子函数 h0
        handle_1.remove()  # 移除钩子函数 h1
        handle_2.remove()  # 移除钩子函数 h2

        v.backward(torch.tensor([1.0, 2.0, 3.0]))  # 再次对结果张量 v 进行反向传播
        # 钩子函数已移除，梯度计算结果将直接应用于张量
        self.assertEqual(v.grad, torch.tensor([7.0, 14.0, 21.0]))  # 断言梯度计算结果是否正确

        self.assertEqual(cnts.frame_count, 1)  # 断言编译计数是否为1

    def test_tensor_register_global_hooks_handles_in_list(self):
        def fn(x):
            global h0  # 声明全局变量 h0
            h0 = x.register_hook(global_hook_0)  # 注册全局钩子函数 global_hook_0，对输入张量进行操作，乘以4
            return x, x * x  # 返回张量 x 和张量 x 的平方

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)  # 对函数 fn 进行优化编译
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)  # 创建一个张量 v，需要计算梯度
        v, r = fn(v)  # 调用优化后的 fn 函数，获取结果张量 v 和其它返回值

        self.assertIsNotNone(h0)  # 断言全局变量 h0 不为 None
        v.backward(torch.tensor([1.0, 2.0, 3.0]))  # 对结果张量 v 进行反向传播，传入梯度张量
        self.assertEqual(v.grad, torch.tensor([4.0, 8.0, 12.0]))  # 断言梯度计算结果是否正确
        h0.remove()  # 移除全局钩子函数 h0

        v.backward(torch.tensor([1.0, 2.0, 3.0]))  # 再次对结果张量 v 进行反向传播
        # 钩子函数已移除，梯度计算结果将直接应用于张量
        self.assertEqual(v.grad, torch.tensor([5.0, 10.0, 15.0]))  # 断言梯度计算结果是否正确

        # NYI!
        self.assertEqual(cnts.frame_count, 0)  # 断言编译计数是否为0

    def test_intermediary_hooks(self):
        # 图形中断，因为未设置 compiled_autograd
        def simple_hook(g):
            return g * 2

        def f(x):
            y = x + 1  # 对输入张量 x 加1
            y.register_hook(simple_hook)  # 注册简单钩子函数 simple_hook 到张量 y
            z = y + 1  # 对张量 y 再加1
            return z  # 返回结果张量 z

        out = torch.randn(1, requires_grad=True)  # 创建一个随机张量 out，需要计算梯度
        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts, nopython=False)(f)  # 对函数 f 进行优化编译，关闭 nopython 选项
        res = fn(out)  # 调用优化后的 fn 函数，传入张量 out，得到结果张量 res
        res.backward()  # 对结果张量 res 进行反向传播
        self.assertEqual(res, f(out))  # 断言结果张量 res 是否与调用未优化的 f 函数得到的结果一致
        self.assertEqual(cnts.frame_count, 2)  # 断言编译计数是否为2
        self.assertEqual(out.grad, torch.Tensor([2.0]))  # 断言输入张量 out 的梯度计算结果是否正确
    def test_intermediary_hooks_same_on_aot_eager(self):
        # 定义一个自定义的梯度钩子函数，将传入的梯度 grad 加上 k 并返回
        def my_hook(grad, *, k=0):
            return grad + k

        # 定义一个继承自 torch.nn.Module 的模块类 MyMod
        class MyMod(torch.nn.Module):
            def forward(self, x):
                # 计算 x 的每个元素乘以 2 得到 y
                y = x.mul(2)
                # 创建两个部分应用了 my_hook 的钩子函数：hook1 加上 k=3，hook2 加上 k=4
                hook1 = functools.partial(my_hook, k=3)
                hook2 = functools.partial(my_hook, k=4)
                # 将 hook1 和 hook2 注册为 y 的钩子函数
                y.register_hook(hook1)
                y.register_hook(hook2)
                # 计算 y 的每个元素乘以 3 得到 z
                z = y.mul(3)
                return (z,)

        # 创建 MyMod 的实例 mod
        mod = MyMod()
        # 创建一个 requires_grad=True 的张量 x0，所有操作对其梯度进行跟踪
        x0 = torch.ones(4, requires_grad=True)
        # 在 eager 模式下进行前向传播和反向传播
        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        # 创建另一个 requires_grad=True 的张量 x1，用于编译后的模块测试
        x1 = torch.ones(4, requires_grad=True)
        # 使用简化的 AOT 模块编译 mod，并进行前向传播和反向传播
        mod_compiled = aot_module_simplified(mod, (x1,), nop)
        aot_out = mod_compiled(x1)
        aot_out[0].backward(torch.ones(4))

        # 创建第三个 requires_grad=True 的张量 x2，用于动态优化模式测试
        x2 = torch.ones(4, requires_grad=True)
        # 在编译的自动求导模式中进行前向传播和反向传播
        with compiled_autograd.enable(compiler_fn):
            dynamo_out = torch._dynamo.optimize("aot_eager", nopython=True)(mod)(x2)
            dynamo_out[0].backward(torch.ones(4))

        # 断言三种模式下的输出应该相同
        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        # 断言三种模式下的梯度应该相同
        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

    def test_input_hooks_same(self):
        # 定义一个自定义的梯度钩子函数，将传入的梯度 grad 加上 k 并返回
        def my_hook(grad, *, k=0):
            return grad + k

        # 定义一个包含三种后端的列表
        backends = ["eager", "aot_eager", "inductor"]
        for backend in backends:
            # 创建部分应用了 my_hook 的钩子函数，设置 k=3
            hook = functools.partial(my_hook, k=3)

            # 定义一个继承自 torch.nn.Module 的模块类 MyMod
            class MyMod(torch.nn.Module):
                def forward(self, x):
                    # 将 hook 注册为 x 的钩子函数
                    x.register_hook(hook)
                    # 计算 x 的每个元素乘以 2 得到 y
                    y = x.mul(2)
                    # 计算 y 的每个元素乘以 3 得到 z
                    z = y.mul(3)
                    return (z,)

            # 创建 MyMod 的实例 mod
            mod = MyMod()
            # 创建一个 requires_grad=True 的张量 x0，所有操作对其梯度进行跟踪
            x0 = torch.ones(4, requires_grad=True)
            # 在当前后端 backend 下进行前向传播和反向传播
            eager_out = mod(x0)
            eager_out[0].backward(torch.ones(4))

            # 创建另一个 requires_grad=True 的张量 x1，用于编译后的模块测试
            x1 = torch.ones(4, requires_grad=True)
            # 使用简化的 AOT 模块编译 mod，并进行前向传播和反向传播
            mod_compiled = aot_module_simplified(mod, (x1,), nop)
            aot_out = mod_compiled(x1)
            aot_out[0].backward(torch.ones(4))

            # 创建第三个 requires_grad=True 的张量 x2，用于当前后端 backend 下的动态优化模式测试
            x2 = torch.ones(4, requires_grad=True)
            # 在当前后端 backend 下进行前向传播和反向传播
            dynamo_out = torch._dynamo.optimize(backend, nopython=True)(mod)(x2)
            with compiled_autograd.enable(compiler_fn):
                dynamo_out[0].backward(torch.ones(4))

            # 断言三种模式下的输出应该相同
            self.assertEqual(dynamo_out, aot_out)
            self.assertEqual(dynamo_out, eager_out)

            # 断言三种模式下的梯度应该相同
            self.assertEqual(x0.grad, x1.grad)
            self.assertEqual(x0.grad, x2.grad)
    def test_intermediary_hooks_same_on_inductor(self):
        # 定义一个自定义的梯度钩子函数
        def my_hook(grad, *, k=0):
            return grad + k

        # 定义一个继承自 torch.nn.Module 的模型类
        class MyMod(torch.nn.Module):
            def forward(self, x):
                # 计算 y = 2 * x
                y = x.mul(2)
                # 创建两个部分应用了不同 k 值的钩子函数
                hook1 = functools.partial(my_hook, k=3)
                hook2 = functools.partial(my_hook, k=4)
                # 注册这两个钩子函数到 y 上
                y.register_hook(hook1)
                y.register_hook(hook2)
                # 计算 z = 3 * y
                z = y.mul(3)
                return (z,)

        # 创建 MyMod 类的实例
        mod = MyMod()
        # 创建一个需要梯度的张量 x0，并将其输入模型得到 eager_out
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0)
        # 对 eager_out[0] 进行反向传播
        eager_out[0].backward(torch.ones(4))

        # 准备下一个测试的输入张量 x1
        x1 = torch.ones(4, requires_grad=True)
        # 使用简化的 Ahead-of-Time (AOT) 编译模块，编译模型并得到 aot_out
        mod_compiled = aot_module_simplified(mod, (x1,), nop)
        aot_out = mod_compiled(x1)
        # 对 aot_out[0] 进行反向传播
        aot_out[0].backward(torch.ones(4))

        # 准备下一个测试的输入张量 x2
        x2 = torch.ones(4, requires_grad=True)
        # 使用动态图优化工具对模型进行优化，得到 dynamo_out
        with compiled_autograd.enable(compiler_fn):
            dynamo_out = torch._dynamo.optimize("inductor", nopython=True)(mod)(x2)
            # 对 dynamo_out[0] 进行反向传播
            dynamo_out[0].backward(torch.ones(4))

        # 断言 dynamo_out 和 aot_out 相等
        self.assertEqual(dynamo_out, aot_out)
        # 断言 dynamo_out 和 eager_out 相等
        self.assertEqual(dynamo_out, eager_out)

        # 断言 x0.grad 和 x1.grad 相等
        self.assertEqual(x0.grad, x1.grad)
        # 断言 x0.grad 和 x2.grad 相等
        self.assertEqual(x0.grad, x2.grad)

    def test_complex_state_mutation_in_intermediary_hooks_same_on_inductor(self):
        # 定义一个 Python 类 SomePyClass，用于处理复杂状态的变化
        class SomePyClass:
            count = 0

            def do_stuff(self, grad):
                if self.count % 2 == 0:
                    r = grad * grad
                else:
                    r = grad + grad
                self.count += 1
                return r

        # 定义一个复杂状态变化的钩子函数
        def complex_state_touching_hook(grad, *, obj):
            return obj.do_stuff(grad)

        # 定义一个继承自 torch.nn.Module 的模型类
        class MyMod(torch.nn.Module):
            def forward(self, x, obj):
                # 计算 y = 2 * x
                y = x.mul(2)
                # 创建两个部分应用了同一个 obj 的钩子函数
                hook1 = functools.partial(complex_state_touching_hook, obj=obj)
                hook2 = functools.partial(complex_state_touching_hook, obj=obj)
                # 注册这两个钩子函数到 y 上
                y.register_hook(hook1)
                y.register_hook(hook2)
                # 计算 z = 3 * y
                z = y.mul(3)
                return (z,)

        # 创建 MyMod 类的实例
        mod = MyMod()
        # 创建 SomePyClass 的实例 obj
        obj = SomePyClass()
        # 创建一个需要梯度的张量 x0，并将其输入模型得到 eager_out
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0, obj)
        # 对 eager_out[0] 进行反向传播
        eager_out[0].backward(torch.ones(4))

        # 断言 obj.count 的值为 2
        self.assertEqual(obj.count, 2)
        # 准备下一个测试的输入张量 x2
        x2 = torch.ones(4, requires_grad=True)
        # 使用动态图优化工具对模型进行优化，得到 dynamo_out
        with compiled_autograd.enable(compiler_fn):
            dynamo_out = torch._dynamo.optimize("inductor", nopython=True)(mod)(x2, obj)
            # 对 dynamo_out[0] 进行反向传播
            dynamo_out[0].backward(torch.ones(4))

        # 断言 dynamo_out 和 eager_out 相等
        self.assertEqual(dynamo_out, eager_out)

        # 断言 obj.count 的值为 4
        self.assertEqual(obj.count, 4)
        # 断言 x0.grad 和 x2.grad 相等
        self.assertEqual(x0.grad, x2.grad)
    ):
        # 定义一个内部类 SomePyClass
        class SomePyClass:
            # 初始化类变量 grad_as_str 和 count
            grad_as_str = "None"
            count = 0

            # 实例方法，将输入的 grad 转换为字符串并做一些操作
            def write_grad_as_str_and_do_stuff(self, grad):
                # 将 grad 转换为字符串并保存到 grad_as_str 中
                self.grad_as_str = str(grad)
                # 根据 count 的奇偶性选择操作
                if self.count % 2 == 0:
                    r = grad * grad  # 偶数次幂操作
                else:
                    r = grad + grad  # 奇数次加法操作
                # 输出调试信息
                print("Break!")
                # 计数器加一
                self.count += 1
                # 返回计算结果 r
                return r

        # 定义一个函数，用于处理复杂的状态变化，利用传入的 obj 调用 SomePyClass 的方法
        def complex_state_touching_hook(grad, *, obj):
            return obj.write_grad_as_str_and_do_stuff(grad)

        # 定义一个继承自 torch.nn.Module 的子类 MyMod
        class MyMod(torch.nn.Module):
            # 定义前向传播方法
            def forward(self, x, obj):
                # 对输入 x 做乘法操作
                y = x.mul(2)
                # 创建两个部分应用的 hook，分别注册到 y 上
                hook1 = functools.partial(complex_state_touching_hook, obj=obj)
                hook2 = functools.partial(complex_state_touching_hook, obj=obj)
                y.register_hook(hook1)
                y.register_hook(hook2)
                # 对 y 做乘法操作
                z = y.mul(3)
                # 返回结果 z
                return (z,)

        # 创建 MyMod 的实例 mod 和 SomePyClass 的实例 obj
        mod = MyMod()
        obj = SomePyClass()
        # 创建一个张量 x0，要求梯度跟踪
        x0 = torch.ones(4, requires_grad=True)
        # 在 mod 上执行前向传播
        eager_out = mod(x0, obj)
        # 对 eager_out 的第一个元素执行反向传播
        eager_out[0].backward(torch.ones(4))

        # 创建一个张量 x2，要求梯度跟踪
        x2 = torch.ones(4, requires_grad=True)
        # 使用编译后自动微分功能，优化 mod 的执行
        with compiled_autograd.enable(compiler_fn):
            dynamo_out = torch._dynamo.optimize("inductor", nopython=True)(mod)(x2, obj)
            # 使用断言检查是否抛出了特定异常
            with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "builtin: str"):
                dynamo_out[0].backward(torch.ones(4))

        # 使用断言检查 obj 的 count 是否为 2
        self.assertEqual(obj.count, 2)

    # 定义一个测试方法 test_register_hook_partial_guarding
    def test_register_hook_partial_guarding(
        self,
    ):
        # 定义一个 hook 函数，根据 obj 的 val 对 grad 做加法操作
        def some_hook(grad, *, obj):
            return grad + obj.val

        # 定义一个继承自 torch.nn.Module 的子类 MyMod
        class MyMod(torch.nn.Module):
            # 定义前向传播方法
            def forward(self, x, obj):
                # 对输入 x 做乘法操作
                y = x.mul(2)
                # 创建部分应用的 hook，注册到 y 上
                hook1 = functools.partial(some_hook, obj=obj)
                y.register_hook(hook1)
                # 对 y 做乘法操作
                z = y.mul(3)
                # 返回结果 z
                return (z,)

        # 创建 MyMod 的实例 mod 和几个 ClassWithVal 的实例
        mod = MyMod()
        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        obj3 = ClassWithVal(11)
        cnt = torch._dynamo.testing.CompileCounter()

        # 创建两个张量 x0 和 x1，要求梯度跟踪
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)

        # 使用编译后自动微分功能，对 mod 进行编译优化，并统计编译帧数
        with compiled_autograd.enable(compiler_fn):
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj1)
            torch.compile(mod, backend=cnt, fullgraph=True)(x1, obj1)
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj2)
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj3)
            # 使用断言检查编译帧数是否为 1
            self.assertEqual(cnt.frame_count, 1)
    # 定义一个测试方法，测试带有闭包的钩子函数
    def test_hook_with_closure(self):
        # 定义一个函数 fn，接受输入 x 和 obj，计算相关数学函数并注册梯度钩子
        def fn(x, obj):
            # 计算 x 的正弦值
            y = x.sin()
            # 注册一个梯度钩子，该钩子会在梯度计算时添加 obj.val 到梯度中
            x.register_hook(lambda grad: grad + obj.val)
            # 再次对 y 计算正弦值
            z = y.sin()
            return z

        # 创建一个编译计数器 cnt_fw
        cnt_fw = torch._dynamo.testing.CompileCounter()
        # 创建另一个编译计数器 cnt_bw
        cnt_bw = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译函数 fn，设置后端为 cnt_fw，并使用完整图形模式
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        # 创建两个 ClassWithVal 对象 obj1 和 obj2，分别使用不同的 tensor 初始化
        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        # 创建四个 requires_grad=True 的张量 x0, x1, x2, x3，初始值均为 1
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        
        # 对 x0 和 obj1 调用 fn，计算结果的和并进行反向传播
        fn(x0, obj1).sum().backward()
        # 对 x1 和 obj2 调用 fn，计算结果的和并进行反向传播
        fn(x1, obj2).sum().backward()

        # 启用编译自动微分功能，使用编译后的函数 opt 和 cnt_bw 编译器
        with compiled_autograd.enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            # 对 x2 和 obj1 调用 opt，计算结果的和并进行反向传播
            opt(x2, obj1).sum().backward()
            # 对 x3 和 obj2 调用 opt，计算结果的和并进行反向传播
            opt(x3, obj2).sum().backward()
            # 断言 cnt_fw 的帧计数为 1
            self.assertEqual(cnt_fw.frame_count, 1)
            # 断言 cnt_bw 的帧计数为 1
            self.assertEqual(cnt_bw.frame_count, 1)

        # 断言 x0 和 x2 的梯度相等
        self.assertEqual(x0.grad, x2.grad)
        # 断言 x1 和 x3 的梯度相等
        self.assertEqual(x1.grad, x3.grad)

    # 定义一个测试方法，测试使用 eager 模式的带闭包的中间钩子函数
    def test_intermediate_hook_with_closure_eager(self):
        # 定义一个函数 fn，接受输入 x 和 obj，计算相关数学函数并注册梯度钩子
        def fn(x, obj):
            # 计算 x 的正弦值
            y = x.sin()
            # 注册一个梯度钩子，该钩子会在梯度计算时添加 obj.val 到梯度中
            y.register_hook(lambda grad: grad + obj.val)
            # 再次对 y 计算正弦值
            z = y.sin()
            return z

        # 创建一个编译计数器 cnt_fw
        cnt_fw = torch._dynamo.testing.CompileCounter()
        # 创建另一个编译计数器 cnt_bw
        cnt_bw = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译函数 fn，设置后端为 cnt_fw，并使用完整图形模式
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        # 创建两个 ClassWithVal 对象 obj1 和 obj2，分别使用不同的 tensor 初始化
        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        # 创建四个 requires_grad=True 的张量 x0, x1, x2, x3，初始值均为 1
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)

        # 对 x0 和 obj1 调用 fn，计算结果的和并进行反向传播
        fn(x0, obj1).sum().backward()
        # 对 x1 和 obj2 调用 fn，计算结果的和并进行反向传播
        fn(x1, obj2).sum().backward()

        # 启用编译自动微分功能，使用编译后的函数 opt 和 cnt_bw 编译器
        with compiled_autograd.enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            # 对 x2 和 obj1 调用 opt，计算结果的和并进行反向传播
            opt(x2, obj1).sum().backward()
            # 对 x3 和 obj2 调用 opt，计算结果的和并进行反向传播
            opt(x3, obj2).sum().backward()
            # 断言 cnt_fw 的帧计数为 1
            self.assertEqual(cnt_fw.frame_count, 1)
            # 断言 cnt_bw 的帧计数为 1
            self.assertEqual(cnt_bw.frame_count, 1)

        # 断言 x0 和 x2 的梯度相等
        self.assertEqual(x0.grad, x2.grad)
        # 断言 x1 和 x3 的梯度相等
        self.assertEqual(x1.grad, x3.grad)
    # 定义一个测试函数，用于测试带闭包的中间挂钩（hook）功能，使用 Ahead-of-Time 编译（AOT）
    def test_intermediate_hook_with_closure_aot(self):
        # 定义一个函数 fn，接受两个参数 x 和 obj
        def fn(x, obj):
            # 对 x 调用 sin() 方法并保存结果到 y
            y = x.sin()
            # 注册一个 hook 函数，用于修改梯度 grad，使其加上 obj.val 的值
            y.register_hook(lambda grad: grad + obj.val)
            # 对 y 调用 sin() 方法并保存结果到 z
            z = y.sin()
            return z

        # 创建一个编译计数器对象
        cnt_bw = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 对 fn 进行编译，选择 backend 为 "aot_eager"，并启用完整图形模式
        opt = torch.compile(fn, backend="aot_eager", fullgraph=True)

        # 创建两个 ClassWithVal 对象，分别使用不同的初始值
        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        # 创建四个张量 x0, x1, x2, x3，每个都设置为 requires_grad=True
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        
        # 分别对 x0 和 x1 调用 fn 函数，求和后进行反向传播
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        # 使用编译后的 opt 对象进行类似操作，验证编译后的结果
        with compiled_autograd.enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            # 断言编译帧计数器的帧数为 1
            self.assertEqual(cnt_bw.frame_count, 1)

        # 比较 x0 和 x2 的梯度，以及 x1 和 x3 的梯度，验证编译后的结果是否与未编译的一致
        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    # 定义一个测试函数，验证在挂钩函数身份更改时不重新编译的行为
    def test_no_recompile_on_hook_identity_change(self):
        # 定义一个自定义的挂钩函数 my_hook，接受 grad 和 k 两个参数
        def my_hook(grad, k=0):
            return grad + k

        # 定义另一个自定义的挂钩函数 my_hook2，用于替换原先的 my_hook 函数
        def my_hook2(grad):
            return grad * 2

        # 定义一个继承自 torch.nn.Module 的自定义模块 MyMod
        class MyMod(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, x):
                # 将输入张量 x 的每个元素乘以 2，保存到 y
                y = x.mul(2)
                # 注册 my_hook 挂钩函数到 y
                y.register_hook(my_hook)
                # 再次注册 my_hook 挂钩函数到 y，验证重复注册行为
                y.register_hook(my_hook)
                # 将 y 中的每个元素乘以 3，保存到 z
                z = y.mul(3)
                return (z,)

        # 创建 MyMod 类的实例 mod
        mod = MyMod()
        # 创建一个 requires_grad=True 的张量 x0，所有元素为 1
        x0 = torch.ones(4, requires_grad=True)
        # 对 mod 的输入 x0 进行前向传播，保存结果到 eager_out
        eager_out = mod(x0)
        # 对 eager_out 的第一个元素进行反向传播，使用全为 1 的张量作为梯度
        eager_out[0].backward(torch.ones(4))

        # 创建另一个 requires_grad=True 的张量 x1，所有元素为 1
        x1 = torch.ones(4, requires_grad=True)
        # 使用编译后的 autograd 机制，编译 mod，并用 cnts 计数器来统计编译帧数
        with compiled_autograd.enable(compiler_fn):
            cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            comp_mod = torch._dynamo.optimize(cnts, nopython=True)(mod)
            # 对编译后的模块 comp_mod 进行前向传播，保存结果到 comp_out
            comp_out = comp_mod(x1)
            # 对 comp_out 的第一个元素进行反向传播，使用全为 1 的张量作为梯度
            comp_out[0].backward(torch.ones(4))

            # 断言编译帧计数器的帧数为 1
            self.assertEqual(cnts.frame_count, 1)
            # 修改全局的 my_hook 函数为 my_hook2，用于验证在 hook 身份更改时的行为
            my_hook = my_hook2  # noqa: F811
            # 比较 x0 和 x1 的梯度，验证编译后的结果是否与未编译的一致
            self.assertEqual(x0.grad, x1.grad)

            # 再次进行非编译模式的前向传播和反向传播，用于比较结果
            eager_out = mod(x0)
            eager_out[0].backward(torch.ones(4))

            comp_out = comp_mod(x1)
            comp_out[0].backward(torch.ones(4))
            # 断言编译帧计数器的帧数为 1
            self.assertEqual(cnts.frame_count, 1)
            # 再次比较 x0 和 x1 的梯度，确认结果保持一致
            self.assertEqual(x0.grad, x1.grad)
    # 定义一个测试函数，用于测试 functools.partial 和 Torch 的编译功能在不同背景下的效果
    def test_functools_arg_vary(self):
        
        # 定义一个预处理钩子函数，接受 grad 参数并乘以 k 后返回
        def pre_hook(grad, *, k):
            return grad * k
        
        # 使用 functools.partial 创建一个新的钩子函数，固定参数 k=1
        hook = functools.partial(pre_hook, k=1)
        
        # 使用 Torch 的编译装饰器，定义一个函数 h，参数为 x，返回 x 乘以 2，并对结果注册钩子 hook，最后乘以 3 返回
        @torch.compile(backend="eager", fullgraph=True)
        def h(x):
            y = x.mul(2)
            y.register_hook(hook)
            return y.mul(3)
        
        # 启用 Torch 的编译自动求导功能
        with compiled_autograd.enable(torch.compile(backend="eager", fullgraph=True)):
            # 创建一个随机张量 x，开启梯度追踪
            x = torch.randn(2, requires_grad=True)
            # 对 h(x) 运行前向传播和反向传播，并对结果求和
            h(x).sum().backward()
            # 保存原始梯度
            orig_grad = x.grad
            # 将 x 的梯度清零
            x.grad = None
            
            # 更新钩子函数为 functools.partial(pre_hook, k=2)
            hook = functools.partial(pre_hook, k=2)
            # 再次对 h(x) 进行前向传播和反向传播，并对结果求和
            h(x).sum().backward()
            # 断言更新后的梯度应为原始梯度乘以 2
            self.assertEqual(orig_grad * 2, x.grad)

    # 定义一个测试后积累梯度钩子函数的函数
    def test_post_acc_grad_hook(self):
        
        # 定义一个钩子函数 hook，对输入张量的每个元素进行平方操作，并将其乘以对应位置的梯度值，然后再将梯度值乘以 5
        def hook(input_t):
            input_t.mul_(input_t.grad)
            input_t.grad.mul_(5)

        # 定义一个函数 reg_and_mul，将钩子函数 hook 注册为 x 的后积累梯度钩子，然后返回 x 与 y 的点乘结果
        def reg_and_mul(x, y):
            x.register_post_accumulate_grad_hook(hook)
            return x * y

        # 定义一个测试函数 test_fn，接受一个函数 fn 作为参数，执行 fn(x, y)，然后对 x 使用 b 进行反向传播
        def test_fn(fn):
            fn(x, y)
            b = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
            x.backward(b)
            if cnts:
                # 断言帧计数器的帧数为 1
                self.assertEqual(cnts.frame_count, 1)
            # 这些相同的断言在急切模式和编译模式下都能运行
            # X 变为 x*2 因为 mul_ 操作
            self.assertEqual(x, torch.tensor([0.5, 0.5, 0.5]) * 2)
            # 这个测试证明了梯度别名工作 -
            self.assertEqual(x.grad, b * 5)

        # 在急切模式下定义张量 x 和 y
        x = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        # 使用 reg_and_mul 运行 test_fn 测试函数
        test_fn(reg_and_mul)

        # 编译模式
        for backend in ["eager", "aot_eager", "inductor"]:
            for compiled_bwd in [False, True]:
                torch._dynamo.reset()
                x = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
                y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

                # 创建带有指定后端的帧计数器 cnts
                cnts = torch._dynamo.testing.CompileCounterWithBackend(backend)
                # 使用 torch._dynamo.optimize 对 reg_and_mul 进行优化并生成编译函数 compiled_fn
                compiled_fn = torch._dynamo.optimize(cnts, nopython=True)(reg_and_mul)

                # 根据 compiled_bwd 的值，设置编译自动求导的上下文
                compiled_bwd_ctx = (
                    compiled_autograd.enable(
                        torch.compile(backend=backend, fullgraph=True)
                    )
                    if compiled_bwd
                    else contextlib.nullcontext()
                )
                # 使用编译后的上下文运行 test_fn 测试函数
                with compiled_bwd_ctx:
                    test_fn(compiled_fn)
    def test_recompile(self):
        # 定义一个用于修改梯度的钩子函数
        def hook(param):
            param.grad *= 2

        # 创建一个包含十个元素并且需要计算梯度的张量
        x = torch.ones(10)
        x.requires_grad = True

        # 定义一个简单的函数，用于执行张量和输入的乘法操作
        def run(input):
            return x * input

        # 将梯度后处理钩子函数注册到张量 x 上
        x.register_post_accumulate_grad_hook(hook)

        # 使用编译后自动求导功能启用编译器函数
        with compiled_autograd.enable(compiler_fn):
            # 循环5次
            for i in range(5):
                with unittest.mock.patch(
                    "torch._dynamo.config.error_on_recompile", True
                ):
                    # 模拟 optimizer.zero_grad() 来清除梯度
                    x.grad = None
                    # 计算 run(i) 的和，并执行反向传播
                    run(i).sum().backward()
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试函数
    run_tests()
```