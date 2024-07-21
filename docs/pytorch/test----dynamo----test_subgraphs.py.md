# `.\pytorch\test\dynamo\test_subgraphs.py`

```
# Owner(s): ["module: dynamo"]
from unittest.mock import patch  # 导入 patch 函数用于模拟测试

import torch  # 导入 PyTorch 库

import torch._dynamo.test_case  # 导入 PyTorch 内部测试框架相关模块
import torch._dynamo.testing  # 导入 PyTorch 内部测试工具模块
from torch._dynamo.testing import unsupported  # 从测试工具模块中导入 unsupported 函数
from torch._dynamo.utils import ifdynstaticdefault  # 导入 PyTorch 内部工具函数

globalmod = torch.nn.ReLU()  # 创建一个全局的 ReLU 模块实例


def indirectly_unsupported(a, b):
    c = a + b  # 计算 a 和 b 的和
    return unsupported(a, c)  # 调用测试工具中的 unsupported 函数


class SubGraphTests(torch._dynamo.test_case.TestCase):
    def _common(self, fn, frame_count, op_count):
        torch._dynamo.reset()  # 重置 PyTorch 内部状态
        v1 = torch.ones(10)  # 创建一个包含十个 1 的张量 v1
        v2 = torch.ones(10) * -2.0  # 创建一个包含十个 -2 的张量 v2
        correct1 = fn(v1, v2)  # 使用给定函数 fn 计算结果
        correct2 = fn(v2, v1)  # 使用给定函数 fn 计算结果
        cnt = torch._dynamo.testing.CompileCounter()  # 创建编译计数器实例
        opt_fn = torch._dynamo.optimize(cnt)(fn)  # 优化给定函数 fn，并返回优化后的函数 opt_fn
        r1 = opt_fn(v1, v2)  # 使用优化后的函数 opt_fn 计算结果
        r2 = opt_fn(v2, v1)  # 使用优化后的函数 opt_fn 计算结果
        self.assertTrue(torch._dynamo.testing.same(r1, correct1))  # 断言优化结果与正确结果相同
        self.assertTrue(torch._dynamo.testing.same(r2, correct2))  # 断言优化结果与正确结果相同
        self.assertEqual(
            cnt.frame_count,
            frame_count,
            f"actual {cnt.frame_count} != expected {frame_count}",  # 检查帧数是否符合预期
        )
        self.assertEqual(cnt.op_count, op_count)  # 检查操作数是否符合预期

    def test_control_flow1(self):
        def fn(a, b):
            c1 = a - b  # 计算 a 和 b 的差
            c2 = b - a  # 计算 b 和 a 的差
            if c1.sum() > c2.sum():  # 如果 c1 的总和大于 c2 的总和
                return c1  # 返回 c1
            else:
                return c2  # 返回 c2

        self._common(fn, 1, 5)  # 调用 _common 方法进行测试

    def test_control_flow2(self):
        def fn(a, b):
            if a.sum() > b.sum():  # 如果 a 的总和大于 b 的总和
                return 1  # 返回 1
            else:
                return 2  # 返回 2

        self._common(fn, 1, 3)  # 调用 _common 方法进行测试

    def test_control_flow3(self):
        def fn(a, b):
            c1 = a - b  # 计算 a 和 b 的差
            c2 = b - a  # 计算 b 和 a 的差
            m = globalmod  # 使用全局的 ReLU 模块
            if c1.sum() > c2.sum():  # 如果 c1 的总和大于 c2 的总和
                return m(c1)  # 对 c1 应用全局模块 m
            else:
                return m(c2)  # 对 c2 应用全局模块 m

        self._common(fn, 3, 7)  # 调用 _common 方法进行测试

    def test_control_flow4(self):
        def fn(a, b):
            tmp1 = a.sum() > b.sum() and a.sum() > 0  # 计算临时变量 tmp1
            if tmp1:  # 如果 tmp1 为真
                return 1  # 返回 1
            else:
                return 2  # 返回 2

        self._common(fn, 3, 5)  # 调用 _common 方法进行测试

    def test_control_flow5(self):
        def fn(a, b):
            tmp1 = a.sum() > b.sum() and a.sum() > 0  # 计算临时变量 tmp1
            tmp2 = a.sum() < b.sum() or b.sum() > 0  # 计算临时变量 tmp2
            if tmp1 and tmp2:  # 如果 tmp1 和 tmp2 同时为真
                return 1, tmp1, tmp2  # 返回结果 1，tmp1，tmp2
            else:
                return 2, tmp1, tmp2  # 返回结果 2，tmp1，tmp2

        self._common(fn, 6, 13)  # 调用 _common 方法进行测试

    def test_capi_call1(self):
        def fn(a, b):
            c1 = a - b  # 计算 a 和 b 的差
            c2 = b - a  # 计算 b 和 a 的差
            return unsupported(c1, c2)  # 调用 unsupported 函数

        self._common(fn, 1, 2)  # 调用 _common 方法进行测试

    def test_capi_call2(self):
        def fn(a, b):
            c1 = a - b  # 计算 a 和 b 的差
            c2 = b - a  # 计算 b 和 a 的差
            return a - (b - unsupported(c1, c2))  # 进行复杂计算，包含 unsupported 函数调用

        self._common(fn, 2, 4)  # 调用 _common 方法进行测试

    def test_capi_call3(self):
        def fn(a, b):
            c1 = a - b  # 计算 a 和 b 的差
            c2 = b - a  # 计算 b 和 a 的差
            return torch._dynamo.testing.unsupported(c1, c2)  # 调用 PyTorch 内部测试工具中的 unsupported 函数

        self._common(fn, 1, 2)  # 调用 _common 方法进行测试
    def test_indirect_unsupported1(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 c1 和 c2 分别为 a - b 和 b - a
            c1 = a - b
            c2 = b - a
            # 调用 indirectly_unsupported 函数处理 c1 和 c2 的结果并返回
            return indirectly_unsupported(c1, c2)

        # 调用测试函数 _common，验证 fn 的行为
        self._common(fn, 2, 3)

    def test_indirect_unsupported2(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 定义本地常量 local_const1 和 local_const2
            local_const1 = 7
            local_const2 = 22
            # 计算 c1 和 c2 分别为 a - b 和 b - a
            c1 = a - b
            c2 = b - a
            # 调用 indirectly_unsupported 处理 c1 和 c2 的结果并与 local_const1 和 local_const2 进行计算
            return local_const1 / (local_const2 - indirectly_unsupported(c1, c2))

        # 调用测试函数 _common，验证 fn 的行为
        self._common(fn, 3, 5)

    def test_indirect_unsupported3(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 构建参数列表 args，包含 a - b 和 b - a
            args = [a - b, b - a]
            # 调用 indirectly_unsupported 函数处理参数列表 args
            return indirectly_unsupported(*args)

        # 调用测试函数 _common，验证 fn 的行为
        self._common(fn, 2, 3)

    def test_stack_state1(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 t1 和 t2
            t1 = 1.23 * a
            t2 = 4.56 * a
            # 计算 c1 和 c2 分别为 a - b 和 b - a
            c1 = a - b
            c2 = b - a
            # 调用 unsupported 处理 c1 和 c2 的结果，并与 t1 和 t2 进行计算
            return t1 / (t2 - unsupported(c1, c2))

        # 调用测试函数 _common，验证 fn 的行为
        self._common(fn, 2, 6)

    def test_stack_state2(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 t1 和 t2
            t1 = 1.23 * a
            t2 = 4.56 * a
            # 计算 c1 和 c2 分别为 a - b 和 b - a
            c1 = a - b
            c2 = b - a
            # 调用 indirectly_unsupported 处理 c1 和 c2 的结果，并与 t1 和 t2 进行计算
            return t1 / (t2 - indirectly_unsupported(c1, c2))

        # 调用测试函数 _common，验证 fn 的行为
        self._common(fn, 3, 7)

    def test_multigraph(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 x 为 a 和 b 的和的平均值
            x = a + b
            x = x / 2.0
            # 如果 x 的总和小于 0，则返回 x 的负值，否则返回 x
            if x.sum() < 0:
                return x * -1.0
            return x

        # 调用测试函数 _common，验证 fn 的行为
        self._common(fn, 2, 5)

    def test_extended_args(self):
        # 构建一个过长的字符串 too_many_adds
        too_many_adds = "+".join(["a", "b"] * 256)
        # 构建源字符串 source，包含表达式 lambda 函数
        source = (
            f"lambda a, b: ({too_many_adds}+a if a.sum() > 0 else {too_many_adds} - b)"
        )
        # 通过 eval 函数将源字符串转换为函数，并调用 _common 进行验证
        self._common(eval(source), 3, 1026)

    def test_resume1(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 x 为 a 和 b 的和的平均值
            x = a + b
            x = x / 2.0
            # 将 x 和 a 作为参数调用 unsupported 函数
            x = unsupported(x, a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        # 调用测试函数 _common，验证 fn 的行为
        self._common(fn, 2, 6)

    def test_resume2(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 x 为 a 和 b 的和的平均值
            x = a + b
            x = x / 2.0
            # 将 x 和 a 作为参数调用 indirectly_unsupported 函数
            x = indirectly_unsupported(x, a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        # 调用测试函数 _common，验证 fn 的行为
        self._common(fn, 3, 7)

    def test_resume3(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 x 为 a 和 b 的和的平均值
            x = a + b
            x = x / 2.0
            # 将 x 和 b 作为参数调用 indirectly_unsupported 函数
            x = indirectly_unsupported(x, b=a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        # 调用测试函数 _common，验证 fn 的行为
        self._common(fn, 3, 7)

    def test_resume4(self):
        # 定义一个内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 x 为 a 和 b 的和的平均值
            x = a + b
            x = x / 2.0
            # 将 a 和 b 作为命名参数调用 indirectly_unsupported 函数
            x = indirectly_unsupported(a=x, b=a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        # 调用测试函数 _common，验证 fn 的行为
        self._common(fn, 3, 7)
    # 定义一个测试函数 test_resume5，用于测试函数 fn
    def test_resume5(self):
        # 定义内部函数 fn，接受参数 a 和 b
        def fn(a, b):
            # 计算 a 和 b 的和赋给 x
            x = a + b
            # 将 x 除以 2.0 赋给 x
            x = x / 2.0
            # 将 x 加上 2.0 赋给 x
            x = x + 2.0
            # 打印 x 的值
            print(x)
            # 连续将 x 加上 2.0 赋给 x
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            # 返回最终的 x 值
            return x

        # 调用测试公共方法 _common，传入 fn 函数和参数 2, 6
        self._common(fn, 2, 6)

    # 定义一个测试函数 test_start1，用于测试函数 fn
    def test_start1(self):
        # 定义内部函数 fn，接受参数 a 和 b
        def fn(a, b):
            # 打印参数 a 的值
            print(a)
            # 计算 a 和 b 的和赋给 x
            x = a + b
            # 连续将 x 加上 2.0 赋给 x
            x = x + 2.0
            x = x + 2.0
            # 返回最终的 x 值
            return x

        # 调用测试公共方法 _common，传入 fn 函数和参数 1, 3
        self._common(fn, 1, 3)

    # 定义一个测试函数 test_start2，用于测试函数 fn
    def test_start2(self):
        # 定义内部函数 fn，接受参数 a 和 b
        def fn(a, b):
            # 调用 indirectly_unsupported 函数处理参数 a 和 b，结果赋给 x
            x = indirectly_unsupported(a, b)
            # 连续将 x 加上 2.0 赋给 x
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            # 返回最终的 x 值
            return x

        # 调用测试公共方法 _common，传入 fn 函数和参数 2, 4
        self._common(fn, 2, 4)

    # 定义一个测试函数 test_start3，用于测试函数 fn
    def test_start3(self):
        # 定义内部函数 fn，接受参数 a 和 b
        def fn(a, b):
            # 调用 unsupported 函数处理参数 a 和 b，结果赋给 x
            x = unsupported(a, b)
            # 连续将 x 加上 2.0 赋给 x
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            # 返回最终的 x 值
            return x

        # 调用测试公共方法 _common，传入 fn 函数和参数 1, 3
        self._common(fn, 1, 3)

    # 定义一个测试函数 test_start4
    def test_start4(self):
        # 定义函数 fn，接受参数 a, b, check
        def fn(a, b, check):
            # 如果 check 为真，返回 a + b + 10，否则返回 a + b - 10
            if check:
                return a + b + 10
            else:
                return a + b - 10

        # 生成随机张量 v1 和 v2
        v1 = torch.randn(10)
        v2 = torch.randn(10)
        # 创建值为 0 的整型张量 f 和值为 1 的整型张量 t
        f = torch.zeros(1, dtype=torch.int32)
        t = torch.ones(1, dtype=torch.int32)
        # 分别使用 fn 函数计算不同参数组合的结果
        correct1 = fn(v1, v2, t)
        correct2 = fn(v1, v2, f)
        # 创建 CompileCounter 实例 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，得到 opt_fn
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 分别使用优化后的 opt_fn 函数计算不同参数组合的结果
        r1 = opt_fn(v1, v2, t)
        r2 = opt_fn(v1, v2, f)
        # 断言优化后的结果与预期结果相同
        self.assertTrue(torch._dynamo.testing.same(r1, correct1))
        self.assertTrue(torch._dynamo.testing.same(r2, correct2))
        # 断言编译帧数和操作数符合预期
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 4)

    # 定义一个测试函数 test_resume_freevars
    def test_resume_freevars(self):
        # 生成随机张量 c1 和 c2
        c1 = torch.randn(10)
        c2 = torch.randn(10)

        # 定义函数 fn，接受参数 a 和 b
        def fn(a, b):
            # 计算 a + b + (c1 - c2) 的结果赋给 x
            x = a + b + (c1 - c2)
            # 调用 unsupported 函数处理 x 和 x，结果赋给 x
            x = unsupported(x, x)
            # 返回 x + (c1 - c2) 的结果
            return x + (c1 - c2)

        # 调用测试公共方法 _common，传入 fn 函数和参数 2, 5
        self._common(fn, 2, 5)

    # 定义一个测试函数 test_restore_state
    def test_restore_state(self):
        # 定义函数 fn，接受参数 a 和 b
        def fn(a, b):
            # 将 len 函数赋给 len_ 变量
            len_ = len
            # 计算 a + b 的结果赋给 x
            x = a + b
            # 调用 unsupported 函数处理 x 和 x，结果再加 1赋给 x
            x = torch.add(unsupported(x, x), 1)
            # 返回 a * x + len_(b) 的结果
            return a * x + len_(b)

        # 调用测试公共方法 _common，传入 fn 函数和参数 2, ifdynstaticdefault(4, 5)
        self._common(fn, 2, ifdynstaticdefault(4, 5))

    # 定义一个测试函数 test_restore_range
    def test_restore_range(self):
        # 定义函数 fn，接受参数 a 和 b
        def fn(a, b):
            # 计算 a + b 的结果赋给 x
            x = a + b
            # 创建步长为 2，起始值为 3，结束值为 8 的范围对象 rng
            rng = range(3, 8, 2)
            # 调用 unsupported 函数处理 x 和 x，结果赋给 x
            x = unsupported(x, x)
            # 遍历范围 rng 中的元素，将其加到 x 上
            for i in rng:
                x = x + i
            # 返回最终的 x 值
            return x

        # 当使用动态形状时，不对 range 进行特化，导致循环无法展开
        # TODO: 考虑在迭代循环时强制特化
        # 调用测试公共方法 _common，传入 fn 函数和参数 ifdynstaticdefault(2, 1), ifdynstaticdefault(4, 1)
        self._common(fn, ifdynstaticdefault(2, 1), ifdynstaticdefault(4, 1))

    # 定义一个测试函数 test_restore_range_iter
    def test_restore_range_iter(self):
        # 定义函数 fn，接受参数 a 和 b
        def fn(a, b):
            # 计算 a + b 的结果赋给 x
            x = a + b
            # 创建迭代器对象 rng，从范围对象 range(3, 8, 2) 中获取下一个元素并加到 x 上
            rng = iter(range(3, 8, 2))
            x = unsupported(x, x)
    # 定义一个测试函数，测试在恢复后弹出操作的行为
    def test_pop_after_resume(self):
        # 定义一个内部函数fn，接受两个参数a和b
        def fn(a, b):
            # 创建临时列表tmp，存储a+1, b+2, a+b三个元素
            tmp = [a + 1, b + 2, a + b]
            # 初始化变量x为a
            x = a
            # 调用unsupported函数，将x传入自身作为参数（此处表明这是一个伪代码）
            x = unsupported(x, x)
            # 迭代临时列表tmp中的元素，每次从末尾弹出一个元素并加到x上
            for i in range(3):
                x += tmp.pop(-1)
            # 返回计算结果x
            return x

        # 调用测试基类的_common方法，验证fn函数在输入2和6时的输出
        self._common(fn, 2, 6)

    # 使用patch修饰器，设置torch._dynamo.config.assume_static_by_default为False
    @patch("torch._dynamo.config.assume_static_by_default", False)
    # 定义测试动态getitem操作的函数
    def test_dynamic_getitem(self):
        # 定义一个接受两个参数a和b的函数fn，返回a的第b.size(0) - 1个元素
        def fn(a, b):
            return a[b.size(0) - 1]

        # 创建CompileCounter对象cnt，用torch._dynamo.optimize修饰fn函数，并赋值给opt_fn
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 多次调用opt_fn函数，传入不同大小的torch.randn张量
        for i in range(3, 12):
            opt_fn(torch.randn(i), torch.randn(i))
        # 断言只生成了一个计算图
        self.assertEqual(cnt.frame_count, 1)

    # 定义测试动态kwargs操作的函数
    def test_dynamic_kwarg(self):
        # 定义一个接受两个参数a和b的函数fn，返回a减去b乘以10的结果
        def fn(a, b):
            return a - b * 10

        # 重置torch._dynamo的状态
        torch._dynamo.reset()
        # 创建CompileCounter对象cnt_dynamic，用torch._dynamo.optimize修饰fn函数（动态模式），并赋值给opt_fn
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt_dynamic, dynamic=True)(fn)
        # 设置起始值start为2，终止值end为12
        start = 2
        end = 12
        steps = end - start
        # 多次调用opt_fn函数，传入不同大小的torch.randn张量
        for i in range(start, end):
            opt_fn(torch.randn(i), torch.randn(i))

        # 断言只生成了一个计算图
        self.assertEqual(cnt_dynamic.frame_count, 1)

    # 定义测试动态duck size操作的函数
    def test_dynamic_duck_size(self):
        # 定义一个接受两个参数a和b的函数fn
        def fn(a, b):
            # 如果a和b的第一维大小相等，则返回a和b的元素级加法结果
            if a.size(0) == b.size(0):
                return a + b
            else:
                # 否则分别对a和b进行求和操作并返回结果
                return a.sum() + b.sum()

        # 重置torch._dynamo的状态
        torch._dynamo.reset()
        # 创建CompileCounter对象cnt_dynamic，用torch._dynamo.optimize修饰fn函数（动态模式），并赋值给opt_fn
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt_dynamic, dynamic=True)(fn)
        # 创建两个torch.randn张量x和y
        x = torch.randn(2)
        y = torch.randn(3)
        # 断言opt_fn对x和x的调用结果与fn函数相同
        self.assertEqual(opt_fn(x, x), fn(x, x))
        # 断言opt_fn对x和y的调用结果与fn函数相同
        self.assertEqual(opt_fn(x, y), fn(x, y))
        # 断言生成了两个计算图
        self.assertEqual(cnt_dynamic.frame_count, 2)

    # 定义测试动态顺序依赖性的函数
    def test_dynamic_order_dependence(self):
        # 定义一个接受两个参数a和b的函数fn，返回a和b的元素级求和结果
        def fn(a, b):
            return a.sum() + b.sum()

        # 重置torch._dynamo的状态
        torch._dynamo.reset()
        # 创建CompileCounter对象cnt_dynamic，用torch._dynamo.optimize修饰fn函数，并赋值给opt_fn
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt_dynamic)(fn)
        # 创建两个torch.randn张量x和y
        x = torch.randn(2)
        y = torch.randn(3)
        # 断言opt_fn对x和y的调用结果与fn函数相同
        self.assertEqual(opt_fn(x, y), fn(x, y))
        # 断言opt_fn对x和x的调用结果与fn函数相同
        self.assertEqual(opt_fn(x, x), fn(x, x))
        # NB: 这里的frame_count可能是2，但我们不测试x和y没有共同的尺寸时的保护条件，
        # 因此我们得到一个通用图，当x和y碰巧有共同的尺寸时也能工作。
        self.assertEqual(cnt_dynamic.frame_count, 2)

        # 重置torch._dynamo的状态
        torch._dynamo.reset()
        # 将cnt_dynamic的frame_count重置为0
        cnt_dynamic.frame_count = 0
        # 断言opt_fn对x和x的调用结果与fn函数相同（这会过度特化！）
        self.assertEqual(opt_fn(x, x), fn(x, x))
        # 断言opt_fn对x和y的调用结果与fn函数相同
        self.assertEqual(opt_fn(x, y), fn(x, y))
        # 断言生成了两个计算图
        self.assertEqual(cnt_dynamic.frame_count, 2)
    def test_dynamic_zero_inference(self):
        # 定义一个函数 fn，接受参数 a，根据 a 的 size(0) 是否为 0 进行条件判断，返回不同计算结果
        def fn(a):
            if a.size(0) != 0:
                return a * 2
            else:
                return a + 1

        # 重置 Torch 的动态优化器状态
        torch._dynamo.reset()
        # 创建一个编译计数器对象 cnt_dynamic
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 进行动态优化，并返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnt_dynamic, dynamic=True)(fn)
        # 创建一个张量 x，形状为 (0,)
        x = torch.randn(0)
        # 创建一个张量 y，形状为 (2,)
        y = torch.randn(2)
        # 断言优化后的函数 opt_fn 对 y 和 x 的计算结果与未优化的函数 fn 相同
        self.assertEqual(opt_fn(y), fn(y))
        self.assertEqual(opt_fn(x), fn(x))
        # 断言编译帧数为 2
        self.assertEqual(cnt_dynamic.frame_count, 2)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_no_graph_break_on_item(self):
        # 定义一个函数 fn，接受参数 a 和 b，执行一系列数学运算并返回结果 x
        def fn(a, b):
            x = a + b - 1.5
            x = x.sum()
            x.item()  # 获取 x 的标量值
            x = x / (a + b)
            return x

        # 调用 _common 方法执行测试，验证 item 操作是否被 DCE（死代码消除）
        self._common(fn, 1, 5)  # item gets DCE'd

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", False)
    def test_graph_break_on_item(self):
        # 定义一个函数 fn，接受参数 a 和 b，执行一系列数学运算并返回结果 x
        def fn(a, b):
            x = a + b - 1.5
            x = x.sum()
            x.item()  # 获取 x 的标量值
            x = x / (a + b)
            return x

        # 调用 _common 方法执行测试，验证 item 操作是否影响计算图
        self._common(fn, 2, 5)

    def test_resume_paths_join(self):
        # 定义一个函数 fn，接受参数 x, c1, c2, c3，根据条件逐步对 x 进行加法操作并返回结果
        def fn(x, c1, c2, c3):
            x = x + 1
            if c1:
                x = x + 2
            x = x + 3
            if c2:
                x = x + 4
            x = x + 5
            if c3:
                x = x + 6
            return x + 7

        # 创建一个形状为 (10,) 的随机张量 v1
        v1 = torch.randn(10)
        # 创建一个值为 True 的张量 t 和值为 False 的张量 f
        t = torch.Tensor([True])
        f = torch.Tensor([False])
        # 创建一个编译计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 进行动态优化，并返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 遍历参数组合 (t, f) * (t, f) * (t, f)，分别传入 opt_fn 进行计算
        for a in (t, f):
            for b in (t, f):
                for c in (t, f):
                    opt_fn(v1, a, b, c)

        # 断言编译帧数为 7
        self.assertEqual(cnt.frame_count, 7)
        # 断言操作计数为 10
        self.assertEqual(cnt.op_count, 10)

    def test_resume_with_no_grad1(self):
        # 定义一个函数 fn，接受参数 a 和 b，根据 torch.no_grad() 上下文进行数学运算并返回结果 x
        def fn(a, b):
            x = a + b
            with torch.no_grad():
                x = x + 1
                x.sum().tolist()  # 在此处中断计算图
                x = x + 2
            x = x + 3
            return x

        # 调用 _common 方法执行测试，验证在不同的 torch.no_grad() 上下文中 fn 的行为
        self._common(fn, 2, 9)
        # 重置 Torch 的动态优化器状态
        torch._dynamo.reset()
        with torch.no_grad():
            self._common(fn, 2, 5)

    def test_resume_with_no_grad2(self):
        # 定义一个函数 fn，接受参数 a 和 b，根据 torch.no_grad() 上下文进行数学运算并返回结果 x
        def fn(a, b):
            x = a + b
            with torch.no_grad():
                x = x + 1
                x.sum().tolist()  # 在此处中断计算图
                x = x + 2
                x.sum().tolist()  # 在此处中断计算图
                x = x + 3
            x = x + 4
            return x

        # 调用 _common 方法执行测试，验证在不同的 torch.no_grad() 上下文中 fn 的行为
        self._common(fn, 3, 13)
    # 定义一个测试函数 test_resume_with_no_grad3，测试在没有梯度情况下的函数执行
    def test_resume_with_no_grad3(self):
        # 定义内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 a + b 并赋给变量 x
            x = a + b
            # 使用 torch.no_grad() 禁止梯度计算
            with torch.no_grad():
                # 内部的第一个 torch.no_grad() 块
                with torch.no_grad():
                    # 对 x 加 1
                    x = x + 1
                    # 使用 torch.enable_grad() 重新启用梯度计算
                    with torch.enable_grad():
                        # 计算 x 的和，并转换为列表形式（graph break 表示断开梯度图）
                        x.sum().tolist()
                        # 取 x 的第一个元素并加 2
                        x = x[0] + 2
                    # 对 x 加 3
                    x = x + 3
            # 对 x 加 4
            x = x + 4
            # 返回计算结果 x
            return x

        # 调用 self._common 函数测试 fn 函数，传入参数 2 和 11
        self._common(fn, 2, 11)

    # 定义一个测试函数 test_resume_tuple_iterator，测试元组迭代器的函数执行
    def test_resume_tuple_iterator(self):
        # 定义内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 计算 a + b 并赋给变量 x
            x = a + b
            # 创建元组范围的迭代器 it
            it = iter(tuple(range(10)))
            # 对 x 加上 it 的下一个元素
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            # 调用 unsupported 函数，将 x 作为参数传入两次
            x = unsupported(x, x)
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            # 返回计算结果 x
            return x

        # 调用 self._common 函数测试 fn 函数，传入参数 2 和 8
        self._common(fn, 2, 8)

    # 定义一个测试函数 test_tuple_iterator_return，测试元组迭代器返回的函数执行
    def test_tuple_iterator_return(self):
        # 定义内部函数 fn，接受一个参数 x
        def fn(x):
            # 创建元组范围的迭代器 it
            it = iter(tuple(range(10)))
            # 对 x 加上 it 的下一个元素
            x = x + next(it)
            x = x + next(it)
            # 调用 unsupported 函数，将 x 作为参数传入两次
            x = unsupported(x, x)
            x = x + next(it)
            x = x + next(it)
            # 调用 unsupported 函数，将 x 作为参数传入两次
            x = unsupported(x, x)
            x = x + next(it)
            x = x + next(it)
            # 返回计算结果 x 和迭代器 it
            return x, it

        # 生成一个包含 10 个随机数的张量 v1
        v1 = torch.randn(10)
        # 调用 fn 函数，传入 v1，并接收返回值 v2 和 it2
        v2, it2 = fn(v1)
        # 创建一个 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化处理，并赋值给 opt_fn
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 分别调用 opt_fn 函数两次，传入 v1，并接收返回值 v3 和 it3，v4 和 it4
        v3, it3 = opt_fn(v1)
        v4, it4 = opt_fn(v1)
        # 断言 v2 和 v3 相等，转换为列表比较
        self.assertEqual(v2.tolist(), v3.tolist())
        # 断言 v2 和 v4 相等，转换为列表比较
        self.assertEqual(v2.tolist(), v4.tolist())
        # 断言 it2 和 it3 的内容相同，转换为列表比较
        self.assertEqual(list(it2), list(it3))
        # 断言 cnt.frame_count 的值为 3
        self.assertEqual(cnt.frame_count, 3)
        # 断言 cnt.op_count 的值为 6
        self.assertEqual(cnt.op_count, 6)

    # 定义一个测试函数 test_tuple_iterator_mutate，测试修改元组迭代器的函数执行
    def test_tuple_iterator_mutate(self):
        # 定义内部函数 fn，接受两个参数 x 和 it
        def fn(x, it):
            # 对 x 加上 it 的下一个元素
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            # 返回计算结果 x
            return x

        # 生成一个包含 10 个随机数的张量 v1
        v1 = torch.randn(10)
        # 创建元组范围的迭代器 it1
        it1 = iter(tuple(range(10)))
        # 对 fn 函数进行优化处理，并赋值给 opt_fn
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 断言优化后的 opt_fn 函数的结果与预期值相等，转换为列表比较
        self.assertEqual(opt_fn(v1, it1).tolist(), (v1 + 1 + 2 + 3).tolist())
        # 断言 it1 的内容为 [4, 5, 6, 7, 8, 9]
        self.assertEqual(list(it1), [4, 5, 6, 7, 8, 9])

    # 定义一个测试函数 test_enumerate_not_break_graph，测试 enumerate 不中断梯度图的函数执行
    def test_enumerate_not_break_graph(self):
        # 定义内部函数 fn，接受两个参数 a 和 b
        def fn(a, b):
            # 对 a 的形状进行枚举，返回索引 i 和值 x
            for i, x in enumerate(a.shape):
                # 对 b 加上 x 的值
                b = b + x
            # 对 b 的形状进行枚举，指定初始索引为 8，返回索引 i 和值 x
            for i, x in enumerate(b.shape, 8):
                # 对 b 加上 x 乘以 i 的值
                b = b + x * i
            # 返回计算结果 b
            return b

        # 调用 self._common 函数测试 fn 函数，传入参数 1 和 ifdynstaticdefault(2, 7) 的结果
        self._common(fn, 1, ifdynstaticdefault(2, 7))
# 如果当前脚本作为主程序执行（而不是被导入作为模块），则执行以下代码块
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试函数，用于执行测试用例并输出结果
    run_tests()
```