# `.\pytorch\test\dynamo\test_nops.py`

```py
# Owner(s): ["module: dynamo"]
# 导入 PyTorch 库
import torch

# 导入测试相关的模块
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import eval_frame
from torch._dynamo.hooks import Hooks

# 设置全局变量 c
c = 10


# 定义函数 fn1，计算 a + b - c 的结果
def fn1(a, b):
    return a + b - c


# 定义函数 fn2，内部包含嵌套函数 modify()，计算 x 和 y 的和
def fn2(a, b):
    x = 0
    y = 1

    def modify():
        nonlocal x
        x += a + b + c

    # 调用 modify() 函数两次
    for _ in range(2):
        modify()

    return x + y


# 定义生成器函数 fn3，产生两个值 1 和 2
def fn3():
    yield 1
    yield 2


# 使用 eval_frame._optimize_catch_errors() 方法调用 debug_insert_nops 函数，返回包含 Hooks 对象的装饰器函数 with_debug_nops
with_debug_nops = eval_frame._optimize_catch_errors(
    torch._dynamo.testing.debug_insert_nops, Hooks(None, None)
)


# 定义测试类 NopTests，继承自 torch._dynamo.test_case.TestCase
class NopTests(torch._dynamo.test_case.TestCase):
    # 使用装饰器 with_debug_nops 修饰的测试方法 test1
    @with_debug_nops
    def test1(self):
        # 断言 fn1(1, 2) 的返回值为 -7
        self.assertEqual(fn1(1, 2), -7)
        # 再次断言 fn1(1, 2) 的返回值为 -7
        self.assertEqual(fn1(1, 2), -7)

    # 使用装饰器 with_debug_nops 修饰的测试方法 test2
    @with_debug_nops
    def test2(self):
        # 断言 fn2(1, 2) 的返回值为 27
        self.assertEqual(fn2(1, 2), 27)
        # 再次断言 fn2(1, 2) 的返回值为 27
        self.assertEqual(fn2(1, 2), 27)

    # 使用装饰器 with_debug_nops 修饰的测试方法 test3
    @with_debug_nops
    def test3(self):
        # 调用 fn3() 生成器，断言第一个 next(t) 返回值为 1
        t = fn3()
        self.assertEqual(next(t), 1)
        # 再次断言 next(t) 返回值为 2
        self.assertEqual(next(t), 2)
        # 使用 lambda 表达式断言调用 next(t) 会引发 StopIteration 异常
        self.assertRaises(StopIteration, lambda: next(t))

    # 测试扩展参数的方法 test_extended_args
    def test_extended_args(self):
        # 构造一个过多添加字符串的表达式
        too_many_adds = "+".join(["a", "b"] * 256)
        # 构造 lambda 表达式 source
        source = (
            f"lambda a, b: ({too_many_adds}+a if a.sum() > 0 else {too_many_adds} - b)"
        )
        # 使用 eval 执行 source，得到函数 fn
        fn = eval(source)
        a = torch.ones(1)
        b = torch.ones(1)
        # 使用 with_debug_nops 装饰 fn 函数
        fn = with_debug_nops(fn)
        # 断言 fn(a, b).sum() 的结果为 513
        self.assertEqual(fn(a, b).sum(), 513)


# 如果当前脚本是主程序
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数并执行
    from torch._dynamo.test_case import run_tests
    run_tests()
```