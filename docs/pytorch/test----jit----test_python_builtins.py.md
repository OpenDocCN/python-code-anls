# `.\pytorch\test\jit\test_python_builtins.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的模块
import os
import random
import sys
import tempfile
from textwrap import dedent

# 导入 PyTorch 相关模块
import torch
from torch.testing._internal.jit_utils import execWrapper, JitTestCase

# 将测试文件夹 test/ 中的文件变为可导入状态
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果作为主程序运行，则抛出运行时错误，提示使用正确的运行方式
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


def get_fn(file_name, script_path):
    # 导入动态生成的 Python 脚本作为模块
    import importlib.util

    # 创建模块的规范
    spec = importlib.util.spec_from_file_location(file_name, script_path)
    # 根据规范加载模块
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # 返回加载的模块中的函数 fn
    fn = module.fn
    return fn


class TestPythonBuiltinOP(JitTestCase):
    def test_add(self):
        # 定义加法函数
        def func(a, b):
            # 执行加法操作
            c = a + b
            # 进行就地加法操作
            c += a
            return c

        # 创建需要梯度的随机张量
        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        # 使用脚本检查函数
        self.checkScript(func, (a, b), optimize=True)

    def test_mul(self):
        # 定义乘法函数
        def func(a, b):
            return a * b

        # 创建需要梯度的随机张量
        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        # 使用脚本检查函数
        self.checkScript(func, (a, b), optimize=True)

    def test_matmul_py3(self):
        # 定义包含矩阵乘法的 Python 3 代码片段
        code = dedent(
            """
        def fn(a, b):
            return a @ b
        """
        )

        # 在临时目录中创建 Python 脚本文件
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)
            # 加载脚本中的函数 fn
            fn = get_fn("test_matmul_py3", script_path)

            # 创建需要梯度的随机张量
            a = torch.rand(4, 3, requires_grad=True)
            b = torch.rand(3, 2, requires_grad=True)
            # 使用脚本检查函数
            self.checkScript(fn, (a, b), optimize=True)

    def test_pow(self):
        # 定义指数函数
        def func(a, b):
            return a**b

        # 定义复杂的指数操作函数
        def func2(a, b, c, d):
            return c + a**b**d

        # 定义带有类型注释的指数函数
        def func3(a, b):
            # type: (int, float) -> float
            return a**b

        # 定义无参数的指数函数
        def func4():
            # type: () -> float
            return 2**-2

        # 定义处理张量的指数函数
        def func5(x, y):
            return x.item() ** y.item()

        # 创建需要梯度的随机张量
        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        c = torch.rand(1, requires_grad=True)
        d = torch.rand(1, requires_grad=True)

        # 使用脚本检查函数
        self.checkScript(func, (a, b), optimize=True)
        self.checkScript(func2, (a, b, c, d), optimize=True)
        self.checkScript(func3, (4, -0.5), optimize=True)
        self.checkScript(func4, ())

        # 定义输入的张量列表
        inputs = [
            torch.tensor(2),
            torch.tensor(-2),
            torch.tensor(0.5),
            torch.tensor(0.2),
        ]
        for x in inputs:
            for y in inputs:
                if x < 0:
                    continue
                else:
                    # 使用脚本检查函数
                    self.checkScript(func5, (x, y))
    def test_stepped_tuple_slicing(self):
        # 定义一个测试函数，用于检查元组的切片操作
        def check_slicing_tuple(slicing, tuple_type, tuple):
            # 准备模板字符串，用于生成切片函数的代码
            template = dedent(
                """
                def func(x):
                    # type: ({}) -> Any
                    # 返回对输入 x 进行 {} 切片操作的结果
                    return x{}
                """
            )
            # 使用模板字符串生成切片函数的代码，并执行检查
            self._check_code(template.format(tuple_type, slicing), "func", [tuple])

        # 执行多组切片测试
        check_slicing_tuple("[-3:3:2]", "Tuple[int, int, int]", (0, 1, 2))
        check_slicing_tuple("[::55]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4))
        check_slicing_tuple("[:4:4]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4))
        check_slicing_tuple("[::-1]", "Tuple[int, int, int, int, int, int, int]", (0, 1, 2, 3, 4, 5, 6))
        check_slicing_tuple("[7:5:2]", "Tuple[int, int, int, int, int, int, int]", (0, 1, 2, 3, 4, 5, 6))
        check_slicing_tuple("[5:7:-2]", "Tuple[int, int, int, int, int, int, int]", (0, 1, 2, 3, 4, 5, 6))
        check_slicing_tuple("[::-2]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4))
        check_slicing_tuple("[:4:-3]", "Tuple[int, int, int, int, int, int]", (0, 1, 2, 3, 4, 5))
        check_slicing_tuple("[3::-2]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4))
    def test_adv_indexing_list(self):
        # 测试使用列表进行高级索引，等同于使用张量进行索引
        def func1(x):
            # 返回索引为[0, 1, 5]的元素
            return x[[0, 1, 5]]

        def func2(x):
            # 返回索引为[(0,0), (1,1)]的元素
            return x[[0, 1], [0, 1]]

        def func3(x):
            # 返回索引为[[(0,0), (1,1)], [(0,0), (1,1)]]的元素
            return x[[[0, 1], [0, 1]], [[0, 1], [0, 1]]]

        def func4(x):
            ls = [0]
            ls.append(1)
            ls.append(2)
            # 返回索引为[0, 1, 2]的元素
            return x[ls]

        def func5(x):
            ls = [0.1, 1.2, 2.3]
            # 返回索引为[0.1, 1.2, 2.3]的元素（可能会进行四舍五入）
            return x[ls]

        input = torch.rand((6, 2))
        self.checkScript(func1, (input,))
        self.checkScript(func2, (input,))
        self.checkScript(func3, (input,))
        self.checkScript(func4, (input,))
        self.checkScript(func5, (input,))

    def test_index_ellipses(self):
        vals = [":", 1, None]
        for _ in range(100):
            indices = [random.choice(vals) for _ in range(4)]
            indices[random.randint(0, len(indices) - 1)] = "..."
            test_str = dedent(
                """
            def f():
                x = torch.ones(10, 9, 8, 7, 6)
                return x{indices}.shape
            """.format(
                    indices=indices
                )
            )
            # 去除字符串中的单引号
            test_str = test_str.replace(r"'", r"")
            scope = {}
            # 执行包装的代码字符串，将结果存储在scope中
            execWrapper(test_str, globals(), scope)
            cu = torch.jit.CompilationUnit(test_str)
            res1 = cu.f()
            res2 = scope["f"]()
            self.assertEqual(res1, res2)

    def test_inf(self):
        @torch.jit.script
        def foo(a):
            # 返回a是否小于正无穷
            return a < float("inf")

        s = torch.rand(1)
        self.assertTrue(foo(s))

        @torch.jit.script
        def bar(a):
            # 返回a是否大于负无穷
            return a > float("-inf")

        s = torch.rand(1)
        self.assertTrue(foo(s))

        # 测试导入源码后的重新赋值
        str = """
        def foo(x):
            # type: (bool)
            a = float("-inf")
            if not x:
                a = float(torch.tensor([5]))
            return a < 4
        """
        cu = torch.jit.CompilationUnit(str)
        self.assertTrue(cu.foo(True))
        self.assertFalse(cu.foo(False))

    def test_str_to_float(self):
        @torch.jit.script
        def foo(a):
            # 检查是否能将字符串转换为浮点数
            return 0.5 == float("0.5 hello")

        s = torch.rand(1)
        with self.assertRaisesRegex(RuntimeError, "could not convert string to float"):
            self.assertTrue(foo(s))

        @torch.jit.script
        def foo(a):
            # 检查是否能将字符串"0.5"正确转换为浮点数
            return 0.5 == float("0.5")

        s = torch.rand(1)
        self.assertTrue(foo(s))

        @torch.jit.script
        def foo(a):
            # 检查是否能将字符串"0"正确转换为浮点数
            return 0.0 == float("0")

        s = torch.rand(1)
        self.assertTrue(foo(s))
```