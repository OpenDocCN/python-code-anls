# `.\pytorch\test\jit\test_recursive_script.py`

```
# Owner(s): ["oncall: jit"]

# 引入必要的标准库和第三方库
import os  # 操作系统相关功能
import re  # 正则表达式模块
import sys  # 系统相关功能
import types  # 类型操作支持
import typing  # 类型提示模块
import typing_extensions  # 类型提示扩展模块
from collections import OrderedDict  # 引入有序字典

# 引入 PyTorch 相关库
import torch  # PyTorch 主库
import torch.jit.frontend  # PyTorch JIT 前端
import torch.nn as nn  # PyTorch 神经网络模块
from torch import Tensor  # 引入 Tensor 类型
from torch.testing import FileCheck  # 引入用于测试的文件检查工具

# 将 test/ 目录下的辅助文件设置为可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import (
    _tmp_donotuse_dont_inline_everything,  # 导入特定的测试工具
    JitTestCase,  # 导入 JIT 测试基类
)

# 如果该脚本文件作为主程序运行，则抛出运行时错误
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestRecursiveScript(JitTestCase):
    # 测试推断出的 NoneType 类型
    def test_inferred_nonetype(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.x = None  # 初始化属性为 None

            def forward(self):
                assert self.x is None  # 断言属性为 None

        m = torch.jit.script(M())  # 对类 M 进行脚本化
        self.checkModule(M(), ())  # 检查模块

    # 测试脚本函数属性
    def test_script_function_attribute(self):
        @torch.jit.script
        def fn1(x):
            return x + x

        @torch.jit.script
        def fn2(x):
            return x - x

        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn  # 初始化属性为脚本函数

            def forward(self, x):
                return self.fn(x)  # 调用属性中的脚本函数

        fn1_mod = M(fn1)
        fn2_mod = M(fn2)

        self.checkModule(fn1_mod, (torch.randn(2, 2),))  # 检查模块
        self.checkModule(fn2_mod, (torch.randn(2, 2),))  # 检查模块

    # 测试 Python 函数属性
    def test_python_function_attribute(self):
        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn  # 初始化属性为 Python 函数

            def forward(self, x):
                return self.fn(x)  # 调用属性中的 Python 函数

        mod = M(torch.sigmoid)

        self.checkModule(mod, (torch.randn(2, 2),))  # 检查模块

    # 测试函数编译失败情况
    def test_failed_function_compilation(self):
        def fn(x):
            return i_dont_exist  # 引发未定义的变量错误

        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn  # 初始化属性为失败的函数

            def forward(self, x):
                return self.fn(x)  # 调用属性中的函数

        m = M(fn)
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "failed to compile", "i_dont_exist"
        ):
            torch.jit.script(m)  # 尝试对模块进行脚本化，期望捕获编译错误信息

    # 测试初始化错误的情况
    def test_init_error(self):
        class M(nn.Module):
            def __init__(self):
                self.x = 2  # 初始化属性为一个值

            def forward(self):
                pass

        with self.assertRaisesRegex(RuntimeError, "has not been initialized"):
            torch.jit.script(M())  # 尝试对模块进行脚本化，期望捕获未初始化属性的错误
    def test_script_after_eval(self):
        # 定义一个继承自nn.Module的内部类M，用于测试模块脚本化后的行为
        class M(nn.Module):
            # 定义模块的前向传播方法
            def forward(self):
                # 如果模块处于训练模式，返回2
                if self.training:
                    return 2
                else:
                    return 0

        # 创建M类的实例m
        m = M()
        # 使用torch.jit.script函数将模块m脚本化
        sm1 = torch.jit.script(m)
        # 将模块m设置为评估模式
        m.eval()
        # 使用torch.jit.script函数将模块m脚本化
        sm2 = torch.jit.script(m)

        # 断言模块m处于评估模式，即self.training为False
        self.assertFalse(m.training)

        # 断言sm1在模块m处于训练模式时创建，即sm1.training为True
        self.assertTrue(sm1.training)
        # 断言sm1的训练状态与其内部C++对象的训练状态一致
        self.assertEqual(sm1.training, sm1._c.getattr("training"))
        # 断言调用sm1()返回结果为2
        self.assertEqual(sm1(), 2)

        # 断言sm2在模块m评估后创建，即sm2.training为False
        self.assertFalse(sm2.training)
        # 断言sm2的训练状态与其内部C++对象的训练状态一致
        self.assertEqual(sm2.training, sm2._c.getattr("training"))
        # 断言调用sm2()返回结果为0
        self.assertEqual(sm2(), 0)

    def test_module_name(self):
        # 定义一个继承自torch.nn.Module的内部类MyModule，用于测试模块脚本化后的图形名称检查
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = 2

            def forward(self, t):
                return t + self.x

        # 使用torch.jit.script函数将MyModule类的实例脚本化，并运行图形名称检查
        m = torch.jit.script(MyModule())
        FileCheck().check("MyModule").run(m.graph)

    def test_repeated_error_stack(self):
        # 定义一系列嵌套函数a、b、c和d，用于测试torch.jit.script函数对异常堆栈的处理
        def d(x):
            return "a" - 2

        def c(x):
            return d(x)

        def b(x):
            return c(x)

        def a(x):
            return b(x)

        try:
            # 尝试对函数a应用torch.jit.script函数，捕获异常并运行文件检查
            torch.jit.script(a)
        except Exception as e:
            FileCheck().check_count("is being compiled", 2).run(str(e))

        try:
            # 再次尝试对函数a应用torch.jit.script函数，捕获异常并运行文件检查
            torch.jit.script(a)
        except Exception as e:
            # 确保前一次失败的条目已被清除
            FileCheck().check_count("is being compiled", 2).run(str(e))

    def test_constants_with_final(self):
        # 定义三个类M1、M2和M3，分别使用不同版本的Final修饰符，用于测试常量的脚本化行为
        class M1(torch.nn.Module):
            x: torch.jit.Final[int]

            def __init__(self):
                super().__init__()
                self.x = 2

            def forward(self, t):
                return t + self.x

        # 调用self.checkModule方法检查M1类的实例
        self.checkModule(M1(), (torch.randn(2, 2),))

        class M2(torch.nn.Module):
            x: typing_extensions.Final[int]

            def __init__(self):
                super().__init__()
                self.x = 2

            def forward(self, t):
                return t + self.x

        # 调用self.checkModule方法检查M2类的实例
        self.checkModule(M2(), (torch.randn(2, 2),))

        class M3(torch.nn.Module):
            x: typing.Final[int]

            def __init__(self):
                super().__init__()
                self.x = 2

            def forward(self, t):
                return t + self.x

        # 调用self.checkModule方法检查M3类的实例
        self.checkModule(M3(), (torch.randn(2, 2),))
    def test_ignore_class(self):
        # 定义一个被 Torch JIT 忽略的类
        @torch.jit.ignore
        class MyScriptClass:
            # 包含一个无法被脚本化的方法
            def unscriptable(self):
                return "a" + 200

        # 定义一个继承自 nn.Module 的测试模块
        class TestModule(torch.nn.Module):
            # 前向传播方法
            def forward(self, x):
                # 返回一个 MyScriptClass 的实例
                return MyScriptClass()

        # 断言捕获 Torch JIT 前端错误，期望出现类无法实例化的异常
        with self.assertRaisesRegexWithHighlight(
            torch.jit.frontend.FrontendError,
            "Cannot instantiate class",
            "MyScriptClass",
        ):
            # 将 TestModule 实例化为 Torch 脚本
            t = torch.jit.script(TestModule())

    def test_method_call(self):
        # 定义一个继承自 nn.Module 的类 M
        class M(nn.Module):
            # 测试方法，返回输入参数 x
            def test(self, x):
                return x

            # 前向传播方法
            def forward(self, z):
                # 调用 test 方法，将结果存储在 y 中
                y = self.test(z)
                # 返回 z + 20 + y 的结果
                return z + 20 + y

        # 检查 M 类的模块
        self.checkModule(M(), (torch.randn(2, 2),))

    def test_module_repr(self):
        # 定义一个继承自 nn.Module 的子模块 Submodule
        class Submodule(nn.Module):
            # 前向传播方法，返回输入参数 x
            def forward(self, x):
                return x

        # 定义一个继承自 nn.Module 的主模块 MyModule
        class MyModule(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加 nn.Conv2d 和 nn.Linear 层，以及 Submodule 的实例
                self.conv = nn.Conv2d(10, 10, 3)
                self.lin = nn.Linear(10, 10)
                self.sub = Submodule()

            # 前向传播方法
            def forward(self, x):
                # 返回线性层、子模块和卷积层的组合结果
                return self.lin(x) + self.sub(x) + self.conv(x)

        # 使用 Torch 脚本化 MyModule
        m = torch.jit.script(MyModule())

        # 捕获标准输出
        with self.capture_stdout() as out:
            # 打印 Torch 脚本化的 MyModule
            print(m)

        # 使用 FileCheck 检查输出结果中是否包含特定字符串
        f = FileCheck()
        f.check("MyModule")
        f.check("Conv2d")
        f.check("Linear")
        f.check("Submodule")
        f.run(out[0])

        # 断言 Torch 脚本化模块的 original_name 属性为 "MyModule"
        self.assertEqual(m.original_name, "MyModule")

    def test_dir(self):
        # 定义一个测试模块的 dir 方法的辅助函数
        def test_module_dir(mod):
            # 获取模块的属性列表
            dir_set = dir(mod)
            # 对模块进行 Torch 脚本化
            scripted_mod = torch.jit.script(mod)
            # 创建一个需要忽略的属性集合
            ignore_set = [
                "training",
                "__delitem__",
                "__setitem__",
                "clear",
                "items",
                "keys",
                "pop",
                "update",
                "values",
            ]
            # 遍历属性列表
            for attr in dir_set:
                # 如果属性在忽略集合中，跳过
                if attr in ignore_set:
                    continue
                # 断言 Torch 脚本化后的模块也包含相同的属性
                self.assertTrue(attr in set(dir(scripted_mod)), attr)

        # 定义一个继承自 nn.Module 的测试模块 MyModule
        class MyModule(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加 nn.Conv2d 和 nn.Linear 层
                self.conv = nn.Conv2d(10, 10, 3)
                self.lin = nn.Linear(10, 10)

            # 前向传播方法
            def forward(self, x):
                # 返回线性层和卷积层的组合结果
                return self.lin(x) + self.conv(x)

        # 测试 MyModule 的属性列表
        test_module_dir(MyModule())

        # 测试容器类的自定义 __dir__ 方法
        conv = nn.Conv2d(10, 10, 3)
        linear = nn.Linear(10, 10)

        # 测试 nn.Sequential 的属性列表
        test_module_dir(nn.Sequential(conv, linear))

        # 测试 nn.ModuleDict 的属性列表
        test_module_dir(
            nn.ModuleDict(OrderedDict([("conv", conv), ("linear", linear)]))
        )
    def test_class_compile(self):
        def other_fn(a: int, b: Tensor) -> Tensor:
            return a * b

        class B:
            def __init__(self, x):
                self.x = 2  # 初始化类 B 的实例属性 x

            def helper(self, a):
                return self.x + a + other_fn(self.x, a)  # 调用外部函数 other_fn，并返回计算结果

        class N(torch.nn.Module):
            def forward(self, x):
                b = B(x)  # 创建类 B 的实例
                return b.helper(x)  # 调用类 B 中的方法 helper，并返回结果

        self.checkModule(N(), (torch.randn(2, 2),))  # 调用 self.checkModule 进行模块检查

    def test_error_stack(self):
        def d(x: int) -> int:
            return x + 10  # 返回输入参数 x 加 10 的结果

        def c(x):
            return d("hello") + d(x)  # 调用函数 d，但将字符串 "hello" 作为参数传递给了 d，然后将两次调用的结果相加

        def b(x):
            return c(x)  # 调用函数 c

        def a(x):
            return b(x)  # 调用函数 b

        try:
            scripted = torch.jit.script(a)  # 对函数 a 进行脚本化
        except RuntimeError as e:
            checker = FileCheck()  # 创建 FileCheck 对象
            checker.check("Expected a value of type 'int'")  # 检查错误消息中是否包含特定字符串
            checker.check("def c(x)")  # 检查错误消息中是否包含特定字符串
            checker.check("def b(x)")  # 检查错误消息中是否包含特定字符串
            checker.check("def a(x)")  # 检查错误消息中是否包含特定字符串
            checker.run(str(e))  # 运行 FileCheck 对象以匹配错误消息内容

    def test_error_stack_module(self):
        def d(x: int) -> int:
            return x + 10  # 返回输入参数 x 加 10 的结果

        def c(x):
            return d("hello") + d(x)  # 调用函数 d，但将字符串 "hello" 作为参数传递给了 d，然后将两次调用的结果相加

        def b(x):
            return c(x)  # 调用函数 c

        class Submodule(torch.nn.Module):
            def forward(self, x):
                return b(x)  # 调用函数 b

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = Submodule()  # 创建 Submodule 类的实例

            def some_method(self, y):
                return y + self.submodule(y)  # 调用 Submodule 实例的 forward 方法

            def forward(self, x):
                return self.some_method(x)  # 调用实例方法 some_method

        try:
            scripted = torch.jit.script(M())  # 对类 M 进行脚本化
        except RuntimeError as e:
            checker = FileCheck()  # 创建 FileCheck 对象
            checker.check("Expected a value of type 'int'")  # 检查错误消息中是否包含特定字符串
            checker.check("'c' is being compiled since it was called from 'b'")  # 检查错误消息中是否包含特定字符串
            checker.check("'b' is being compiled since it was called from")  # 检查错误消息中是否包含特定字符串
            checker.run(str(e))  # 运行 FileCheck 对象以匹配错误消息内容

    @_tmp_donotuse_dont_inline_everything
    def test_script_basic(self):
        def a_python_fn(a, b, c):
            return a + b + c  # 返回三个参数的和

        @torch.jit.script
        def a_script_fn(d, e, f):
            return a_python_fn(d, e, f)  # 对 a_python_fn 进行脚本化

        graph = str(a_script_fn.graph)  # 获取脚本化函数的计算图的字符串表示
        FileCheck().check("prim::CallFunction").run(graph)  # 使用 FileCheck 检查计算图中是否有函数调用的原语
        FileCheck().check_not("^a_python_fn").run(graph)  # 使用 FileCheck 检查计算图中是否没有直接的 a_python_fn 函数调用
        t = torch.ones(2, 2)  # 创建一个全为1的张量
        self.assertEqual(a_script_fn(t, t, t), t + t + t)  # 断言脚本化函数的结果与预期的张量相等

    def test_error_stack_class(self):
        class X:
            def bad_fn(self):
                import pdb  # noqa: F401  # 导入 pdb 模块，但此行不会被使用到

        def fn(x) -> X:
            return X(10)  # 返回类 X 的实例，但传递了一个整数参数

        try:
            torch.jit.script(fn)  # 对函数 fn 进行脚本化
        except Exception as e:
            checker = FileCheck()  # 创建 FileCheck 对象
            checker.check("import statements")  # 检查错误消息中是否包含特定字符串
            checker.check("is being compiled since it was called from")  # 检查错误消息中是否包含特定字符串
            checker.run(str(e))  # 运行 FileCheck 对象以匹配错误消息内容
    def test_error_stack_annotation(self):
        # 定义内部类 X，包含一个问题函数 bad_fn
        class X:
            def bad_fn(self):
                import pdb  # noqa: F401  # 导入 pdb 模块（用于调试），但未使用

        # 定义函数 fn，返回类型为 X
        def fn(x) -> X:
            return X(10)

        # 尝试对函数 fn 进行 Torch 脚本化
        try:
            torch.jit.script(fn)
        except Exception as e:
            # 创建 FileCheck 对象进行检查
            checker = FileCheck()
            # 检查是否包含 "import statements"
            checker.check("import statements")
            # 检查是否包含 "is being compiled since it was called from"
            checker.check("is being compiled since it was called from")
            # 检查是否包含 "-> X"
            checker.check("-> X")
            # 运行检查器并传入异常 e 的字符串形式作为参数
            checker.run(str(e))

    def test_module_basic(self):
        # 定义内部类 Other，继承自 torch.nn.Module
        class Other(torch.nn.Module):
            __constants__ = ["x"]

            # 构造函数，初始化 x 和 param
            def __init__(self, x):
                super().__init__()
                self.x = x
                self.param = torch.nn.Parameter(torch.ones(2, 2))

            # 一个无法脚本化的方法
            def some_unscriptable_method(self):
                a = 2
                a = [2]
                return a

            # 前向传播函数，返回输入 t 加上 x 和 param 的结果
            def forward(self, t):
                return t + self.x + self.param

        # 定义内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 构造函数，初始化 Other 类的实例
            def __init__(self):
                super().__init__()
                self.other = Other(200)

            # 前向传播函数，返回 self.other(t) 的结果乘以 2
            def forward(self, t):
                return self.other(t) * 2

        # 使用 checkModule 方法对 M 的实例进行检查
        self.checkModule(M(), (torch.ones(2, 2),))

    def test_module_function_export(self):
        # 定义内部类 Other，继承自 torch.nn.Module
        class Other(torch.nn.Module):
            __constants__ = ["x"]

            # 构造函数，初始化 x 和 param
            def __init__(self, x):
                super().__init__()
                self.x = x
                self.param = torch.nn.Parameter(torch.ones(2, 2))

            # 使用 @torch.jit.export 装饰器导出的方法，对输入 y 加上 20 并返回
            @torch.jit.export
            def some_entry_point(self, y):
                return y + 20

            # 前向传播函数，返回输入 t 加上 x 和 param 的结果
            def forward(self, t):
                return t + self.x + self.param

        # 定义内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 构造函数，初始化 Other 类的实例
            def __init__(self):
                super().__init__()
                self.other = Other(200)

            # 前向传播函数，返回 self.other(t) 的结果乘以 2
            def forward(self, t):
                return self.other(t) * 2

        # 使用 checkModule 方法对 M 的实例进行检查
        self.checkModule(M(), (torch.ones(2, 2),))

    def test_iterable_modules(self):
        # 定义内部类 Inner，继承自 torch.nn.Module
        class Inner(torch.nn.Module):
            # 前向传播函数，返回输入 x 加上 10 的结果
            def forward(self, x):
                return x + 10

        # 定义内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 构造函数，初始化 nn.Sequential 和 nn.ModuleList
            def __init__(self):
                super().__init__()
                self.sequential = nn.Sequential(
                    Inner(), Inner(), nn.Sequential(Inner(), Inner())
                )
                self.module_list = nn.ModuleList([Inner(), Inner()])

            # 前向传播函数，对 module_list 中的每个模块进行前向传播并累加结果，再加上 sequential 的前向传播结果
            def forward(self, x):
                for mod in self.module_list:
                    x += mod(x)
                x += self.sequential(x)
                return x

        # 使用 checkModule 方法对 M 的实例进行检查
        self.checkModule(M(), (torch.randn(5, 5),))
    def test_prepare_scriptable_basic(self):
        # 定义一个 SELU 的子类，但在脚本化时返回一个 ReLU 模块
        class SeluButReluWhenScripted(torch.nn.SELU):
            def __prepare_scriptable__(self):
                return nn.ReLU()

        # 创建一个 5x5 的张量
        t = torch.randn(5, 5)
        # 实例化 SeluButReluWhenScripted 类
        m = SeluButReluWhenScripted()
        # 对 m 进行即时脚本化
        sm = torch.jit.script(m)
        # 在非脚本模式下使用 m 处理张量 t
        eager_out = m(t)
        # 在脚本模式下使用 sm 处理张量 t
        script_out = sm(t)
        # 断言非脚本化输出与脚本化输出不相等
        self.assertNotEqual(eager_out, script_out)

    def test_prepare_scriptable_iterable_modules(self):
        # 定义一个 SELU 的子类，但在脚本化时返回一个 ReLU 模块
        class SeluButReluWhenScripted(torch.nn.SELU):
            def __prepare_scriptable__(self):
                return nn.ReLU()

        # 定义一个包含多个模块的类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建共享的 SeluButReluWhenScripted 实例
                shared = SeluButReluWhenScripted()
                # 定义一个序列容器 sequential，包含多个 SeluButReluWhenScripted 模块
                self.sequential = nn.Sequential(
                    SeluButReluWhenScripted(),
                    SeluButReluWhenScripted(),
                    nn.Sequential(
                        SeluButReluWhenScripted(), shared, SeluButReluWhenScripted()
                    ),
                    shared,
                )
                # 定义一个模块列表 module_list，包含多个 SeluButReluWhenScripted 模块
                self.module_list = nn.ModuleList(
                    [SeluButReluWhenScripted(), shared, SeluButReluWhenScripted()]
                )

            def forward(self, x):
                # 对 module_list 中的模块进行迭代并处理输入张量 x
                for mod in self.module_list:
                    x += mod(x)
                # 对 sequential 容器处理输入张量 x
                x += self.sequential(x)
                return x

        # 创建一个 5x5 的张量
        t = torch.randn(5, 5)
        # 实例化类 M
        m = M()
        # 在非脚本模式下使用 m 处理张量 t 的克隆
        eager_out = m(t.clone())
        # 对类 M 进行即时脚本化
        sm = torch.jit.script(m)
        # 在脚本模式下使用 sm 处理张量 t 的克隆
        script_out = sm(t.clone())
        # 断言非脚本化输出与脚本化输出不相等
        self.assertNotEqual(eager_out, script_out)

    def test_prepare_scriptable_cycle(self):
        # 创建一个 5x5 的张量
        t = torch.randn(5, 5)
        # 创建两个空的 Module 实例 c 和 p
        c = torch.nn.Module()
        p = torch.nn.Module()
        # 设置 c 的字典属性 "_p" 为 p
        c.__dict__["_p"] = p
        # 设置 p 的字典属性 "_c" 为 c
        p.__dict__["_c"] = c

        # 对 p 进行脚本化
        sm = torch.jit.script(p)

    def test_prepare_scriptable_escape_hatch(self):
        # 定义一个非脚本化类 NonJitableClass
        class NonJitableClass:
            def __call__(self, int1, int2, *args):
                total = int1 + int2
                for arg in args:
                    total += arg
                return total

        # 创建 NonJitableClass 的实例 obj
        obj = NonJitableClass()

        # 断言 obj(1, 2) 的结果为 3
        self.assertEqual(obj(1, 2), 3)
        # 断言 obj(1, 2, 3, 4) 的结果为 10
        self.assertEqual(obj(1, 2, 3, 4), 10)
        # 使用 torch.jit.script 尝试对 obj 进行脚本化，期望抛出 NotSupportedError
        with self.assertRaisesRegex(
            torch.jit.frontend.NotSupportedError,
            expected_regex="can't take variable number of arguments",
        ):
            torch.jit.script(obj)

        # 定义一个函数 escape_hatch，对两个整数进行加法并返回结果
        def escape_hatch(int1: int, int2: int) -> int:
            return int1 + int2

        # 定义一个继承自 NonJitableClass 的子类 NonJitableClassWithEscapeHatch
        class NonJitableClassWithEscapeHatch(NonJitableClass):
            # 覆盖 __prepare_scriptable__ 方法，返回 escape_hatch 函数
            def __prepare_scriptable__(self):
                return escape_hatch

        # 对 NonJitableClassWithEscapeHatch 类进行脚本化
        jit_obj = torch.jit.script(NonJitableClassWithEscapeHatch())

        # 断言脚本化对象 jit_obj 的结果与预期结果相等
        self.assertEqual(jit_obj(1, 2), 3)
        # 使用 torch.jit.script 尝试对 jit_obj(1, 2, 3, 4) 进行脚本化，期望抛出 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape(
                "expected at most 2 argument(s) but received 4 argument(s)"
            ),
        ):
            jit_obj(1, 2, 3, 4)
    def test_function_attribute_in_submodule(self):
        # 定义一个内部测试函数，用于测试子模块中的函数属性设置
        class N(nn.Module):
            def __init__(self, norm):
                super().__init__()
                # 设置激活函数为 ReLU
                self.activation = torch.nn.functional.relu
                # 设置模块的归一化层
                self.norm = norm

            def forward(self, src):
                output = src
                # 对输入数据应用归一化操作
                output = self.norm(output)
                return output

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 ReLU 归一化层
                encoder_norm = nn.ReLU()
                # 创建一个 N 类的实例，并将其设置为编码器
                self.encoder = N(encoder_norm)

            def forward(self, x):
                # 调用编码器的前向传播方法
                return self.encoder(x)

        # 创建 M 类的实例
        m = M()
        # 调用外部方法检查模块
        self.checkModule(m, (torch.randn(5, 5),))

    def test_inner_traced_module(self):
        # 定义一个测试内部追踪模块的函数
        class Dummy(nn.Module):
            def forward(self, x):
                return x

        class Model(nn.Module):
            def __init__(self, dummies):
                super().__init__()
                # 将给定的 dummies 列表保存为模块的私有属性
                self._dummies = dummies

            def forward(self, x):
                out = []
                # 对每个 dummy 模块进行前向传播，并将结果保存到 out 列表中
                for dummy in self._dummies:
                    out.append(dummy(x))
                return out

        # 使用 torch.jit.trace 方法对 Dummy 类进行追踪，得到一个追踪模块
        dummy = torch.jit.trace(Dummy(), torch.randn(1, 2))
        # 创建一个包含追踪模块的模块列表
        dummies = nn.ModuleList([dummy])
        # 创建一个 Model 类的实例
        model = Model(dummies)
        # 调用外部方法检查模块
        self.checkModule(model, (torch.rand(5, 5),))

    def test_script_loaded_module(self):
        """
        Test that we can hold a loaded ScriptModule as a submodule.
        """
        # 测试能否将加载的 ScriptModule 作为子模块保存的函数

        class Dummy(nn.Module):
            def forward(self, x):
                return x

        # 使用 torch.jit.script 方法对 Dummy 类进行脚本化
        dummy = torch.jit.script(Dummy())
        # 调用外部方法，获取脚本化后的模块的导出和导入副本
        dummy = self.getExportImportCopy(dummy)

        class ContainsLoaded(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 将脚本化后的 Dummy 模块作为编码器保存在 ContainsLoaded 模块中
                self.encoder = dummy

            def forward(self, input):
                # 调用编码器的前向传播方法
                return self.encoder(input)

        # 创建 ContainsLoaded 类的实例并调用外部方法检查模块
        self.checkModule(ContainsLoaded(), (torch.rand(2, 3),))

    def test_optional_module(self):
        # 定义一个测试可选模块行为的函数
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个线性层模块
                self.foo = nn.Linear(2, 2)

            def forward(self, x):
                # 如果 foo 不为空，则对输入 x 应用线性层 foo
                if self.foo is not None:
                    return self.foo(x)
                # 否则直接返回输入 x
                return x

        # 创建 Dummy 类的实例
        mod = Dummy()
        # 调用外部方法检查模块
        self.checkModule(mod, (torch.rand(2, 2),))
        # 将模块的 foo 属性设置为 None
        mod.foo = None
        # 再次调用外部方法检查模块
        self.checkModule(mod, (torch.rand(2, 2),))
    def test_override_instance_method_ignore(self):
        # 定义一个名为 M 的子类 torch.nn.Module
        class M(torch.nn.Module):
            # 定义一个被 torch.jit.ignore 忽略的实例方法 i_am_ignored
            @torch.jit.ignore
            def i_am_ignored(self):
                return "old"

        # 创建 M 类的实例 m
        m = M()

        # 通过绑定一个新方法来覆盖被忽略的方法
        @torch.jit.ignore
        def i_am_ignored(self):
            return "new"

        # 将新定义的方法绑定到实例 m 上
        m.i_am_ignored = types.MethodType(i_am_ignored, m)
        # 断言实例 m 调用 i_am_ignored 方法返回 "new"
        self.assertEqual(m.i_am_ignored(), "new")

        # 验证 ScriptModule 正确反映了方法的覆盖
        s = torch.jit.script(m)
        # 断言 ScriptModule 实例 s 调用 i_am_ignored 方法返回 "new"
        self.assertEqual(s.i_am_ignored(), "new")
```