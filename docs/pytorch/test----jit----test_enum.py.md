# `.\pytorch\test\jit\test_enum.py`

```py
# Owner(s): ["oncall: jit"]

# 引入必要的库和模块
import os
import sys
from enum import Enum
from typing import Any, List

import torch
from torch.testing import FileCheck

# 将测试目录下的辅助文件加入路径
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, make_global

# 如果作为主程序运行，抛出运行时错误并提供使用说明
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestEnum，继承自 JitTestCase
class TestEnum(JitTestCase):

    # 测试枚举值类型
    def test_enum_value_types(self):

        # 定义整数枚举类 IntEnum
        class IntEnum(Enum):
            FOO = 1
            BAR = 2

        # 定义浮点数枚举类 FloatEnum
        class FloatEnum(Enum):
            FOO = 1.2
            BAR = 2.3

        # 定义字符串枚举类 StringEnum
        class StringEnum(Enum):
            FOO = "foo as in foo bar"
            BAR = "bar as in foo bar"

        # 将定义的枚举类设置为全局可用
        make_global(IntEnum, FloatEnum, StringEnum)

        # 定义支持枚举类型作为参数的脚本函数
        @torch.jit.script
        def supported_enum_types(a: IntEnum, b: FloatEnum, c: StringEnum):
            return (a.name, b.name, c.name)

        # 使用 FileCheck 检查生成的脚本函数的图形表示
        FileCheck().check("IntEnum").check("FloatEnum").check("StringEnum").run(
            str(supported_enum_types.graph)
        )

        # 定义包含张量的枚举类 TensorEnum
        class TensorEnum(Enum):
            FOO = torch.tensor(0)
            BAR = torch.tensor(1)

        # 将张量枚举类设置为全局可用
        make_global(TensorEnum)

        # 定义不支持的枚举类型作为参数的函数
        def unsupported_enum_types(a: TensorEnum):
            return a.name

        # 使用断言检查脚本化该函数时是否引发预期的异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Cannot create Enum with value type 'Tensor'", ""
        ):
            torch.jit.script(unsupported_enum_types)

    # 测试枚举类型的比较操作
    def test_enum_comp(self):

        # 定义颜色枚举类 Color
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 将颜色枚举类设置为全局可用
        make_global(Color)

        # 定义比较枚举类型的脚本函数
        @torch.jit.script
        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        # 使用 FileCheck 检查生成的脚本函数的图形表示
        FileCheck().check("aten::eq").run(str(enum_comp.graph))

        # 使用断言检查枚举比较的预期行为
        self.assertEqual(enum_comp(Color.RED, Color.RED), True)
        self.assertEqual(enum_comp(Color.RED, Color.GREEN), False)

    # 测试不同枚举类之间的比较操作
    def test_enum_comp_diff_classes(self):

        # 定义枚举类 Foo
        class Foo(Enum):
            ITEM1 = 1
            ITEM2 = 2

        # 定义枚举类 Bar
        class Bar(Enum):
            ITEM1 = 1
            ITEM2 = 2

        # 将定义的枚举类设置为全局可用
        make_global(Foo, Bar)

        # 定义比较不同枚举类之间的脚本函数
        @torch.jit.script
        def enum_comp(x: Foo) -> bool:
            return x == Bar.ITEM1

        # 使用 FileCheck 检查生成的脚本函数的图形表示
        FileCheck().check("prim::Constant").check_same("Bar.ITEM1").check(
            "aten::eq"
        ).run(str(enum_comp.graph))

        # 使用断言检查枚举比较的预期行为
        self.assertEqual(enum_comp(Foo.ITEM1), False)
    # 定义一个单元测试函数，用于测试枚举类型中不同值类型的错误
    def test_heterogenous_value_type_enum_error(self):
        # 定义一个枚举类型 Color，包含 RED 和 GREEN 两个成员，分别对应整数和字符串类型的值
        class Color(Enum):
            RED = 1
            GREEN = "green"

        # 将 Color 枚举类型注册为全局变量
        make_global(Color)

        # 定义一个函数 enum_comp，接受两个 Color 类型的参数并返回布尔值
        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        # 使用 assertRaisesRegexWithHighlight 上下文管理器来捕获 RuntimeError 异常，
        # 并检查异常信息中是否包含特定文本 "Could not unify type list"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Could not unify type list", ""
        ):
            # 对 enum_comp 函数进行 Torch 脚本化处理
            torch.jit.script(enum_comp)

    # 定义一个单元测试函数，用于测试枚举类型的名称获取
    def test_enum_name(self):
        # 定义一个枚举类型 Color，包含 RED 和 GREEN 两个成员
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 将 Color 枚举类型注册为全局变量
        make_global(Color)

        # 定义一个 Torch 脚本函数 enum_name，接受 Color 类型的参数并返回其名称
        @torch.jit.script
        def enum_name(x: Color) -> str:
            return x.name

        # 使用 FileCheck 对 enum_name 函数生成的图形进行检查，验证图中是否包含特定指令顺序
        FileCheck().check("Color").check_next("prim::EnumName").check_next(
            "return"
        ).run(str(enum_name.graph))

        # 断言调用 enum_name 函数返回的结果与 Color.RED 和 Color.GREEN 的名称相符
        self.assertEqual(enum_name(Color.RED), Color.RED.name)
        self.assertEqual(enum_name(Color.GREEN), Color.GREEN.name)

    # 定义一个单元测试函数，用于测试枚举类型的值获取
    def test_enum_value(self):
        # 定义一个枚举类型 Color，包含 RED 和 GREEN 两个成员
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 将 Color 枚举类型注册为全局变量
        make_global(Color)

        # 定义一个 Torch 脚本函数 enum_value，接受 Color 类型的参数并返回其数值
        @torch.jit.script
        def enum_value(x: Color) -> int:
            return x.value

        # 使用 FileCheck 对 enum_value 函数生成的图形进行检查，验证图中是否包含特定指令顺序
        FileCheck().check("Color").check_next("prim::EnumValue").check_next(
            "return"
        ).run(str(enum_value.graph))

        # 断言调用 enum_value 函数返回的结果与 Color.RED 和 Color.GREEN 的数值相符
        self.assertEqual(enum_value(Color.RED), Color.RED.value)
        self.assertEqual(enum_value(Color.GREEN), Color.GREEN.value)

    # 定义一个单元测试函数，用于测试枚举类型作为常量的比较
    def test_enum_as_const(self):
        # 定义一个枚举类型 Color，包含 RED 和 GREEN 两个成员
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 将 Color 枚举类型注册为全局变量
        make_global(Color)

        # 定义一个 Torch 脚本函数 enum_const，接受 Color 类型的参数并返回其与 Color.RED 的比较结果
        @torch.jit.script
        def enum_const(x: Color) -> bool:
            return x == Color.RED

        # 使用 FileCheck 对 enum_const 函数生成的图形进行检查，验证图中是否包含特定指令顺序
        FileCheck().check(
            "prim::Constant[value=__torch__.jit.test_enum.Color.RED]"
        ).check_next("aten::eq").check_next("return").run(str(enum_const.graph))

        # 断言调用 enum_const 函数返回的结果与 Color.RED 和 Color.GREEN 的比较结果相符
        self.assertEqual(enum_const(Color.RED), True)
        self.assertEqual(enum_const(Color.GREEN), False)

    # 定义一个单元测试函数，用于测试不存在的枚举成员的处理
    def test_non_existent_enum_value(self):
        # 定义一个枚举类型 Color，包含 RED 和 GREEN 两个成员
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 将 Color 枚举类型注册为全局变量
        make_global(Color)

        # 定义一个函数 enum_const，接受 Color 类型的参数并检查其是否为 Color.PURPLE
        def enum_const(x: Color) -> bool:
            if x == Color.PURPLE:
                return True
            else:
                return False

        # 使用 assertRaisesRegexWithHighlight 上下文管理器来捕获 RuntimeError 异常，
        # 并检查异常信息中是否包含特定文本 "has no attribute 'PURPLE'"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "has no attribute 'PURPLE'", "Color.PURPLE"
        ):
            # 对 enum_const 函数进行 Torch 脚本化处理
            torch.jit.script(enum_const)
    # 定义一个测试方法，用于测试枚举类型和 Torch 脚本
    def test_enum_ivalue_type(self):
        # 定义枚举类型 Color，包含 RED 和 GREEN 两个成员
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 将 Color 枚举类型注册为全局变量
        make_global(Color)

        # 定义一个 Torch 脚本函数，检查输入是否为 Color 枚举类型
        @torch.jit.script
        def is_color_enum(x: Any):
            return isinstance(x, Color)

        # 运行 FileCheck 检查 Torch 脚本生成的图形式表示
        FileCheck().check(
            "prim::isinstance[types=[Enum<__torch__.jit.test_enum.Color>]]"
        ).check_next("return").run(str(is_color_enum.graph))

        # 断言函数对 Color.RED 和 Color.GREEN 的判定结果为 True，对于整数 1 的判定结果为 False
        self.assertEqual(is_color_enum(Color.RED), True)
        self.assertEqual(is_color_enum(Color.GREEN), True)
        self.assertEqual(is_color_enum(1), False)

    # 定义测试方法，测试闭包引用的枚举常量和 Torch 脚本
    def test_closed_over_enum_constant(self):
        # 定义枚举类型 Color，包含 RED 和 GREEN 两个成员
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 将 Color 枚举类型的引用赋给变量 a
        a = Color

        # 定义一个 Torch 脚本函数，返回变量 a 引用的 RED 的值
        @torch.jit.script
        def closed_over_aliased_type():
            return a.RED.value

        # 运行 FileCheck 检查 Torch 脚本生成的图形式表示
        FileCheck().check("prim::Constant[value={}]".format(a.RED.value)).check_next(
            "return"
        ).run(str(closed_over_aliased_type.graph))

        # 断言闭包函数返回的值等于 Color.RED 的值
        self.assertEqual(closed_over_aliased_type(), Color.RED.value)

        # 将 Color.RED 的引用赋给变量 b
        b = Color.RED

        # 定义一个 Torch 脚本函数，返回变量 b 引用的值
        @torch.jit.script
        def closed_over_aliased_value():
            return b.value

        # 运行 FileCheck 检查 Torch 脚本生成的图形式表示
        FileCheck().check("prim::Constant[value={}]".format(b.value)).check_next(
            "return"
        ).run(str(closed_over_aliased_value.graph))

        # 断言闭包函数返回的值等于 Color.RED 的值
        self.assertEqual(closed_over_aliased_value(), Color.RED.value)

    # 定义测试方法，测试枚举类型作为模块属性和 Torch 脚本
    def test_enum_as_module_attribute(self):
        # 定义枚举类型 Color，包含 RED 和 GREEN 两个成员
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 定义一个继承自 Torch 模块的测试模块类 TestModule
        class TestModule(torch.nn.Module):
            def __init__(self, e: Color):
                super().__init__()
                self.e = e

            def forward(self):
                return self.e.value

        # 创建一个 TestModule 的实例 m，传入 Color.RED 枚举类型作为参数
        m = TestModule(Color.RED)
        # 对模块进行 Torch 脚本化
        scripted = torch.jit.script(m)

        # 运行 FileCheck 检查 Torch 脚本生成的图形式表示
        FileCheck().check("TestModule").check_next("Color").check_same(
            'prim::GetAttr[name="e"]'
        ).check_next("prim::EnumValue").check_next("return").run(str(scripted.graph))

        # 断言 Torch 脚本化后的模块实例 m 执行结果等于 Color.RED 的值
        self.assertEqual(scripted(), Color.RED.value)

    # 定义测试方法，测试字符串作为枚举类型的值作为模块属性和 Torch 脚本
    def test_string_enum_as_module_attribute(self):
        # 定义枚举类型 Color，包含 RED 和 GREEN 两个成员，其值为字符串类型
        class Color(Enum):
            RED = "red"
            GREEN = "green"

        # 定义一个继承自 Torch 模块的测试模块类 TestModule
        class TestModule(torch.nn.Module):
            def __init__(self, e: Color):
                super().__init__()
                self.e = e

            def forward(self):
                return (self.e.name, self.e.value)

        # 将 Color 枚举类型注册为全局变量
        make_global(Color)
        # 创建一个 TestModule 的实例 m，传入 Color.RED 枚举类型作为参数
        m = TestModule(Color.RED)
        # 对模块进行 Torch 脚本化
        scripted = torch.jit.script(m)

        # 断言 Torch 脚本化后的模块实例 m 执行结果等于 Color.RED 的名称和值
        self.assertEqual(scripted(), (Color.RED.name, Color.RED.value))

    # 定义测试方法，测试枚举类型作为函数返回值和 Torch 脚本
    def test_enum_return(self):
        # 定义枚举类型 Color，包含 RED 和 GREEN 两个成员
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 将 Color 枚举类型注册为全局变量
        make_global(Color)

        # 定义一个 Torch 脚本函数，根据条件返回不同的枚举成员
        @torch.jit.script
        def return_enum(cond: bool):
            if cond:
                return Color.RED
            else:
                return Color.GREEN

        # 断言函数返回值等于 Color.RED 或 Color.GREEN
        self.assertEqual(return_enum(True), Color.RED)
        self.assertEqual(return_enum(False), Color.GREEN)
    # 定义一个测试方法，测试枚举类型作为返回值
    def test_enum_module_return(self):
        # 定义一个枚举类型 Color，包含 RED 和 GREEN 两个成员
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 定义一个继承自 torch.nn.Module 的测试模块类 TestModule
        class TestModule(torch.nn.Module):
            # 初始化方法，接受一个 Color 类型的参数 e
            def __init__(self, e: Color):
                super().__init__()
                self.e = e  # 将参数 e 存储在实例属性中

            # 前向传播方法
            def forward(self):
                return self.e  # 返回实例属性 e

        # 将 Color 类型变量作为全局变量
        make_global(Color)
        # 创建一个 TestModule 实例 m，传入 Color.RED 作为参数
        m = TestModule(Color.RED)
        # 对 TestModule 进行脚本化
        scripted = torch.jit.script(m)

        # 对脚本化后的图形进行检查，验证返回值等
        FileCheck().check("TestModule").check_next("Color").check_same(
            'prim::GetAttr[name="e"]'
        ).check_next("return").run(str(scripted.graph))

        # 断言脚本化后的模块调用结果与 Color.RED 相等
        self.assertEqual(scripted(), Color.RED)

    # 定义一个测试方法，测试枚举类型的迭代
    def test_enum_iterate(self):
        # 定义一个枚举类型 Color，包含 RED、GREEN 和 BLUE 三个成员
        class Color(Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        # 定义一个函数 iterate_enum，接受一个 Color 类型的参数 x
        def iterate_enum(x: Color):
            # 初始化一个空列表 res，用于存储枚举成员的值
            res: List[int] = []
            # 遍历 Color 枚举类型的所有成员
            for e in Color:
                # 如果枚举成员 e 不等于参数 x，则将其值添加到 res 中
                if e != x:
                    res.append(e.value)
            return res  # 返回结果列表

        # 将 Color 类型变量作为全局变量
        make_global(Color)
        # 对 iterate_enum 函数进行脚本化
        scripted = torch.jit.script(iterate_enum)

        # 对脚本化后的图形进行检查，验证枚举类型及其成员的存在
        FileCheck().check("Enum<__torch__.jit.test_enum.Color>[]").check_same(
            "Color.RED"
        ).check_same("Color.GREEN").check_same("Color.BLUE").run(str(scripted.graph))

        # 验证函数调用结果符合预期
        # PURPLE 总是最后出现，因为遵循 Python 的枚举定义顺序
        self.assertEqual(scripted(Color.RED), [Color.GREEN.value, Color.BLUE.value])
        self.assertEqual(scripted(Color.GREEN), [Color.RED.value, Color.BLUE.value])

    # 测试显式和/或重复脚本化枚举类是被允许的
    def test_enum_explicit_script(self):
        # 使用 torch.jit.script 显式脚本化一个枚举类 Color
        @torch.jit.script
        class Color(Enum):
            RED = 1
            GREEN = 2

        torch.jit.script(Color)  # 再次对枚举类 Color 进行脚本化

    # 用于修复 https://github.com/pytorch/pytorch/issues/108933 的回归测试
    def test_typed_enum(self):
        # 定义一个枚举类型 Color，它继承自 int 和 Enum
        class Color(int, Enum):
            RED = 1
            GREEN = 2

        # 使用 torch.jit.script 脚本化一个函数 is_red，接受一个 Color 类型参数 x，返回一个布尔值
        @torch.jit.script
        def is_red(x: Color) -> bool:
            return x == Color.RED
```