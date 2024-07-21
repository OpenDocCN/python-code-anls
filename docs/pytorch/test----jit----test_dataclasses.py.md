# `.\pytorch\test\jit\test_dataclasses.py`

```py
# Owner(s): ["oncall: jit"]
# flake8: noqa

# 导入必要的模块和库
import sys
import unittest
from dataclasses import dataclass, field, InitVar  # 导入 dataclass 相关的模块和装饰器
from enum import Enum  # 导入枚举类型的模块
from typing import List, Optional  # 导入类型提示相关的模块

from hypothesis import given, settings, strategies as st  # 导入 hypothesis 相关模块

import torch  # 导入 PyTorch 模块
from torch.testing._internal.jit_utils import JitTestCase  # 导入 PyTorch 内部测试工具的测试用例


# Example jittable dataclass
@dataclass(order=True)  # 定义一个支持排序的 dataclass
class Point:
    x: float  # x 坐标，浮点型
    y: float  # y 坐标，浮点型
    norm: Optional[torch.Tensor] = None  # 标准化值，可以为空的 Torch 张量

    def __post_init__(self):
        # 在初始化后计算标准化值
        self.norm = (torch.tensor(self.x) ** 2 + torch.tensor(self.y) ** 2) ** 0.5


class MixupScheme(Enum):
    INPUT = ["input"]  # Mixup 方案枚举值为 "input"

    MANIFOLD = [
        "input",
        "before_fusion_projection",
        "after_fusion_projection",
        "after_classifier_projection",
    ]


@dataclass
class MixupParams:
    def __init__(self, alpha: float = 0.125, scheme: MixupScheme = MixupScheme.INPUT):
        self.alpha = alpha  # 混合参数 alpha，默认值为 0.125
        self.scheme = scheme  # 混合方案，默认为 MixupScheme.INPUT


class MixupScheme2(Enum):
    A = 1  # 第二种 Mixup 方案枚举值 A
    B = 2  # 第二种 Mixup 方案枚举值 B


@dataclass
class MixupParams2:
    def __init__(self, alpha: float = 0.125, scheme: MixupScheme2 = MixupScheme2.A):
        self.alpha = alpha  # 混合参数 alpha，默认值为 0.125
        self.scheme = scheme  # 混合方案，默认为 MixupScheme2.A


@dataclass
class MixupParams3:
    def __init__(self, alpha: float = 0.125, scheme: MixupScheme2 = MixupScheme2.A):
        self.alpha = alpha  # 混合参数 alpha，默认值为 0.125
        self.scheme = scheme  # 混合方案，默认为 MixupScheme2.A


# Make sure the Meta internal tooling doesn't raise an overflow error
NonHugeFloats = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False)  # 定义非常大的浮点数范围


class TestDataclasses(JitTestCase):
    @classmethod
    def tearDownClass(cls):
        torch._C._jit_clear_class_registry()  # 清除 PyTorch 的类注册表

    def test_init_vars(self):
        @torch.jit.script
        @dataclass(order=True)
        class Point2:
            x: float  # x 坐标，浮点型
            y: float  # y 坐标，浮点型
            norm_p: InitVar[int] = 2  # 标准化参数，初始化为 2
            norm: Optional[torch.Tensor] = None  # 标准化值，可以为空的 Torch 张量

            def __post_init__(self, norm_p: int):
                # 在初始化后计算标准化值，使用指定的参数 norm_p
                self.norm = (
                    torch.tensor(self.x) ** norm_p + torch.tensor(self.y) ** norm_p
                ) ** (1 / norm_p)

        def fn(x: float, y: float, p: int):
            pt = Point2(x, y, p)  # 创建 Point2 实例
            return pt.norm  # 返回计算得到的标准化值

        self.checkScript(fn, (1.0, 2.0, 3))  # 调用 checkScript 方法进行脚本化测试

    # Sort of tests both __post_init__ and optional fields
    @settings(deadline=None)
    @given(NonHugeFloats, NonHugeFloats)
    def test__post_init__(self, x, y):
        P = torch.jit.script(Point)  # 使用 Torch 脚本化 Point 类

        def fn(x: float, y: float):
            pt = P(x, y)  # 创建 Point 实例
            return pt.norm  # 返回计算得到的标准化值

        self.checkScript(fn, [x, y])  # 调用 checkScript 方法进行脚本化测试

    @settings(deadline=None)
    @given(
        st.tuples(NonHugeFloats, NonHugeFloats), st.tuples(NonHugeFloats, NonHugeFloats)
    )
    # 定义测试方法，比较两个点的坐标
    def test_comparators(self, pt1, pt2):
        # 解包点1和点2的坐标
        x1, y1 = pt1
        x2, y2 = pt2
        # 用 torch.jit.script 将 Point 类型脚本化
        P = torch.jit.script(Point)

        # 定义比较函数，比较两个点的关系
        def compare(x1: float, y1: float, x2: float, y2: float):
            # 使用脚本化的 Point 类创建点1和点2的对象
            pt1 = P(x1, y1)
            pt2 = P(x2, y2)
            return (
                pt1 == pt2,  # 比较相等
                # pt1 != pt2,   # TODO: 修改解释器，使得没有 __ne__ 方法时 (a != b) 自动转为 not (a == b)
                pt1 < pt2,   # 比较小于
                pt1 <= pt2,  # 比较小于等于
                pt1 > pt2,   # 比较大于
                pt1 >= pt2,  # 比较大于等于
            )

        # 调用 self.checkScript 方法检查比较函数的脚本化版本
        self.checkScript(compare, [x1, y1, x2, y2])

    # 测试默认工厂函数
    def test_default_factories(self):
        # 定义一个使用默认工厂函数的数据类 Foo
        @dataclass
        class Foo(object):
            x: List[int] = field(default_factory=list)

        # 断言期望抛出 NotImplementedError 异常
        with self.assertRaises(NotImplementedError):
            torch.jit.script(Foo)

            # 定义一个返回 Foo 实例的函数
            def fn():
                foo = Foo()
                return foo.x

            # 对函数 fn 进行脚本化调用
            torch.jit.script(fn)()

    # 用户应该能够自定义 __eq__ 方法而不被覆盖
    def test_custom__eq__(self):
        # 定义一个带有自定义 __eq__ 方法的数据类 CustomEq
        @torch.jit.script
        @dataclass
        class CustomEq:
            a: int
            b: int

            # 自定义 __eq__ 方法，只比较字段 a
            def __eq__(self, other: "CustomEq") -> bool:
                return self.a == other.a  # 忽略字段 b

        # 定义一个函数，创建两个 CustomEq 实例并比较它们
        def fn(a: int, b1: int, b2: int):
            pt1 = CustomEq(a, b1)
            pt2 = CustomEq(a, b2)
            return pt1 == pt2

        # 调用 self.checkScript 方法检查函数 fn 的脚本化版本
        self.checkScript(fn, [1, 2, 3])

    # 测试不提供源代码时的行为
    def test_no_source(self):
        # 断言期望抛出 RuntimeError 异常，因为 Enum 中使用列表不受支持
        with self.assertRaises(RuntimeError):
            # 使用 MixupParams 类进行脚本化，其中使用列表不受支持
            torch.jit.script(MixupParams)

        # 对 MixupParams2 类进行脚本化调用，不应抛出异常
        torch.jit.script(MixupParams2)  # 不抛出异常

    # 测试使用未注册的数据类时是否引发异常
    def test_use_unregistered_dataclass_raises(self):
        # 定义一个函数，接受 MixupParams3 类型的参数
        def f(a: MixupParams3):
            return 0

        # 断言期望抛出 OSError 异常，因为未注册 MixupParams3 类
        with self.assertRaises(OSError):
            torch.jit.script(f)
```