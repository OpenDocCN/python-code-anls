# `.\pytorch\test\torch_np\numpy_tests\core\test_numerictypes.py`

```
# Owner(s): ["module: dynamo"]

# 导入必要的模块和函数
import functools
import itertools
import sys

# 导入测试相关的函数和装饰器
from unittest import skipIf as skipif
from pytest import raises as assert_raises

# 导入 Torch 相关的测试工具函数和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

# 根据 TEST_WITH_TORCHDYNAMO 的值选择正确的 NumPy 包
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_

# 定义一个跳过装饰器的函数
skip = functools.partial(skipif, True)


@xpassIfTorchDynamo  # （
#    reason="We do not disctinguish between scalar and array types."
#    " Thus, scalars can upcast arrays."
# )
class TestCommonType(TestCase):
    # 测试函数：测试标量 'f4', 'f4', 'i2' 的最常见数据类型
    def test_scalar_loses1(self):
        res = np.find_common_type(["f4", "f4", "i2"], ["f8"])
        assert_(res == "f4")

    # 测试函数：测试标量 'f4', 'f4' 的最常见数据类型
    def test_scalar_loses2(self):
        res = np.find_common_type(["f4", "f4"], ["i8"])
        assert_(res == "f4")

    # 测试函数：测试标量 'f4', 'f4', 'i2' 和 'c8' 的最常见数据类型
    def test_scalar_wins(self):
        res = np.find_common_type(["f4", "f4", "i2"], ["c8"])
        assert_(res == "c8")

    # 测试函数：测试标量 'u4', 'i4', 'i4' 和 'f4' 的最常见数据类型
    def test_scalar_wins2(self):
        res = np.find_common_type(["u4", "i4", "i4"], ["f4"])
        assert_(res == "f8")

    # 测试函数：测试标量 'u8', 'i8', 'i8' 和 'f8' 的最常见数据类型
    def test_scalar_wins3(self):  # doesn't go up to 'f16' on purpose
        res = np.find_common_type(["u8", "i8", "i8"], ["f8"])
        assert_(res == "f8")


class TestIsSubDType(TestCase):
    # scalar types can be promoted into dtypes
    wrappers = [np.dtype, lambda x: x]

    # 测试函数：测试 np.floating 是否为 np.inexact 的子类型
    def test_both_abstract(self):
        assert_(np.issubdtype(np.floating, np.inexact))
        assert_(not np.issubdtype(np.inexact, np.floating))

    # 测试函数：测试相同的数据类型是否可以被升级
    def test_same(self):
        for cls in (np.float32, np.int32):
            for w1, w2 in itertools.product(self.wrappers, repeat=2):
                assert_(np.issubdtype(w1(cls), w2(cls)))

    # 测试函数：测试数据类型是否可以被升级为浮点数
    def test_subclass(self):
        # note we cannot promote floating to a dtype, as it would turn into a
        # concrete type
        for w in self.wrappers:
            assert_(np.issubdtype(w(np.float32), np.floating))
            assert_(np.issubdtype(w(np.float64), np.floating))

    # 测试函数：测试数据类型是否可以被反向升级
    def test_subclass_backwards(self):
        for w in self.wrappers:
            assert_(not np.issubdtype(np.floating, w(np.float32)))
            assert_(not np.issubdtype(np.floating, w(np.float64)))

    # 测试函数：测试兄弟类型之间是否存在子类型关系
    def test_sibling_class(self):
        for w1, w2 in itertools.product(self.wrappers, repeat=2):
            assert_(not np.issubdtype(w1(np.float32), w2(np.float64)))
            assert_(not np.issubdtype(w1(np.float64), w2(np.float32)))
    # 定义一个测试函数，用于测试数据类型不是 `NoneType` 或标量类型的情况
    def test_nondtype_nonscalartype(self):
        # 从 gh-14619 和 gh-9505 中得知，这些测试是为了修复引入的过时功能
        # 这些测试直接来源于 gh-9505
        # 检查 np.float32 是否不是 np.float64 的子类型
        assert not np.issubdtype(np.float32, "float64")
        # 检查 np.float32 是否不是 np.float64 的子类型（"f8" 是 "float64" 的别名）
        assert not np.issubdtype(np.float32, "f8")
        # 检查 np.int32 是否不是 np.int64 的子类型
        assert not np.issubdtype(np.int32, "int64")
        # 对于以下情况，正确的拼写应该是 np.integer、np.floating 或 np.complexfloating
        # 检查 np.int8 是否不是 int 的子类型（np.int8 从不是 np.int_）
        assert not np.issubdtype(np.int8, int)
        # 检查 np.float32 是否不是 float 的子类型
        assert not np.issubdtype(np.float32, float)
        # 检查 np.complex64 是否不是 complex 的子类型
        assert not np.issubdtype(np.complex64, complex)
        # 检查 np.float32 是否不是 float 的子类型
        assert not np.issubdtype(np.float32, "float")
        # 检查 np.float64 是否不是 np.float 的子类型
        assert not np.issubdtype(np.float64, "f")

        # 测试相同情况下的正确数据类型和抽象类型
        # 在 int、float、complex 的情况下：
        # 检查 np.float64 是否是 np.float64 的子类型
        assert np.issubdtype(np.float64, "float64")
        # 检查 np.float64 是否是 np.float64 的子类型（"f8" 是 "float64" 的别名）
        assert np.issubdtype(np.float64, "f8")
        # 检查 np.int64 是否是 np.int64 的子类型
        assert np.issubdtype(np.int64, "int64")
        # 检查 np.int8 是否是 np.integer 的子类型
        assert np.issubdtype(np.int8, np.integer)
        # 检查 np.float32 是否是 np.floating 的子类型
        assert np.issubdtype(np.float32, np.floating)
        # 检查 np.complex64 是否是 np.complexfloating 的子类型
        assert np.issubdtype(np.complex64, np.complexfloating)
        # 检查 np.float64 是否是 np.float 的子类型
        assert np.issubdtype(np.float64, "float")
        # 检查 np.float32 是否是 np.float 的子类型
        assert np.issubdtype(np.float32, "f")
# 定义一个测试类 TestBitName，用于测试 np.core.numerictypes.bitname 函数
@xpassIfTorchDynamo  # 标记为 xpassIfTorchDynamo，用于条件性跳过测试
# 类级别的注释，说明为何要跳过测试
# reason="We do not have (or need) np.core.numerictypes."
# " Our type aliases are in _dtypes.py."
# )
class TestBitName(TestCase):
    def test_abstract(self):
        # 测试函数 test_abstract，验证 np.core.numerictypes.bitname 是否会引发 ValueError 异常
        assert_raises(ValueError, np.core.numerictypes.bitname, np.floating)


@skip(reason="Docstrings for scalar types, not yet.")
@skipif(
    sys.flags.optimize > 1,
    # 类级别的注释，说明为何要跳过测试
    reason="no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1",
)
class TestDocStrings(TestCase):
    def test_platform_dependent_aliases(self):
        # 测试函数 test_platform_dependent_aliases，根据条件验证 np.int64 或 np.longlong 的文档字符串中是否包含 "int64"
        if np.int64 is np.int_:
            assert_("int64" in np.int_.__doc__)
        elif np.int64 is np.longlong:
            assert_("int64" in np.longlong.__doc__)


@instantiate_parametrized_tests
# 实例化参数化测试的注释
class TestScalarTypeNames(TestCase):
    # gh-9799

    # 定义数值类型列表
    numeric_types = [
        np.byte,
        np.short,
        np.intc,
        np.int_,  # , np.longlong, NB: torch does not properly have longlong
        np.ubyte,
        np.half,
        np.single,
        np.double,
        np.csingle,
        np.cdouble,
    ]

    def test_names_are_unique(self):
        # 测试函数 test_names_are_unique，确保数值类型列表中元素互不为别名
        assert len(set(self.numeric_types)) == len(self.numeric_types)

        # 测试函数 test_names_are_unique，确保数值类型列表中元素的名称唯一
        names = [t.__name__ for t in self.numeric_types]
        assert len(set(names)) == len(names)

    @parametrize("t", numeric_types)
    def test_names_reflect_attributes(self, t):
        """Test that names correspond to where the type is under ``np.``"""
        # 测试函数 test_names_reflect_attributes，验证数值类型名称在 np 下的对应性
        assert getattr(np, t.__name__) is t

    @skipIfTorchDynamo()  # XXX: weird, some names are not OK
    @parametrize("t", numeric_types)
    def test_names_are_undersood_by_dtype(self, t):
        """Test the dtype constructor maps names back to the type"""
        # 测试函数 test_names_are_undersood_by_dtype，验证 dtype 构造函数将名称映射回类型
        assert np.dtype(t.__name__).type is t


if __name__ == "__main__":
    # 运行测试
    run_tests()
```