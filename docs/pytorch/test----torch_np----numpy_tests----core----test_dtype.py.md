# `.\pytorch\test\torch_np\numpy_tests\core\test_dtype.py`

```
# Owner(s): ["module: dynamo"]

# 导入必要的模块和函数
import functools
import operator

import pickle
import sys
import types
from itertools import permutations
from typing import Any

# 导入测试相关的模块和函数
from unittest import skipIf as skipif

import pytest
from pytest import raises as assert_raises

# 导入 Torch 相关的测试辅助函数和测试用例
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
    xpassIfTorchDynamo,
)

# 设置 skip 作为 skipIf(True) 的别名
skip = functools.partial(skipif, True)

# 根据 TEST_WITH_TORCHDYNAMO 导入不同的 numpy 实现
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_, assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_, assert_equal

import numpy


# 定义一个断言函数，用于检查两个 dtype 是否相等
def assert_dtype_equal(a, b):
    assert_equal(a, b)
    assert_equal(
        hash(a), hash(b), "two equivalent types do not hash to the same value !"
    )


# 定义一个断言函数，用于检查两个 dtype 是否不相等
def assert_dtype_not_equal(a, b):
    assert_(a != b)
    assert_(hash(a) != hash(b), "two different types hash to the same value !")


# 实例化参数化测试类
@instantiate_parametrized_tests
class TestBuiltin(TestCase):
    # 参数化测试函数，测试不同类型的 dtype 对象
    @parametrize("t", [int, float, complex, np.int32])
    def test_run(self, t):
        """Only test hash runs at all."""
        dt = np.dtype(t)
        hash(dt)

    # 测试等价的 dtype 对象是否具有相同的哈希值
    def test_equivalent_dtype_hashing(self):
        intp = np.dtype(np.intp)
        if intp.itemsize == 4:
            left = intp
            right = np.dtype(np.int32)
        else:
            left = intp
            right = np.dtype(np.int64)
        assert_(left == right)
        assert_(hash(left) == hash(right))

    # 根据 TorchDynamo 的状态标记该测试为预期失败的测试
    @xfailIfTorchDynamo  # TypeError -> InternalTorchDynamoError
    def test_invalid_types(self):
        # 确保传入无效类型字符串时会触发 TypeError 异常

        assert_raises(TypeError, np.dtype, "O3")
        assert_raises(TypeError, np.dtype, "O5")
        assert_raises(TypeError, np.dtype, "O7")
        assert_raises(TypeError, np.dtype, "b3")
        assert_raises(TypeError, np.dtype, "h4")
        assert_raises(TypeError, np.dtype, "I5")
        assert_raises(TypeError, np.dtype, "e3")
        assert_raises(TypeError, np.dtype, "f5")

        if np.dtype("l").itemsize == 8:
            assert_raises(TypeError, np.dtype, "l4")
            assert_raises(TypeError, np.dtype, "L4")
        else:
            assert_raises(TypeError, np.dtype, "l8")
            assert_raises(TypeError, np.dtype, "L8")

    # XXX: 'q' 是什么？在我的 64 位 Ubuntu 机器上是 int64，与 'l' 相同
    #       if np.dtype('q').itemsize == 8:
    #           assert_raises(TypeError, np.dtype, 'q4')
    #           assert_raises(TypeError, np.dtype, 'Q4')
    #       else:
    #           assert_raises(TypeError, np.dtype, 'q8')
    #           assert_raises(TypeError, np.dtype, 'Q8')
    # 定义测试函数，验证不合法的数据类型在相等比较中返回False/True
    def test_richcompare_invalid_dtype_equality(self):
        # 确保不能转换为有效数据类型的对象在与有效数据类型比较时返回False/True
        # 这里的数字7不能转换为数据类型。不应该引发任何异常

        assert not np.dtype(np.int32) == 7, "dtype richcompare failed for =="
        assert np.dtype(np.int32) != 7, "dtype richcompare failed for !="

    # 使用参数化装饰器，测试不合法数据类型的比较操作是否引发TypeError异常
    @parametrize("operation", [operator.le, operator.lt, operator.ge, operator.gt])
    def test_richcompare_invalid_dtype_comparison(self, operation):
        # 确保对于不合法的数据类型，比较操作符会引发TypeError异常。这里的数字7是一个不合法的数据类型。

        with pytest.raises(TypeError):
            operation(np.dtype(np.int32), 7)

    # 使用条件跳过装饰器，根据numpy版本号跳过测试。旧版numpy会发出DeprecationWarnings。
    @skipif(
        numpy.__version__ < "1.24",
        reason="older numpies emit DeprecatioWarnings instead",
    )
    # 参数化装饰器，测试数值样式的数据类型是否为不合法
    @parametrize(
        "dtype",
        [
            "Bool",
            "Bytes0",
            "Complex32",
            "Complex64",
            "Datetime64",
            "Float16",
            "Float32",
            "Float64",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Object0",
            "Str0",
            "Timedelta64",
            "UInt8",
            "UInt16",
            "Uint32",
            "UInt32",
            "Uint64",
            "UInt64",
            "Void0",
            "Float128",
            "Complex128",
        ],
    )
    # 测试数值样式的数据类型是否为不合法，应该引发TypeError异常
    def test_numeric_style_types_are_invalid(self, dtype):
        with assert_raises(TypeError):
            np.dtype(dtype)
# 装饰器，用于跳过此测试类，原因是 "dtype attributes not yet implemented"
@skip(reason="dtype attributes not yet implemented")
class TestDtypeAttributeDeletion(TestCase):
    # 测试删除不可写的 dtype 属性时是否引发 AttributeError 异常
    def test_dtype_non_writable_attributes_deletion(self):
        # 创建一个双精度浮点型的 dtype 对象
        dt = np.dtype(np.double)
        # 定义需要删除的属性列表
        attr = [
            "subdtype",
            "descr",
            "str",
            "name",
            "base",
            "shape",
            "isbuiltin",
            "isnative",
            "isalignedstruct",
            "fields",
            "metadata",
            "hasobject",
        ]

        # 遍历属性列表，断言删除每个属性时是否引发 AttributeError 异常
        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)

    # 测试删除可写的 dtype 属性时是否引发 AttributeError 异常
    def test_dtype_writable_attributes_deletion(self):
        # 创建一个双精度浮点型的 dtype 对象
        dt = np.dtype(np.double)
        # 定义需要删除的属性列表
        attr = ["names"]
        # 遍历属性列表，断言删除每个属性时是否引发 AttributeError 异常
        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)


# 装饰器，用于实例化参数化测试
@instantiate_parametrized_tests
class TestPickling(TestCase):
    # 检查 dtype 对象的序列化和反序列化
    def check_pickling(self, dtype):
        # 对于支持的所有 pickle 协议版本
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # 序列化 dtype 对象
            buf = pickle.dumps(dtype, proto)
            # 检查序列化后的数据中是否包含特定字段
            # 如果 dtype 被序列化为单例，则应存储在缓冲区中
            assert b"_DType_reconstruct" not in buf
            assert b"dtype" in buf
            # 反序列化对象
            pickled = pickle.loads(buf)
            # 断言反序列化后的对象与原对象相等
            assert_equal(pickled, dtype)

            # XXX: 我们的 dtype 没有 .descr
            # assert_equal(pickled.descr, dtype.descr)
            # if dtype.metadata is not None:
            #     assert_equal(pickled.metadata, dtype.metadata)
            
            # 检查重建的 dtype 对象是否可用
            x = np.zeros(3, dtype=dtype)
            y = np.zeros(3, dtype=pickled)
            assert_equal(x, y)
            assert_equal(x[0], y[0])

    # 参数化测试，测试内置数据类型的序列化和反序列化
    @parametrize("t", [int, float, complex, np.int32, bool])
    def test_builtin(self, t):
        self.check_pickling(np.dtype(t))

    # 参数化测试，测试所有可能的数据类型的序列化和反序列化
    @parametrize(
        "DType",
        [
            subtest(type(np.dtype(t)), name=f"{np.dtype(t).name}_{i}")
            for i, t in enumerate(np.typecodes["All"])
        ]
        + [np.dtype],
    )
    def test_pickle_types(self, DType):
        # 对于支持的所有 pickle 协议版本，测试 DTypes 类型的序列化和反序列化
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            roundtrip_DType = pickle.loads(pickle.dumps(DType, proto))
            assert roundtrip_DType is DType


# 装饰器，用于跳过此测试类，原因是 "XXX: value-based promotions, we don't have."
@skip(reason="XXX: value-based promotions, we don't have.")
@instantiate_parametrized_tests
class TestPromotion(TestCase):
    """Test cases related to more complex DType promotions.  Further promotion
    tests are defined in `test_numeric.py`
    """
    @parametrize(
        "other, expected, expected_weak",
        [
            (2**16 - 1, np.complex64, None),  # 参数化测试: other为2**16-1时，期望返回np.complex64，无弱类型提升期望
            (2**32 - 1, np.complex128, np.complex64),  # 参数化测试: other为2**32-1时，期望返回np.complex128，弱类型提升期望为np.complex64
            subtest((np.float16(2), np.complex64, None), name="float16_complex64_None"),  # 子测试: 对(np.float16(2), np.complex64, None)进行测试，命名为"float16_complex64_None"
            subtest((np.float32(2), np.complex64, None), name="float32_complex64_None"),  # 子测试: 对(np.float32(2), np.complex64, None)进行测试，命名为"float32_complex64_None"
            # 复数标量的重复测试:
            subtest(
                (np.complex64(2), np.complex64, None), name="complex64_complex64_None"
            ),  # 子测试: 对(np.complex64(2), np.complex64, None)进行测试，命名为"complex64_complex64_None"
        ],
    )
    def test_complex_other_value_based(
        self, weak_promotion, other, expected, expected_weak
    ):
        if weak_promotion and expected_weak is not None:
            expected = expected_weak

        # 如果我们修改基于值的提升，这里会改变
        min_complex = np.dtype(np.complex64)

        res = np.result_type(other, min_complex)
        assert res == expected
        # 使用相同逻辑的简单ufunc调用进行相同检查：
        res = np.minimum(other, np.ones(3, dtype=min_complex)).dtype
        assert res == expected

    @parametrize(
        "other, expected",
        [
            (np.bool_, np.complex128),
            (np.int64, np.complex128),
            (np.float16, np.complex64),
            (np.float32, np.complex64),
            (np.float64, np.complex128),
            (np.complex64, np.complex64),
            (np.complex128, np.complex128),
        ],
    )
    def test_complex_scalar_value_based(self, other, expected):
        # 如果我们修改基于值的提升，这里会改变
        complex_scalar = 1j

        res = np.result_type(other, complex_scalar)
        assert res == expected
        # 使用相同逻辑的简单ufunc调用进行相同检查：
        res = np.minimum(np.ones(3, dtype=other), complex_scalar).dtype
        assert res == expected

    @parametrize("val", [2, 2**32, 2**63, 2**64, 2 * 100])
    def test_python_integer_promotion(self, val):
        # 如果我们只传递标量（主要是Python的！），结果必须考虑整数可能被认为是int32、int64、uint64或对象，这里测试这些路径！
        expected_dtype = np.result_type(np.array(val).dtype, np.array(0).dtype)
        assert np.result_type(val, 0) == expected_dtype
        # 为了完整起见，还要使用NumPy标量作为第二个参数进行检查：
        assert np.result_type(val, np.int8(0)) == expected_dtype
    @parametrize(
        "dtypes, expected",
        [
            # 这些类型提升不是关联/交换的：
            ([np.int16, np.float16], np.float32),
            ([np.int8, np.float16], np.float32),
            ([np.uint8, np.int16, np.float16], np.float32),
            # 下面的类型提升不是模棱两可的，但涵盖了抽象提升的代码路径（未测试任何特定逻辑）
            ([1, 1, np.float64], np.float64),
            ([1, 1.0, np.complex128], np.complex128),
            ([1, 1j, np.float64], np.complex128),
            ([1.0, 1.0, np.int64], np.float64),
            ([1.0, 1j, np.float64], np.complex128),
            ([1j, 1j, np.float64], np.complex128),
            ([1, True, np.bool_], np.int_),
        ],
    )
    # 测试排列组合对结果没有影响。在上面的示例中，一些无符号整数和整数的组合会提升为更大的整数类型，然后会提升为不必要更大的浮点数类型。
    def test_permutations_do_not_influence_result(self, dtypes, expected):
        # 对于 dtypes 的所有排列组合进行遍历
        for perm in permutations(dtypes):
            # 断言 np.result_type(*perm) 的结果等于 expected
            assert np.result_type(*perm) == expected
class TestMisc(TestCase):
    # 测试是否布尔值为真
    def test_dtypes_are_true(self):
        # test for gh-6294
        # 检查 np.dtype("f8") 是否为真
        assert bool(np.dtype("f8"))
        # 检查 np.dtype("i8") 是否为真
        assert bool(np.dtype("i8"))

    @xpassIfTorchDynamo  # (reason="No keyword arg for dtype ctor.")
    # 测试关键字参数
    def test_keyword_argument(self):
        # test for https://github.com/numpy/numpy/pull/16574#issuecomment-642660971
        # 断言使用关键字参数构造的 np.dtype 是否等于使用普通参数构造的 np.dtype
        assert np.dtype(dtype=np.float64) == np.dtype(np.float64)

    @skipif(sys.version_info >= (3, 9), reason="Requires python 3.9")
    # 测试类的索引操作在 Python 3.8 下的情况
    def test_class_getitem_38(self) -> None:
        match = "Type subscription requires python >= 3.9"
        # 使用 pytest 来检查是否会抛出 TypeError 异常
        with pytest.raises(TypeError):  # , match=match):
            np.dtype[Any]


class TestFromDTypeAttribute(TestCase):
    # 测试简单情况
    def test_simple(self):
        class dt:
            dtype = np.dtype("f8")

        # 断言自定义类 dt 的 np.dtype 是否等于 np.float64
        assert np.dtype(dt) == np.float64
        # 断言 dt 实例的 np.dtype 是否等于 np.float64
        assert np.dtype(dt()) == np.float64

    @skip(
        reason="We simply require the .name attribute, so this "
        "fails with an AttributeError."
    )
    # 测试递归情况
    def test_recursion(self):
        class dt:
            pass

        dt.dtype = dt
        # 使用 pytest 来检查是否会抛出 RecursionError 异常
        with pytest.raises(RecursionError):
            np.dtype(dt)

        dt_instance = dt()
        dt_instance.dtype = dt
        # 使用 pytest 来检查是否会抛出 RecursionError 异常
        with pytest.raises(RecursionError):
            np.dtype(dt_instance)


@skip(reason="Parameteric dtypes, our stuff is simpler.")
@skipif(sys.version_info < (3, 9), reason="Requires python 3.9")
@instantiate_parametrized_tests
class TestClassGetItem(TestCase):
    # 测试 dtype 属性
    def test_dtype(self) -> None:
        alias = np.dtype[Any]
        # 断言 alias 是否是 types.GenericAlias 类的实例
        assert isinstance(alias, types.GenericAlias)
        # 断言 alias 的原始类型是否为 np.dtype
        assert alias.__origin__ is np.dtype

    @parametrize("code", np.typecodes["All"])
    # 测试 dtype 的子类
    def test_dtype_subclass(self, code: str) -> None:
        cls = type(np.dtype(code))
        alias = cls[Any]
        # 断言 alias 是否是 types.GenericAlias 类的实例
        assert isinstance(alias, types.GenericAlias)
        # 断言 alias 的原始类型是否为 cls
        assert alias.__origin__ is cls

    @parametrize("arg_len", range(4))
    # 测试元组的索引操作
    def test_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            assert np.dtype[arg_tup]
        else:
            # 使用 pytest 来检查是否会抛出 TypeError 异常
            with pytest.raises(TypeError):
                np.dtype[arg_tup]

    # 测试标量的索引操作
    def test_subscript_scalar(self) -> None:
        assert np.dtype[Any]


if __name__ == "__main__":
    run_tests()
```