# `.\numpy\numpy\_core\tests\test_deprecations.py`

```py
"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.

"""
# 导入必要的库和模块
import datetime  # 导入日期时间模块
import operator  # 导入操作符模块
import warnings  # 导入警告模块
import pytest  # 导入 pytest 测试框架
import tempfile  # 导入临时文件模块
import re  # 导入正则表达式模块
import sys  # 导入系统相关模块

# 导入 numpy 库及其测试相关的模块和函数
import numpy as np
from numpy.testing import (
    assert_raises, assert_warns, assert_, assert_array_equal, SkipTest,
    KnownFailureException, break_cycles, temppath
    )

# 导入 C API 相关测试模块
from numpy._core._multiarray_tests import fromstring_null_term_c_api

# 尝试导入 pytz 库，标记是否成功
try:
    import pytz
    _has_pytz = True
except ImportError:
    _has_pytz = False


class _DeprecationTestCase:
    # 警告信息的起始部分必须匹配，因为 warnings 使用了 re.match
    message = ''
    warning_cls = DeprecationWarning

    def setup_method(self):
        # 捕获警告信息并记录
        self.warn_ctx = warnings.catch_warnings(record=True)
        self.log = self.warn_ctx.__enter__()

        # 不要忽略其它 DeprecationWarnings，因为忽略可能导致混乱
        warnings.filterwarnings("always", category=self.warning_cls)
        warnings.filterwarnings("always", message=self.message,
                                category=self.warning_cls)

    def teardown_method(self):
        # 退出警告记录环境
        self.warn_ctx.__exit__()

    def assert_not_deprecated(self, function, args=(), kwargs={}):
        """Test that warnings are not raised.

        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
        # 断言不会出现警告
        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)


class _VisibleDeprecationTestCase(_DeprecationTestCase):
    # 使用可见性 DeprecationWarning 类
    warning_cls = np.exceptions.VisibleDeprecationWarning


class TestDTypeAttributeIsDTypeDeprecation(_DeprecationTestCase):
    # 警告消息指定 `dtype` 属性已废弃，从 NumPy 1.21 开始
    message = r".*`.dtype` attribute"

    def test_deprecation_dtype_attribute_is_dtype(self):
        # 定义两个测试类，验证其 dtype 属性是否会触发警告
        class dt:
            dtype = "f8"

        class vdt(np.void):
            dtype = "f,f"

        # 断言各种情况下会触发 dtype 相关警告
        self.assert_deprecated(lambda: np.dtype(dt))
        self.assert_deprecated(lambda: np.dtype(dt()))
        self.assert_deprecated(lambda: np.dtype(vdt))
        self.assert_deprecated(lambda: np.dtype(vdt(1)))


class TestTestDeprecated:
    # 这里测试的部分被省略，需要继续添加注释
    # 定义一个测试方法 test_assert_deprecated，用于测试断言是否捕获了弃用警告
    def test_assert_deprecated(self):
        # 创建 _DeprecationTestCase 的实例
        test_case_instance = _DeprecationTestCase()
        # 调用 setup_method 方法来设置测试环境
        test_case_instance.setup_method()
        # 断言捕获 Assertion 错误，期望 test_case_instance.assert_deprecated 调用时会引发 AssertionError
        assert_raises(AssertionError,
                      test_case_instance.assert_deprecated,
                      lambda: None)

        # 定义一个函数 foo，该函数会发出一个 DeprecationWarning 警告
        def foo():
            warnings.warn("foo", category=DeprecationWarning, stacklevel=2)

        # 调用 test_case_instance.assert_deprecated 来检查是否捕获 foo 函数发出的弃用警告
        test_case_instance.assert_deprecated(foo)
        # 调用 teardown_method 方法来清理测试环境
        test_case_instance.teardown_method()
# _DeprecationTestCase 的子类，用于测试 ndarray.conjugate 在非数值数据类型上的行为
class TestNonNumericConjugate(_DeprecationTestCase):

    """
    Deprecate no-op behavior of ndarray.conjugate on non-numeric dtypes,
    which conflicts with the error behavior of np.conjugate.
    """

    # 测试 ndarray.conjugate 方法的行为
    def test_conjugate(self):
        # 对于 np.array(5) 和 np.array(5j)，检查其不应该被弃用
        for a in np.array(5), np.array(5j):
            self.assert_not_deprecated(a.conjugate)
        # 对于 np.array('s'), np.array('2016', 'M'), np.array((1, 2), [('a', int), ('b', int)])，
        # 检查其应该被弃用
        for a in (np.array('s'), np.array('2016', 'M'),
                np.array((1, 2), [('a', int), ('b', int)])):
            self.assert_deprecated(a.conjugate)


# _DeprecationTestCase 的子类，测试 datetime 和 timedelta 的事件
class TestDatetimeEvent(_DeprecationTestCase):

    # 2017-08-11, 1.14.0
    # 测试 3 元组的情况
    def test_3_tuple(self):
        # 对于 np.datetime64 和 np.timedelta64，检查其应该不被弃用
        for cls in (np.datetime64, np.timedelta64):
            # 两种有效用法 - (unit, num) 和 (unit, num, den, None)
            self.assert_not_deprecated(cls, args=(1, ('ms', 2)))
            self.assert_not_deprecated(cls, args=(1, ('ms', 2, 1, None)))

            # 尝试使用事件参数，在 1.7.0 版本中被移除，因此应该被弃用
            # 事件参数曾经是 uint8 类型
            self.assert_deprecated(cls, args=(1, ('ms', 2, 'event')))
            self.assert_deprecated(cls, args=(1, ('ms', 2, 63)))
            self.assert_deprecated(cls, args=(1, ('ms', 2, 1, 'event')))
            self.assert_deprecated(cls, args=(1, ('ms', 2, 1, 63)))


# _DeprecationTestCase 的子类，测试空数组的真值测试
class TestTruthTestingEmptyArrays(_DeprecationTestCase):

    # 2017-09-25, 1.14.0
    message = '.*truth value of an empty array is ambiguous.*'

    # 测试 1 维数组的情况
    def test_1d(self):
        # 检查空数组的布尔值测试应该被弃用
        self.assert_deprecated(bool, args=(np.array([]),))

    # 测试 2 维数组的情况
    def test_2d(self):
        # 检查不同形状的空数组的布尔值测试应该被弃用
        self.assert_deprecated(bool, args=(np.zeros((1, 0)),))
        self.assert_deprecated(bool, args=(np.zeros((0, 1)),))
        self.assert_deprecated(bool, args=(np.zeros((0, 0)),))


# _DeprecationTestCase 的子类，测试 np.bincount 方法的 minlength 参数
class TestBincount(_DeprecationTestCase):

    # 2017-06-01, 1.14.0
    # 测试 bincount 方法的 minlength 参数
    def test_bincount_minlength(self):
        # 检查使用 None 作为 minlength 参数的情况应该被弃用
        self.assert_deprecated(lambda: np.bincount([1, 2, 3], minlength=None))


# _DeprecationTestCase 的子类，测试生成器作为 np.sum 的输入
class TestGeneratorSum(_DeprecationTestCase):

    # 2018-02-25, 1.15.0
    # 测试生成器作为 np.sum 的输入
    def test_generator_sum(self):
        # 检查使用生成器作为输入的情况应该被弃用
        self.assert_deprecated(np.sum, args=((i for i in range(5)),))


# _DeprecationTestCase 的子类，测试 np.fromstring 方法的使用
class TestFromstring(_DeprecationTestCase):

    # 2017-10-19, 1.14
    # 测试 fromstring 方法的使用
    def test_fromstring(self):
        # 检查使用 fromstring 方法的情况应该被弃用
        self.assert_deprecated(np.fromstring, args=('\x00'*80,))


# _DeprecationTestCase 的子类，测试无效数据作为字符串或文件输入的情况
class TestFromStringAndFileInvalidData(_DeprecationTestCase):

    # 2019-06-08, 1.17.0
    # 测试字符串或文件输入无法完全读取的情况
    # 当弃用完成后，这些测试应该被移到实际测试中
    message = "string or file could not be read to its end"

    @pytest.mark.parametrize("invalid_str", [",invalid_data", "invalid_sep"])
    # 定义一个测试方法，用于测试处理不可解析数据文件时的行为
    def test_deprecate_unparsable_data_file(self, invalid_str):
        # 创建一个包含浮点数的 NumPy 数组
        x = np.array([1.51, 2, 3.51, 4], dtype=float)

        # 使用临时文件来操作数据
        with tempfile.TemporaryFile(mode="w") as f:
            # 将数组 x 的数据以指定格式写入到临时文件 f 中，数据项之间用逗号分隔，保留两位小数
            x.tofile(f, sep=',', format='%.2f')
            # 向临时文件 f 中写入不可解析的字符串 invalid_str
            f.write(invalid_str)

            # 将文件指针移动到文件开头
            f.seek(0)
            # 使用 assert_deprecated 方法断言调用 np.fromfile 时会发出 DeprecationWarning
            self.assert_deprecated(lambda: np.fromfile(f, sep=","))
            
            # 将文件指针移动到文件开头
            f.seek(0)
            # 使用 assert_deprecated 方法断言调用 np.fromfile 时会发出 DeprecationWarning，并指定读取的数据项数目为 5
            self.assert_deprecated(lambda: np.fromfile(f, sep=",", count=5))
            
            # 不应该触发警告：
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                # 将文件指针移动到文件开头
                f.seek(0)
                # 读取临时文件 f 中的数据，数据项之间用逗号分隔，期望的数据项数目为 4，将结果与数组 x 进行比较
                res = np.fromfile(f, sep=",", count=4)
                assert_array_equal(res, x)

    # 使用参数化测试来测试处理不可解析的字符串时的行为
    @pytest.mark.parametrize("invalid_str", [",invalid_data", "invalid_sep"])
    def test_deprecate_unparsable_string(self, invalid_str):
        # 创建一个包含浮点数的 NumPy 数组
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        # 创建一个包含不可解析字符串的字符串 x_str
        x_str = "1.51,2,3.51,4{}".format(invalid_str)

        # 使用 assert_deprecated 方法断言调用 np.fromstring 时会发出 DeprecationWarning
        self.assert_deprecated(lambda: np.fromstring(x_str, sep=","))

        # 使用 assert_deprecated 方法断言调用 np.fromstring 时会发出 DeprecationWarning，并指定读取的数据项数目为 5
        self.assert_deprecated(lambda: np.fromstring(x_str, sep=",", count=5))

        # 测试 C 语言级别的 API，使用 0 结尾的字符串来创建数组
        bytestr = x_str.encode("ascii")
        self.assert_deprecated(lambda: fromstring_null_term_c_api(bytestr))

        # 使用 assert_warns 方法断言调用 np.fromstring 时会产生警告，警告类型为 DeprecationWarning
        with assert_warns(DeprecationWarning):
            # 调用 np.fromstring 从字符串 x_str 中读取数据，数据项之间用逗号分隔，期望的数据项数目为 5，
            # 将结果与数组 x 进行比较，排除最后一个元素
            res = np.fromstring(x_str, sep=",", count=5)
            assert_array_equal(res[:-1], x)

        # 关闭 DeprecationWarning 警告，并确保不会触发其它警告
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            # 不应该触发警告：
            # 调用 np.fromstring 从字符串 x_str 中读取数据，数据项之间用逗号分隔，期望的数据项数目为 4，
            # 将结果与数组 x 进行比较
            res = np.fromstring(x_str, sep=",", count=4)
            assert_array_equal(res, x)
# 继承自 _DeprecationTestCase 的测试类，用于测试 tostring() 方法的废弃情况
class TestToString(_DeprecationTestCase):
    # 设置错误消息，使用 re.escape 转义特殊字符
    message = re.escape("tostring() is deprecated. Use tobytes() instead.")

    # 测试 tostring() 方法的废弃情况
    def test_tostring(self):
        # 创建一个包含特定字节序列的 numpy 数组
        arr = np.array(list(b"test\xFF"), dtype=np.uint8)
        # 断言使用了废弃的 tostring() 方法
        self.assert_deprecated(arr.tostring)

    # 测试 tostring() 方法与 tobytes() 方法是否匹配
    def test_tostring_matches_tobytes(self):
        # 创建一个包含特定字节序列的 numpy 数组
        arr = np.array(list(b"test\xFF"), dtype=np.uint8)
        # 将数组转换为字节流
        b = arr.tobytes()
        # 使用 assert_warns 检测 DeprecationWarning
        with assert_warns(DeprecationWarning):
            # 使用废弃的 tostring() 方法
            s = arr.tostring()
        # 断言 tostring() 方法的结果与 tobytes() 方法的结果相同
        assert s == b


# 继承自 _DeprecationTestCase 的测试类，用于测试类型强制转换的废弃情况
class TestDTypeCoercion(_DeprecationTestCase):
    # 设置错误消息，指出正在废弃的类型转换情况
    message = "Converting .* to a dtype .*is deprecated"
    # 包含所有废弃类型的列表
    deprecated_types = [
        # 内置的标量超类型:
        np.generic, np.flexible, np.number,
        np.inexact, np.floating, np.complexfloating,
        np.integer, np.unsignedinteger, np.signedinteger,
        # 字符串类型是一个特殊的 S1 废弃案例:
        np.character,
    ]

    # 测试各种类型的强制转换是否废弃
    def test_dtype_coercion(self):
        for scalar_type in self.deprecated_types:
            # 断言废弃类型转换
            self.assert_deprecated(np.dtype, args=(scalar_type,))

    # 测试数组构造中的类型强制转换是否废弃
    def test_array_construction(self):
        for scalar_type in self.deprecated_types:
            # 断言数组构造中的废弃类型转换
            self.assert_deprecated(np.array, args=([], scalar_type,))

    # 测试未废弃的类型转换
    def test_not_deprecated(self):
        # 所有特定类型都未废弃:
        for group in np._core.sctypes.values():
            for scalar_type in group:
                # 断言未废弃的类型转换
                self.assert_not_deprecated(np.dtype, args=(scalar_type,))

        # 对于如 type、dict、list、tuple 等典型的 Python 类型，当前被强制转换为对象:
        for scalar_type in [type, dict, list, tuple]:
            # 断言未废弃的类型转换
            self.assert_not_deprecated(np.dtype, args=(scalar_type,))


# 继承自 _DeprecationTestCase 的测试类，用于测试复数类型的 round() 方法的废弃情况
class BuiltInRoundComplexDType(_DeprecationTestCase):
    # 包含废弃复数类型的列表
    deprecated_types = [np.csingle, np.cdouble, np.clongdouble]
    # 包含未废弃的数值类型的列表
    not_deprecated_types = [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
    ]

    # 测试废弃的复数类型的 round() 方法
    def test_deprecated(self):
        for scalar_type in self.deprecated_types:
            scalar = scalar_type(0)
            # 断言废弃的 round() 方法
            self.assert_deprecated(round, args=(scalar,))
            self.assert_deprecated(round, args=(scalar, 0))
            self.assert_deprecated(round, args=(scalar,), kwargs={'ndigits': 0})

    # 测试未废弃的数值类型的 round() 方法
    def test_not_deprecated(self):
        for scalar_type in self.not_deprecated_types:
            scalar = scalar_type(0)
            # 断言未废弃的 round() 方法
            self.assert_not_deprecated(round, args=(scalar,))
            self.assert_not_deprecated(round, args=(scalar, 0))
            self.assert_not_deprecated(round, args=(scalar,), kwargs={'ndigits': 0})


# 继承自 _DeprecationTestCase 的测试类，用于测试在高级索引中空结果的错误情况
class TestIncorrectAdvancedIndexWithEmptyResult(_DeprecationTestCase):
    # 错误消息，指出已经不再忽略的越界索引
    message = "Out of bound index found. This was previously ignored.*"

    # 使用 pytest.mark.parametrize 注入参数化的测试索引
    @pytest.mark.parametrize("index", [([3, 0],), ([0, 0], [3, 0])])
    # 定义一个测试方法，用于测试空子空间情况
    def test_empty_subspace(self, index):
        # 测试单个和多个高级索引的情况
        # 这些情况在未来可能会引发 IndexError
        arr = np.ones((2, 2, 0))
        # 断言 arr.__getitem__ 方法会被弃用
        self.assert_deprecated(arr.__getitem__, args=(index,))
        # 断言 arr.__setitem__ 方法会被弃用，将 index 作为参数
        self.assert_deprecated(arr.__setitem__, args=(index, 0.))

        # 对于这个数组，子空间只有在应用切片后才为空
        arr2 = np.ones((2, 2, 1))
        # 创建一个新的索引 index2，通过在开头插入 slice(0, 0) 来扩展 index
        index2 = (slice(0, 0),) + index
        # 断言 arr2.__getitem__ 方法会被弃用，将 index2 作为参数
        self.assert_deprecated(arr2.__getitem__, args=(index2,))
        # 断言 arr2.__setitem__ 方法会被弃用，将 index2 和 0 作为参数
        self.assert_deprecated(arr2.__setitem__, args=(index2, 0.))

    # 定义一个测试方法，用于测试空索引广播不会被弃用的情况
    def test_empty_index_broadcast_not_deprecated(self):
        arr = np.ones((2, 2, 2))

        # 定义一个广播索引 index，将会产生一个空的结果
        index = ([[3], [2]], [])  # broadcast to an empty result.
        # 断言 arr.__getitem__ 方法不会被弃用，将 index 作为参数
        self.assert_not_deprecated(arr.__getitem__, args=(index,))
        # 断言 arr.__setitem__ 方法不会被弃用，将 index 和空数组作为参数
        self.assert_not_deprecated(arr.__setitem__,
                                   args=(index, np.empty((2, 0, 2))))
class TestNonExactMatchDeprecation(_DeprecationTestCase):
    # 继承自 _DeprecationTestCase 的测试类，测试非精确匹配的警告信息
    def test_non_exact_match(self):
        # 创建一个二维 NumPy 数组
        arr = np.array([[3, 6, 6], [4, 5, 1]])
        # 断言检查：期望在使用 np.ravel_multi_index 函数时产生警告，参数 mode 的值拼写错误
        self.assert_deprecated(lambda: np.ravel_multi_index(arr, (7, 6), mode='Cilp'))
        # 断言检查：期望在使用 np.searchsorted 函数时产生警告，参数 side 的值以 'R' 开头完全不同
        self.assert_deprecated(lambda: np.searchsorted(arr[0], 4, side='Random'))


class TestMatrixInOuter(_DeprecationTestCase):
    # 继承自 _DeprecationTestCase 的测试类，测试 np.matrix 在 np.add.outer 中的警告
    # 2020-05-13 NumPy 1.20.0
    message = (r"add.outer\(\) was passed a numpy matrix as "
               r"(first|second) argument.")

    def test_deprecated(self):
        # 创建一个一维 NumPy 数组
        arr = np.array([1, 2, 3])
        # 创建一个 np.matrix 对象
        m = np.array([1, 2, 3]).view(np.matrix)
        # 断言检查：期望在使用 np.add.outer 函数时产生警告，传递 np.matrix 对象作为参数
        self.assert_deprecated(np.add.outer, args=(m, m), num=2)
        self.assert_deprecated(np.add.outer, args=(arr, m))
        self.assert_deprecated(np.add.outer, args=(m, arr))
        # 断言检查：期望在使用 np.add.outer 函数时不会产生警告，传递两个一维数组作为参数
        self.assert_not_deprecated(np.add.outer, args=(arr, arr))


class FlatteningConcatenateUnsafeCast(_DeprecationTestCase):
    # 继承自 _DeprecationTestCase 的测试类，测试使用 axis=None 的 np.concatenate 的警告
    # NumPy 1.20, 2020-09-03
    message = "concatenate with `axis=None` will use same-kind casting"

    def test_deprecated(self):
        # 断言检查：期望在使用 np.concatenate 函数时产生警告，设置 axis=None 并使用不安全的类型转换
        self.assert_deprecated(np.concatenate,
                args=(([0.], [1.]),),
                kwargs=dict(axis=None, out=np.empty(2, dtype=np.int64)))

    def test_not_deprecated(self):
        # 断言检查：期望在使用 np.concatenate 函数时不会产生警告，设置 axis=None 并使用安全的类型转换
        self.assert_not_deprecated(np.concatenate,
                args=(([0.], [1.]),),
                kwargs={'axis': None, 'out': np.empty(2, dtype=np.int64),
                        'casting': "unsafe"})

        # 使用 assert_raises 检查是否会抛出 TypeError 异常
        with assert_raises(TypeError):
            # 断言检查：确保在传递参数时首先会注意到警告
            np.concatenate(([0.], [1.]), out=np.empty(2, dtype=np.int64),
                           casting="same_kind")


class TestDeprecatedUnpickleObjectScalar(_DeprecationTestCase):
    # 继承自 _DeprecationTestCase 的测试类，测试反序列化 numpy 对象标量的警告
    # Deprecated 2020-11-24, NumPy 1.20
    """
    技术上，应该不可能创建 numpy 对象标量，但存在一个理论上允许的反序列化路径。该路径无效且必须导致警告。
    """
    message = "Unpickling a scalar with object dtype is deprecated."

    def test_deprecated(self):
        # lambda 函数测试：使用 np._core.multiarray.scalar 创建一个具有对象类型的标量，期望产生警告
        ctor = np._core.multiarray.scalar
        self.assert_deprecated(lambda: ctor(np.dtype("O"), 1))


class TestSingleElementSignature(_DeprecationTestCase):
    # 继承自 _DeprecationTestCase 的测试类，测试使用长度为1的签名的警告
    # Deprecated 2021-04-01, NumPy 1.21
    message = r"The use of a length 1"

    def test_deprecated(self):
        # lambda 函数测试：使用 np.add 函数时，使用长度为 1 的签名，期望产生警告
        self.assert_deprecated(lambda: np.add(1, 2, signature="d"))
        self.assert_deprecated(lambda: np.add(1, 2, sig=(np.dtype("l"),)))


class TestCtypesGetter(_DeprecationTestCase):
    # 继承自 _DeprecationTestCase 的测试类，测试使用 ctypes 获取属性的警告
    # Deprecated 2021-05-18, Numpy 1.21.0
    warning_cls = DeprecationWarning
    ctypes = np.array([1]).ctypes

    @pytest.mark.parametrize(
        "name", ["get_data", "get_shape", "get_strides", "get_as_parameter"]
    )
    # 定义一个测试方法，用于测试废弃的功能是否触发了警告
    def test_deprecated(self, name: str) -> None:
        # 获取self.ctypes对象中名为name的属性或方法，并赋值给func变量
        func = getattr(self.ctypes, name)
        # 断言调用func()函数会触发废弃警告
        self.assert_deprecated(lambda: func())
    
    # 使用pytest的参数化装饰器，定义多个参数化测试用例，每个用例测试一个属性名
    @pytest.mark.parametrize(
        "name", ["data", "shape", "strides", "_as_parameter_"]
    )
    # 定义一个测试方法，用于测试未废弃的功能是否没有触发警告
    def test_not_deprecated(self, name: str) -> None:
        # 断言调用self.ctypes对象中名为name的属性或方法不会触发废弃警告
        self.assert_not_deprecated(lambda: getattr(self.ctypes, name))
# 定义一个字典，包含了不同的分区方法和函数作为值，用于测试
PARTITION_DICT = {
    "partition method": np.arange(10).partition,
    "argpartition method": np.arange(10).argpartition,
    "partition function": lambda kth: np.partition(np.arange(10), kth),
    "argpartition function": lambda kth: np.argpartition(np.arange(10), kth),
}

# 使用 pytest 的参数化装饰器，对 PARTITION_DICT 中的每个值进行单独的测试
@pytest.mark.parametrize("func", PARTITION_DICT.values(), ids=PARTITION_DICT)
class TestPartitionBoolIndex(_DeprecationTestCase):
    # Deprecated 2021-09-29, NumPy 1.22
    # 使用 DeprecationWarning 来标记测试中的弃用警告
    warning_cls = DeprecationWarning
    message = "Passing booleans as partition index is deprecated"

    # 测试用例，检查传递布尔值作为分区索引是否会触发弃用警告
    def test_deprecated(self, func):
        self.assert_deprecated(lambda: func(True))
        self.assert_deprecated(lambda: func([False, True]))

    # 测试用例，检查传递其他类型参数是否不会触发弃用警告
    def test_not_deprecated(self, func):
        self.assert_not_deprecated(lambda: func(1))
        self.assert_not_deprecated(lambda: func([0, 1]))


class TestMachAr(_DeprecationTestCase):
    # Deprecated 2022-11-22, NumPy 1.25
    # 使用 DeprecationWarning 来标记测试中的弃用警告
    warning_cls = DeprecationWarning

    # 测试用例，检查模块中的 MachAr 是否会触发弃用警告
    def test_deprecated_module(self):
        self.assert_deprecated(lambda: getattr(np._core, "MachAr"))


class TestQuantileInterpolationDeprecation(_DeprecationTestCase):
    # Deprecated 2021-11-08, NumPy 1.22
    # 使用 pytest 的参数化装饰器，对多个函数进行测试
    @pytest.mark.parametrize("func",
        [np.percentile, np.quantile, np.nanpercentile, np.nanquantile])
    def test_deprecated(self, func):
        # 检查使用线性插值时是否会触发弃用警告
        self.assert_deprecated(
            lambda: func([0., 1.], 0., interpolation="linear"))
        # 检查使用最近邻插值时是否会触发弃用警告
        self.assert_deprecated(
            lambda: func([0., 1.], 0., interpolation="nearest"))

    # 测试用例，检查在使用两个特定参数时是否会引发 TypeError 异常
    @pytest.mark.parametrize("func",
            [np.percentile, np.quantile, np.nanpercentile, np.nanquantile])
    def test_both_passed(self, func):
        with warnings.catch_warnings():
            # 设置警告过滤器以捕获 DeprecationWarning
            warnings.simplefilter("always", DeprecationWarning)
            # 检查是否会因 method 参数而触发 TypeError 异常
            with pytest.raises(TypeError):
                func([0., 1.], 0., interpolation="nearest", method="nearest")


class TestArrayFinalizeNone(_DeprecationTestCase):
    # 使用 DeprecationWarning 来标记测试中的弃用警告
    message = "Setting __array_finalize__ = None"

    # 测试用例，检查设置 __array_finalize__ = None 是否会触发弃用警告
    def test_use_none_is_deprecated(self):
        # 定义一个没有 __array_finalize__ 方法的 ndarray 子类
        class NoFinalize(np.ndarray):
            __array_finalize__ = None

        # 检查创建视图时是否会触发弃用警告
        self.assert_deprecated(lambda: np.array(1).view(NoFinalize))


class TestLoadtxtParseIntsViaFloat(_DeprecationTestCase):
    # Deprecated 2022-07-03, NumPy 1.23
    # 本测试在弃用后可以移除而不需要替换。
    # 此处的消息文本指定了一个需要移除的警告过滤器。
    message = r"loadtxt\(\): Parsing an integer via a float is deprecated.*"

    # 使用 pytest 的参数化装饰器，对多种整数数据类型进行测试
    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    # 在单元测试中，测试特定数据类型的 `np.loadtxt` 函数是否会触发 DeprecationWarning
    def test_deprecated_warning(self, dtype):
        # 使用 pytest 的 warn 断言来检查是否会触发 DeprecationWarning，并且匹配特定的警告信息
        with pytest.warns(DeprecationWarning, match=self.message):
            # 调用 np.loadtxt 以特定的数据类型读取文件 ["10.5"]
            np.loadtxt(["10.5"], dtype=dtype)

    # 使用 pytest 的参数化标记，测试所有整数类型的 dtype
    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_deprecated_raised(self, dtype):
        # DeprecationWarning 在抛出时会被链接，因此需要手动测试：
        with warnings.catch_warnings():
            # 设置警告过滤器，将 DeprecationWarning 设置为错误，这样会抛出异常
            warnings.simplefilter("error", DeprecationWarning)
            try:
                # 尝试调用 np.loadtxt 以特定的数据类型读取文件 ["10.5"]
                np.loadtxt(["10.5"], dtype=dtype)
            except ValueError as e:
                # 检查异常是否是 DeprecationWarning 引起的
                assert isinstance(e.__cause__, DeprecationWarning)
class TestScalarConversion(_DeprecationTestCase):
    # 2023-01-02, 1.25.0
    # 此类测试标量类型转换的过时行为

    def test_float_conversion(self):
        # 断言 float 的行为已被弃用，传入参数为 numpy 数组 [3.14]
        self.assert_deprecated(float, args=(np.array([3.14]),))

    def test_behaviour(self):
        # 测试特定行为
        b = np.array([[3.14]])
        c = np.zeros(5)
        with pytest.warns(DeprecationWarning):
            # 通过向 c[0] 赋值 b 的值来触发警告
            c[0] = b

class TestPyIntConversion(_DeprecationTestCase):
    message = r".*stop allowing conversion of out-of-bound.*"
    # 此类测试 Python 整数转换的过时行为，消息模式指示出界转换的停止

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_deprecated_scalar(self, dtype):
        # 测试特定类型的标量行为
        dtype = np.dtype(dtype)
        info = np.iinfo(dtype)

        # Cover the most common creation paths (all end up in the
        # same place):
        # 覆盖最常见的创建路径（都会最终到达相同的地方）
        def scalar(value, dtype):
            # 创建标量值
            dtype.type(value)

        def assign(value, dtype):
            # 分配值给数组元素
            arr = np.array([0, 0, 0], dtype=dtype)
            arr[2] = value

        def create(value, dtype):
            # 创建包含值的数组
            np.array([value], dtype=dtype)

        for creation_func in [scalar, assign, create]:
            try:
                self.assert_deprecated(
                        lambda: creation_func(info.min - 1, dtype))
            except OverflowError:
                pass  # OverflowErrors always happened also before and are OK.

            try:
                self.assert_deprecated(
                        lambda: creation_func(info.max + 1, dtype))
            except OverflowError:
                pass  # OverflowErrors always happened also before and are OK.


@pytest.mark.parametrize("name", ["str", "bytes", "object"])
def test_future_scalar_attributes(name):
    # FutureWarning added 2022-11-17, NumPy 1.24,
    # 测试未来标量属性的警告，这些属性在未来版本中可能会被移除
    assert name not in dir(np)  # we may want to not add them
    with pytest.warns(FutureWarning,
            match=f"In the future .*{name}"):
        assert not hasattr(np, name)

    # Unfortunately, they are currently still valid via `np.dtype()`
    # 不幸的是，它们目前仍然通过 `np.dtype()` 是有效的
    np.dtype(name)
    name in np._core.sctypeDict


# Ignore the above future attribute warning for this test.
# 忽略上述未来属性警告以进行此测试
@pytest.mark.filterwarnings("ignore:In the future:FutureWarning")
class TestRemovedGlobals:
    # Removed 2023-01-12, NumPy 1.24.0
    # Not a deprecation, but the large error was added to aid those who missed
    # the previous deprecation, and should be removed similarly to one
    # (or faster).
    # 此类测试移除的全局变量，该变更不是废弃，但为了帮助错过先前废弃的人们，添加了大的错误信息，并应该类似废弃移除。

    @pytest.mark.parametrize("name",
            ["object", "float", "complex", "str", "int"])
    def test_attributeerror_includes_info(self, name):
        # 测试 AttributeError 是否包含特定信息
        msg = f".*\n`np.{name}` was a deprecated alias for the builtin"
        with pytest.raises(AttributeError, match=msg):
            getattr(np, name)


class TestDeprecatedFinfo(_DeprecationTestCase):
    # Deprecated in NumPy 1.25, 2023-01-16
    # 此类测试 NumPy 1.25 中废弃的 np.finfo

    def test_deprecated_none(self):
        # 断言 np.finfo(None) 的行为已被废弃
        self.assert_deprecated(np.finfo, args=(None,))


class TestMathAlias(_DeprecationTestCase):
    # 此类测试 np.lib.math 的过时别名
    def test_deprecated_np_lib_math(self):
        self.assert_deprecated(lambda: np.lib.math)


class TestLibImports(_DeprecationTestCase):
    # 空类，用于测试库导入的过时行为
    # 在Numpy 1.26.0中弃用，预计在2023年9月不再支持使用
    def test_lib_functions_deprecation_call(self):
        # 导入必要的函数和类
        from numpy.lib._utils_impl import safe_eval
        from numpy.lib._npyio_impl import recfromcsv, recfromtxt
        from numpy.lib._function_base_impl import disp
        from numpy.lib._shape_base_impl import get_array_wrap
        from numpy._core.numerictypes import maximum_sctype
        from numpy.lib.tests.test_io import TextIO
        from numpy import in1d, row_stack, trapz
        
        # 断言函数safe_eval("None")已经被弃用
        self.assert_deprecated(lambda: safe_eval("None"))
        
        # 定义一个生成数据的lambda函数，用于测试recfromcsv函数
        data_gen = lambda: TextIO('A,B\n0,1\n2,3')
        kwargs = dict(delimiter=",", missing_values="N/A", names=True)
        # 断言函数recfromcsv(data_gen())已经被弃用
        self.assert_deprecated(lambda: recfromcsv(data_gen()))
        # 断言函数recfromtxt(data_gen(), **kwargs)已经被弃用
        self.assert_deprecated(lambda: recfromtxt(data_gen(), **kwargs))
        
        # 断言函数disp("test")已经被弃用
        self.assert_deprecated(lambda: disp("test"))
        # 断言函数get_array_wrap()已经被弃用
        self.assert_deprecated(lambda: get_array_wrap())
        # 断言函数maximum_sctype(int)已经被弃用
        self.assert_deprecated(lambda: maximum_sctype(int))
        
        # 断言函数in1d([1], [1])已经被弃用
        self.assert_deprecated(lambda: in1d([1], [1]))
        # 断言函数row_stack([[]])已经被弃用
        self.assert_deprecated(lambda: row_stack([[]]))
        # 断言函数trapz([1], [1])已经被弃用
        self.assert_deprecated(lambda: trapz([1], [1]))
        # 断言np.chararray已经被弃用
        self.assert_deprecated(lambda: np.chararray)
# 创建一个测试类 TestDeprecatedDTypeAliases，继承自 _DeprecationTestCase
class TestDeprecatedDTypeAliases(_DeprecationTestCase):

    # 定义一个辅助方法 _check_for_warning，用于检查是否发出了警告
    def _check_for_warning(self, func):
        # 使用 warnings 模块捕获警告
        with warnings.catch_warnings(record=True) as caught_warnings:
            func()  # 执行传入的函数
        # 断言捕获的警告数量为1
        assert len(caught_warnings) == 1
        # 获取第一个捕获的警告对象
        w = caught_warnings[0]
        # 断言警告类别为 DeprecationWarning
        assert w.category is DeprecationWarning
        # 断言警告消息包含特定文本
        assert "alias 'a' was deprecated in NumPy 2.0" in str(w.message)

    # 定义测试方法 test_a_dtype_alias
    def test_a_dtype_alias(self):
        # 遍历测试的数据类型列表
        for dtype in ["a", "a10"]:
            # 定义匿名函数 f，用于创建指定 dtype 的 numpy 数据类型对象
            f = lambda: np.dtype(dtype)
            # 检查警告
            self._check_for_warning(f)
            # 断言该函数调用已被弃用
            self.assert_deprecated(f)
            # 使用匿名函数 f 创建 numpy 数组，并将其类型转换为指定 dtype
            f = lambda: np.array(["hello", "world"]).astype("a10")
            # 检查警告
            self._check_for_warning(f)
            # 断言该函数调用已被弃用
            self.assert_deprecated(f)



# 创建一个测试类 TestDeprecatedArrayWrap，继承自 _DeprecationTestCase
class TestDeprecatedArrayWrap(_DeprecationTestCase):
    # 定义类变量 message，表示警告消息包含的正则表达式模式
    message = "__array_wrap__.*"

    # 定义测试方法 test_deprecated
    def test_deprecated(self):
        # 定义 Test1 类
        class Test1:
            # 定义 __array__ 方法
            def __array__(self, dtype=None, copy=None):
                return np.arange(4)

            # 定义 __array_wrap__ 方法
            def __array_wrap__(self, arr, context=None):
                self.called = True
                return 'pass context'

        # 定义 Test2 类，继承自 Test1
        class Test2(Test1):
            # 重写 __array_wrap__ 方法
            def __array_wrap__(self, arr):
                self.called = True
                return 'pass'

        # 创建 Test1 的实例 test1
        test1 = Test1()
        # 创建 Test2 的实例 test2
        test2 = Test2()
        # 断言对 test1 执行 np.negative 函数时已被弃用
        self.assert_deprecated(lambda: np.negative(test1))
        # 断言 test1 实例中的 called 属性为 True
        assert test1.called
        # 断言对 test2 执行 np.negative 函数时已被弃用
        self.assert_deprecated(lambda: np.negative(test2))
        # 断言 test2 实例中的 called 属性为 True
        assert test2.called



# 创建一个测试类 TestDeprecatedDTypeParenthesizedRepeatCount，继承自 _DeprecationTestCase
class TestDeprecatedDTypeParenthesizedRepeatCount(_DeprecationTestCase):
    # 定义类变量 message，表示警告消息包含的文本
    message = "Passing in a parenthesized single number"

    # 使用 pytest 的参数化装饰器标记测试方法 test_parenthesized_repeat_count
    @pytest.mark.parametrize("string", ["(2)i,", "(3)3S,", "f,(2)f"])
    def test_parenthesized_repeat_count(self, string):
        # 断言调用 np.dtype 函数时已被弃用，传入参数 string
        self.assert_deprecated(np.dtype, args=(string,))



# 创建一个测试类 TestDeprecatedSaveFixImports，继承自 _DeprecationTestCase
class TestDeprecatedSaveFixImports(_DeprecationTestCase):
    # 类变量 message 指示警告消息内容，指出 'fix_imports' 标志已弃用
    message = "The 'fix_imports' flag is deprecated and has no effect."

    # 定义测试方法 test_deprecated
    def test_deprecated(self):
        # 使用 temppath 上下文管理器创建临时路径，文件后缀为 .npy
        with temppath(suffix='.npy') as path:
            # 准备样本参数
            sample_args = (path, np.array(np.zeros((1024, 10))))
            # 断言对 np.save 函数的调用未被弃用，传入样本参数
            self.assert_not_deprecated(np.save, args=sample_args)
            # 断言对 np.save 函数的调用已被弃用，传入样本参数和 fix_imports=True 的关键字参数
            self.assert_deprecated(np.save, args=sample_args,
                                kwargs={'fix_imports': True})
            # 断言对 np.save 函数的调用已被弃用，传入样本参数和 fix_imports=False 的关键字参数
            self.assert_deprecated(np.save, args=sample_args,
                                kwargs={'fix_imports': False})
            # 遍历 allow_pickle 的取值 [True, False]
            for allow_pickle in [True, False]:
                # 断言对 np.save 函数的调用未被弃用，传入样本参数和 allow_pickle 的关键字参数
                self.assert_not_deprecated(np.save, args=sample_args,
                                        kwargs={'allow_pickle': allow_pickle})
                # 断言对 np.save 函数的调用已被弃用，传入样本参数、allow_pickle 和 fix_imports=True 的关键字参数
                self.assert_deprecated(np.save, args=sample_args,
                                    kwargs={'allow_pickle': allow_pickle,
                                            'fix_imports': True})
                # 断言对 np.save 函数的调用已被弃用，传入样本参数、allow_pickle 和 fix_imports=False 的关键字参数
                self.assert_deprecated(np.save, args=sample_args,
                                    kwargs={'allow_pickle': allow_pickle,
                                            'fix_imports': False})
```