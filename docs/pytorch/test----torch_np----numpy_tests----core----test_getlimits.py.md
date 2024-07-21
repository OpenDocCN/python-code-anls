# `.\pytorch\test\torch_np\numpy_tests\core\test_getlimits.py`

```py
# Owner(s): ["module: dynamo"]

""" Test functions for limits module.

"""
import functools  # 导入 functools 模块，用于创建偏函数
import warnings  # 导入 warnings 模块，用于处理警告

# from numpy.core.getlimits import _discovered_machar, _float_ma

from unittest import expectedFailure as xfail, skipIf  # 从 unittest 模块导入 expectedFailure 别名为 xfail，以及 skipIf 函数

import numpy  # 导入 numpy 库

from pytest import raises as assert_raises  # 从 pytest 模块导入 raises 函数别名为 assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入测试实例化参数化的函数
    parametrize,  # 导入参数化装饰器
    run_tests,  # 导入运行测试函数
    subtest,  # 导入子测试函数
    TEST_WITH_TORCHDYNAMO,  # 导入测试是否使用 torchdynamo 的标记
    TestCase,  # 导入测试用例类
    xpassIfTorchDynamo,  # 导入如果使用 torchdynamo 则跳过测试的装饰器
)

if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 如果 TEST_WITH_TORCHDYNAMO 为真，则导入 numpy 库并命名为 np
    from numpy import double, finfo, half, iinfo, single  # 导入 numpy 库中的数据类型和函数
    from numpy.testing import assert_, assert_equal  # 导入 numpy.testing 模块中的断言函数
else:
    import torch._numpy as np  # 如果 TEST_WITH_TORCHDYNAMO 为假，则导入 torch._numpy 库并命名为 np
    from torch._numpy import double, finfo, half, iinfo, single  # 导入 torch._numpy 库中的数据类型和函数
    from torch._numpy.testing import assert_, assert_equal  # 导入 torch._numpy.testing 模块中的断言函数


skip = functools.partial(skipIf, True)  # 创建一个跳过测试的偏函数，总是返回 True


##################################################


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestPythonFloat(TestCase):  # 定义测试类 TestPythonFloat，继承自 TestCase
    def test_singleton(self):  # 定义测试单例的方法
        ftype = finfo(float)  # 获取 float 类型的 finfo 对象
        ftype2 = finfo(float)  # 再次获取 float 类型的 finfo 对象
        assert_equal(id(ftype), id(ftype2))  # 断言两个 finfo 对象的 id 相同


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestHalf(TestCase):  # 定义测试类 TestHalf，继承自 TestCase
    def test_singleton(self):  # 定义测试单例的方法
        ftype = finfo(half)  # 获取 half 类型的 finfo 对象
        ftype2 = finfo(half)  # 再次获取 half 类型的 finfo 对象
        assert_equal(id(ftype), id(ftype2))  # 断言两个 finfo 对象的 id 相同


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestSingle(TestCase):  # 定义测试类 TestSingle，继承自 TestCase
    def test_singleton(self):  # 定义测试单例的方法
        ftype = finfo(single)  # 获取 single 类型的 finfo 对象
        ftype2 = finfo(single)  # 再次获取 single 类型的 finfo 对象
        assert_equal(id(ftype), id(ftype2))  # 断言两个 finfo 对象的 id 相同


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestDouble(TestCase):  # 定义测试类 TestDouble，继承自 TestCase
    def test_singleton(self):  # 定义测试单例的方法
        ftype = finfo(double)  # 获取 double 类型的 finfo 对象
        ftype2 = finfo(double)  # 再次获取 double 类型的 finfo 对象
        assert_equal(id(ftype), id(ftype2))  # 断言两个 finfo 对象的 id 相同


class TestFinfo(TestCase):  # 定义测试类 TestFinfo，继承自 TestCase
    @skipIf(numpy.__version__ < "1.23", reason=".smallest_normal is new")
    def test_basic(self):  # 定义基础测试方法
        dts = list(  # 创建数据类型列表
            zip(
                ["f2", "f4", "f8", "c8", "c16"],  # 浮点数和复数数据类型的列表
                [np.float16, np.float32, np.float64, np.complex64, np.complex128],  # 对应的 numpy 数据类型
            )
        )
        for dt1, dt2 in dts:  # 遍历数据类型列表
            for attr in (  # 遍历属性列表
                "bits",
                "eps",
                "max",
                "min",
                "resolution",
                "tiny",
                "smallest_normal",
            ):
                assert_equal(  # 断言两种数据类型的相同属性值
                    getattr(finfo(dt1), attr), getattr(finfo(dt2), attr), attr
                )
        with assert_raises((TypeError, ValueError)):  # 断言引发 TypeError 或 ValueError 异常
            finfo("i4")

    @skip  # (reason="Some of these attributes are not implemented vs NP versions")
    # 定义一个测试函数，用于检查特定数据类型的属性是否存在
    def test_basic_missing(self):
        # 设定数据类型为 np.float32
        dt = np.float32
        # 遍历属性列表，逐个检查数据类型的属性是否存在
        for attr in [
            "epsneg",                 # 最小负数，即最大负数的绝对值
            "iexp",                   # 指数的位数
            "machep",                 # 机器精度，满足 |1 - eps| > 0 的最小正数 eps
            "maxexp",                 # 指数的最大可能值
            "minexp",                 # 指数的最小可能值
            "negep",                  # 负的机器精度，满足 |1 - eps| > 0 的最小负数 eps
            "nexp",                   # 负指数的位数
            "nmant",                  # 尾数的位数
            "precision",              # 数字的有效位数
            "smallest_subnormal",     # 最小非规范化数
        ]:
            # 获取当前属性名对应的属性值，如果属性不存在会引发 AttributeError 异常
            getattr(finfo(dt), attr)
@instantiate_parametrized_tests
class TestIinfo(TestCase):
    # 定义测试类 TestIinfo，用于测试 np.iinfo 相关功能

    def test_basic(self):
        # 定义基本测试方法 test_basic
        dts = list(
            zip(
                ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"],
                [
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                ],
            )
        )
        # 创建包含数据类型名称和对应 numpy 类型的元组列表 dts
        for dt1, dt2 in dts:
            # 遍历 dts 中的每对数据类型名称和 numpy 类型
            for attr in ("bits", "min", "max"):
                # 遍历属性名列表，断言两个 iinfo 对象的对应属性值相等
                assert_equal(getattr(iinfo(dt1), attr), getattr(iinfo(dt2), attr), attr)
        # 断言对于 "f4" 类型，调用 iinfo 会引发 TypeError 或 ValueError 异常
        with assert_raises((TypeError, ValueError)):
            iinfo("f4")

    @parametrize(
        "T",
        [
            np.uint8,
            # xfail: unsupported add (uint[16,32,64])
            subtest(np.uint16, decorators=[xfail]),
            subtest(np.uint32, decorators=[xfail]),
            subtest(np.uint64, decorators=[xfail]),
        ],
    )
    def test_unsigned_max(self, T):
        # 定义测试无符号整数类型的最大值计算方法 test_unsigned_max
        max_calculated = T(0) - T(1)
        # 计算给定类型 T 的最大值
        assert_equal(iinfo(T).max, max_calculated)


class TestRepr(TestCase):
    # 定义测试类 TestRepr，用于测试 np.iinfo 和 np.finfo 的字符串表示

    def test_iinfo_repr(self):
        # 定义测试方法 test_iinfo_repr
        expected = "iinfo(min=-32768, max=32767, dtype=int16)"
        # 预期的 np.iinfo(np.int16) 的字符串表示
        assert_equal(repr(np.iinfo(np.int16)), expected)

    @skipIf(TEST_WITH_TORCHDYNAMO, reason="repr differs")
    def test_finfo_repr(self):
        # 定义测试方法 test_finfo_repr，测试 np.finfo 的字符串表示
        repr_f32 = repr(np.finfo(np.float32))
        # 获取 np.float32 的字符串表示
        assert "finfo(resolution=1e-06, min=-3.40282e+38," in repr_f32
        assert "dtype=float32" in repr_f32


def assert_ma_equal(discovered, ma_like):
    # 定义断言函数 assert_ma_equal，用于比较 MachAr 类型对象和计算出的 MachAr 实例
    # 检查 MachAr 类型对象与计算得到的 MachAr 实例的每个属性值是否相等
    for key, value in discovered.__dict__.items():
        assert_equal(value, getattr(ma_like, key))
        if hasattr(value, "shape"):
            assert_equal(value.shape, getattr(ma_like, key).shape)
            assert_equal(value.dtype, getattr(ma_like, key).dtype)


class TestMisc(TestCase):
    # 定义测试类 TestMisc，包含各种杂项测试方法

    @skip(reason="Instantiate {i,f}info from dtypes.")
    def test_instances(self):
        # 定义测试方法 test_instances，测试实例化 iinfo 和 finfo 类型
        iinfo(10)
        finfo(3.0)

    @skip(reason="MachAr no implemented (does it need to)?")
    def test_known_types(self):
        # 定义测试方法 test_known_types，测试已知类型的参数编译情况
        # 对于给定的浮点类型 ftype 和对应的 MachAr 类型 ma_like
        for ftype, ma_like in (
            (np.float16, _float_ma[16]),
            (np.float32, _float_ma[32]),
            (np.float64, _float_ma[64]),
        ):
            # 断言通过 _discovered_machar 获取的 MachAr 对象与预定义的 ma_like 相等
            assert_ma_equal(_discovered_machar(ftype), ma_like)
        # 在 PPC 平台上，针对双精度浮点数的特殊处理
        ld_ma = _discovered_machar(np.longdouble)
        bytes = np.dtype(np.longdouble).itemsize
        if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
            # 80 位扩展精度浮点数的 MachAr 对象
            assert_ma_equal(ld_ma, _float_ma[80])
        elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
            # IEE 754 128 位双精度浮点数的 MachAr 对象
            assert_ma_equal(ld_ma, _float_ma[128])
    def test_subnormal_warning(self):
        """Test that the subnormal is zero warning is not being raised."""
        # 获取 np.longdouble 类型的机器精度信息
        ld_ma = _discovered_machar(np.longdouble)
        # 获取 np.longdouble 类型的字节大小
        bytes = np.dtype(np.longdouble).itemsize
        # 设置警告捕获，确保警告被记录
        with warnings.catch_warnings(record=True) as w:
            # 设置警告过滤器，始终显示警告
            warnings.simplefilter("always")
            # 根据机器精度和字节大小进行条件判断
            if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
                # 对于 80 位扩展精度
                ld_ma.smallest_subnormal
                # 断言没有警告被触发
                assert len(w) == 0
            elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
                # 对于 IEEE 754 128 位
                ld_ma.smallest_subnormal
                # 断言没有警告被触发
                assert len(w) == 0
            else:
                # 对于双倍精度
                ld_ma.smallest_subnormal
                # 可能在某些平台上失败的测试
                assert len(w) == 0

    @xpassIfTorchDynamo  # (reason="None of nmant, minexp, maxexp is implemented.")
    def test_plausible_finfo(self):
        # 断言 finfo 对所有类型返回合理结果
        for ftype in np.sctypes["float"] + np.sctypes["complex"]:
            # 获取指定类型的 finfo 信息
            info = np.finfo(ftype)
            # 断言 nmant 大于 1
            assert_(info.nmant > 1)
            # 断言 minexp 小于 -1
            assert_(info.minexp < -1)
            # 断言 maxexp 大于 1
            assert_(info.maxexp > 1)
# 如果当前模块作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```