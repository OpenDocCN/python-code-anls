# `.\numpy\numpy\lib\tests\test_nanfunctions.py`

```py
# 引入警告模块，用于管理和控制警告信息的显示
import warnings
# 引入 pytest 模块，用于编写和运行测试用例
import pytest
# 引入 inspect 模块，提供了对 Python 对象内部结构的访问
import inspect
# 从 functools 模块中引入 partial 函数，用于部分应用函数
from functools import partial

# 引入 numpy 库，用于科学计算
import numpy as np
# 引入 normalize_axis_tuple 函数，用于规范化轴元组
from numpy._core.numeric import normalize_axis_tuple
# 引入 AxisError 和 ComplexWarning 异常类，用于处理轴错误和复数警告
from numpy.exceptions import AxisError, ComplexWarning
# 引入 _nan_mask 和 _replace_nan 函数，用于处理 NaN 值的掩码和替换
from numpy.lib._nanfunctions_impl import _nan_mask, _replace_nan
# 引入 numpy.testing 模块中的测试函数，用于进行各种断言和测试
from numpy.testing import (
    assert_, assert_equal, assert_almost_equal, assert_raises,
    assert_raises_regex, assert_array_equal, suppress_warnings
    )

# 测试数据，包含 NaN 值
_ndat = np.array([[0.6244, np.nan, 0.2692, 0.0116, np.nan, 0.1170],
                  [0.5351, -0.9403, np.nan, 0.2100, 0.4759, 0.2833],
                  [np.nan, np.nan, np.nan, 0.1042, np.nan, -0.5954],
                  [0.1610, np.nan, np.nan, 0.1859, 0.3146, np.nan]])

# 移除 NaN 值后的数据行
_rdat = [np.array([0.6244, 0.2692, 0.0116, 0.1170]),
         np.array([0.5351, -0.9403, 0.2100, 0.4759, 0.2833]),
         np.array([0.1042, -0.5954]),
         np.array([0.1610, 0.1859, 0.3146])]

# 将 NaN 值替换为 1.0 后的数据
_ndat_ones = np.array([[0.6244, 1.0, 0.2692, 0.0116, 1.0, 0.1170],
                       [0.5351, -0.9403, 1.0, 0.2100, 0.4759, 0.2833],
                       [1.0, 1.0, 1.0, 0.1042, 1.0, -0.5954],
                       [0.1610, 1.0, 1.0, 0.1859, 0.3146, 1.0]])

# 将 NaN 值替换为 0.0 后的数据
_ndat_zeros = np.array([[0.6244, 0.0, 0.2692, 0.0116, 0.0, 0.1170],
                        [0.5351, -0.9403, 0.0, 0.2100, 0.4759, 0.2833],
                        [0.0, 0.0, 0.0, 0.1042, 0.0, -0.5954],
                        [0.1610, 0.0, 0.0, 0.1859, 0.3146, 0.0]])


class TestSignatureMatch:
    # 定义一个字典，将 numpy 中处理 NaN 的函数映射到其对应的非 NaN 版本
    NANFUNCS = {
        np.nanmin: np.amin,
        np.nanmax: np.amax,
        np.nanargmin: np.argmin,
        np.nanargmax: np.argmax,
        np.nansum: np.sum,
        np.nanprod: np.prod,
        np.nancumsum: np.cumsum,
        np.nancumprod: np.cumprod,
        np.nanmean: np.mean,
        np.nanmedian: np.median,
        np.nanpercentile: np.percentile,
        np.nanquantile: np.quantile,
        np.nanvar: np.var,
        np.nanstd: np.std,
    }
    # 使用函数名作为参数化测试的标识
    IDS = [k.__name__ for k in NANFUNCS]

    @staticmethod
    def get_signature(func, default="..."):
        """构造函数签名并替换所有默认参数值。"""
        # 初始化参数列表
        prm_list = []
        # 获取函数的签名信息
        signature = inspect.signature(func)
        # 遍历签名中的每个参数
        for prm in signature.parameters.values():
            # 如果参数没有默认值，则直接添加到参数列表中
            if prm.default is inspect.Parameter.empty:
                prm_list.append(prm)
            else:
                # 否则，用指定的默认值替换参数的默认值
                prm_list.append(prm.replace(default=default))
        # 返回替换后的函数签名对象
        return inspect.Signature(prm_list)

    # 使用 pytest 的参数化装饰器，传入处理 NaN 的函数和对应的非 NaN 函数
    @pytest.mark.parametrize("nan_func,func", NANFUNCS.items(), ids=IDS)
    # 测试函数签名是否匹配的方法
    def test_signature_match(self, nan_func, func):
        # 忽略默认参数值，因为它们有时可能不同
        # 一个函数可能为 `False`，而另一个可能为 `np._NoValue`
        signature = self.get_signature(func)
        nan_signature = self.get_signature(nan_func)
        # 使用 NumPy 的测试工具检查两个函数的签名是否相等
        np.testing.assert_equal(signature, nan_signature)

    # 测试方法，验证所有的 NaN 函数是否都被测试到
    def test_exhaustiveness(self):
        """Validate that all nan functions are actually tested."""
        # 使用 NumPy 的测试工具，比较已测试的函数集合和 NumPy 内部所有 NaN 函数的集合
        np.testing.assert_equal(
            set(self.IDS), set(np.lib._nanfunctions_impl.__all__)
        )
# 定义一个测试类 TestNanFunctions_MinMax，用于测试处理 NaN 值的函数 np.nanmin 和 np.nanmax
class TestNanFunctions_MinMax:

    # 初始化类变量，包含处理 NaN 的函数列表和标准函数列表
    nanfuncs = [np.nanmin, np.nanmax]
    stdfuncs = [np.min, np.max]

    # 测试数组是否被修改的方法
    def test_mutation(self):
        # 复制原始数组 _ndat 到 ndat，确保不修改原始数据
        ndat = _ndat.copy()
        # 对 nanfuncs 中的每个函数 f，应用于 ndat
        for f in self.nanfuncs:
            f(ndat)
            # 断言 ndat 未被修改
            assert_equal(ndat, _ndat)

    # 测试 keepdims 参数的方法
    def test_keepdims(self):
        # 创建一个3x3的单位矩阵 mat
        mat = np.eye(3)
        # 对 nanfuncs 和 stdfuncs 中的每一对函数 nf 和 rf 进行迭代
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            # 对于每个可能的轴 axis：None, 0, 1
            for axis in [None, 0, 1]:
                # 计算使用 rf 函数在 mat 上的结果 tgt，并保持维度不变
                tgt = rf(mat, axis=axis, keepdims=True)
                # 计算使用 nf 函数在 mat 上的结果 res，并保持维度不变
                res = nf(mat, axis=axis, keepdims=True)
                # 断言 res 的维度与 tgt 的维度相同
                assert_(res.ndim == tgt.ndim)

    # 测试 out 参数的方法
    def test_out(self):
        # 创建一个3x3的单位矩阵 mat
        mat = np.eye(3)
        # 对 nanfuncs 和 stdfuncs 中的每一对函数 nf 和 rf 进行迭代
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            # 创建一个用于存储输出结果的 resout 数组
            resout = np.zeros(3)
            # 使用 rf 函数计算 mat 的结果 tgt，仅在 axis=1 时
            tgt = rf(mat, axis=1)
            # 使用 nf 函数计算 mat 的结果 res，将结果存储在 resout 中
            res = nf(mat, axis=1, out=resout)
            # 断言 res 与 resout 的值接近
            assert_almost_equal(res, resout)
            # 断言 res 与 tgt 的值接近
            assert_almost_equal(res, tgt)

    # 测试根据输入的 dtype 类型来确定输出的 dtype 的方法
    def test_dtype_from_input(self):
        # 定义一组 dtype 代码
        codes = 'efdgFDG'
        # 对 nanfuncs 和 stdfuncs 中的每一对函数 nf 和 rf 进行迭代
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            # 对于每个 dtype 代码 c
            for c in codes:
                # 创建一个 dtype 为 c 的3x3单位矩阵 mat
                mat = np.eye(3, dtype=c)
                # 使用 rf 函数计算 mat 的结果 tgt，仅在 axis=1 时，并确定其 dtype 类型
                tgt = rf(mat, axis=1).dtype.type
                # 使用 nf 函数计算 mat 的结果 res，并确定其 dtype 类型
                res = nf(mat, axis=1).dtype.type
                # 断言 res 的 dtype 类型与 tgt 的 dtype 类型相同
                assert_(res is tgt)
                # 在标量情况下进行断言
                tgt = rf(mat, axis=None).dtype.type
                res = nf(mat, axis=None).dtype.type
                assert_(res is tgt)

    # 测试函数返回值的方法
    def test_result_values(self):
        # 对 nanfuncs 和 stdfuncs 中的每一对函数 nf 和 rf 进行迭代
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            # 计算标准函数 rf 在 _rdat 中每个数组 d 上的结果列表 tgt
            tgt = [rf(d) for d in _rdat]
            # 使用 nf 函数计算 _ndat 在 axis=1 上的结果 res
            res = nf(_ndat, axis=1)
            # 断言 res 与 tgt 的值接近
            assert_almost_equal(res, tgt)

    # 使用 pytest.mark.parametrize 标记测试用例的方法，测试处理全为 NaN 的数组情况
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),  # 0维数组情况
        np.full((3, 3), np.nan),  # 2维数组情况
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        # 如果 axis 不为 None 且 array 的维度为 0，则跳过该测试用例
        if axis is not None and array.ndim == 0:
            pytest.skip(f"`axis != None` not supported for 0d arrays")

        # 将 array 转换为指定的 dtype 类型
        array = array.astype(dtype)
        # 匹配字符串 "All-NaN slice encountered"
        match = "All-NaN slice encountered"
        # 对 nanfuncs 中的每个函数 func 进行迭代
        for func in self.nanfuncs:
            # 使用 pytest.warns 检查是否会发出 RuntimeWarning 警告，匹配警告信息为 match
            with pytest.warns(RuntimeWarning, match=match):
                # 使用 func 函数计算 array 在指定 axis 上的结果 out
                out = func(array, axis=axis)
            # 断言 out 中所有的值都为 NaN
            assert np.isnan(out).all()
            # 断言 out 的 dtype 类型与 array 的 dtype 类型相同
            assert out.dtype == array.dtype

    # 测试处理带掩码的数组的方法
    def test_masked(self):
        # 创建一个包含无效值修正的 _ndat 的掩码数组 mat
        mat = np.ma.fix_invalid(_ndat)
        # 复制 mat 的掩码到 msk
        msk = mat._mask.copy()
        # 对于函数 f 中的每个函数 f，仅使用 np.nanmin
        for f in [np.nanmin]:
            # 使用 f 函数计算 mat 在 axis=1 上的结果 res
            res = f(mat, axis=1)
            # 使用 f 函数计算 _ndat 在 axis=1 上的结果 tgt
            tgt = f(_ndat, axis=1)
            # 断言 res 等于 tgt
            assert_equal(res, tgt)
            # 断言 mat 的掩码与 msk 相同
            assert_equal(mat._mask, msk)
            # 断言 mat 中不包含任何无穷值
            assert_(not np.isinf(mat).any())

    # 测试处理标量输入的方法
    def test_scalar(self):
        # 对 nanfuncs 中的每个函数 f 进行迭代
        for f in self.nanfuncs:
            # 断言 f 函数在输入为标量 0.0 时的结果为 0.0
            assert_(f(0.) == 0.)
    def test_subclass(self):
        # 定义一个自定义的 ndarray 子类 MyNDArray
        class MyNDArray(np.ndarray):
            pass

        # 创建一个 3x3 的单位矩阵，并将其视图转换为 MyNDArray 类型
        mine = np.eye(3).view(MyNDArray)

        # 对每个在 self.nanfuncs 中的函数进行测试
        for f in self.nanfuncs:
            # 测试沿 axis=0 方向的函数调用结果
            res = f(mine, axis=0)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == (3,))

            # 测试沿 axis=1 方向的函数调用结果
            res = f(mine, axis=1)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == (3,))

            # 测试没有指定 axis 的函数调用结果
            res = f(mine)
            assert_(res.shape == ())

        # 对包含 NaN 的行进行处理的测试 (#4628)
        mine[1] = np.nan
        for f in self.nanfuncs:
            # 捕获可能的警告信息
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')

                # 测试沿 axis=0 方向处理 NaN 行的结果
                res = f(mine, axis=0)
                assert_(isinstance(res, MyNDArray))
                assert_(not np.any(np.isnan(res)))
                assert_(len(w) == 0)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')

                # 测试沿 axis=1 方向处理 NaN 行的结果
                res = f(mine, axis=1)
                assert_(isinstance(res, MyNDArray))
                assert_(np.isnan(res[1]) and not np.isnan(res[0])
                        and not np.isnan(res[2]))
                assert_(len(w) == 1, 'no warning raised')
                assert_(issubclass(w[0].category, RuntimeWarning))

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')

                # 测试没有指定 axis 处理 NaN 的结果
                res = f(mine)
                assert_(res.shape == ())
                assert_(res != np.nan)
                assert_(len(w) == 0)

    def test_object_array(self):
        # 创建一个包含 NaN 的对象数组
        arr = np.array([[1.0, 2.0], [np.nan, 4.0], [np.nan, np.nan]], dtype=object)

        # 测试 np.nanmin 在对象数组上的表现
        assert_equal(np.nanmin(arr), 1.0)
        assert_equal(np.nanmin(arr, axis=0), [1.0, 2.0])

        # 测试对对象数组使用 np.nanmin 时的警告情况
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            # 对比 np.nanmin 在 axis=1 上的结果
            # 注意：assert_equal 在处理对象数组的 NaN 时不适用
            assert_equal(list(np.nanmin(arr, axis=1)), [1.0, 4.0, np.nan])
            assert_(len(w) == 1, 'no warning raised')
            assert_(issubclass(w[0].category, RuntimeWarning))

    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_initial(self, dtype):
        # 定义一个自定义的 ndarray 子类 MyNDArray
        class MyNDArray(np.ndarray):
            pass

        # 创建一个浮点类型的数组，并将前五个元素设为 NaN
        ar = np.arange(9).astype(dtype)
        ar[:5] = np.nan

        for f in self.nanfuncs:
            initial = 100 if f is np.nanmax else 0

            # 测试带有 initial 参数的函数调用结果
            ret1 = f(ar, initial=initial)
            assert ret1.dtype == dtype
            assert ret1 == initial

            # 测试对 MyNDArray 类型的视图进行函数调用的结果
            ret2 = f(ar.view(MyNDArray), initial=initial)
            assert ret2.dtype == dtype
            assert ret2 == initial
    # 定义一个测试方法，用于测试特定数据类型的函数
    def test_where(self, dtype):
        # 定义一个继承自 numpy.ndarray 的子类 MyNDArray
        class MyNDArray(np.ndarray):
            pass

        # 创建一个3x3的数组，元素为0到8，转换为指定的数据类型并设置第一行为 NaN
        ar = np.arange(9).reshape(3, 3).astype(dtype)
        ar[0, :] = np.nan

        # 创建一个与 ar 形状相同的全为 True 的布尔数组 where，并将第一列设为 False
        where = np.ones_like(ar, dtype=np.bool)
        where[:, 0] = False

        # 遍历 nanfuncs 中的每一个函数 f
        for f in self.nanfuncs:
            # 如果 f 是 np.nanmin，则 reference 为 4；否则为 8
            reference = 4 if f is np.nanmin else 8

            # 使用函数 f 计算 ar 数组中符合条件的最小值或最大值，初始值为 5
            ret1 = f(ar, where=where, initial=5)
            # 断言返回值的数据类型与指定的 dtype 相同
            assert ret1.dtype == dtype
            # 断言返回值等于预期的 reference 值
            assert ret1 == reference

            # 使用函数 f 计算 ar 数组（视图）中符合条件的最小值或最大值，初始值为 5
            ret2 = f(ar.view(MyNDArray), where=where, initial=5)
            # 断言返回值的数据类型与指定的 dtype 相同
            assert ret2.dtype == dtype
            # 断言返回值等于预期的 reference 值
            assert ret2 == reference
class TestNanFunctions_ArgminArgmax:
    # 定义一个测试类，用于测试 np.nanargmin 和 np.nanargmax 函数
    nanfuncs = [np.nanargmin, np.nanargmax]

    def test_mutation(self):
        # 检查传入的数组不会被修改
        ndat = _ndat.copy()
        for f in self.nanfuncs:
            f(ndat)
            assert_equal(ndat, _ndat)

    def test_result_values(self):
        for f, fcmp in zip(self.nanfuncs, [np.greater, np.less]):
            for row in _ndat:
                with suppress_warnings() as sup:
                    sup.filter(RuntimeWarning, "invalid value encountered in")
                    ind = f(row)
                    val = row[ind]
                    # 比较 NaN 可能有些棘手，因为结果总是 False，除了 NaN != NaN
                    assert_(not np.isnan(val))
                    assert_(not fcmp(val, row).any())
                    assert_(not np.equal(val, row[:ind]).any())

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),
        np.full((3, 3), np.nan),
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip(f"`axis != None` not supported for 0d arrays")

        array = array.astype(dtype)
        for func in self.nanfuncs:
            with pytest.raises(ValueError, match="All-NaN slice encountered"):
                func(array, axis=axis)

    def test_empty(self):
        mat = np.zeros((0, 3))
        for f in self.nanfuncs:
            for axis in [0, None]:
                assert_raises_regex(
                        ValueError,
                        "attempt to get argm.. of an empty sequence",
                        f, mat, axis=axis)
            for axis in [1]:
                res = f(mat, axis=axis)
                assert_equal(res, np.zeros(0))

    def test_scalar(self):
        for f in self.nanfuncs:
            assert_(f(0.) == 0.)

    def test_subclass(self):
        class MyNDArray(np.ndarray):
            pass

        # 检查函数能正常工作，并且类型和形状得到保留
        mine = np.eye(3).view(MyNDArray)
        for f in self.nanfuncs:
            res = f(mine, axis=0)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == (3,))
            res = f(mine, axis=1)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == (3,))
            res = f(mine)
            assert_(res.shape == ())

    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_keepdims(self, dtype):
        ar = np.arange(9).astype(dtype)
        ar[:5] = np.nan

        for f in self.nanfuncs:
            reference = 5 if f is np.nanargmin else 8
            ret = f(ar, keepdims=True)
            assert ret.ndim == ar.ndim
            assert ret == reference

    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    # 定义一个测试方法，用于测试NaN处理函数的行为
    def test_out(self, dtype):
        # 创建一个包含0到8的数组，并将其转换为指定数据类型(dtype)
        ar = np.arange(9).astype(dtype)
        # 将数组的前5个元素设置为NaN
        ar[:5] = np.nan
    
        # 遍历NaN处理函数列表
        for f in self.nanfuncs:
            # 创建一个dtype为np.intp的零维数组out，用于接收函数的输出
            out = np.zeros((), dtype=np.intp)
            # 根据函数类型设置参考值，如果是np.nanargmin，则参考值为5，否则为8
            reference = 5 if f is np.nanargmin else 8
            # 调用NaN处理函数f，将ar作为输入，将结果存入out
            ret = f(ar, out=out)
            # 断言返回值ret与out相同
            assert ret is out
            # 断言返回值ret与参考值reference相同
            assert ret == reference
# 定义测试用例中使用的示例数组集合
_TEST_ARRAYS = {
    "0d": np.array(5),                             # 创建一个0维的NumPy数组，包含单个整数5
    "1d": np.array([127, 39, 93, 87, 46])           # 创建一个1维的NumPy数组，包含多个整数
}

# 设置所有数组为不可写以确保测试不会修改它们
for _v in _TEST_ARRAYS.values():
    _v.setflags(write=False)

# 使用pytest的参数化标记定义多个测试参数
@pytest.mark.parametrize(
    "dtype",                                         # 参数名为dtype，用于测试不同的数据类型
    np.typecodes["AllInteger"] + np.typecodes["AllFloat"] + "O",  # 测试所有整数、浮点数和对象类型
)
@pytest.mark.parametrize(
    "mat", _TEST_ARRAYS.values(),                    # 参数名为mat，用于测试_TEST_ARRAYS中的所有数组
    ids=_TEST_ARRAYS.keys()                          # 用_TEST_ARRAYS中的键作为数组的标识符
)
class TestNanFunctions_NumberTypes:
    # 定义NaN相关函数与其对应的标准函数的映射关系
    nanfuncs = {
        np.nanmin: np.min,                           # NaN最小值与最小值函数的映射关系
        np.nanmax: np.max,                           # NaN最大值与最大值函数的映射关系
        np.nanargmin: np.argmin,                     # NaN最小值位置与最小值位置函数的映射关系
        np.nanargmax: np.argmax,                     # NaN最大值位置与最大值位置函数的映射关系
        np.nansum: np.sum,                           # NaN求和与求和函数的映射关系
        np.nanprod: np.prod,                         # NaN累积乘积与累积乘积函数的映射关系
        np.nancumsum: np.cumsum,                     # NaN累积求和与累积求和函数的映射关系
        np.nancumprod: np.cumprod,                   # NaN累积乘积与累积乘积函数的映射关系
        np.nanmean: np.mean,                         # NaN均值与均值函数的映射关系
        np.nanmedian: np.median,                     # NaN中位数与中位数函数的映射关系
        np.nanvar: np.var,                           # NaN方差与方差函数的映射关系
        np.nanstd: np.std                            # NaN标准差与标准差函数的映射关系
    }
    nanfunc_ids = [i.__name__ for i in nanfuncs]      # 提取函数名用于测试标识符的参数化

    # 使用参数化标记定义测试函数，测试NaN函数与其对应的标准函数
    @pytest.mark.parametrize("nanfunc,func", nanfuncs.items(), ids=nanfunc_ids)
    @np.errstate(over="ignore")
    def test_nanfunc(self, mat, dtype, nanfunc, func):
        mat = mat.astype(dtype)                       # 将mat数组转换为指定的数据类型
        tgt = func(mat)                               # 计算标准函数的结果
        out = nanfunc(mat)                            # 计算NaN函数的结果

        assert_almost_equal(out, tgt)                 # 断言NaN函数的结果与标准函数的结果几乎相等
        if dtype == "O":
            assert type(out) is type(tgt)             # 如果数据类型为对象类型，断言NaN函数与标准函数的结果类型相同
        else:
            assert out.dtype == tgt.dtype             # 否则，断言NaN函数的结果与标准函数的结果的数据类型相同

    # 使用参数化标记定义测试函数，测试NaN分位数和百分位数函数
    @pytest.mark.parametrize(
        "nanfunc,func",
        [(np.nanquantile, np.quantile), (np.nanpercentile, np.percentile)],
        ids=["nanquantile", "nanpercentile"],
    )
    def test_nanfunc_q(self, mat, dtype, nanfunc, func):
        mat = mat.astype(dtype)                       # 将mat数组转换为指定的数据类型
        if mat.dtype.kind == "c":
            assert_raises(TypeError, func, mat, q=1)  # 复数数组不支持分位数和百分位数计算，断言引发TypeError
            assert_raises(TypeError, nanfunc, mat, q=1)

        else:
            tgt = func(mat, q=1)                      # 计算标准分位数或百分位数的结果
            out = nanfunc(mat, q=1)                   # 计算NaN分位数或百分位数的结果

            assert_almost_equal(out, tgt)             # 断言NaN函数的结果与标准函数的结果几乎相等

            if dtype == "O":
                assert type(out) is type(tgt)         # 如果数据类型为对象类型，断言NaN函数与标准函数的结果类型相同
            else:
                assert out.dtype == tgt.dtype         # 否则，断言NaN函数的结果与标准函数的结果的数据类型相同

    # 使用参数化标记定义测试函数，测试NaN方差和标准差函数的ddof参数
    @pytest.mark.parametrize(
        "nanfunc,func",
        [(np.nanvar, np.var), (np.nanstd, np.std)],
        ids=["nanvar", "nanstd"],
    )
    def test_nanfunc_ddof(self, mat, dtype, nanfunc, func):
        mat = mat.astype(dtype)                       # 将mat数组转换为指定的数据类型
        tgt = func(mat, ddof=0.5)                     # 计算标准函数的结果，使用ddof参数为0.5
        out = nanfunc(mat, ddof=0.5)                  # 计算NaN函数的结果，使用ddof参数为0.5

        assert_almost_equal(out, tgt)                 # 断言NaN函数的结果与标准函数的结果几乎相等
        if dtype == "O":
            assert type(out) is type(tgt)             # 如果数据类型为对象类型，断言NaN函数与标准函数的结果类型相同
        else:
            assert out.dtype == tgt.dtype             # 否则，断言NaN函数的结果与标准函数的结果的数据类型相同

    # 使用参数化标记定义测试函数，测试NaN方差和标准差函数的correction参数
    @pytest.mark.parametrize(
        "nanfunc", [np.nanvar, np.nanstd]
    )
    def test_nanfunc_correction(self, mat, dtype, nanfunc):
        mat = mat.astype(dtype)                       # 将mat数组转换为指定的数据类型
        assert_almost_equal(
            nanfunc(mat, correction=0.5),             # 断言使用correction参数0.5计算的NaN函数结果与使用ddof参数0.5计算的结果几乎相等
            nanfunc(mat, ddof=0.5)
        )

        err_msg = "ddof and correction can't be provided simultaneously."
        with assert_raises_regex(ValueError, err_msg):
            nanfunc(mat, ddof=0.5, correction=0.5)    # 断言当同时提供ddof和correction参数时，会引发ValueError异常

        with assert_raises_regex(ValueError, err_msg):
            nanfunc(mat, ddof=1, correction=0)        # 断言当提供ddof参数为1和correction参数时，会引发ValueError异常
    def test_mutation(self):
        # 检查传入的数组未被修改
        ndat = _ndat.copy()  # 复制 _ndat 数组以防止修改原始数据
        for f in self.nanfuncs:
            f(ndat)  # 调用函数 f 对 ndat 进行操作
            assert_equal(ndat, _ndat)  # 断言 ndat 与 _ndat 相等，验证未修改原始数据

    def test_keepdims(self):
        mat = np.eye(3)  # 创建一个 3x3 的单位矩阵
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for axis in [None, 0, 1]:
                tgt = rf(mat, axis=axis, keepdims=True)  # 调用 rf 函数计算结果并保持维度
                res = nf(mat, axis=axis, keepdims=True)  # 调用 nf 函数计算结果并保持维度
                assert_(res.ndim == tgt.ndim)  # 断言 res 和 tgt 的维度相同

    def test_out(self):
        mat = np.eye(3)  # 创建一个 3x3 的单位矩阵
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            resout = np.zeros(3)  # 创建一个长度为 3 的零向量
            tgt = rf(mat, axis=1)  # 调用 rf 函数计算结果
            res = nf(mat, axis=1, out=resout)  # 调用 nf 函数计算结果，并将结果存储到 resout 中
            assert_almost_equal(res, resout)  # 断言 nf 计算结果与 resout 几乎相等
            assert_almost_equal(res, tgt)  # 断言 nf 计算结果与 rf 计算结果几乎相等

    def test_dtype_from_dtype(self):
        mat = np.eye(3)  # 创建一个 3x3 的单位矩阵
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                with suppress_warnings() as sup:
                    if nf in {np.nanstd, np.nanvar} and c in 'FDG':
                        sup.filter(ComplexWarning)  # 过滤掉复杂类型警告
                    tgt = rf(mat, dtype=np.dtype(c), axis=1).dtype.type  # 指定数据类型进行计算并获取类型
                    res = nf(mat, dtype=np.dtype(c), axis=1).dtype.type  # 使用 nf 函数计算相同数据类型的结果类型
                    assert_(res is tgt)  # 断言 nf 和 rf 的结果类型相同
                    # scalar case
                    tgt = rf(mat, dtype=np.dtype(c), axis=None).dtype.type  # 沿单个轴进行计算并获取类型
                    res = nf(mat, dtype=np.dtype(c), axis=None).dtype.type  # 使用 nf 函数计算相同数据类型的结果类型
                    assert_(res is tgt)  # 断言 nf 和 rf 的结果类型相同

    def test_dtype_from_char(self):
        mat = np.eye(3)  # 创建一个 3x3 的单位矩阵
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                with suppress_warnings() as sup:
                    if nf in {np.nanstd, np.nanvar} and c in 'FDG':
                        sup.filter(ComplexWarning)  # 过滤掉复杂类型警告
                    tgt = rf(mat, dtype=c, axis=1).dtype.type  # 使用字符指定数据类型进行计算并获取类型
                    res = nf(mat, dtype=c, axis=1).dtype.type  # 使用 nf 函数计算相同数据类型的结果类型
                    assert_(res is tgt)  # 断言 nf 和 rf 的结果类型相同
                    # scalar case
                    tgt = rf(mat, dtype=c, axis=None).dtype.type  # 沿单个轴进行计算并获取类型
                    res = nf(mat, dtype=c, axis=None).dtype.type  # 使用 nf 函数计算相同数据类型的结果类型
                    assert_(res is tgt)  # 断言 nf 和 rf 的结果类型相同

    def test_dtype_from_input(self):
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                mat = np.eye(3, dtype=c)  # 创建指定数据类型的 3x3 单位矩阵
                tgt = rf(mat, axis=1).dtype.type  # 指定轴进行计算并获取结果类型
                res = nf(mat, axis=1).dtype.type  # 使用 nf 函数计算相同轴的结果类型
                assert_(res is tgt, "res %s, tgt %s" % (res, tgt))  # 断言 nf 和 rf 的结果类型相同，否则输出详细信息
                # scalar case
                tgt = rf(mat, axis=None).dtype.type  # 沿单个轴进行计算并获取类型
                res = nf(mat, axis=None).dtype.type  # 使用 nf 函数计算相同轴的结果类型
                assert_(res is tgt)  # 断言 nf 和 rf 的结果类型相同
    # 定义测试方法以验证结果值
    def test_result_values(self):
        # 对于每对自定义函数和标准函数，分别进行测试
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            # 创建目标结果列表，其中包含对标准函数在_rdat数据上的应用结果
            tgt = [rf(d) for d in _rdat]
            # 对自定义函数在_ndat上执行轴向为1的操作，得到结果
            res = nf(_ndat, axis=1)
            # 断言结果近似等于目标结果列表
            assert_almost_equal(res, tgt)

    # 定义测试标量情况的方法
    def test_scalar(self):
        # 对于每个自定义函数，测试其对标量0.的返回值是否为0.
        for f in self.nanfuncs:
            assert_(f(0.) == 0.)

    # 定义测试子类情况的方法
    def test_subclass(self):
        # 定义一个继承自np.ndarray的子类MyNDArray
        class MyNDArray(np.ndarray):
            pass

        # 创建一个3x3的单位矩阵
        array = np.eye(3)
        # 将array视图转换为MyNDArray类型的对象mine
        mine = array.view(MyNDArray)

        # 对于每个自定义函数，验证其在不同轴上操作后返回的类型和形状与预期一致
        for f in self.nanfuncs:
            # 预期轴为0时的形状
            expected_shape = f(array, axis=0).shape
            # 对mine在轴为0上执行函数f，得到结果res
            res = f(mine, axis=0)
            # 断言res的类型为MyNDArray
            assert_(isinstance(res, MyNDArray))
            # 断言res的形状与预期一致
            assert_(res.shape == expected_shape)

            # 预期轴为1时的形状
            expected_shape = f(array, axis=1).shape
            # 对mine在轴为1上执行函数f，得到结果res
            res = f(mine, axis=1)
            # 断言res的类型为MyNDArray
            assert_(isinstance(res, MyNDArray))
            # 断言res的形状与预期一致
            assert_(res.shape == expected_shape)

            # 对于不指定轴的情况，验证返回结果的形状
            expected_shape = f(array).shape
            # 对mine执行函数f，得到结果res
            res = f(mine)
            # 断言res的类型为MyNDArray
            assert_(isinstance(res, MyNDArray))
            # 断言res的形状与预期一致
            assert_(res.shape == expected_shape)
# 定义一个测试类 TestNanFunctions_SumProd，继承自 SharedNanFunctionsTestsMixin
class TestNanFunctions_SumProd(SharedNanFunctionsTestsMixin):

    # nanfuncs 列表包含 np.nansum 和 np.nanprod 函数
    nanfuncs = [np.nansum, np.nanprod]
    # stdfuncs 列表包含 np.sum 和 np.prod 函数
    stdfuncs = [np.sum, np.prod]

    # 使用 pytest.mark.parametrize 标记的参数化测试方法
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),  # 创建一个包含单个 NaN 值的数组
        np.full((3, 3), np.nan),  # 创建一个全部元素为 NaN 的 3x3 数组
    ], ids=["0d", "2d"])  # 分别用 "0d" 和 "2d" 标识两个测试用例
    def test_allnans(self, axis, dtype, array):
        # 如果 axis 不为 None 且 array 的维度为 0，则跳过测试，并显示相应的提示信息
        if axis is not None and array.ndim == 0:
            pytest.skip(f"`axis != None` not supported for 0d arrays")

        # 将 array 转换为指定的 dtype 类型
        array = array.astype(dtype)
        # 对于 nanfuncs 列表中的每个函数 func 和对应的 identity 值
        for func, identity in zip(self.nanfuncs, [0, 1]):
            # 调用 func 函数计算 array 在指定 axis 上的操作结果
            out = func(array, axis=axis)
            # 断言结果 out 中所有的元素等于预期的 identity 值
            assert np.all(out == identity)
            # 断言结果 out 的数据类型与 array 的数据类型相同
            assert out.dtype == array.dtype

    # 定义测试空数组情况的方法
    def test_empty(self):
        # 对于 nanfuncs 列表中的每个函数 f 和其对应的目标值 tgt_value
        for f, tgt_value in zip([np.nansum, np.nanprod], [0, 1]):
            # 创建一个形状为 (0, 3) 的全零数组 mat
            mat = np.zeros((0, 3))
            # 设置目标值 tgt 为长度为 3 的列表，其元素均为 tgt_value
            tgt = [tgt_value]*3
            # 调用函数 f 计算 mat 在 axis=0 上的结果 res
            res = f(mat, axis=0)
            # 断言 res 与目标值 tgt 相等
            assert_equal(res, tgt)
            # 设置目标值 tgt 为空列表
            tgt = []
            # 调用函数 f 计算 mat 在 axis=1 上的结果 res
            res = f(mat, axis=1)
            # 断言 res 与目标值 tgt 相等
            assert_equal(res, tgt)
            # 设置目标值 tgt 为单一的 tgt_value
            tgt = tgt_value
            # 调用函数 f 计算 mat 在 axis=None 上的结果 res
            res = f(mat, axis=None)
            # 断言 res 等于目标值 tgt
            assert_equal(res, tgt)

    # 使用 pytest.mark.parametrize 标记的参数化测试方法
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_initial(self, dtype):
        # 创建一个长度为 9 的数组 ar，并转换为指定的 dtype 类型
        ar = np.arange(9).astype(dtype)
        # 将数组 ar 中的前 5 个元素设为 NaN
        ar[:5] = np.nan

        # 对于 nanfuncs 列表中的每个函数 f
        for f in self.nanfuncs:
            # 设置参考值 reference，根据 f 是 np.nansum 还是 np.nanprod 不同而不同
            reference = 28 if f is np.nansum else 3360
            # 调用函数 f 计算 ar 的结果 ret，并指定 initial 参数为 2
            ret = f(ar, initial=2)
            # 断言 ret 的数据类型与 dtype 相同
            assert ret.dtype == dtype
            # 断言 ret 等于参考值 reference
            assert ret == reference

    # 使用 pytest.mark.parametrize 标记的参数化测试方法
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_where(self, dtype):
        # 创建一个形状为 (3, 3) 的数组 ar，并转换为指定的 dtype 类型
        ar = np.arange(9).reshape(3, 3).astype(dtype)
        # 将数组 ar 的第一行所有元素设为 NaN
        ar[0, :] = np.nan
        # 创建一个与 ar 相同形状的布尔数组 where，并将所有元素初始化为 True
        where = np.ones_like(ar, dtype=np.bool)
        # 将 where 的第一列所有元素设为 False
        where[:, 0] = False

        # 对于 nanfuncs 列表中的每个函数 f
        for f in self.nanfuncs:
            # 设置参考值 reference，根据 f 是 np.nansum 还是 np.nanprod 不同而不同
            reference = 26 if f is np.nansum else 2240
            # 调用函数 f 计算 ar 在给定 where 和 initial=2 的条件下的结果 ret
            ret = f(ar, where=where, initial=2)
            # 断言 ret 的数据类型与 dtype 相同
            assert ret.dtype == dtype
            # 断言 ret 等于参考值 reference
            assert ret == reference


# 定义一个测试类 TestNanFunctions_CumSumProd，继承自 SharedNanFunctionsTestsMixin
class TestNanFunctions_CumSumProd(SharedNanFunctionsTestsMixin):

    # nanfuncs 列表包含 np.nancumsum 和 np.nancumprod 函数
    nanfuncs = [np.nancumsum, np.nancumprod]
    # stdfuncs 列表包含 np.cumsum 和 np.cumprod 函数
    stdfuncs = [np.cumsum, np.cumprod]

    # 使用 pytest.mark.parametrize 标记的参数化测试方法
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),  # 创建一个包含单个 NaN 值的数组
        np.full((3, 3), np.nan)  # 创建一个全部元素为 NaN 的 3x3 数组
    ], ids=["0d", "2d"])  # 分别用 "0d" 和 "2d" 标识两个测试用例
    def test_allnans(self, axis, dtype, array):
        # 如果 axis 不为 None 且 array 的维度为 0，则跳过测试，并显示相应的提示信息
        if axis is not None and array.ndim == 0:
            pytest.skip(f"`axis != None` not supported for 0d arrays")

        # 将 array 转换为指定的 dtype 类型
        array = array.astype(dtype)
        # 对于 nanfuncs 列表中的每个函数 func 和对应的 identity 值
        for func, identity in zip(self.nanfuncs, [0, 1]):
            # 调用 func 函数计算 array 上的累积操作结果 out
            out = func(array)
            # 断言结果 out 中所有的元素等于预期的 identity 值
            assert np.all(out == identity)
            # 断言结果 out 的数据类型与 array 的数据类型相同
            assert out.dtype == array.dtype
    # 测试空矩阵情况下各函数的行为
    def test_empty(self):
        # 遍历 nanfuncs 和对应的目标值列表
        for f, tgt_value in zip(self.nanfuncs, [0, 1]):
            # 创建一个空的 0x3 的矩阵
            mat = np.zeros((0, 3))
            # 创建一个与 mat 相同形状的矩阵，填充为目标值的倍数
            tgt = tgt_value * np.ones((0, 3))
            # 使用函数 f 计算 mat 沿 axis=0 的结果
            res = f(mat, axis=0)
            # 断言 res 与目标值 tgt 相等
            assert_equal(res, tgt)
            # 将目标值设为 mat 自身
            tgt = mat
            # 使用函数 f 计算 mat 沿 axis=1 的结果
            res = f(mat, axis=1)
            # 断言 res 与目标值 tgt 相等
            assert_equal(res, tgt)
            # 创建一个空的 0 维数组
            tgt = np.zeros((0))
            # 使用函数 f 计算 mat 沿 axis=None 的结果
            res = f(mat, axis=None)
            # 断言 res 与目标值 tgt 相等
            assert_equal(res, tgt)

    # 测试 keepdims 参数对函数行为的影响
    def test_keepdims(self):
        # 遍历 nanfuncs 和 stdfuncs
        for f, g in zip(self.nanfuncs, self.stdfuncs):
            # 创建一个 3x3 的单位矩阵
            mat = np.eye(3)
            # 遍历 axis 参数的可能取值
            for axis in [None, 0, 1]:
                # 使用函数 f 计算 mat 的结果，并且不指定输出
                tgt = f(mat, axis=axis, out=None)
                # 使用函数 g 计算 mat 的结果，并且不指定输出
                res = g(mat, axis=axis, out=None)
                # 断言 res 和 tgt 的维度相等
                assert_(res.ndim == tgt.ndim)

        # 再次遍历 nanfuncs
        for f in self.nanfuncs:
            # 创建一个形状为 (3, 5, 7, 11) 的全为 1 的数组
            d = np.ones((3, 5, 7, 11))
            # 随机将一些元素设为 NaN
            rs = np.random.RandomState(0)
            d[rs.rand(*d.shape) < 0.5] = np.nan
            # 使用函数 f 计算数组 d 沿 axis=None 的结果
            res = f(d, axis=None)
            # 断言 res 的形状为 (1155,)
            assert_equal(res.shape, (1155,))
            # 遍历 axis 的所有可能取值
            for axis in np.arange(4):
                # 使用函数 f 计算数组 d 沿指定 axis 的结果
                res = f(d, axis=axis)
                # 断言 res 的形状与数组 d 相同
                assert_equal(res.shape, (3, 5, 7, 11))

    # 测试结果值是否正确的断言
    def test_result_values(self):
        # 遍历 axis 的多个可能取值
        for axis in (-2, -1, 0, 1, None):
            # 计算 _ndat_ones 沿指定 axis 的累积乘积
            tgt = np.cumprod(_ndat_ones, axis=axis)
            # 计算 _ndat 沿指定 axis 的累积乘积，跳过 NaN 值
            res = np.nancumprod(_ndat, axis=axis)
            # 断言 res 和 tgt 的近似相等
            assert_almost_equal(res, tgt)
            # 计算 _ndat_zeros 沿指定 axis 的累积和
            tgt = np.cumsum(_ndat_zeros, axis=axis)
            # 计算 _ndat 沿指定 axis 的累积和，跳过 NaN 值
            res = np.nancumsum(_ndat, axis=axis)
            # 断言 res 和 tgt 的近似相等
            assert_almost_equal(res, tgt)

    # 测试输出参数 out 对函数行为的影响
    def test_out(self):
        # 创建一个 3x3 的单位矩阵
        mat = np.eye(3)
        # 遍历 nanfuncs 和 stdfuncs
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            # 创建一个与 mat 相同形状的单位矩阵作为输出容器
            resout = np.eye(3)
            # 遍历 axis 的多个可能取值
            for axis in (-2, -1, 0, 1):
                # 使用函数 rf 计算 mat 沿指定 axis 的结果
                tgt = rf(mat, axis=axis)
                # 使用函数 nf 计算 mat 沿指定 axis 的结果，并将结果写入 resout
                res = nf(mat, axis=axis, out=resout)
                # 断言 res 与 resout 的近似相等
                assert_almost_equal(res, resout)
                # 断言 res 与 tgt 的近似相等
                assert_almost_equal(res, tgt)
class TestNanFunctions_MeanVarStd(SharedNanFunctionsTestsMixin):
    # 继承自 SharedNanFunctionsTestsMixin 的测试类，用于测试 NaN 相关函数的行为

    nanfuncs = [np.nanmean, np.nanvar, np.nanstd]
    # 包含 NaN 函数的列表：nanmean, nanvar, nanstd

    stdfuncs = [np.mean, np.var, np.std]
    # 标准函数的列表：mean, var, std

    def test_dtype_error(self):
        # 测试数据类型错误的情况
        for f in self.nanfuncs:
            for dtype in [np.bool, np.int_, np.object_]:
                # 对于每个 NaN 函数和指定的数据类型
                assert_raises(TypeError, f, _ndat, axis=1, dtype=dtype)

    def test_out_dtype_error(self):
        # 测试输出数据类型错误的情况
        for f in self.nanfuncs:
            for dtype in [np.bool, np.int_, np.object_]:
                # 对于每个 NaN 函数和指定的数据类型
                out = np.empty(_ndat.shape[0], dtype=dtype)
                assert_raises(TypeError, f, _ndat, axis=1, out=out)

    def test_ddof(self):
        # 测试自由度参数 ddof 的影响
        nanfuncs = [np.nanvar, np.nanstd]
        stdfuncs = [np.var, np.std]
        for nf, rf in zip(nanfuncs, stdfuncs):
            for ddof in [0, 1]:
                # 对于每个 NaN 方差和标准差函数，以及不同的 ddof 值
                tgt = [rf(d, ddof=ddof) for d in _rdat]
                res = nf(_ndat, axis=1, ddof=ddof)
                assert_almost_equal(res, tgt)

    def test_ddof_too_big(self):
        # 测试 ddof 参数过大的情况
        nanfuncs = [np.nanvar, np.nanstd]
        stdfuncs = [np.var, np.std]
        dsize = [len(d) for d in _rdat]
        for nf, rf in zip(nanfuncs, stdfuncs):
            for ddof in range(5):
                # 对于每个 NaN 方差和标准差函数，以及不同的 ddof 值
                with suppress_warnings() as sup:
                    sup.record(RuntimeWarning)
                    sup.filter(ComplexWarning)
                    tgt = [ddof >= d for d in dsize]
                    res = nf(_ndat, axis=1, ddof=ddof)
                    assert_equal(np.isnan(res), tgt)
                    if any(tgt):
                        assert_(len(sup.log) == 1)
                    else:
                        assert_(len(sup.log) == 0)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),
        np.full((3, 3), np.nan),
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        # 测试所有元素为 NaN 的情况
        if axis is not None and array.ndim == 0:
            pytest.skip(f"`axis != None` not supported for 0d arrays")

        array = array.astype(dtype)
        match = "(Degrees of freedom <= 0 for slice.)|(Mean of empty slice)"
        for func in self.nanfuncs:
            with pytest.warns(RuntimeWarning, match=match):
                out = func(array, axis=axis)
            assert np.isnan(out).all()

            # `nanvar` and `nanstd` convert complex inputs to their
            # corresponding floating dtype
            if func is np.nanmean:
                assert out.dtype == array.dtype
            else:
                assert out.dtype == np.abs(array).dtype
    # 定义一个测试方法，测试处理空数组的情况
    def test_empty(self):
        # 创建一个形状为 (0, 3) 的全零数组
        mat = np.zeros((0, 3))
        # 遍历 nanfuncs 列表中的函数
        for f in self.nanfuncs:
            # 对于 axis 参数为 [0, None] 的情况
            for axis in [0, None]:
                # 使用 warnings.catch_warnings 捕获警告信息
                with warnings.catch_warnings(record=True) as w:
                    # 设置警告过滤器，捕获所有警告
                    warnings.simplefilter('always')
                    # 断言调用 f 函数处理 mat 时所有结果都是 NaN
                    assert_(np.isnan(f(mat, axis=axis)).all())
                    # 断言警告列表 w 的长度为 1
                    assert_(len(w) == 1)
                    # 断言第一个警告是 RuntimeWarning 的子类
                    assert_(issubclass(w[0].category, RuntimeWarning))
            # 对于 axis 参数为 1 的情况
            for axis in [1]:
                # 使用 warnings.catch_warnings 捕获警告信息
                with warnings.catch_warnings(record=True) as w:
                    # 设置警告过滤器，捕获所有警告
                    warnings.simplefilter('always')
                    # 断言调用 f 函数处理 mat 时结果与形状为 [] 的全零数组相等
                    assert_equal(f(mat, axis=axis), np.zeros([]))
                    # 断言警告列表 w 的长度为 0
                    assert_(len(w) == 0)

    # 使用 pytest.mark.parametrize 装饰器，参数为 np.typecodes["AllFloat"] 中的数据类型
    def test_where(self, dtype):
        # 创建一个形状为 (3, 3) 的数组 ar，转换为指定数据类型 dtype
        ar = np.arange(9).reshape(3, 3).astype(dtype)
        # 将第一行设置为 NaN
        ar[0, :] = np.nan
        # 创建一个与 ar 相同形状的布尔数组 where，并设置第一列为 False
        where = np.ones_like(ar, dtype=np.bool)
        where[:, 0] = False

        # 遍历 nanfuncs 和 stdfuncs 列表中的函数
        for f, f_std in zip(self.nanfuncs, self.stdfuncs):
            # 使用 where 数组的条件，对 ar 的第三行及之后的数据应用 f_std 函数计算参考值
            reference = f_std(ar[where][2:])
            # 如果 f 是 np.nanmean，则使用指定数据类型 dtype 作为参考值的数据类型
            dtype_reference = dtype if f is np.nanmean else ar.real.dtype

            # 调用 f 函数处理 ar 和 where 数组
            ret = f(ar, where=where)
            # 断言返回结果的数据类型与 dtype_reference 相同
            assert ret.dtype == dtype_reference
            # 使用 np.testing.assert_allclose 断言返回结果与 reference 的接近程度

    # 定义一个测试方法，测试带有 mean 关键字参数的 np.nanstd 函数
    def test_nanstd_with_mean_keyword(self):
        # 设置随机种子以保证测试的可复现性
        rng = np.random.RandomState(1234)
        # 创建一个形状为 (10, 20, 5) 的随机数组 A，并添加 NaN 值
        A = rng.randn(10, 20, 5) + 0.5
        A[:, 5, :] = np.nan

        # 创建形状为 (10, 1, 5) 的全零数组 mean_out 和 std_out
        mean_out = np.zeros((10, 1, 5))
        std_out = np.zeros((10, 1, 5))

        # 使用 np.nanmean 计算 A 的均值，输出到 mean_out，沿 axis=1，保持维度为 True
        mean = np.nanmean(A,
                          out=mean_out,
                          axis=1,
                          keepdims=True)

        # 断言 mean_out 与 mean 是同一个对象
        assert mean_out is mean

        # 使用 np.nanstd 计算 A 的标准差，输出到 std_out，沿 axis=1，保持维度为 True，指定 mean 参数为 mean
        std = np.nanstd(A,
                        out=std_out,
                        axis=1,
                        keepdims=True,
                        mean=mean)

        # 断言 std_out 与 std 是同一个对象
        assert std_out is std

        # 断言 mean 和 std 的形状相同，应为 (10, 1, 5)
        assert std.shape == mean.shape
        assert std.shape == (10, 1, 5)

        # 使用 np.nanstd 计算 A 的标准差，沿 axis=1，保持维度为 True，作为旧方法的参考值
        std_old = np.nanstd(A, axis=1, keepdims=True)

        # 断言 std_old 与 mean 的形状相同
        assert std_old.shape == mean.shape
        # 使用 assert_almost_equal 断言 std 与 std_old 的接近程度
_TIME_UNITS = (
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"
)
# 定义了时间单位的元组

# All `inexact` + `timdelta64` type codes
_TYPE_CODES = list(np.typecodes["AllFloat"])
_TYPE_CODES += [f"m8[{unit}]" for unit in _TIME_UNITS]
# 将所有浮点数类型代码添加到 _TYPE_CODES 列表中，同时添加了时间单位对应的 m8[unit] 类型代码

class TestNanFunctions_Median:

    def test_mutation(self):
        # 检查传递的数组未被修改
        ndat = _ndat.copy()
        np.nanmedian(ndat)
        assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for axis in [None, 0, 1]:
            tgt = np.median(mat, axis=axis, out=None, overwrite_input=False)
            res = np.nanmedian(mat, axis=axis, out=None, overwrite_input=False)
            assert_(res.ndim == tgt.ndim)

        d = np.ones((3, 5, 7, 11))
        # 随机将一些元素设为 NaN：
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            res = np.nanmedian(d, axis=None, keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanmedian(d, axis=(0, 1), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 11))
            res = np.nanmedian(d, axis=(0, 3), keepdims=True)
            assert_equal(res.shape, (1, 5, 7, 1))
            res = np.nanmedian(d, axis=(1,), keepdims=True)
            assert_equal(res.shape, (3, 1, 7, 11))
            res = np.nanmedian(d, axis=(0, 1, 2, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanmedian(d, axis=(0, 1, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 1))

    @pytest.mark.parametrize(
        argnames='axis',
        argvalues=[
            None,
            1,
            (1, ),
            (0, 1),
            (-3, -1),
        ]
    )
    @pytest.mark.filterwarnings("ignore:All-NaN slice:RuntimeWarning")
    def test_keepdims_out(self, axis):
        d = np.ones((3, 5, 7, 11))
        # 随机将一些元素设为 NaN：
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        out = np.empty(shape_out)
        result = np.nanmedian(d, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)
    def test_out(self):
        # 创建一个 3x3 的随机数矩阵
        mat = np.random.rand(3, 3)
        # 在矩阵中插入 NaN 值，每行插入两个 NaN
        nan_mat = np.insert(mat, [0, 2], np.nan, axis=1)
        # 创建一个全零的长度为 3 的数组
        resout = np.zeros(3)
        # 计算原始矩阵每行的中位数
        tgt = np.median(mat, axis=1)
        # 计算插入 NaN 值后的矩阵每行的中位数，结果存入 resout 中
        res = np.nanmedian(nan_mat, axis=1, out=resout)
        # 检查计算结果与预期是否几乎相等
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)
        
        # 对于零维输出：
        resout = np.zeros(())
        # 计算原始矩阵所有元素的中位数
        tgt = np.median(mat, axis=None)
        # 计算插入 NaN 值后的矩阵所有元素的中位数，结果存入 resout 中
        res = np.nanmedian(nan_mat, axis=None, out=resout)
        # 检查计算结果与预期是否几乎相等
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)
        
        # 计算插入 NaN 值后的矩阵在指定轴（0 和 1）上的中位数，结果存入 resout 中
        res = np.nanmedian(nan_mat, axis=(0, 1), out=resout)
        # 检查计算结果与预期是否几乎相等
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)

    def test_small_large(self):
        # 测试小型和大型代码路径，当前截断为 400 个元素
        for s in [5, 20, 51, 200, 1000]:
            # 创建一个大小为 4x(s+1) 的随机数矩阵
            d = np.random.randn(4, s)
            # 随机将部分元素设为 NaN
            w = np.random.randint(0, d.size, size=d.size // 5)
            d.ravel()[w] = np.nan
            d[:, 0] = 1.  # 确保至少有一个有效值
            # 使用没有 NaN 的普通中位数进行比较
            tgt = []
            for x in d:
                nonan = np.compress(~np.isnan(x), x)
                tgt.append(np.median(nonan, overwrite_input=True))
            
            # 检查 np.nanmedian 函数计算结果与预期是否相等
            assert_array_equal(np.nanmedian(d, axis=-1), tgt)

    def test_result_values(self):
        # 计算 _ndat 沿第二个轴的每行的中位数
        tgt = [np.median(d) for d in _rdat]
        res = np.nanmedian(_ndat, axis=1)
        # 检查计算结果与预期是否几乎相等
        assert_almost_equal(res, tgt)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", _TYPE_CODES)
    def test_allnans(self, dtype, axis):
        # 创建一个全为 NaN 的 3x3 数组，并转换为指定的 dtype
        mat = np.full((3, 3), np.nan).astype(dtype)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)

            # 计算 mat 在指定轴上的中位数
            output = np.nanmedian(mat, axis=axis)
            # 检查输出的数据类型与 mat 的数据类型是否相等
            assert output.dtype == mat.dtype
            # 检查输出是否全为 NaN
            assert np.isnan(output).all()

            if axis is None:
                # 如果 axis 为 None，检查警告记录数是否为 1
                assert_(len(sup.log) == 1)
            else:
                # 如果 axis 不为 None，检查警告记录数是否为 3
                assert_(len(sup.log) == 3)

            # 检查标量情况下的中位数计算
            scalar = np.array(np.nan).astype(dtype)[()]
            output_scalar = np.nanmedian(scalar)
            # 检查输出的数据类型与标量的数据类型是否相等
            assert output_scalar.dtype == scalar.dtype
            # 检查输出是否为 NaN
            assert np.isnan(output_scalar)

            if axis is None:
                # 如果 axis 为 None，检查警告记录数是否为 2
                assert_(len(sup.log) == 2)
            else:
                # 如果 axis 不为 None，检查警告记录数是否为 4
                assert_(len(sup.log) == 4)
    # 定义测试函数，测试处理空数组时的行为
    def test_empty(self):
        # 创建一个空的 0x3 的 NumPy 数组
        mat = np.zeros((0, 3))
        # 针对不同的轴进行循环测试
        for axis in [0, None]:
            # 捕获警告信息
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                # 使用 np.nanmedian 计算空数组的中位数，并检查是否全为 NaN
                assert_(np.isnan(np.nanmedian(mat, axis=axis)).all())
                # 断言捕获到一条警告
                assert_(len(w) == 1)
                # 断言该警告是 RuntimeWarning 的子类
                assert_(issubclass(w[0].category, RuntimeWarning))
        # 针对 axis=1 的情况进行测试
        for axis in [1]:
            # 捕获警告信息
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                # 使用 np.nanmedian 计算空数组的中位数，并与空数组 [] 进行比较
                assert_equal(np.nanmedian(mat, axis=axis), np.zeros([]))
                # 断言没有捕获到任何警告
                assert_(len(w) == 0)

    # 定义测试函数，测试处理标量时的行为
    def test_scalar(self):
        # 断言 np.nanmedian(0.) 的结果等于 0.
        assert_(np.nanmedian(0.) == 0.)

    # 定义测试函数，测试处理超出范围的轴参数时的行为
    def test_extended_axis_invalid(self):
        # 创建一个形状为 (3, 5, 7, 11) 的全为 1 的 NumPy 数组
        d = np.ones((3, 5, 7, 11))
        # 断言处理超出负轴索引的 AxisError 异常
        assert_raises(AxisError, np.nanmedian, d, axis=-5)
        # 断言处理包含负轴索引的 AxisError 异常
        assert_raises(AxisError, np.nanmedian, d, axis=(0, -5))
        # 断言处理超出正轴索引的 AxisError 异常
        assert_raises(AxisError, np.nanmedian, d, axis=4)
        # 断言处理包含超出正轴索引的 AxisError 异常
        assert_raises(AxisError, np.nanmedian, d, axis=(0, 4))
        # 断言处理重复轴的 ValueError 异常
        assert_raises(ValueError, np.nanmedian, d, axis=(1, 1))
    # 定义测试函数，用于测试处理特殊浮点数情况的函数
    def test_float_special(self):
        # 使用 suppress_warnings 上下文管理器，过滤掉 RuntimeWarning 警告
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            
            # 对于正无穷和负无穷两种情况进行迭代测试
            for inf in [np.inf, -np.inf]:
                # 创建包含特殊值的二维数组 a
                a = np.array([[inf,  np.nan], [np.nan, np.nan]])
                # 检查按列计算忽略 NaN 后的中位数是否符合预期
                assert_equal(np.nanmedian(a, axis=0), [inf,  np.nan])
                # 检查按行计算忽略 NaN 后的中位数是否符合预期
                assert_equal(np.nanmedian(a, axis=1), [inf,  np.nan])
                # 检查忽略 NaN 后的整体中位数是否符合预期
                assert_equal(np.nanmedian(a), inf)
                
                # 最小填充值检查
                a = np.array([[np.nan, np.nan, inf],
                             [np.nan, np.nan, inf]])
                # 检查忽略 NaN 后的整体中位数是否符合预期
                assert_equal(np.nanmedian(a), inf)
                # 检查按列计算忽略 NaN 后的中位数是否符合预期
                assert_equal(np.nanmedian(a, axis=0), [np.nan, np.nan, inf])
                # 检查按行计算忽略 NaN 后的中位数是否符合预期
                assert_equal(np.nanmedian(a, axis=1), inf)
                
                # 无遮罩路径
                a = np.array([[inf, inf], [inf, inf]])
                # 检查按行计算忽略 NaN 后的中位数是否符合预期
                assert_equal(np.nanmedian(a, axis=1), inf)
                
                # 创建包含特殊浮点数的二维数组 a
                a = np.array([[inf, 7, -inf, -9],
                              [-10, np.nan, np.nan, 5],
                              [4, np.nan, np.nan, inf]],
                              dtype=np.float32)
                # 根据正无穷的值进行条件判断，检查按列计算忽略 NaN 后的中位数是否符合预期
                if inf > 0:
                    assert_equal(np.nanmedian(a, axis=0), [4., 7., -inf, 5.])
                    assert_equal(np.nanmedian(a), 4.5)
                else:
                    assert_equal(np.nanmedian(a, axis=0), [-10., 7., -inf, -9.])
                    assert_equal(np.nanmedian(a), -2.5)
                # 检查按行计算忽略 NaN 后的中位数是否符合预期
                assert_equal(np.nanmedian(a, axis=-1), [-1., -2.5, inf])
                
                # 针对不同长度的 i 和 j 进行迭代测试
                for i in range(0, 10):
                    for j in range(1, 10):
                        # 创建特殊值数组 a
                        a = np.array([([np.nan] * i) + ([inf] * j)] * 2)
                        # 检查忽略 NaN 后的整体中位数是否符合预期
                        assert_equal(np.nanmedian(a), inf)
                        # 检查按行计算忽略 NaN 后的中位数是否符合预期
                        assert_equal(np.nanmedian(a, axis=1), inf)
                        # 检查按列计算忽略 NaN 后的中位数是否符合预期
                        assert_equal(np.nanmedian(a, axis=0),
                                     ([np.nan] * i) + [inf] * j)
                        
                        # 创建特殊值数组 a
                        a = np.array([([np.nan] * i) + ([-inf] * j)] * 2)
                        # 检查忽略 NaN 后的整体中位数是否符合预期
                        assert_equal(np.nanmedian(a), -inf)
                        # 检查按行计算忽略 NaN 后的中位数是否符合预期
                        assert_equal(np.nanmedian(a, axis=1), -inf)
                        # 检查按列计算忽略 NaN 后的中位数是否符合预期
                        assert_equal(np.nanmedian(a, axis=0),
                                     ([np.nan] * i) + [-inf] * j)
class TestNanFunctions_Percentile:

    def test_mutation(self):
        # 检查传入的数组是否被修改
        ndat = _ndat.copy()  # 复制_ndat数组的副本，确保不改变原始数据
        np.nanpercentile(ndat, 30)  # 计算ndat数组的30th百分位数，忽略NaN值
        assert_equal(ndat, _ndat)  # 断言复制后的数组与原始数组相等，验证原始数据未被修改

    def test_keepdims(self):
        mat = np.eye(3)  # 创建一个3x3的单位矩阵
        for axis in [None, 0, 1]:
            tgt = np.percentile(mat, 70, axis=axis, out=None,
                                overwrite_input=False)
            res = np.nanpercentile(mat, 70, axis=axis, out=None,
                                   overwrite_input=False)
            assert_(res.ndim == tgt.ndim)  # 断言计算结果的维度与目标维度相等

        d = np.ones((3, 5, 7, 11))  # 创建一个全为1的4维数组
        # 随机将一些元素设为NaN：
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan  # 在d数组中随机设置一些元素为NaN
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            # 测试不同轴上的百分位数计算，保持维度为True
            res = np.nanpercentile(d, 90, axis=None, keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanpercentile(d, 90, axis=(0, 1), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 11))
            res = np.nanpercentile(d, 90, axis=(0, 3), keepdims=True)
            assert_equal(res.shape, (1, 5, 7, 1))
            res = np.nanpercentile(d, 90, axis=(1,), keepdims=True)
            assert_equal(res.shape, (3, 1, 7, 11))
            res = np.nanpercentile(d, 90, axis=(0, 1, 2, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanpercentile(d, 90, axis=(0, 1, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 1))

    @pytest.mark.parametrize('q', [7, [1, 7]])
    @pytest.mark.parametrize(
        argnames='axis',
        argvalues=[
            None,
            1,
            (1,),
            (0, 1),
            (-3, -1),
        ]
    )
    @pytest.mark.filterwarnings("ignore:All-NaN slice:RuntimeWarning")
    def test_keepdims_out(self, q, axis):
        d = np.ones((3, 5, 7, 11))  # 创建一个全为1的4维数组
        # 随机将一些元素设为NaN：
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan  # 在d数组中随机设置一些元素为NaN
        if axis is None:
            shape_out = (1,) * d.ndim  # 如果axis为None，输出形状为全1
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            # 根据指定的轴计算输出形状
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        shape_out = np.shape(q) + shape_out  # 在q的形状前加上计算得到的输出形状

        out = np.empty(shape_out)  # 创建一个空数组作为输出
        result = np.nanpercentile(d, q, axis=axis, keepdims=True, out=out)
        assert result is out  # 断言返回的结果与指定的输出数组相同
        assert_equal(result.shape, shape_out)  # 断言返回的结果的形状与预期的形状相同

    @pytest.mark.parametrize("weighted", [False, True])
    # 定义一个测试方法，用于测试特定条件下的函数行为
    def test_out(self, weighted):
        # 创建一个 3x3 的随机数矩阵
        mat = np.random.rand(3, 3)
        # 在矩阵中插入 NaN 值，构成一个包含 NaN 的矩阵
        nan_mat = np.insert(mat, [0, 2], np.nan, axis=1)
        # 创建一个长度为 3 的零向量，用于存储结果
        resout = np.zeros(3)
        # 根据权重条件选择参数
        if weighted:
            # 如果使用权重，定义带有权重和方法的参数字典
            w_args = {"weights": np.ones_like(mat), "method": "inverted_cdf"}
            nan_w_args = {
                "weights": np.ones_like(nan_mat), "method": "inverted_cdf"
            }
        else:
            # 否则，参数字典为空
            w_args = dict()
            nan_w_args = dict()
        # 计算 mat 矩阵的百分位数，返回目标数组
        tgt = np.percentile(mat, 42, axis=1, **w_args)
        # 计算 nan_mat 矩阵的带有 NaN 的百分位数，将结果存储到 resout 中
        res = np.nanpercentile(nan_mat, 42, axis=1, out=resout, **nan_w_args)
        # 断言结果近似相等
        assert_almost_equal(res, resout)
        # 断言结果近似相等
        assert_almost_equal(res, tgt)
        # 处理 0 维输出的情况：
        resout = np.zeros(())
        # 计算 mat 矩阵的全局百分位数，返回目标值
        tgt = np.percentile(mat, 42, axis=None, **w_args)
        # 计算 nan_mat 矩阵的带有 NaN 的全局百分位数，将结果存储到 resout 中
        res = np.nanpercentile(
            nan_mat, 42, axis=None, out=resout, **nan_w_args
        )
        # 断言结果近似相等
        assert_almost_equal(res, resout)
        # 断言结果近似相等
        assert_almost_equal(res, tgt)
        # 计算 nan_mat 矩阵在多轴 (0, 1) 上的带有 NaN 的百分位数，将结果存储到 resout 中
        res = np.nanpercentile(
            nan_mat, 42, axis=(0, 1), out=resout, **nan_w_args
        )
        # 断言结果近似相等
        assert_almost_equal(res, resout)
        # 断言结果近似相等
        assert_almost_equal(res, tgt)

    # 定义一个测试复杂情况的方法
    def test_complex(self):
        # 创建一个复数数组，测试在复数数组上调用 nanpercentile 会引发 TypeError
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='G')
        assert_raises(TypeError, np.nanpercentile, arr_c, 0.5)
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='D')
        assert_raises(TypeError, np.nanpercentile, arr_c, 0.5)
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='F')
        assert_raises(TypeError, np.nanpercentile, arr_c, 0.5)

    # 使用 pytest 的参数化装饰器，定义测试不同参数组合下的函数行为
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("use_out", [False, True])
    def test_result_values(self, weighted, use_out):
        # 根据 weighted 参数选择相应的百分位数函数和生成权重的函数
        if weighted:
            percentile = partial(np.percentile, method="inverted_cdf")
            nanpercentile = partial(np.nanpercentile, method="inverted_cdf")

            def gen_weights(d):
                return np.ones_like(d)

        else:
            percentile = np.percentile
            nanpercentile = np.nanpercentile

            def gen_weights(d):
                return None

        # 对给定数据集 _rdat 计算目标百分位数，并存储到 tgt 中
        tgt = [percentile(d, 28, weights=gen_weights(d)) for d in _rdat]
        # 根据 use_out 参数决定是否使用 out 参数
        out = np.empty_like(tgt) if use_out else None
        # 计算 _ndat 数据集的带有 NaN 的百分位数，存储到 res 中
        res = nanpercentile(_ndat, 28, axis=1,
                            weights=gen_weights(_ndat), out=out)
        # 断言结果近似相等
        assert_almost_equal(res, tgt)
        # 将结果数组转置以符合 numpy.percentile 的输出约定
        tgt = np.transpose([percentile(d, (28, 98), weights=gen_weights(d))
                            for d in _rdat])
        # 根据 use_out 参数决定是否使用 out 参数
        out = np.empty_like(tgt) if use_out else None
        # 计算 _ndat 数据集的带有 NaN 的多轴 (1, 2) 百分位数，存储到 res 中
        res = nanpercentile(_ndat, (28, 98), axis=1,
                            weights=gen_weights(_ndat), out=out)
        # 断言结果近似相等
        assert_almost_equal(res, tgt)

    # 使用 pytest 的参数化装饰器，测试不同的轴和浮点数类型
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["Float"])
    # 使用 pytest 的参数化装饰器，为测试方法 test_allnans 提供不同的输入数组
    @pytest.mark.parametrize("array", [
        # 创建一个包含单个 NaN 值的 NumPy 数组
        np.array(np.nan),
        # 创建一个 3x3 的 NumPy 数组，每个元素都是 NaN
        np.full((3, 3), np.nan),
    ], ids=["0d", "2d"])
    # 测试所有元素为 NaN 的情况
    def test_allnans(self, axis, dtype, array):
        # 如果指定了 axis 且数组维度为 0，则跳过测试，并给出相应提示
        if axis is not None and array.ndim == 0:
            pytest.skip(f"`axis != None` not supported for 0d arrays")

        # 将数组转换为指定的数据类型
        array = array.astype(dtype)
        # 在执行计算时，捕获所有的 RuntimeWarning，其中包含 All-NaN slice encountered 的警告
        with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
            # 计算数组的第 60 百分位数，可以指定计算的轴
            out = np.nanpercentile(array, 60, axis=axis)
        # 断言计算结果中所有元素都是 NaN
        assert np.isnan(out).all()
        # 断言计算结果的数据类型与原始数组的数据类型相同
        assert out.dtype == array.dtype

    # 测试空数组的情况
    def test_empty(self):
        # 创建一个空的 0x3 的 NumPy 数组
        mat = np.zeros((0, 3))
        # 分别测试 axis 为 0 和 None 的情况
        for axis in [0, None]:
            # 在测试期间捕获所有警告
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                # 断言对空数组计算百分位数时所有元素都是 NaN
                assert_(np.isnan(np.nanpercentile(mat, 40, axis=axis)).all())
                # 断言捕获到一条警告
                assert_(len(w) == 1)
                # 断言该警告是 RuntimeWarning 类型的
                assert_(issubclass(w[0].category, RuntimeWarning))
        # 对于 axis 为 1 的情况，不应有警告产生
        for axis in [1]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                # 断言对空数组在 axis=1 上计算百分位数的结果是一个空的数组
                assert_equal(np.nanpercentile(mat, 40, axis=axis), np.zeros([]))
                # 断言没有捕获到任何警告
                assert_(len(w) == 0)

    # 测试标量输入的情况
    def test_scalar(self):
        # 断言对标量 0 计算百分位数得到的结果是 0
        assert_equal(np.nanpercentile(0., 100), 0.)
        # 创建一个 0 到 5 的数组
        a = np.arange(6)
        # 计算数组的第 50 百分位数
        r = np.nanpercentile(a, 50, axis=0)
        # 断言计算结果为 2.5
        assert_equal(r, 2.5)
        # 断言计算结果是一个标量
        assert_(np.isscalar(r))

    # 测试扩展轴参数无效的情况
    def test_extended_axis_invalid(self):
        # 创建一个形状为 (3, 5, 7, 11) 的全为 1 的数组
        d = np.ones((3, 5, 7, 11))
        # 断言对于超出范围的轴索引会抛出 AxisError
        assert_raises(AxisError, np.nanpercentile, d, q=5, axis=-5)
        # 断言对于同时指定有效和无效轴索引的情况会抛出 AxisError
        assert_raises(AxisError, np.nanpercentile, d, q=5, axis=(0, -5))
        # 断言对于超出范围的轴索引会抛出 AxisError
        assert_raises(AxisError, np.nanpercentile, d, q=5, axis=4)
        # 断言对于同时指定有效和超出范围的轴索引的情况会抛出 AxisError
        assert_raises(AxisError, np.nanpercentile, d, q=5, axis=(0, 4))
        # 断言当指定轴索引为重复时会抛出 ValueError
        assert_raises(ValueError, np.nanpercentile, d, q=5, axis=(1, 1))
    # 定义一个测试方法，用于测试多个百分位数的计算
    def test_multiple_percentiles(self):
        # 设定百分位数的列表
        perc = [50, 100]
        # 创建一个4x3的全1矩阵
        mat = np.ones((4, 3))
        # 创建一个与mat相同大小的NaN矩阵
        nan_mat = np.nan * mat
        # 在更高维度情况下检查一致性
        large_mat = np.ones((3, 4, 5))
        # 将large_mat的第1维和第3维的特定切片置为0
        large_mat[:, 0:2:4, :] = 0
        # 将large_mat的第3维后的所有元素乘以2
        large_mat[:, :, 3:] *= 2

        # 遍历不同的轴和保持维度的选项
        for axis in [None, 0, 1]:
            for keepdim in [False, True]:
                # 使用suppress_warnings上下文管理器以过滤特定的运行时警告
                with suppress_warnings() as sup:
                    # 过滤掉特定的运行时警告信息
                    sup.filter(RuntimeWarning, "All-NaN slice encountered")
                    # 计算mat的百分位数，返回值val
                    val = np.percentile(mat, perc, axis=axis, keepdims=keepdim)
                    # 计算nan_mat的百分位数，返回值nan_val
                    nan_val = np.nanpercentile(nan_mat, perc, axis=axis,
                                               keepdims=keepdim)
                    # 断言nan_val的形状与val的形状相同
                    assert_equal(nan_val.shape, val.shape)

                    # 计算large_mat的百分位数，返回值val
                    val = np.percentile(large_mat, perc, axis=axis,
                                        keepdims=keepdim)
                    # 计算large_mat中NaN值排除后的百分位数，返回值nan_val
                    nan_val = np.nanpercentile(large_mat, perc, axis=axis,
                                               keepdims=keepdim)
                    # 断言nan_val等于val
                    assert_equal(nan_val, val)

        # 创建一个更大的矩阵megamat，形状为3x4x5x6
        megamat = np.ones((3, 4, 5, 6))
        # 断言计算megamat在指定轴(1, 2)上的NaN值排除后的百分位数的形状
        assert_equal(
            np.nanpercentile(megamat, perc, axis=(1, 2)).shape, (2, 3, 6)
        )

    # 使用pytest的参数化标记定义一个测试方法，用于测试带有权重的NaN值处理
    @pytest.mark.parametrize("nan_weight", [0, 1, 2, 3, 1e200])
    def test_nan_value_with_weight(self, nan_weight):
        # 创建一个包含NaN的列表x
        x = [1, np.nan, 2, 3]
        # 预期的非NaN位置上的结果
        result = np.float64(2.0)
        # 计算未加权情况下的百分位数，返回值q_unweighted
        q_unweighted = np.nanpercentile(x, 50, method="inverted_cdf")
        # 断言q_unweighted等于预期结果result
        assert_equal(q_unweighted, result)

        # 创建一个权重列表w，在NaN位置处的权重值为nan_weight
        w = [1.0, nan_weight, 1.0, 1.0]
        # 计算带权重情况下的百分位数，返回值q_weighted
        q_weighted = np.nanpercentile(x, 50, weights=w, method="inverted_cdf")
        # 断言q_weighted等于预期结果result
        assert_equal(q_weighted, result)
    # 定义一个测试方法，用于测试带有权重和多维数组的 NaN 值处理
    def test_nan_value_with_weight_ndim(self, axis):
        # 创建一个多维数组进行测试
        np.random.seed(1)
        x_no_nan = np.random.random(size=(100, 99, 2))
        
        # 将部分位置设置为 NaN（不是特别聪明的做法），以确保始终存在非 NaN 值
        x = x_no_nan.copy()
        x[np.arange(99), np.arange(99), 0] = np.nan

        # 设置权重为全 1 数组，但在下面的 NaN 位置用 0 或 1e200 替换
        weights = np.ones_like(x)

        # 对比使用带有 NaN 权重的加权正常百分位，其中 NaN 位置的权重为 0（没有 NaN）
        weights[np.isnan(x)] = 0
        p_expected = np.percentile(
            x_no_nan, p, axis=axis, weights=weights, method="inverted_cdf")

        # 使用 np.nanpercentile 计算未加权的百分位
        p_unweighted = np.nanpercentile(
            x, p, axis=axis, method="inverted_cdf")
        
        # 正常版本和未加权版本应该是相同的：
        assert_equal(p_unweighted, p_expected)

        # 将 NaN 位置的权重设置为 1e200（一个很大的值，不应影响结果）
        weights[np.isnan(x)] = 1e200
        p_weighted = np.nanpercentile(
            x, p, axis=axis, weights=weights, method="inverted_cdf")
        
        # 断言加权版本的结果与预期结果相等
        assert_equal(p_weighted, p_expected)

        # 还可以传递输出数组进行检查：
        out = np.empty_like(p_weighted)
        res = np.nanpercentile(
            x, p, axis=axis, weights=weights, out=out, method="inverted_cdf")
        
        # 断言结果数组是传递的输出数组，并且其内容与预期结果相等
        assert res is out
        assert_equal(out, p_expected)
class TestNanFunctions_Quantile:
    # most of this is already tested by TestPercentile

    @pytest.mark.parametrize("weighted", [False, True])
    def test_regression(self, weighted):
        # 创建一个3维的浮点数数组，形状为(2, 3, 4)，数值为0到23
        ar = np.arange(24).reshape(2, 3, 4).astype(float)
        # 将第一个子数组的第二个子数组全部设为NaN
        ar[0][1] = np.nan
        # 根据weighted参数设置权重参数w_args
        if weighted:
            w_args = {"weights": np.ones_like(ar), "method": "inverted_cdf"}
        else:
            w_args = dict()

        # 断言np.nanquantile和np.nanpercentile的结果相等
        assert_equal(np.nanquantile(ar, q=0.5, **w_args),
                     np.nanpercentile(ar, q=50, **w_args))
        assert_equal(np.nanquantile(ar, q=0.5, axis=0, **w_args),
                     np.nanpercentile(ar, q=50, axis=0, **w_args))
        assert_equal(np.nanquantile(ar, q=0.5, axis=1, **w_args),
                     np.nanpercentile(ar, q=50, axis=1, **w_args))
        assert_equal(np.nanquantile(ar, q=[0.5], axis=1, **w_args),
                     np.nanpercentile(ar, q=[50], axis=1, **w_args))
        assert_equal(np.nanquantile(ar, q=[0.25, 0.5, 0.75], axis=1, **w_args),
                     np.nanpercentile(ar, q=[25, 50, 75], axis=1, **w_args))

    def test_basic(self):
        # 创建一个包含8个元素的浮点数数组
        x = np.arange(8) * 0.5
        # 断言np.nanquantile的结果与预期相等
        assert_equal(np.nanquantile(x, 0), 0.)
        assert_equal(np.nanquantile(x, 1), 3.5)
        assert_equal(np.nanquantile(x, 0.5), 1.75)

    def test_complex(self):
        # 创建一个复数数组，包含三个复数元素
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='G')
        # 断言对于复数数组，调用np.nanquantile会引发TypeError异常
        assert_raises(TypeError, np.nanquantile, arr_c, 0.5)
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='D')
        assert_raises(TypeError, np.nanquantile, arr_c, 0.5)
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='F')
        assert_raises(TypeError, np.nanquantile, arr_c, 0.5)

    def test_no_p_overwrite(self):
        # 这个测试值得重新测试，因为quantile函数不会创建副本
        p0 = np.array([0, 0.75, 0.25, 0.5, 1.0])
        p = p0.copy()
        # 调用np.nanquantile，验证参数p是否被修改
        np.nanquantile(np.arange(100.), p, method="midpoint")
        assert_array_equal(p, p0)

        p0 = p0.tolist()
        p = p.tolist()
        # 调用np.nanquantile，验证参数p是否被修改
        np.nanquantile(np.arange(100.), p, method="midpoint")
        assert_array_equal(p, p0)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["Float"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),
        np.full((3, 3), np.nan),
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            # 对于0维数组，不支持axis参数不为None的情况，跳过测试
            pytest.skip(f"`axis != None` not supported for 0d arrays")

        # 将array转换为指定dtype
        array = array.astype(dtype)
        # 断言调用np.nanquantile后，返回的结果全为NaN，并且dtype与array一致
        with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
            out = np.nanquantile(array, 1, axis=axis)
        assert np.isnan(out).all()
        assert out.dtype == array.dtype
    (np.array([1, 5, 7, 9], dtype=np.int64),
     True),
    # 创建一个包含整数的一维数组，数据类型为64位整数，不包含 NaN
    (np.array([False, True, False, True]),
     True),
    # 创建一个包含布尔值的一维数组，所有值都是布尔类型，不包含 NaN
    (np.array([[np.nan, 5.0],
               [np.nan, np.inf]], dtype=np.complex64),
     np.array([[False, True],
               [False, True]])),
    # 创建一个包含复数的二维数组，数据类型为64位复数，包含 NaN 和无穷大（inf）
    # 同时创建一个布尔类型的二维数组，标识对应位置是否包含 NaN 或无穷大
    ])
# 测试函数，验证 _nan_mask 函数的行为是否符合预期
def test__nan_mask(arr, expected):
    # 针对两种输出情况进行循环测试
    for out in [None, np.empty(arr.shape, dtype=np.bool)]:
        # 调用 _nan_mask 函数计算实际结果
        actual = _nan_mask(arr, out=out)
        # 断言实际结果与期望结果相等
        assert_equal(actual, expected)
        # 如果期望结果不是 np.ndarray 类型，则需进一步验证 actual 是否为 True
        # 用于无法包含 NaN 的数据类型，确保 actual 是 True 而非 True 数组
        if type(expected) is not np.ndarray:
            assert actual is True


# 测试函数，验证 _replace_nan 函数的不同数据类型情况下的行为
def test__replace_nan():
    """ Test that _replace_nan returns the original array if there are no
    NaNs, not a copy.
    """
    # 针对不同的数据类型进行测试
    for dtype in [np.bool, np.int32, np.int64]:
        # 创建指定类型的数组 arr
        arr = np.array([0, 1], dtype=dtype)
        # 调用 _replace_nan 函数，替换 NaN，并获取结果及 mask
        result, mask = _replace_nan(arr, 0)
        # 断言 mask 为 None，表明没有 NaN 存在时不进行复制操作
        assert mask is None
        # 断言 result 与 arr 是同一个对象，即不进行复制
        assert result is arr

    # 针对浮点类型进行测试
    for dtype in [np.float32, np.float64]:
        # 创建指定类型的数组 arr
        arr = np.array([0, 1], dtype=dtype)
        # 调用 _replace_nan 函数，替换 NaN，并获取结果及 mask
        result, mask = _replace_nan(arr, 2)
        # 断言 mask 全为 False，表明没有 NaN 存在时不进行复制操作
        assert (mask == False).all()
        # 断言 result 不是 arr，表明需要进行复制操作
        assert result is not arr
        # 断言 result 与 arr 的内容相等
        assert_equal(result, arr)

        # 创建包含 NaN 的数组 arr_nan
        arr_nan = np.array([0, 1, np.nan], dtype=dtype)
        # 调用 _replace_nan 函数，替换 NaN，并获取结果及 mask
        result_nan, mask_nan = _replace_nan(arr_nan, 2)
        # 断言 mask_nan 的值与预期一致
        assert_equal(mask_nan, np.array([False, False, True]))
        # 断言 result_nan 不是 arr_nan，表明需要进行复制操作
        assert result_nan is not arr_nan
        # 断言 result_nan 的内容与预期一致
        assert_equal(result_nan, np.array([0, 1, 2]))
        # 断言 arr_nan 最后一个元素仍然为 NaN
        assert np.isnan(arr_nan[-1])
```