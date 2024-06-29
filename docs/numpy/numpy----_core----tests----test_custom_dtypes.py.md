# `.\numpy\numpy\_core\tests\test_custom_dtypes.py`

```py
import sys  # 导入sys模块，用于系统相关操作

from tempfile import NamedTemporaryFile  # 导入NamedTemporaryFile类，用于创建临时文件

import pytest  # 导入pytest模块，用于编写和运行测试用例

import numpy as np  # 导入NumPy库，用于科学计算
from numpy.testing import assert_array_equal  # 导入assert_array_equal函数，用于比较NumPy数组是否相等
from numpy._core._multiarray_umath import (  # 导入NumPy内部函数
    _discover_array_parameters as discover_array_params, _get_sfloat_dtype)

SF = _get_sfloat_dtype()  # 获取特定类型的浮点数作为SF的常量

class TestSFloat:
    def _get_array(self, scaling, aligned=True):
        if not aligned:
            a = np.empty(3*8 + 1, dtype=np.uint8)[1:]  # 创建一个未对齐的NumPy数组
            a = a.view(np.float64)  # 将数组视图转换为np.float64类型
            a[:] = [1., 2., 3.]  # 数组赋值为[1., 2., 3.]
        else:
            a = np.array([1., 2., 3.])  # 创建一个包含[1., 2., 3.]的NumPy数组

        a *= 1./scaling  # 数组中每个元素乘以scaling的倒数
        return a.view(SF(scaling))  # 返回数组的SF类型视图

    def test_sfloat_rescaled(self):
        sf = SF(1.)  # 创建SF对象，缩放因子为1.0
        sf2 = sf.scaled_by(2.)  # 使用sf对象的scaled_by方法创建sf2对象，缩放因子为2.0
        assert sf2.get_scaling() == 2.  # 断言sf2对象的缩放因子为2.0
        sf6 = sf2.scaled_by(3.)  # 使用sf2对象的scaled_by方法创建sf6对象，缩放因子为6.0
        assert sf6.get_scaling() == 6.  # 断言sf6对象的缩放因子为6.0

    def test_class_discovery(self):
        # 这个测试并不多，因为我们总是发现缩放因子为1.0。
        # 但是当写入时，大多数NumPy不理解DType类。
        dt, _ = discover_array_params([1., 2., 3.], dtype=SF)  # 调用discover_array_params函数获取数组参数
        assert dt == SF(1.)  # 断言dt对象为SF(1.)，即缩放因子为1.0的SF对象

    @pytest.mark.parametrize("scaling", [1., -1., 2.])
    def test_scaled_float_from_floats(self, scaling):
        a = np.array([1., 2., 3.], dtype=SF(scaling))  # 创建指定缩放因子的SF类型的NumPy数组

        assert a.dtype.get_scaling() == scaling  # 断言数组的dtype的缩放因子为scaling
        assert_array_equal(scaling * a.view(np.float64), [1., 2., 3.])  # 断言scaling乘以数组视图的结果与[1., 2., 3.]相等

    def test_repr(self):
        # 检查repr，主要是为了覆盖代码路径：
        assert repr(SF(scaling=1.)) == "_ScaledFloatTestDType(scaling=1.0)"  # 断言SF对象的repr为"_ScaledFloatTestDType(scaling=1.0)"

    def test_dtype_name(self):
        assert SF(1.).name == "_ScaledFloatTestDType64"  # 断言SF(1.)的名称为"_ScaledFloatTestDType64"

    def test_sfloat_structured_dtype_printing(self):
        dt = np.dtype([("id", int), ("value", SF(0.5))])  # 创建包含SF(0.5)的结构化dtype对象
        # 结构化dtype的repr需要特殊处理，因为实现绕过了对象repr
        assert "('value', '_ScaledFloatTestDType64')" in repr(dt)  # 断言结构化dtype的repr包含"('value', '_ScaledFloatTestDType64')"

    @pytest.mark.parametrize("scaling", [1., -1., 2.])
    def test_sfloat_from_float(self, scaling):
        a = np.array([1., 2., 3.]).astype(dtype=SF(scaling))  # 创建指定缩放因子的SF类型的NumPy数组

        assert a.dtype.get_scaling() == scaling  # 断言数组的dtype的缩放因子为scaling
        assert_array_equal(scaling * a.view(np.float64), [1., 2., 3.])  # 断言scaling乘以数组视图的结果与[1., 2., 3.]相等

    @pytest.mark.parametrize("aligned", [True, False])
    @pytest.mark.parametrize("scaling", [1., -1., 2.])
    def test_sfloat_getitem(self, aligned, scaling):
        a = self._get_array(1., aligned)  # 调用_get_array方法创建数组a
        assert a.tolist() == [1., 2., 3.]  # 断言数组a转换为列表后为[1., 2., 3.]

    @pytest.mark.parametrize("aligned", [True, False])
    # 测试单精度浮点数转换函数的功能，使用指定的对齐方式
    def test_sfloat_casts(self, aligned):
        # 获取一个包含单个浮点数的数组，使用给定的对齐方式
        a = self._get_array(1., aligned)

        # 检查是否可以将数组 a 转换为单精度浮点数 SF(-1.)，等效转换
        assert np.can_cast(a, SF(-1.), casting="equiv")
        # 检查是否不可以将数组 a 转换为单精度浮点数 SF(-1.)，禁止转换
        assert not np.can_cast(a, SF(-1.), casting="no")
        # 将数组 a 转换为单精度浮点数 SF(-1.) 的结果，且其负数值等效于 a 的浮点数视图
        na = a.astype(SF(-1.))
        assert_array_equal(-1 * na.view(np.float64), a.view(np.float64))

        # 检查是否可以将数组 a 转换为单精度浮点数 SF(2.)，同类型转换
        assert np.can_cast(a, SF(2.), casting="same_kind")
        # 检查是否不可以将数组 a 转换为单精度浮点数 SF(2.)，安全转换
        assert not np.can_cast(a, SF(2.), casting="safe")
        # 将数组 a 转换为单精度浮点数 SF(2.) 的结果，且其正数值等效于 a 的浮点数视图
        a2 = a.astype(SF(2.))
        assert_array_equal(2 * a2.view(np.float64), a.view(np.float64))

    # 使用参数化测试，测试单精度浮点数转换中的内部错误处理
    @pytest.mark.parametrize("aligned", [True, False])
    def test_sfloat_cast_internal_errors(self, aligned):
        # 获取一个包含非常大的浮点数的数组，使用给定的对齐方式
        a = self._get_array(2e300, aligned)

        # 测试在转换为单精度浮点数 SF(2e-300) 时是否引发 TypeError 异常，并匹配指定错误信息
        with pytest.raises(TypeError,
                match="error raised inside the core-loop: non-finite factor!"):
            a.astype(SF(2e-300))

    # 测试单精度浮点数的类型提升功能
    def test_sfloat_promotion(self):
        # 检查单精度浮点数 SF(2.) 和 SF(3.) 的类型提升结果是否为 SF(3.)
        assert np.result_type(SF(2.), SF(3.)) == SF(3.)
        # 检查单精度浮点数 SF(3.) 和 SF(2.) 的类型提升结果是否为 SF(3.)
        assert np.result_type(SF(3.), SF(2.)) == SF(3.)
        # 检查将浮点数类型 Float64 转换为单精度浮点数 SF(1.)，然后正常提升，两者结果相同
        assert np.result_type(SF(3.), np.float64) == SF(3.)
        assert np.result_type(np.float64, SF(0.5)) == SF(1.)

        # 测试未定义的类型提升
        with pytest.raises(TypeError):
            np.result_type(SF(1.), np.int64)

    # 测试基本的乘法操作
    def test_basic_multiply(self):
        # 获取两个包含浮点数的数组
        a = self._get_array(2.)
        b = self._get_array(4.)

        # 计算数组 a 和 b 的乘积
        res = a * b
        # 检查乘积的 dtype 缩放因子是否为 8.
        assert res.dtype.get_scaling() == 8.
        # 计算预期的浮点数视图乘积
        expected_view = a.view(np.float64) * b.view(np.float64)
        assert_array_equal(res.view(np.float64), expected_view)

    # 测试可能和不可能的约简操作
    def test_possible_and_impossible_reduce(self):
        # 对于约简操作，第一个和最后一个操作数必须具有相同的 dtype。
        a = self._get_array(2.)
        # 执行加法约简操作，需要指定初始值为 0.
        res = np.add.reduce(a, initial=0.)
        assert res == a.astype(np.float64).sum()

        # 由于每次乘法都会改变因子，因此不可能进行乘法约简操作
        with pytest.raises(TypeError,
                match="the resolved dtypes are not compatible"):
            np.multiply.reduce(a)

    # 测试基本的 ufunc 的 at 方法
    def test_basic_ufunc_at(self):
        # 创建一个浮点数数组 float_a 和 self._get_array 返回的数组 b
        float_a = np.array([1., 2., 3.])
        b = self._get_array(2.)

        # 创建 float_b 的副本，复制其浮点数视图
        float_b = b.view(np.float64).copy()
        # 使用 ufunc 的 at 方法，将 float_a 应用于 float_b 的指定索引位置
        np.multiply.at(float_b, [1, 1, 1], float_a)
        # 使用 ufunc 的 at 方法，将 float_a 应用于数组 b 的指定索引位置
        np.multiply.at(b, [1, 1, 1], float_a)

        # 检查修改后的 b 的浮点数视图是否与修改后的 float_b 的浮点数视图相等
        assert_array_equal(b.view(np.float64), float_b)
    def test_basic_multiply_promotion(self):
        float_a = np.array([1., 2., 3.])  # 创建一个包含浮点数的 NumPy 数组 float_a
        b = self._get_array(2.)  # 调用 self._get_array 方法获取一个数组 b，元素为浮点数

        res1 = float_a * b  # 数组 float_a 和数组 b 相乘的结果
        res2 = b * float_a  # 数组 b 和数组 float_a 相乘的结果

        # 检查结果数组的数据类型是否与数组 b 的数据类型相同
        assert res1.dtype == res2.dtype == b.dtype
        expected_view = float_a * b.view(np.float64)
        assert_array_equal(res1.view(np.float64), expected_view)  # 检查结果数组视图与预期视图是否相等
        assert_array_equal(res2.view(np.float64), expected_view)  # 检查结果数组视图与预期视图是否相等

        # 检查使用 'out' 参数时类型提升是否有效
        np.multiply(b, float_a, out=res2)
        with pytest.raises(TypeError):
            # 该推广器接受此操作（也许不应该），但 SFloat 结果不能转换为整数：
            np.multiply(b, float_a, out=np.arange(3))

    def test_basic_addition(self):
        a = self._get_array(2.)  # 调用 self._get_array 方法获取一个数组 a，元素为浮点数
        b = self._get_array(4.)  # 调用 self._get_array 方法获取一个数组 b，元素为浮点数

        res = a + b  # 数组 a 和数组 b 相加的结果
        # 加法使用结果的类型提升规则：
        assert res.dtype == np.result_type(a.dtype, b.dtype)
        expected_view = (a.astype(res.dtype).view(np.float64) +
                         b.astype(res.dtype).view(np.float64))
        assert_array_equal(res.view(np.float64), expected_view)  # 检查结果数组视图与预期视图是否相等

    def test_addition_cast_safety(self):
        """The addition method is special for the scaled float, because it
        includes the "cast" between different factors, thus cast-safety
        is influenced by the implementation.
        """
        a = self._get_array(2.)  # 调用 self._get_array 方法获取一个数组 a，元素为浮点数
        b = self._get_array(-2.)  # 调用 self._get_array 方法获取一个数组 b，元素为浮点数
        c = self._get_array(3.)  # 调用 self._get_array 方法获取一个数组 c，元素为浮点数

        # 符号变化是 "equiv"：
        np.add(a, b, casting="equiv")
        with pytest.raises(TypeError):
            np.add(a, b, casting="no")  # 使用不允许的类型转换引发异常

        # 不同的因子是 "same_kind"（默认），因此检查 "safe" 失败
        with pytest.raises(TypeError):
            np.add(a, c, casting="safe")

        # 检查输出类型转换也失败（由 ufunc 完成）
        with pytest.raises(TypeError):
            np.add(a, a, out=c, casting="safe")

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_casts_to_bool(self, ufunc):
        a = self._get_array(2.)  # 调用 self._get_array 方法获取一个数组 a，元素为浮点数
        a[0] = 0.  # 确保第一个元素被认为是 False

        float_equiv = a.astype(float)
        expected = ufunc(float_equiv, float_equiv)
        res = ufunc(a, a)
        assert_array_equal(res, expected)  # 检查逻辑操作的结果是否与预期相等

        # 检查对约简操作也是同样适用：
        expected = ufunc.reduce(float_equiv)
        res = ufunc.reduce(a)
        assert_array_equal(res, expected)  # 检查约简操作的结果是否与预期相等

        # 输出类型转换与 bool, bool -> bool 循环不匹配：
        with pytest.raises(TypeError):
            ufunc(a, a, out=np.empty(a.shape, dtype=int), casting="equiv")
    def test_wrapped_and_wrapped_reductions(self):
        # 创建一个数组 a，其中元素为浮点数，并将其视作等价的浮点数数组
        a = self._get_array(2.)
        float_equiv = a.astype(float)

        # 计算浮点数数组的各元素的直角三角形斜边长度，作为期望结果
        expected = np.hypot(float_equiv, float_equiv)
        # 计算数组 a 中各元素的直角三角形斜边长度
        res = np.hypot(a, a)
        # 断言结果数组的数据类型与数组 a 的数据类型相同
        assert res.dtype == a.dtype
        # 将结果数组视作 np.float64 类型，并乘以 2
        res_float = res.view(np.float64) * 2
        # 断言乘以 2 后的结果数组与期望结果数组相等
        assert_array_equal(res_float, expected)

        # 进行归约计算，保持维度（由于错误的获取项目）
        res = np.hypot.reduce(a, keepdims=True)
        # 断言归约结果数组的数据类型与数组 a 的数据类型相同
        assert res.dtype == a.dtype
        # 计算浮点数数组的归约结果，并保持维度
        expected = np.hypot.reduce(float_equiv, keepdims=True)
        # 断言乘以 2 后的归约结果数组与期望结果数组相等
        assert res.view(np.float64) * 2 == expected

    def test_astype_class(self):
        # 非常简单的测试，验证在类上也接受 `.astype()`
        # ScaledFloat 总是返回默认描述符，但它会检查相关的代码路径
        arr = np.array([1., 2., 3.], dtype=object)

        # 对数组进行类型转换，传入 SF 类
        res = arr.astype(SF)  # 传入类 class
        # 预期结果是经过 SF(1.) 类型转换后的数组
        expected = arr.astype(SF(1.))  # 上述代码将发现 1. 的缩放
        # 断言转换后的数组视作 np.float64 类型后与期望结果数组相等
        assert_array_equal(res.view(np.float64), expected.view(np.float64))

    def test_creation_class(self):
        # 传入 dtype 类应该返回默认的描述符
        arr1 = np.array([1., 2., 3.], dtype=SF)
        # 断言数组的 dtype 与 SF(1.) 相同
        assert arr1.dtype == SF(1.)
        arr2 = np.array([1., 2., 3.], dtype=SF(1.))
        # 断言转换后的数组视作 np.float64 类型后与 arr1 视作 np.float64 类型后相等
        assert_array_equal(arr1.view(np.float64), arr2.view(np.float64))
        # 断言 arr1 和 arr2 的 dtype 相同
        assert arr1.dtype == arr2.dtype

        # 测试创建空数组、空形同数组、全零数组、全零形同数组时的 dtype
        assert np.empty(3, dtype=SF).dtype == SF(1.)
        assert np.empty_like(arr1, dtype=SF).dtype == SF(1.)
        assert np.zeros(3, dtype=SF).dtype == SF(1.)
        assert np.zeros_like(arr1, dtype=SF).dtype == SF(1.)

    def test_np_save_load(self):
        # 必须进行这种猴子补丁，因为 pickle 使用类型的 repr 来重建它
        np._ScaledFloatTestDType = SF

        arr = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))

        # 从 np.save 测试中适配而来的代码，进行数组保存和加载的测试
        with NamedTemporaryFile("wb", delete=False, suffix=".npz") as f:
            with pytest.warns(UserWarning) as record:
                np.savez(f.name, arr)

        # 断言警告的数量为 1
        assert len(record) == 1

        # 使用 np.load 加载保存的数据
        with np.load(f.name, allow_pickle=True) as data:
            larr = data["arr_0"]
        # 断言加载后的数组视作 np.float64 类型后与原数组视作 np.float64 类型后相等
        assert_array_equal(arr.view(np.float64), larr.view(np.float64))
        # 断言加载后的数组的 dtype 与原数组的 dtype 与 SF(1.0) 相同
        assert larr.dtype == arr.dtype == SF(1.0)

        del np._ScaledFloatTestDType

    def test_flatiter(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))

        # 使用 flatiter 遍历数组的每个元素
        for i, val in enumerate(arr.flat):
            # 断言数组索引 i 处的元素与 flatiter 返回的值相等
            assert arr[i] == val

    @pytest.mark.parametrize(
        "index", [
            [1, 2], ..., slice(None, 2, None),
            np.array([True, True, False]), np.array([0, 1])
        ], ids=["int_list", "ellipsis", "slice", "bool_array", "int_array"])
    # 定义一个测试方法，用于测试 flatiter 的索引功能
    def test_flatiter_index(self, index):
        # 创建一个包含浮点数的 NumPy 数组，使用特定的类型SF(1.0)
        arr = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))
        # 断言：验证通过 flat 属性的索引得到的视图是否与直接索引得到的视图相等
        np.testing.assert_array_equal(
            arr[index].view(np.float64), arr.flat[index].view(np.float64))
    
        # 复制数组 arr 到 arr2
        arr2 = arr.copy()
        # 通过索引将 arr 中的元素设置为 5.0
        arr[index] = 5.0
        # 通过 flat 属性的索引将 arr2 中的元素设置为 5.0
        arr2.flat[index] = 5.0
        # 断言：验证修改后的 arr 和 arr2 是否相等
        np.testing.assert_array_equal(
            arr.view(np.float64), arr2.view(np.float64))
# 测试使用 pickle 模块处理 SF 对象的序列化和反序列化
def test_type_pickle():
    # 导入 pickle 模块
    import pickle

    # 将 SF 对象作为被序列化对象
    np._ScaledFloatTestDType = SF

    # 使用 pickle.dumps 将 SF 对象序列化为字节流 s
    s = pickle.dumps(SF)
    
    # 使用 pickle.loads 将字节流 s 反序列化为对象 res
    res = pickle.loads(s)
    
    # 断言反序列化后的对象 res 与原始 SF 对象相同
    assert res is SF

    # 删除测试时添加的 SF 对象属性
    del np._ScaledFloatTestDType


# 测试 SF 对象的 _is_numeric 属性是否为真
def test_is_numeric():
    # 断言 SF 对象的 _is_numeric 属性为真
    assert SF._is_numeric
```