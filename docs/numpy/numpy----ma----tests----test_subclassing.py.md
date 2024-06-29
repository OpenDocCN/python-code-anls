# `.\numpy\numpy\ma\tests\test_subclassing.py`

```
# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for MaskedArray & subclassing.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: test_subclassing.py 3473 2007-10-29 15:18:13Z jarrod.millman $

"""
# 导入 NumPy 库
import numpy as np
# 导入 NumPy 数组操作的混合类
from numpy.lib.mixins import NDArrayOperatorsMixin
# 导入 NumPy 测试工具
from numpy.testing import assert_, assert_raises
# 导入 NumPy 掩码数组相关测试工具
from numpy.ma.testutils import assert_equal
# 导入 NumPy 掩码数组核心功能
from numpy.ma.core import (
    array, arange, masked, MaskedArray, masked_array, log, add, hypot,
    divide, asarray, asanyarray, nomask
    )

# 定义一个自定义断言函数，用于比较字符串开头
def assert_startswith(a, b):
    # 比 assert_(a.startswith(b)) 提供更好的错误信息
    assert_equal(a[:len(b)], b)

# 定义一个 NumPy 的子类 SubArray，存储在字典 `info` 中的一些元数据
class SubArray(np.ndarray):
    def __new__(cls,arr,info={}):
        x = np.asanyarray(arr).view(cls)
        x.info = info.copy()
        return x

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.info = getattr(obj, 'info', {}).copy()
        return

    def __add__(self, other):
        result = super().__add__(other)
        result.info['added'] = result.info.get('added', 0) + 1
        return result

    def __iadd__(self, other):
        result = super().__iadd__(other)
        result.info['iadded'] = result.info.get('iadded', 0) + 1
        return result

# 创建一个别名 subarray 指向 SubArray 类
subarray = SubArray

# 定义一个纯粹的 MaskedArray 子类 SubMaskedArray，保留了一些子类信息
class SubMaskedArray(MaskedArray):
    def __new__(cls, info=None, **kwargs):
        obj = super().__new__(cls, **kwargs)
        obj._optinfo['info'] = info
        return obj

# 定义一个同时继承 SubArray 和 MaskedArray 的 MSubArray 子类
class MSubArray(SubArray, MaskedArray):

    def __new__(cls, data, info={}, mask=nomask):
        subarr = SubArray(data, info)
        _data = MaskedArray.__new__(cls, data=subarr, mask=mask)
        _data.info = subarr.info
        return _data

    @property
    def _series(self):
        _view = self.view(MaskedArray)
        _view._sharedmask = False
        return _view

# 创建一个别名 msubarray 指向 MSubArray 类
msubarray = MSubArray

# 定义 CSAIterator 类，用作扁平化迭代器对象
class CSAIterator:
    """
    Flat iterator object that uses its own setter/getter
    (works around ndarray.flat not propagating subclass setters/getters
    see https://github.com/numpy/numpy/issues/4564)
    roughly following MaskedIterator
    """
    def __init__(self, a):
        self._original = a
        self._dataiter = a.view(np.ndarray).flat

    def __iter__(self):
        return self

# 代码块结束
    # 定义特殊方法 __getitem__，用于实现索引操作
    def __getitem__(self, indx):
        # 从 _dataiter 中获取索引为 indx 的元素
        out = self._dataiter.__getitem__(indx)
        # 如果获取的元素不是 ndarray 类型，则将其转换为 ndarray 类型
        if not isinstance(out, np.ndarray):
            out = out.__array__()
        # 将 ndarray 类型的数据视图转换为原始类型 self._original
        out = out.view(type(self._original))
        # 返回转换后的数据
        return out

    # 定义特殊方法 __setitem__，用于实现设置索引操作
    def __setitem__(self, index, value):
        # 将 value 参数传递给 self._original._validate_input 方法进行验证和处理，然后设置到 self._dataiter 的 index 索引位置
        self._dataiter[index] = self._original._validate_input(value)

    # 定义特殊方法 __next__，用于实现迭代器的下一个元素获取操作
    def __next__(self):
        # 获取下一个迭代器 _dataiter 的元素，并将其转换为 ndarray 类型，再将其视图转换为原始类型 self._original
        return next(self._dataiter).__array__().view(type(self._original))
class TestSubclassing:
    # Test suite for masked subclasses of ndarray.

    def setup_method(self):
        # 设置测试方法的初始数据
        x = np.arange(5, dtype='float')
        # 创建一个包含浮点数的 ndarray
        mx = msubarray(x, mask=[0, 1, 0, 0, 0])
        # 创建一个带有掩码的子类数组
        self.data = (x, mx)

    def test_data_subclassing(self):
        # Tests whether the subclass is kept.
        # 测试子类是否被保留
        x = np.arange(5)
        # 创建一个普通的 ndarray
        m = [0, 0, 1, 0, 0]
        xsub = SubArray(x)
        # 使用 SubArray 创建一个子类数组
        xmsub = masked_array(xsub, mask=m)
        # 使用掩码创建一个 MaskedArray
        assert_(isinstance(xmsub, MaskedArray))
        # 断言 xmsub 是 MaskedArray 的实例
        assert_equal(xmsub._data, xsub)
        # 断言 xmsub 的 _data 属性与 xsub 相同
        assert_(isinstance(xmsub._data, SubArray))
        # 断言 xmsub 的 _data 属性是 SubArray 的实例
    # 测试 MaskedArray 的子类化
    def test_maskedarray_subclassing(self):
        # 获取测试数据中的 x 和 mx
        (x, mx) = self.data
        # 断言 mx._data 是 subarray 类的实例
        assert_(isinstance(mx._data, subarray))

    # 测试掩码一元操作
    def test_masked_unary_operations(self):
        # 获取测试数据中的 x 和 mx
        (x, mx) = self.data
        # 在忽略除法警告的上下文中
        with np.errstate(divide='ignore'):
            # 断言 log(mx) 是 msubarray 类的实例
            assert_(isinstance(log(mx), msubarray))
            # 断言 log(x) 等于 np.log(x)
            assert_equal(log(x), np.log(x))

    # 测试掩码二元操作
    def test_masked_binary_operations(self):
        # 获取测试数据中的 x 和 mx
        (x, mx) = self.data
        # 结果应该是 msubarray 类的实例
        assert_(isinstance(add(mx, mx), msubarray))
        assert_(isinstance(add(mx, x), msubarray))
        # 断言加法操作的结果正确
        assert_equal(add(mx, x), mx+x)
        assert_(isinstance(add(mx, mx)._data, subarray))
        assert_(isinstance(add.outer(mx, mx), msubarray))
        assert_(isinstance(hypot(mx, mx), msubarray))
        assert_(isinstance(hypot(mx, x), msubarray))

    # 测试 domained_masked_binary_operation
    def test_masked_binary_operations2(self):
        # 获取测试数据中的 x 和 mx
        (x, mx) = self.data
        # 创建 xmx 对象，使用 mx 的数据和掩码
        xmx = masked_array(mx.data.__array__(), mask=mx.mask)
        assert_(isinstance(divide(mx, mx), msubarray))
        assert_(isinstance(divide(mx, x), msubarray))
        # 断言 divide(mx, mx) 等于 divide(xmx, xmx)
        assert_equal(divide(mx, mx), divide(xmx, xmx))

    # 测试属性传播
    def test_attributepropagation(self):
        # 创建带有掩码的数组 x
        x = array(arange(5), mask=[0]+[1]*4)
        # 创建 masked_array my 和 msubarray ym
        my = masked_array(subarray(x))
        ym = msubarray(x)
        #
        z = (my+1)
        # 断言 z 是 MaskedArray 类的实例，但不是 MSubArray 的实例
        assert_(isinstance(z, MaskedArray))
        assert_(not isinstance(z, MSubArray))
        assert_(isinstance(z._data, SubArray))
        assert_equal(z._data.info, {})
        #
        z = (ym+1)
        # 断言 z 是 MaskedArray 和 MSubArray 的实例
        assert_(isinstance(z, MaskedArray))
        assert_(isinstance(z, MSubArray))
        assert_(isinstance(z._data, SubArray))
        assert_(z._data.info['added'] > 0)
        # 测试数据的原地方法是否被使用 (gh-4617)
        ym += 1
        assert_(isinstance(ym, MaskedArray))
        assert_(isinstance(ym, MSubArray))
        assert_(isinstance(ym._data, SubArray))
        assert_(ym._data.info['iadded'] > 0)
        #
        ym._set_mask([1, 0, 0, 0, 1])
        assert_equal(ym._mask, [1, 0, 0, 0, 1])
        ym._series._set_mask([0, 0, 0, 0, 1])
        assert_equal(ym._mask, [0, 0, 0, 0, 1])
        #
        xsub = subarray(x, info={'name':'x'})
        mxsub = masked_array(xsub)
        # 断言 mxsub 具有 'info' 属性，并且其值等于 xsub 的 'info'
        assert_(hasattr(mxsub, 'info'))
        assert_equal(mxsub.info, xsub.info)
    def test_subclasspreservation(self):
        # 检查使用 masked_array(...,subok=True) 是否能够保留类的类型。
        x = np.arange(5)  # 创建一个长度为5的 NumPy 数组 x
        m = [0, 0, 1, 0, 0]  # 创建一个掩码数组 m
        xinfo = [(i, j) for (i, j) in zip(x, m)]  # 创建一个包含元组的列表 xinfo，元组由 x 和 m 的对应元素组成
        xsub = MSubArray(x, mask=m, info={'xsub':xinfo})  # 创建一个自定义子类 MSubArray 的对象 xsub

        # 使用 subok=False 创建 masked_array 对象 mxsub，并进行断言检查
        mxsub = masked_array(xsub, subok=False)
        assert_(not isinstance(mxsub, MSubArray))  # 断言 mxsub 不是 MSubArray 类的实例
        assert_(isinstance(mxsub, MaskedArray))  # 断言 mxsub 是 MaskedArray 类的实例
        assert_equal(mxsub._mask, m)  # 断言 mxsub 的掩码与 m 相等

        # 使用 asarray(xsub) 创建 mxsub，并进行断言检查
        mxsub = asarray(xsub)
        assert_(not isinstance(mxsub, MSubArray))  # 断言 mxsub 不是 MSubArray 类的实例
        assert_(isinstance(mxsub, MaskedArray))  # 断言 mxsub 是 MaskedArray 类的实例
        assert_equal(mxsub._mask, m)  # 断言 mxsub 的掩码与 m 相等

        # 使用 subok=True 创建 masked_array 对象 mxsub，并进行断言检查
        mxsub = masked_array(xsub, subok=True)
        assert_(isinstance(mxsub, MSubArray))  # 断言 mxsub 是 MSubArray 类的实例
        assert_equal(mxsub.info, xsub.info)  # 断言 mxsub 的信息与 xsub 的信息相等
        assert_equal(mxsub._mask, xsub._mask)  # 断言 mxsub 的掩码与 xsub 的掩码相等

        # 使用 asanyarray(xsub) 创建 mxsub，并进行断言检查
        mxsub = asanyarray(xsub)
        assert_(isinstance(mxsub, MSubArray))  # 断言 mxsub 是 MSubArray 类的实例
        assert_equal(mxsub.info, xsub.info)  # 断言 mxsub 的信息与 xsub 的信息相等
        assert_equal(mxsub._mask, m)  # 断言 mxsub 的掩码与 m 相等

    def test_subclass_items(self):
        """测试 getter 和 setter 是否通过基类进行"""
        x = np.arange(5)  # 创建一个长度为5的 NumPy 数组 x
        xcsub = ComplicatedSubArray(x)  # 创建一个复杂子类 ComplicatedSubArray 的对象 xcsub
        mxcsub = masked_array(xcsub, mask=[True, False, True, False, False])  # 使用掩码创建 masked_array 对象 mxcsub

        # getter 应该返回 ComplicatedSubArray，即使是单个项
        # 首先检查我们正确地编写了 ComplicatedSubArray
        assert_(isinstance(xcsub[1], ComplicatedSubArray))  # 断言 xcsub[1] 是 ComplicatedSubArray 类的实例
        assert_(isinstance(xcsub[1,...], ComplicatedSubArray))  # 断言 xcsub[1,...] 是 ComplicatedSubArray 类的实例
        assert_(isinstance(xcsub[1:4], ComplicatedSubArray))  # 断言 xcsub[1:4] 是 ComplicatedSubArray 类的实例

        # 现在它在 MaskedArray 内部传播
        assert_(isinstance(mxcsub[1], ComplicatedSubArray))  # 断言 mxcsub[1] 是 ComplicatedSubArray 类的实例
        assert_(isinstance(mxcsub[1,...].data, ComplicatedSubArray))  # 断言 mxcsub[1,...].data 是 ComplicatedSubArray 类的实例
        assert_(mxcsub[0] is masked)  # 断言 mxcsub[0] 是 masked
        assert_(isinstance(mxcsub[0,...].data, ComplicatedSubArray))  # 断言 mxcsub[0,...].data 是 ComplicatedSubArray 类的实例
        assert_(isinstance(mxcsub[1:4].data, ComplicatedSubArray))  # 断言 mxcsub[1:4].data 是 ComplicatedSubArray 类的实例

        # 对扁平化版本进行同样的测试（通过 MaskedIterator）
        assert_(isinstance(mxcsub.flat[1].data, ComplicatedSubArray))  # 断言 mxcsub.flat[1].data 是 ComplicatedSubArray 类的实例
        assert_(mxcsub.flat[0] is masked)  # 断言 mxcsub.flat[0] 是 masked
        assert_(isinstance(mxcsub.flat[1:4].base, ComplicatedSubArray))  # 断言 mxcsub.flat[1:4].base 是 ComplicatedSubArray 类的实例

        # setter 只能接受 ComplicatedSubArray 类型的输入
        # 首先检查我们正确地编写了 ComplicatedSubArray
        assert_raises(ValueError, xcsub.__setitem__, 1, x[4])  # 断言对 xcsub[1] 设置 x[4] 会引发 ValueError
        # 现在它在 MaskedArray 内部传播
        assert_raises(ValueError, mxcsub.__setitem__, 1, x[4])  # 断言对 mxcsub[1] 设置 x[4] 会引发 ValueError
        assert_raises(ValueError, mxcsub.__setitem__, slice(1, 4), x[1:4])  # 断言对 mxcsub[1:4] 设置 x[1:4] 会引发 ValueError
        mxcsub[1] = xcsub[4]  # 设置 mxcsub[1] 为 xcsub[4]
        mxcsub[1:4] = xcsub[1:4]  # 设置 mxcsub[1:4] 为 xcsub[1:4]

        # 对扁平化版本进行同样的测试（通过 MaskedIterator）
        assert_raises(ValueError, mxcsub.flat.__setitem__, 1, x[4])  # 断言对 mxcsub.flat[1] 设置 x[4] 会引发 ValueError
        assert_raises(ValueError, mxcsub.flat.__setitem__, slice(1, 4), x[1:4])  # 断言对 mxcsub.flat[1:4] 设置 x[1:4] 会引发 ValueError
        mxcsub.flat[1] = xcsub[4]  # 设置 mxcsub.flat[1] 为 xcsub[4]
        mxcsub.flat[1:4] = xcsub[1:4]  # 设置 mxcsub.flat[1:4] 为 xcsub[1:4]
    def test_subclass_nomask_items(self):
        x = np.arange(5)
        xcsub = ComplicatedSubArray(x)  # 创建一个名为xcsub的ComplicatedSubArray对象，使用x作为数据
        mxcsub_nomask = masked_array(xcsub)  # 创建一个名为mxcsub_nomask的masked_array对象，使用xcsub作为数据

        assert_(isinstance(mxcsub_nomask[1,...].data, ComplicatedSubArray))  # 断言mxcsub_nomask的第二个元素数据为ComplicatedSubArray类型
        assert_(isinstance(mxcsub_nomask[0,...].data, ComplicatedSubArray))  # 断言mxcsub_nomask的第一个元素数据为ComplicatedSubArray类型

        assert_(isinstance(mxcsub_nomask[1], ComplicatedSubArray))  # 断言mxcsub_nomask的第二个元素为ComplicatedSubArray类型
        assert_(isinstance(mxcsub_nomask[0], ComplicatedSubArray))  # 断言mxcsub_nomask的第一个元素为ComplicatedSubArray类型

    def test_subclass_repr(self):
        """test that repr uses the name of the subclass
        and 'array' for np.ndarray"""
        x = np.arange(5)
        mx = masked_array(x, mask=[True, False, True, False, False])  # 创建一个名为mx的masked_array对象，使用x作为数据和指定的掩码

        assert_startswith(repr(mx), 'masked_array')  # 断言mx的repr字符串以'masked_array'开头
        xsub = SubArray(x)  # 创建一个名为xsub的SubArray对象，使用x作为数据
        mxsub = masked_array(xsub, mask=[True, False, True, False, False])  # 创建一个名为mxsub的masked_array对象，使用xsub作为数据和指定的掩码
        assert_startswith(repr(mxsub),
            f'masked_{SubArray.__name__}(data=[--, 1, --, 3, 4]')  # 断言mxsub的repr字符串以'masked_SubArray'开头，并包含指定格式的数据表示

    def test_subclass_str(self):
        """test str with subclass that has overridden str, setitem"""
        # first without override
        x = np.arange(5)
        xsub = SubArray(x)  # 创建一个名为xsub的SubArray对象，使用x作为数据
        mxsub = masked_array(xsub, mask=[True, False, True, False, False])  # 创建一个名为mxsub的masked_array对象，使用xsub作为数据和指定的掩码
        assert_equal(str(mxsub), '[-- 1 -- 3 4]')  # 断言mxsub的str字符串为指定格式的字符串表示

        xcsub = ComplicatedSubArray(x)  # 创建一个名为xcsub的ComplicatedSubArray对象，使用x作为数据
        assert_raises(ValueError, xcsub.__setitem__, 0,
                      np.ma.core.masked_print_option)  # 使用xcsub对象尝试调用__setitem__方法，并断言引发ValueError异常
        mxcsub = masked_array(xcsub, mask=[True, False, True, False, False])  # 创建一个名为mxcsub的masked_array对象，使用xcsub作为数据和指定的掩码
        assert_equal(str(mxcsub), 'myprefix [-- 1 -- 3 4] mypostfix')  # 断言mxcsub的str字符串为指定格式的字符串表示

    def test_pure_subclass_info_preservation(self):
        # Test that ufuncs and methods conserve extra information consistently;
        # see gh-7122.
        arr1 = SubMaskedArray('test', data=[1,2,3,4,5,6])  # 创建一个名为arr1的SubMaskedArray对象，使用指定的信息字符串和数据
        arr2 = SubMaskedArray(data=[0,1,2,3,4,5])  # 创建一个名为arr2的SubMaskedArray对象，使用数据作为数据
        diff1 = np.subtract(arr1, arr2)  # 使用np.subtract函数计算arr1和arr2的差异，并赋值给diff1
        assert_('info' in diff1._optinfo)  # 断言diff1的_optinfo字典中包含'info'键
        assert_(diff1._optinfo['info'] == 'test')  # 断言diff1的_optinfo字典中'info'键的值为'test'
        diff2 = arr1 - arr2  # 计算arr1和arr2的差异，并赋值给diff2
        assert_('info' in diff2._optinfo)  # 断言diff2的_optinfo字典中包含'info'键
        assert_(diff2._optinfo['info'] == 'test')  # 断言diff2的_optinfo字典中'info'键的值为'test'
class ArrayNoInheritance:
    """Quantity-like class that does not inherit from ndarray"""
    
    def __init__(self, data, units):
        # Initialize with data and units
        self.magnitude = data  # Store data in the magnitude attribute
        self.units = units  # Store units information
    
    def __getattr__(self, attr):
        # Delegate attribute access to self.magnitude
        return getattr(self.magnitude, attr)


def test_array_no_inheritance():
    # Create a masked array with data [1, 2, 3] and mask [True, False, True]
    data_masked = np.ma.array([1, 2, 3], mask=[True, False, True])
    # Create an instance of ArrayNoInheritance with the masked array and 'meters' units
    data_masked_units = ArrayNoInheritance(data_masked, 'meters')

    # Get the masked representation of the Quantity-like class
    new_array = np.ma.array(data_masked_units)
    assert_equal(data_masked.data, new_array.data)  # Check data equality
    assert_equal(data_masked.mask, new_array.mask)  # Check mask equality
    # Test sharing the mask
    data_masked.mask = [True, False, False]  # Change the mask of data_masked
    assert_equal(data_masked.mask, new_array.mask)  # Assert that new_array's mask also changes
    assert_(new_array.sharedmask)  # Assert that new_array shares its mask

    # Get the masked representation of the Quantity-like class with copy=True
    new_array = np.ma.array(data_masked_units, copy=True)
    assert_equal(data_masked.data, new_array.data)  # Check data equality
    assert_equal(data_masked.mask, new_array.mask)  # Check mask equality
    # Test that the mask is not shared when copy=True
    data_masked.mask = [True, False, True]  # Change the mask of data_masked
    assert_equal([True, False, False], new_array.mask)  # Assert new_array's mask remains unchanged
    assert_(not new_array.sharedmask)  # Assert new_array does not share its mask

    # Get the masked representation of the Quantity-like class with keep_mask=False
    new_array = np.ma.array(data_masked_units, keep_mask=False)
    assert_equal(data_masked.data, new_array.data)  # Check data equality
    # The change did not affect the original mask
    assert_equal(data_masked.mask, [True, False, True])  # Check original mask remains unchanged
    # Test that the mask is False and not shared when keep_mask=False
    assert_(not new_array.mask)  # Assert new_array's mask is False
    assert_(not new_array.sharedmask)  # Assert new_array does not share its mask


class TestClassWrapping:
    # Test suite for classes that wrap MaskedArrays

    def setup_method(self):
        # Setup method to create masked and wrapped arrays
        m = np.ma.masked_array([1, 3, 5], mask=[False, True, False])
        wm = WrappedArray(m)  # Assuming WrappedArray is defined elsewhere
        self.data = (m, wm)  # Store masked and wrapped arrays in self.data

    def test_masked_unary_operations(self):
        # Tests masked_unary_operation
        (m, wm) = self.data  # Unpack self.data into masked array (m) and wrapped array (wm)
        with np.errstate(divide='ignore'):
            assert_(isinstance(np.log(wm), WrappedArray))  # Check if np.log(wm) is instance of WrappedArray
    # 定义测试函数，用于测试掩码二进制操作
    def test_masked_binary_operations(self):
        # 测试 masked_binary_operation 函数
        (m, wm) = self.data
        # 断言结果应为 WrappedArray 类型
        assert_(isinstance(np.add(wm, wm), WrappedArray))
        assert_(isinstance(np.add(m, wm), WrappedArray))
        assert_(isinstance(np.add(wm, m), WrappedArray))
        # add 和 '+' 应调用相同的 ufunc
        assert_equal(np.add(m, wm), m + wm)
        assert_(isinstance(np.hypot(m, wm), WrappedArray))
        assert_(isinstance(np.hypot(wm, m), WrappedArray))
        # 测试包含定义域的二进制操作
        assert_(isinstance(np.divide(wm, m), WrappedArray))
        assert_(isinstance(np.divide(m, wm), WrappedArray))
        assert_equal(np.divide(wm, m) * m, np.divide(m, m) * wm)
        # 测试广播功能
        m2 = np.stack([m, m])
        assert_(isinstance(np.divide(wm, m2), WrappedArray))
        assert_(isinstance(np.divide(m2, wm), WrappedArray))
        assert_equal(np.divide(m2, wm), np.divide(wm, m2))

    # 定义测试函数，用于验证混合类是否具有 slots 属性
    def test_mixins_have_slots(self):
        mixin = NDArrayOperatorsMixin()
        # 应抛出 AttributeError 异常
        assert_raises(AttributeError, mixin.__setattr__, "not_a_real_attr", 1)

        m = np.ma.masked_array([1, 3, 5], mask=[False, True, False])
        wm = WrappedArray(m)
        # 应抛出 AttributeError 异常
        assert_raises(AttributeError, wm.__setattr__, "not_an_attr", 2)
```