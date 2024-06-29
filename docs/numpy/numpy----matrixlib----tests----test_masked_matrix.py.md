# `.\numpy\numpy\matrixlib\tests\test_masked_matrix.py`

```
import pickle  # 导入 pickle 库，用于对象的序列化和反序列化

import numpy as np  # 导入 NumPy 库，进行科学计算
from numpy.testing import assert_warns  # 导入 NumPy 测试模块中的 assert_warns 函数
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
                                assert_array_equal)  # 导入 NumPy 测试工具中的断言函数
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
                           MaskType, getmask, MaskedArray, nomask,
                           log, add, hypot, divide)  # 导入 NumPy 掩码数组相关的函数和类
from numpy.ma.extras import mr_  # 导入 NumPy 掩码数组额外的函数


class MMatrix(MaskedArray, np.matrix,):  # 定义 MMatrix 类，继承自 MaskedArray 和 np.matrix

    def __new__(cls, data, mask=nomask):
        mat = np.matrix(data)  # 将输入数据转换为 np.matrix 对象
        _data = MaskedArray.__new__(cls, data=mat, mask=mask)  # 创建 MaskedArray 对象
        return _data

    def __array_finalize__(self, obj):
        np.matrix.__array_finalize__(self, obj)  # 在创建后处理 np.matrix 对象
        MaskedArray.__array_finalize__(self, obj)  # 在创建后处理 MaskedArray 对象
        return

    @property
    def _series(self):
        _view = self.view(MaskedArray)  # 创建一个 MaskedArray 的视图
        _view._sharedmask = False  # 设置共享掩码为 False
        return _view  # 返回视图对象


class TestMaskedMatrix:
    def test_matrix_indexing(self):
        # Tests conversions and indexing
        x1 = np.matrix([[1, 2, 3], [4, 3, 2]])  # 创建一个 np.matrix 对象
        x2 = masked_array(x1, mask=[[1, 0, 0], [0, 1, 0]])  # 创建一个掩码数组对象 x2
        x3 = masked_array(x1, mask=[[0, 1, 0], [1, 0, 0]])  # 创建一个掩码数组对象 x3
        x4 = masked_array(x1)  # 创建一个掩码数组对象 x4
        # test conversion to strings
        str(x2)  # 转换为字符串，可能会引发异常
        repr(x2)  # 返回对象的字符串表示，可能会引发异常
        # tests of indexing
        assert_(type(x2[1, 0]) is type(x1[1, 0]))  # 断言 x2 和 x1 在指定位置上的类型相同
        assert_(x1[1, 0] == x2[1, 0])  # 断言 x1 和 x2 在指定位置上的值相等
        assert_(x2[1, 1] is masked)  # 断言 x2 在指定位置上是否是掩码
        assert_equal(x1[0, 2], x2[0, 2])  # 断言 x1 和 x2 在指定位置上的值相等
        assert_equal(x1[0, 1:], x2[0, 1:])  # 断言 x1 和 x2 在指定范围上的值相等
        assert_equal(x1[:, 2], x2[:, 2])  # 断言 x1 和 x2 在指定列上的值相等
        assert_equal(x1[:], x2[:])  # 断言 x1 和 x2 的值相等
        assert_equal(x1[1:], x3[1:])  # 断言 x1 和 x3 在指定范围上的值相等
        x1[0, 2] = 9  # 修改 x1 的值
        x2[0, 2] = 9  # 修改 x2 的值
        assert_equal(x1, x2)  # 断言 x1 和 x2 的值相等
        x1[0, 1:] = 99  # 修改 x1 的值
        x2[0, 1:] = 99  # 修改 x2 的值
        assert_equal(x1, x2)  # 断言 x1 和 x2 的值相等
        x2[0, 1] = masked  # 在 x2 中指定位置设置为掩码
        assert_equal(x1, x2)  # 断言 x1 和 x2 的值相等
        x2[0, 1:] = masked  # 在 x2 中指定范围设置为掩码
        assert_equal(x1, x2)  # 断言 x1 和 x2 的值相等
        x2[0, :] = x1[0, :]  # 修改 x2 的部分值
        x2[0, 1] = masked  # 在 x2 中指定位置设置为掩码
        assert_(allequal(getmask(x2), np.array([[0, 1, 0], [0, 1, 0]])))  # 断言 x2 的掩码是否与指定数组相等
        x3[1, :] = masked_array([1, 2, 3], [1, 1, 0])  # 修改 x3 的部分值
        assert_(allequal(getmask(x3)[1], masked_array([1, 1, 0])))  # 断言 x3 的掩码是否与指定数组相等
        assert_(allequal(getmask(x3[1]), masked_array([1, 1, 0])))  # 断言 x3 的掩码是否与指定数组相等
        x4[1, :] = masked_array([1, 2, 3], [1, 1, 0])  # 修改 x4 的部分值
        assert_(allequal(getmask(x4[1]), masked_array([1, 1, 0])))  # 断言 x4 的掩码是否与指定数组相等
        assert_(allequal(x4[1], masked_array([1, 2, 3])))  # 断言 x4 在指定位置上的值是否与指定数组相等
        x1 = np.matrix(np.arange(5) * 1.0)  # 创建一个新的 np.matrix 对象
        x2 = masked_values(x1, 3.0)  # 使用指定值创建掩码数组对象
        assert_equal(x1, x2)  # 断言 x1 和 x2 的值相等
        assert_(allequal(masked_array([0, 0, 0, 1, 0], dtype=MaskType),
                         x2.mask))  # 断言 x2 的掩码是否与指定数组相等
        assert_equal(3.0, x2.fill_value)  # 断言 x2 的填充值是否等于指定值
    # 定义一个测试方法，用于测试带有子类 ndarray 的序列化
    def test_pickling_subbaseclass(self):
        # 创建一个带有掩码的 masked_array 对象，其中包含一个 10 个元素的矩阵
        a = masked_array(np.matrix(list(range(10))), mask=[1, 0, 1, 0, 0] * 2)
        # 对不同协议版本进行循环测试序列化和反序列化操作
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 序列化对象 a，并使用指定的协议版本进行反序列化
            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
            # 断言反序列化后的对象的掩码与原始对象的掩码相同
            assert_equal(a_pickled._mask, a._mask)
            # 断言反序列化后的对象与原始对象相等
            assert_equal(a_pickled, a)
            # 断言反序列化后的对象的 _data 属性是一个 np.matrix 类型的对象
            assert_(isinstance(a_pickled._data, np.matrix))

    # 定义一个测试方法，测试带有矩阵的 masked_array 对象的计数和平均值计算
    def test_count_mean_with_matrix(self):
        # 创建一个带有矩阵和全零掩码的 masked_array 对象
        m = masked_array(np.matrix([[1, 2], [3, 4]]), mask=np.zeros((2, 2)))

        # 断言沿着 axis=0 方向计数的结果形状为 (1, 2)
        assert_equal(m.count(axis=0).shape, (1, 2))
        # 断言沿着 axis=1 方向计数的结果形状为 (2, 1)
        assert_equal(m.count(axis=1).shape, (2, 1))

        # 确保在 mean 和 var 方法内部进行广播计算正常工作
        assert_equal(m.mean(axis=0), [[2., 3.]])
        assert_equal(m.mean(axis=1), [[1.5], [3.5]])

    # 定义一个测试方法，测试 flat 属性在矩阵中的使用情况
    def test_flat(self):
        # 创建一个带有矩阵和部分掩码的 masked_array 对象，用于测试 flat 属性
        test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        # 断言通过 flat 访问第二个元素返回正确的值 2
        assert_equal(test.flat[1], 2)
        # 断言通过 flat 访问第三个元素返回 masked
        assert_equal(test.flat[2], masked)
        # 断言通过 flat 访问切片 [0:2] 返回与指定范围的元素相等的结果
        assert_(np.all(test.flat[0:2] == test[0, 0:2]))

        # 创建另一个带有矩阵和部分掩码的 masked_array 对象，并修改其 flat 属性
        test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        test.flat = masked_array([3, 2, 1], mask=[1, 0, 0])
        # 创建一个作为对照的 masked_array 对象
        control = masked_array(np.matrix([[3, 2, 1]]), mask=[1, 0, 0])
        # 断言修改 flat 属性后的对象与对照对象相等
        assert_equal(test, control)

        # 再次创建带有矩阵和部分掩码的 masked_array 对象，测试 flat 属性的设置
        test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        # 获取 flat 属性的引用
        testflat = test.flat
        # 使用切片操作修改 flat 属性的值，使其与 control 对象相等
        testflat[:] = testflat[[2, 1, 0]]
        # 断言修改后的对象与对照对象相等
        assert_equal(test, control)

        # 对 flat 属性的单独元素进行修改测试
        testflat[0] = 9
        # 断言修改后的对象与 control 对象依然相等
        # 测试矩阵保持正确的形状（#4615）
        a = masked_array(np.matrix(np.eye(2)), mask=0)
        b = a.flat
        b01 = b[:2]
        # 断言修改后的对象数据与预期值相等
        assert_equal(b01.data, np.array([[1., 0.]]))
        # 断言修改后的对象掩码与预期值相等
        assert_equal(b01.mask, np.array([[False, False]]))
    def test_allany_onmatrices(self):
        # 创建一个 NumPy 数组 x
        x = np.array([[0.13, 0.26, 0.90],
                      [0.28, 0.33, 0.63],
                      [0.31, 0.87, 0.70]])
        # 将数组 x 转换为 NumPy 矩阵 X
        X = np.matrix(x)
        # 创建一个布尔类型的 NumPy 数组 m
        m = np.array([[True, False, False],
                      [False, False, False],
                      [True, True, False]], dtype=np.bool)
        # 使用 m 创建一个 MaskedArray 对象 mX
        mX = masked_array(X, mask=m)
        # 创建两个布尔类型的 MaskedArray 对象 mXbig 和 mXsmall
        mXbig = (mX > 0.5)
        mXsmall = (mX < 0.5)

        # 断言：mXbig 中不是所有元素都大于 0.5
        assert_(not mXbig.all())
        # 断言：mXbig 中至少有一个元素大于 0.5
        assert_(mXbig.any())
        # 断言：mXbig 沿着列的维度上所有元素都大于 0.5
        assert_equal(mXbig.all(0), np.matrix([False, False, True]))
        # 断言：mXbig 沿着行的维度上所有元素都大于 0.5
        assert_equal(mXbig.all(1), np.matrix([False, False, True]).T)
        # 断言：mXbig 沿着列的维度上至少有一个元素大于 0.5
        assert_equal(mXbig.any(0), np.matrix([False, False, True]))
        # 断言：mXbig 沿着行的维度上至少有一个元素大于 0.5
        assert_equal(mXbig.any(1), np.matrix([True, True, True]).T)

        # 断言：mXsmall 中不是所有元素都小于 0.5
        assert_(not mXsmall.all())
        # 断言：mXsmall 中至少有一个元素小于 0.5
        assert_(mXsmall.any())
        # 断言：mXsmall 沿着列的维度上所有元素都小于 0.5
        assert_equal(mXsmall.all(0), np.matrix([True, True, False]))
        # 断言：mXsmall 沿着行的维度上所有元素都小于 0.5
        assert_equal(mXsmall.all(1), np.matrix([False, False, False]).T)
        # 断言：mXsmall 沿着列的维度上至少有一个元素小于 0.5
        assert_equal(mXsmall.any(0), np.matrix([True, True, False]))
        # 断言：mXsmall 沿着行的维度上至少有一个元素小于 0.5
        assert_equal(mXsmall.any(1), np.matrix([True, True, False]).T)

    def test_compressed(self):
        # 创建一个 NumPy 矩阵 a，并为其创建一个 MaskedArray 对象 b
        a = masked_array(np.matrix([1, 2, 3, 4]), mask=[0, 0, 0, 0])
        # 使用 compressed 方法压缩 MaskedArray 对象 a，得到 b
        b = a.compressed()
        # 断言：压缩后的 b 应当与原始的 a 相等
        assert_equal(b, a)
        # 断言：b 应当是一个 NumPy 矩阵
        assert_(isinstance(b, np.matrix))
        # 修改 a 中的一个元素为 masked，并重新压缩得到 b
        a[0, 0] = masked
        b = a.compressed()
        # 断言：压缩后的 b 应当是 [[2, 3, 4]]
        assert_equal(b, [[2, 3, 4]])

    def test_ravel(self):
        # 创建一个 NumPy 矩阵 a，并为其创建一个 MaskedArray 对象 aravel
        a = masked_array(np.matrix([1, 2, 3, 4, 5]), mask=[[0, 1, 0, 0, 0]])
        aravel = a.ravel()
        # 断言：aravel 应当是一个形状为 (1, 5) 的矩阵
        assert_equal(aravel.shape, (1, 5))
        # 断言：aravel 的掩码应当与 a 的形状相同
        assert_equal(aravel._mask.shape, a.shape)

    def test_view(self):
        # 测试带有灵活数据类型的视图功能
        iterator = list(zip(np.arange(10), np.random.rand(10)))
        data = np.array(iterator)
        # 创建一个 MaskedArray 对象 a，使用自定义的数据类型 [('a', float), ('b', float)] 
        a = masked_array(iterator, dtype=[('a', float), ('b', float)])
        # 将 a 的第一个元素的掩码设置为 (1, 0)
        a.mask[0] = (1, 0)
        # 使用 view 方法创建一个新的 NumPy 矩阵 test
        test = a.view((float, 2), np.matrix)
        # 断言：test 应当与原始数据 data 相等
        assert_equal(test, data)
        # 断言：test 应当是一个 NumPy 矩阵
        assert_(isinstance(test, np.matrix))
        # 断言：test 不是一个 MaskedArray 对象
        assert_(not isinstance(test, MaskedArray))
class TestSubclassing:
    # Test suite for masked subclasses of ndarray.

    def setup_method(self):
        # 创建一个长度为5的浮点数 ndarray
        x = np.arange(5, dtype='float')
        # 使用 MMatrix 类型创建一个带有掩码的 ndarray
        mx = MMatrix(x, mask=[0, 1, 0, 0, 0])
        # 将 x 和 mx 存储到实例变量 self.data 中
        self.data = (x, mx)

    def test_maskedarray_subclassing(self):
        # Tests subclassing MaskedArray
        (x, mx) = self.data
        # 检查 mx._data 是否为 np.matrix 类型
        assert_(isinstance(mx._data, np.matrix))

    def test_masked_unary_operations(self):
        # Tests masked_unary_operation
        (x, mx) = self.data
        # 在忽略除法时进行操作
        with np.errstate(divide='ignore'):
            # 检查 log(mx) 是否为 MMatrix 类型
            assert_(isinstance(log(mx), MMatrix))
            # 检查 log(x) 是否与 np.log(x) 相等
            assert_equal(log(x), np.log(x))

    def test_masked_binary_operations(self):
        # Tests masked_binary_operation
        (x, mx) = self.data
        # 结果应为 MMatrix 类型
        assert_(isinstance(add(mx, mx), MMatrix))
        assert_(isinstance(add(mx, x), MMatrix))
        # 检查 add(mx, x) 是否等于 mx+x
        assert_equal(add(mx, x), mx+x)
        # 检查 add(mx, mx)._data 是否为 np.matrix 类型
        assert_(isinstance(add(mx, mx)._data, np.matrix))
        # 检查 add.outer(mx, mx) 是否为 MMatrix 类型，并发出 DeprecationWarning
        with assert_warns(DeprecationWarning):
            assert_(isinstance(add.outer(mx, mx), MMatrix))
        # 检查 hypot(mx, mx) 和 hypot(mx, x) 是否为 MMatrix 类型
        assert_(isinstance(hypot(mx, mx), MMatrix))
        assert_(isinstance(hypot(mx, x), MMatrix))

    def test_masked_binary_operations2(self):
        # Tests domained_masked_binary_operation
        (x, mx) = self.data
        # 使用 mx.data 创建一个带有掩码的 masked_array 对象
        xmx = masked_array(mx.data.__array__(), mask=mx.mask)
        # 检查 divide(mx, mx) 和 divide(mx, x) 是否为 MMatrix 类型
        assert_(isinstance(divide(mx, mx), MMatrix))
        assert_(isinstance(divide(mx, x), MMatrix))
        # 检查 divide(mx, mx) 和 divide(xmx, xmx) 是否相等
        assert_equal(divide(mx, mx), divide(xmx, xmx))

class TestConcatenator:
    # Tests for mr_, the equivalent of r_ for masked arrays.

    def test_matrix_builder(self):
        # 测试 mr_['1, 2; 3, 4'] 是否引发 np.ma.MAError 异常
        assert_raises(np.ma.MAError, lambda: mr_['1, 2; 3, 4'])

    def test_matrix(self):
        # Test consistency with unmasked version.  If we ever deprecate
        # matrix, this test should either still pass, or both actual and
        # expected should fail to be build.
        # 检查 mr_['r', 1, 2, 3] 是否与预期的 np.ma.array(np.r_['r', 1, 2, 3]) 一致
        actual = mr_['r', 1, 2, 3]
        expected = np.ma.array(np.r_['r', 1, 2, 3])
        assert_array_equal(actual, expected)

        # outer type is masked array, inner type is matrix
        # 检查 actual 和 expected 的类型是否一致，并且它们的数据类型也一致
        assert_equal(type(actual), type(expected))
        assert_equal(type(actual.data), type(expected.data))
```