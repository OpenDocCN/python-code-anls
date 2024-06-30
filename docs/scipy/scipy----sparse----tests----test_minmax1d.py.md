# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_minmax1d.py`

```
"""Test of min-max 1D features of sparse array classes"""

# 导入pytest库，用于测试
import pytest

# 导入numpy库，并简化命名为np，用于数值计算
import numpy as np

# 从numpy.testing中导入断言函数，用于测试时比较数组是否相等
from numpy.testing import assert_equal, assert_array_equal

# 从scipy.sparse库中导入稀疏数组的不同类型，如coo_array, csr_array等
from scipy.sparse import coo_array, csr_array, csc_array, bsr_array
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, bsr_matrix

# 导入_scipy.sparse._sputils模块中的isscalarlike函数，用于判断对象是否类似于标量
from scipy.sparse._sputils import isscalarlike


# 定义一个函数toarray，用于将输入转换为numpy数组（如果已经是数组或者类似标量则直接返回）
def toarray(a):
    if isinstance(a, np.ndarray) or isscalarlike(a):
        return a
    return a.toarray()


# 定义一个列表formats_for_minmax，包含了用于测试的稀疏数组类型，如bsr_array, coo_array等
formats_for_minmax = [bsr_array, coo_array, csc_array, csr_array]

# 定义一个列表formats_for_minmax_supporting_1d，包含了支持一维数组操作的稀疏数组类型，如coo_array, csr_array等
formats_for_minmax_supporting_1d = [coo_array, csr_array]

# 使用pytest的parametrize装饰器，为每个稀疏数组类型spcreator创建一个测试类Test_MinMaxMixin1D
@pytest.mark.parametrize("spcreator", formats_for_minmax_supporting_1d)
class Test_MinMaxMixin1D:
    # 定义一个测试方法test_minmax，测试稀疏数组的最大最小值计算功能
    def test_minmax(self, spcreator):
        # 创建一个numpy数组D，包含0到4的整数
        D = np.arange(5)
        # 使用稀疏数组类型spcreator创建一个稀疏数组X
        X = spcreator(D)

        # 断言稀疏数组X的最小值为0
        assert_equal(X.min(), 0)
        # 断言稀疏数组X的最大值为4
        assert_equal(X.max(), 4)
        # 断言稀疏数组-X的最小值为-4
        assert_equal((-X).min(), -4)
        # 断言稀疏数组-X的最大值为0
        assert_equal((-X).max(), 0)

    # 定义一个测试方法test_minmax_axis，测试稀疏数组在指定轴上的最大最小值计算功能
    def test_minmax_axis(self, spcreator):
        # 创建一个包含0到49的numpy数组D
        D = np.arange(50)
        # 使用稀疏数组类型spcreator创建一个稀疏数组X
        X = spcreator(D)

        # 遍历轴列表[0, -1]
        for axis in [0, -1]:
            # 断言稀疏数组X在指定轴上的最大值与numpy数组D在相同轴上的最大值相等
            assert_array_equal(
                toarray(X.max(axis=axis)), D.max(axis=axis, keepdims=True)
            )
            # 断言稀疏数组X在指定轴上的最小值与numpy数组D在相同轴上的最小值相等
            assert_array_equal(
                toarray(X.min(axis=axis)), D.min(axis=axis, keepdims=True)
            )
        
        # 遍历轴列表[-2, 1]
        for axis in [-2, 1]:
            # 使用pytest的raises断言，验证在指定轴上计算最小值和最大值时抛出ValueError异常
            with pytest.raises(ValueError, match="axis out of range"):
                X.min(axis=axis)
            with pytest.raises(ValueError, match="axis out of range"):
                X.max(axis=axis)

    # 定义一个测试方法test_numpy_minmax，测试稀疏数组与numpy函数在最大最小值计算上的一致性
    def test_numpy_minmax(self, spcreator):
        # 创建一个包含[0, 1, 2]的numpy数组dat
        dat = np.array([0, 1, 2])
        # 使用稀疏数组类型spcreator创建一个稀疏数组datsp
        datsp = spcreator(dat)
        # 断言稀疏数组datsp与numpy数组dat在最小值上相等
        assert_array_equal(np.min(datsp), np.min(dat))
        # 断言稀疏数组datsp与numpy数组dat在最大值上相等
        assert_array_equal(np.max(datsp), np.max(dat))

    # 定义一个测试方法test_argmax，测试稀疏数组在argmax和argmin计算功能上的正确性
    def test_argmax(self, spcreator):
        # 创建多个包含不同数据的numpy数组D1到D5
        D1 = np.array([-1, 5, 2, 3])
        D2 = np.array([0, 0, -1, -2])
        D3 = np.array([-1, -2, -3, -4])
        D4 = np.array([1, 2, 3, 4])
        D5 = np.array([1, 2, 0, 0])

        # 遍历所有numpy数组D1到D5
        for D in [D1, D2, D3, D4, D5]:
            # 使用稀疏数组类型spcreator创建一个稀疏数组mat
            mat = spcreator(D)

            # 断言稀疏数组mat的argmax结果与numpy数组D的argmax结果相等
            assert_equal(mat.argmax(), np.argmax(D))
            # 断言稀疏数组mat的argmin结果与numpy数组D的argmin结果相等
            assert_equal(mat.argmin(), np.argmin(D))

            # 断言稀疏数组mat在轴0上的argmax结果与numpy数组D在轴0上的argmax结果相等
            assert_equal(mat.argmax(axis=0), np.argmax(D, axis=0))
            # 断言稀疏数组mat在轴0上的argmin结果与numpy数组D在轴0上的argmin结果相等
            assert_equal(mat.argmin(axis=0), np.argmin(D, axis=0))

        # 创建一个空的numpy数组D6
        D6 = np.empty((0,))

        # 遍历轴列表[None, 0]
        for axis in [None, 0]:
            # 使用稀疏数组类型spcreator创建一个稀疏数组mat
            mat = spcreator(D6)
            # 使用pytest的raises断言，验证在空数组上计算argmin和argmax时抛出ValueError异常
            with pytest.raises(ValueError, match="to an empty matrix"):
                mat.argmin(axis=axis)
            with pytest.raises(ValueError, match="to an empty matrix"):
                mat.argmax(axis=axis)


# 使用pytest的parametrize装饰器，为每个稀疏数组类型spcreator创建一个测试类Test_ShapeMinMax2DWithAxis
@pytest.mark.parametrize("spcreator", formats_for_minmax)
class Test_ShapeMinMax2DWithAxis:
    # 定义一个测试函数，用于测试稀疏矩阵的最小最大值函数
    def test_minmax(self, spcreator):
        # 创建一个包含整数的 NumPy 数组
        dat = np.array([[-1, 5, 0, 3], [0, 0, -1, -2], [0, 0, 1, 2]])
        # 使用给定的 spcreator 函数创建稀疏矩阵
        datsp = spcreator(dat)

        # 对每个稀疏矩阵的最小最大值函数进行测试
        for (spminmax, npminmax) in [
            (datsp.min, np.min),         # 测试稀疏矩阵的最小值函数与 NumPy 的最小值函数对比
            (datsp.max, np.max),         # 测试稀疏矩阵的最大值函数与 NumPy 的最大值函数对比
            (datsp.nanmin, np.nanmin),   # 测试稀疏矩阵的忽略 NaN 值的最小值函数与 NumPy 的对比
            (datsp.nanmax, np.nanmax),   # 测试稀疏矩阵的忽略 NaN 值的最大值函数与 NumPy 的对比
        ]:
            # 对每个轴进行测试
            for ax, result_shape in [(0, (4,)), (1, (3,))]:
                # 断言稀疏矩阵函数在指定轴上的结果与 NumPy 函数的结果一致
                assert_equal(toarray(spminmax(axis=ax)), npminmax(dat, axis=ax))
                # 断言稀疏矩阵函数在指定轴上的形状与预期形状一致
                assert_equal(spminmax(axis=ax).shape, result_shape)
                # 断言稀疏矩阵函数返回的对象格式为 "coo"
                assert spminmax(axis=ax).format == "coo"

        # 对稀疏矩阵的 argmin 和 argmax 函数进行测试
        for spminmax in [datsp.argmin, datsp.argmax]:
            # 对每个轴进行测试
            for ax in [0, 1]:
                # 断言稀疏矩阵函数返回的对象为 NumPy 数组
                assert isinstance(spminmax(axis=ax), np.ndarray)

        # 验证稀疏矩阵的行为
        spmat_form = {
            'coo': coo_matrix,
            'csr': csr_matrix,
            'csc': csc_matrix,
            'bsr': bsr_matrix,
        }
        # 使用相应的格式化函数将原始数组转换为稀疏矩阵
        datspm = spmat_form[datsp.format](dat)

        # 对每个稀疏矩阵的最小最大值函数进行测试
        for spm, npm in [
            (datspm.min, np.min),         # 测试稀疏矩阵的最小值函数与 NumPy 的最小值函数对比
            (datspm.max, np.max),         # 测试稀疏矩阵的最大值函数与 NumPy 的最大值函数对比
            (datspm.nanmin, np.nanmin),   # 测试稀疏矩阵的忽略 NaN 值的最小值函数与 NumPy 的对比
            (datspm.nanmax, np.nanmax),   # 测试稀疏矩阵的忽略 NaN 值的最大值函数与 NumPy 的对比
        ]:
            # 对每个轴进行测试
            for ax, result_shape in [(0, (1, 4)), (1, (3, 1))]:
                # 断言稀疏矩阵函数在指定轴上的结果与 NumPy 函数的结果一致，保持维度为 True
                assert_equal(toarray(spm(axis=ax)), npm(dat, axis=ax, keepdims=True))
                # 断言稀疏矩阵函数在指定轴上的形状与预期形状一致
                assert_equal(spm(axis=ax).shape, result_shape)
                # 断言稀疏矩阵函数返回的对象格式为 "coo"
                assert spm(axis=ax).format == "coo"

        # 对稀疏矩阵的 argmin 和 argmax 函数进行测试
        for spminmax in [datspm.argmin, datspm.argmax]:
            # 对每个轴进行测试
            for ax in [0, 1]:
                # 断言稀疏矩阵函数返回的对象为 NumPy 数组
                assert isinstance(spminmax(axis=ax), np.ndarray)
```