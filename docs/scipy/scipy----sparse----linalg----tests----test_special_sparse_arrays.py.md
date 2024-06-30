# `D:\src\scipysrc\scipy\scipy\sparse\linalg\tests\test_special_sparse_arrays.py`

```
import pytest  # 导入 pytest 测试框架
import numpy as np  # 导入 NumPy 库并使用 np 别名
from numpy.testing import assert_array_equal, assert_allclose  # 从 NumPy 的 testing 模块导入两个断言函数

from scipy.sparse import diags, csgraph  # 从 SciPy 的 sparse 模块导入 diags 和 csgraph 子模块
from scipy.linalg import eigh  # 从 SciPy 的 linalg 模块导入 eigh 函数

from scipy.sparse.linalg import LaplacianNd  # 从 SciPy 的 sparse.linalg 模块导入 LaplacianNd 类
from scipy.sparse.linalg._special_sparse_arrays import Sakurai  # 导入特定的类 Sakurai
from scipy.sparse.linalg._special_sparse_arrays import MikotaPair  # 导入特定的类 MikotaPair

INT_DTYPES = [np.int8, np.int16, np.int32, np.int64]  # 整数类型的 NumPy 数据类型列表
REAL_DTYPES = [np.float32, np.float64]  # 浮点数类型的 NumPy 数据类型列表
COMPLEX_DTYPES = [np.complex64, np.complex128]  # 复数类型的 NumPy 数据类型列表
ALLDTYPES = INT_DTYPES + REAL_DTYPES + COMPLEX_DTYPES  # 所有数据类型的组合列表

class TestLaplacianNd:
    """
    LaplacianNd tests
    """

    @pytest.mark.parametrize('bc', ['neumann', 'dirichlet', 'periodic'])  # 参数化测试，参数为 bc，值为 'neumann', 'dirichlet', 'periodic'
    def test_1d_specific_shape(self, bc):
        lap = LaplacianNd(grid_shape=(6, ), boundary_conditions=bc)  # 创建 LaplacianNd 对象，传入网格形状和边界条件
        lapa = lap.toarray()  # 获取 LaplacianNd 对象的稀疏矩阵表示，并转换为密集数组
        if bc == 'neumann':
            a = np.array(
                [
                    [-1, 1, 0, 0, 0, 0],  # Neumann 边界条件下的 Laplacian 矩阵
                    [1, -2, 1, 0, 0, 0],
                    [0, 1, -2, 1, 0, 0],
                    [0, 0, 1, -2, 1, 0],
                    [0, 0, 0, 1, -2, 1],
                    [0, 0, 0, 0, 1, -1],
                ]
            )
        elif bc == 'dirichlet':
            a = np.array(
                [
                    [-2, 1, 0, 0, 0, 0],  # Dirichlet 边界条件下的 Laplacian 矩阵
                    [1, -2, 1, 0, 0, 0],
                    [0, 1, -2, 1, 0, 0],
                    [0, 0, 1, -2, 1, 0],
                    [0, 0, 0, 1, -2, 1],
                    [0, 0, 0, 0, 1, -2],
                ]
            )
        else:
            a = np.array(
                [
                    [-2, 1, 0, 0, 0, 1],  # 周期性边界条件下的 Laplacian 矩阵
                    [1, -2, 1, 0, 0, 0],
                    [0, 1, -2, 1, 0, 0],
                    [0, 0, 1, -2, 1, 0],
                    [0, 0, 0, 1, -2, 1],
                    [1, 0, 0, 0, 1, -2],
                ]
            )
        assert_array_equal(a, lapa)  # 断言 lap.toarray() 的结果与预期矩阵 a 相等

    def test_1d_with_graph_laplacian(self):
        n = 6  # 网格大小
        G = diags(np.ones(n - 1), 1, format='dia')  # 创建一个带状矩阵 G
        Lf = csgraph.laplacian(G, symmetrized=True, form='function')  # 使用函数形式计算图拉普拉斯矩阵
        La = csgraph.laplacian(G, symmetrized=True, form='array')  # 使用数组形式计算图拉普拉斯矩阵
        grid_shape = (n,)  # 网格形状
        bc = 'neumann'  # 边界条件为 Neumann
        lap = LaplacianNd(grid_shape, boundary_conditions=bc)  # 创建 LaplacianNd 对象
        assert_array_equal(lap(np.eye(n)), -Lf(np.eye(n)))  # 断言 LaplacianNd 对象应用于单位矩阵的结果与负的 Lf 应用结果相等
        assert_array_equal(lap.toarray(), -La.toarray())  # 断言 LaplacianNd 对象的稀疏矩阵表示与负的 La 的稀疏矩阵表示相等
        # https://github.com/numpy/numpy/issues/24351
        assert_array_equal(lap.tosparse().toarray(), -La.toarray())  # 断言 LaplacianNd 对象转换为稀疏矩阵后的密集数组与 La 的稀疏矩阵表示相等

    @pytest.mark.parametrize('grid_shape', [(6, ), (2, 3), (2, 3, 4)])  # 参数化测试，参数为 grid_shape，值为 (6,), (2,3), (2,3,4)
    @pytest.mark.parametrize('bc', ['neumann', 'dirichlet', 'periodic'])  # 参数化测试，参数为 bc，值为 'neumann', 'dirichlet', 'periodic'
    # 定义一个测试方法，用于测试 LaplacianNd 类的特征值功能
    def test_eigenvalues(self, grid_shape, bc):
        # 创建 LaplacianNd 类的实例，生成 Laplacian 矩阵，数据类型为 np.float64
        lap = LaplacianNd(grid_shape, boundary_conditions=bc, dtype=np.float64)
        # 将 Laplacian 矩阵转换为密集数组 L
        L = lap.toarray()
        # 使用 eigh 函数计算 Laplacian 矩阵 L 的特征值
        eigvals = eigh(L, eigvals_only=True)
        # 计算总元素个数 n
        n = np.prod(grid_shape)
        # 调用 LaplacianNd 实例的 eigenvalues 方法，获取其特征值
        eigenvalues = lap.eigenvalues()
        # 获取特征值数组的数据类型
        dtype = eigenvalues.dtype
        # 计算允许的数值误差阈值 atol
        atol = n * n * np.finfo(dtype).eps
        # 断言默认情况下特征值数组与 eigh 计算的特征值数组的吻合性
        assert_allclose(eigenvalues, eigvals, atol=atol)
        # 对每个 m > 0 进行测试
        for m in np.arange(1, n + 1):
            # 断言 LaplacianNd 实例的 eigenvalues(m) 方法与预期的特征值数组的部分吻合
            assert_array_equal(lap.eigenvalues(m), eigenvalues[-m:])

    # 使用 pytest 的参数化标记，定义测试方法，用于测试 LaplacianNd 类的特征向量功能
    @pytest.mark.parametrize('grid_shape', [(6, ), (2, 3), (2, 3, 4)])
    @pytest.mark.parametrize('bc', ['neumann', 'dirichlet', 'periodic'])
    def test_eigenvectors(self, grid_shape, bc):
        # 创建 LaplacianNd 类的实例，设置网格形状和边界条件，数据类型为 np.float64
        lap = LaplacianNd(grid_shape, boundary_conditions=bc, dtype=np.float64)
        # 计算总元素个数 n
        n = np.prod(grid_shape)
        # 调用 LaplacianNd 实例的 eigenvalues 方法，获取特征值数组
        eigenvalues = lap.eigenvalues()
        # 调用 LaplacianNd 实例的 eigenvectors 方法，获取特征向量矩阵
        eigenvectors = lap.eigenvectors()
        # 获取特征向量数组的数据类型
        dtype = eigenvectors.dtype
        # 计算允许的数值误差阈值 atol
        atol = n * n * max(np.finfo(dtype).eps, np.finfo(np.double).eps)
        # 对每个特征向量进行测试，验证 Laplacian 矩阵与特征向量的关系
        for i in np.arange(n):
            r = lap.toarray() @ eigenvectors[:, i] - eigenvectors[:, i] * eigenvalues[i]
            assert_allclose(r, np.zeros_like(r), atol=atol)
        # 对每个 m > 0 进行测试，验证 Laplacian 矩阵与特征向量的关系
        for m in np.arange(1, n + 1):
            e = lap.eigenvalues(m)
            ev = lap.eigenvectors(m)
            r = lap.toarray() @ ev - ev @ np.diag(e)
            assert_allclose(r, np.zeros_like(r), atol=atol)

    # 使用 pytest 的参数化标记，定义测试方法，验证 LaplacianNd 类中 toarray 方法与 tosparse 方法的一致性
    @pytest.mark.parametrize('grid_shape', [(6, ), (2, 3), (2, 3, 4)])
    @pytest.mark.parametrize('bc', ['neumann', 'dirichlet', 'periodic'])
    def test_toarray_tosparse_consistency(self, grid_shape, bc):
        # 创建 LaplacianNd 类的实例，设置网格形状和边界条件
        lap = LaplacianNd(grid_shape, boundary_conditions=bc)
        # 计算总元素个数 n
        n = np.prod(grid_shape)
        # 断言 Laplacian 矩阵转换为密集数组与使用单位矩阵计算结果的一致性
        assert_array_equal(lap.toarray(), lap(np.eye(n)))
        # 断言 Laplacian 矩阵转换为稀疏数组再转换为密集数组与直接转换为密集数组的一致性
        assert_array_equal(lap.tosparse().toarray(), lap.toarray())
    # 测试线性算子的形状和数据类型是否正确
    def test_linearoperator_shape_dtype(self, grid_shape, bc, dtype):
        # 创建一个 LaplacianNd 对象，指定网格形状和边界条件，并指定数据类型
        lap = LaplacianNd(grid_shape, boundary_conditions=bc, dtype=dtype)
        # 计算网格形状的总元素个数
        n = np.prod(grid_shape)
        # 断言 LaplacianNd 对象的形状为 (n, n)
        assert lap.shape == (n, n)
        # 断言 LaplacianNd 对象的数据类型为指定的 dtype
        assert lap.dtype == dtype
        # 断言转换为稀疏矩阵后的 LaplacianNd 对象与直接指定 dtype 的对比结果相等
        assert_array_equal(
            LaplacianNd(
                grid_shape, boundary_conditions=bc, dtype=dtype
            ).toarray(),
            LaplacianNd(grid_shape, boundary_conditions=bc)
            .toarray()
            .astype(dtype),
        )
        # 断言转换为稀疏矩阵后的 LaplacianNd 对象与直接指定 dtype 的对比结果相等
        assert_array_equal(
            LaplacianNd(grid_shape, boundary_conditions=bc, dtype=dtype)
            .tosparse()
            .toarray(),
            LaplacianNd(grid_shape, boundary_conditions=bc)
            .tosparse()
            .toarray()
            .astype(dtype),
        )

    @pytest.mark.parametrize('dtype', ALLDTYPES)
    @pytest.mark.parametrize('grid_shape', [(6, ), (2, 3), (2, 3, 4)])
    @pytest.mark.parametrize('bc', ['neumann', 'dirichlet', 'periodic'])
    # 测试 dot 乘积的类型保留和一致性
    def test_dot(self, grid_shape, bc, dtype):
        """ Test the dot-product for type preservation and consistency.
        """
        # 创建一个 LaplacianNd 对象，指定网格形状和边界条件
        lap = LaplacianNd(grid_shape, boundary_conditions=bc)
        # 计算网格形状的总元素个数
        n = np.prod(grid_shape)
        # 创建一个长度为 n 的等差数列数组
        x0 = np.arange(n)
        # 将 x0 重塑为 n 行 1 列的数组
        x1 = x0.reshape((-1, 1))
        # 创建一个形状为 (n, 2) 的等差数列数组
        x2 = np.arange(2 * n).reshape((n, 2))
        # 输入集合，包含 x0, x1, x2 三种不同形状的数组
        input_set = [x0, x1, x2]
        # 对于每个输入数组 x 进行测试
        for x in input_set:
            # 计算 LaplacianNd 对象与数组 x 的点乘结果，指定数据类型为 dtype
            y = lap.dot(x.astype(dtype))
            # 断言计算后的 y 的形状与 x 相同
            assert x.shape == y.shape
            # 断言计算后的 y 的数据类型为指定的 dtype
            assert y.dtype == dtype
            # 如果 x 是二维数组
            if x.ndim == 2:
                # 计算 LaplacianNd 对象转换为稀疏矩阵后与 x 点乘的结果
                yy = lap.toarray() @ x.astype(dtype)
                # 断言计算后的 yy 的数据类型为指定的 dtype
                assert yy.dtype == dtype
                # 断言 numpy 数组 y 与 yy 相等
                np.array_equal(y, yy)

    # 测试边界条件错误是否会引发 ValueError 异常
    def test_boundary_conditions_value_error(self):
        # 使用 pytest 断言检测是否会引发 ValueError 异常，异常消息中包含 "Unknown value 'robin'"
        with pytest.raises(ValueError, match="Unknown value 'robin'"):
            # 创建一个 LaplacianNd 对象，指定网格形状和未知的边界条件 'robin'
            LaplacianNd(grid_shape=(6, ), boundary_conditions='robin')
# 定义测试类 TestSakurai，用于测试 Sakurai 类的功能
class TestSakurai:
    """
    Sakurai tests
    """

    # 测试特定形状的 Sakurai 对象
    def test_specific_shape(self):
        # 创建一个 Sakurai 对象，参数为 6，用于后续测试
        sak = Sakurai(6)
        # 断言 Sakurai 对象的 toarray 方法返回与输入 np.eye(6) 相同的数组
        assert_array_equal(sak.toarray(), sak(np.eye(6)))
        
        # 定义一个 6x6 的 numpy 数组 a
        a = np.array(
            [
                [ 5, -4,  1,  0,  0,  0],
                [-4,  6, -4,  1,  0,  0],
                [ 1, -4,  6, -4,  1,  0],
                [ 0,  1, -4,  6, -4,  1],
                [ 0,  0,  1, -4,  6, -4],
                [ 0,  0,  0,  1, -4,  5]
            ]
        )

        # 断言 Sakurai 对象的 toarray 方法返回与数组 a 相同的数组
        np.array_equal(a, sak.toarray())
        
        # 断言 Sakurai 对象的 tosparse 方法返回的稀疏矩阵转换为数组后与 Sakurai 对象的 toarray 方法返回的数组相同
        np.array_equal(sak.tosparse().toarray(), sak.toarray())
        
        # 定义一个 3x6 的 numpy 数组 ab
        ab = np.array(
            [
                [ 1,  1,  1,  1,  1,  1],
                [-4, -4, -4, -4, -4, -4],
                [ 5,  6,  6,  6,  6,  5]
            ]
        )
        
        # 断言 Sakurai 对象的 tobanded 方法返回与数组 ab 相同的数组
        np.array_equal(ab, sak.tobanded())
        
        # 定义一个包含 6 个浮点数的 numpy 数组 e
        e = np.array(
                [0.03922866, 0.56703972, 2.41789479, 5.97822974,
                 10.54287655, 14.45473055]
            )
        
        # 断言 Sakurai 对象的 eigenvalues 方法返回与数组 e 相同的数组
        np.array_equal(e, sak.eigenvalues())
        
        # 断言 Sakurai 对象的 eigenvalues 方法返回的前两个元素与数组 e 的前两个元素相同
        np.array_equal(e[:2], sak.eigenvalues(2))

    # 使用参数化测试对 Sakurai 类的形状和数据类型进行测试
    @pytest.mark.parametrize('dtype', ALLDTYPES)
    def test_linearoperator_shape_dtype(self, dtype):
        # 设置矩阵维度为 7，创建一个具有指定数据类型的 Sakurai 对象
        n = 7
        sak = Sakurai(n, dtype=dtype)
        
        # 断言 Sakurai 对象的形状为 (7, 7)
        assert sak.shape == (n, n)
        
        # 断言 Sakurai 对象的数据类型与设定的 dtype 相同
        assert sak.dtype == dtype
        
        # 断言 Sakurai 对象转换为数组后与未指定数据类型的 Sakurai 对象转换为数组后再转换为指定数据类型 dtype 相同
        assert_array_equal(sak.toarray(), Sakurai(n).toarray().astype(dtype))
        
        # 断言 Sakurai 对象转换为稀疏矩阵后再转换为数组与未指定数据类型的 Sakurai 对象转换为稀疏矩阵后再转换为数组后再转换为指定数据类型 dtype 相同
        assert_array_equal(sak.tosparse().toarray(),
                           Sakurai(n).tosparse().toarray().astype(dtype))

    # 使用参数化测试对 Sakurai 类的 dot 方法进行测试
    @pytest.mark.parametrize('dtype', ALLDTYPES)
    @pytest.mark.parametrize('argument_dtype', ALLDTYPES)
    def test_dot(self, dtype, argument_dtype):
        """ Test the dot-product for type preservation and consistency.
        """
        # 根据参数设定推断结果数据类型
        result_dtype = np.promote_types(argument_dtype, dtype)
        
        # 设置矩阵维度为 5，创建一个 Sakurai 对象
        n = 5
        sak = Sakurai(n)
        
        # 创建不同维度的输入数组集合
        x0 = np.arange(n)
        x1 = x0.reshape((-1, 1))
        x2 = np.arange(2 * n).reshape((n, 2))
        input_set = [x0, x1, x2]
        
        # 对每个输入数组进行测试
        for x in input_set:
            # 对 Sakurai 对象的 dot 方法应用参数化数据类型后的输入数组 x，结果保存为 y
            y = sak.dot(x.astype(argument_dtype))
            
            # 断言 y 的形状与 x 相同
            assert x.shape == y.shape
            
            # 断言 y 的数据类型可以转换为预期结果数据类型 result_dtype
            assert np.can_cast(y.dtype, result_dtype)
            
            # 如果输入数组 x 的维度为 2
            if x.ndim == 2:
                # 计算标准矩阵乘积 sak.toarray() @ x.astype(argument_dtype)，并断言其与 y 相等
                ya = sak.toarray() @ x.astype(argument_dtype)
                np.array_equal(y, ya)
                
                # 断言 ya 的数据类型可以转换为预期结果数据类型 result_dtype
                assert np.can_cast(ya.dtype, result_dtype)
                
                # 计算稀疏矩阵乘积 sak.tosparse() @ x.astype(argument_dtype)，并断言其与 y 相等
                ys = sak.tosparse() @ x.astype(argument_dtype)
                np.array_equal(y, ys)
                
                # 断言 ys 的数据类型可以转换为预期结果数据类型 result_dtype
                assert np.can_cast(ys.dtype, result_dtype)

# 定义测试类 TestMikotaPair，用于测试 MikotaPair 类的功能
class TestMikotaPair:
    """
    MikotaPair tests
    """
    
    # 定义被测试的 MikotaPair 类的数据类型集合
    # MikotaK 的默认 dtype 是 np.int32，因为其条目是整数
    # MikotaM 包含逆运算，因此最小且仍准确的 dtype 是 np.float32
    tested_types = REAL_DTYPES + COMPLEX_DTYPES
    # 测试特定形状的 MikotaPair 对象
    def test_specific_shape(self):
        # 设置矩阵维度为 6
        n = 6
        # 创建一个 MikotaPair 对象，传入维度参数 n
        mik = MikotaPair(n)
        # 获取 MikotaPair 对象的 k 属性（对称矩阵 k）
        mik_k = mik.k
        # 获取 MikotaPair 对象的 m 属性（对角矩阵 m）
        mik_m = mik.m
        # 断言：验证 mik_k 转换为数组后与 mik_k(np.eye(n)) 的值相等
        assert_array_equal(mik_k.toarray(), mik_k(np.eye(n)))
        # 断言：验证 mik_m 转换为数组后与 mik_m(np.eye(n)) 的值相等
        assert_array_equal(mik_m.toarray(), mik_m(np.eye(n)))

        # 定义一个对称矩阵 k
        k = np.array(
            [
                [11, -5,  0,  0,  0,  0],
                [-5,  9, -4,  0,  0,  0],
                [ 0, -4,  7, -3,  0,  0],
                [ 0,  0, -3,  5, -2,  0],
                [ 0,  0,  0, -2,  3, -1],
                [ 0,  0,  0,  0, -1,  1]
            ]
        )
        # 断言：验证 mik_k 转换为数组后与 k 的值相等
        np.array_equal(k, mik_k.toarray())
        # 断言：验证 mik_k 转换为稀疏矩阵后再转换为数组与 k 的值相等
        np.array_equal(mik_k.tosparse().toarray(), k)

        # 定义一个带状矩阵 kb
        kb = np.array(
            [
                [ 0, -5, -4, -3, -2, -1],
                [11,  9,  7,  5,  3,  1]
            ]
        )
        # 断言：验证 mik_k 转换为带状矩阵与 kb 的值相等
        np.array_equal(kb, mik_k.tobanded())

        # 定义一个长度为 n 的数组 minv
        minv = np.arange(1, n + 1)
        # 断言：验证 mik_m 转换为数组后与对角矩阵 diag(1. / minv) 的值相等
        np.array_equal(np.diag(1. / minv), mik_m.toarray())
        # 断言：验证 mik_m 转换为稀疏矩阵后再转换为数组与 mik_m.toarray() 的值相等
        np.array_equal(mik_m.tosparse().toarray(), mik_m.toarray())
        # 断言：验证 mik_m 转换为带状矩阵与 1. / minv 的值相等
        np.array_equal(1. / minv, mik_m.tobanded())

        # 定义一个数组 e，包含一些特定的数值
        e = np.array([ 1,  4,  9, 16, 25, 36])
        # 断言：验证 mik 对象的特征值计算结果与数组 e 的值相等
        np.array_equal(e, mik.eigenvalues())
        # 断言：验证 mik 对象的前两个特征值计算结果与数组 e 的前两个值相等
        np.array_equal(e[:2], mik.eigenvalues(2))

    # 使用 pytest 的参数化装饰器，测试 MikotaPair 对象的形状和数据类型
    @pytest.mark.parametrize('dtype', tested_types)
    def test_linearoperator_shape_dtype(self, dtype):
        # 设置矩阵维度为 7，创建 MikotaPair 对象，并指定数据类型为 dtype
        n = 7
        mik = MikotaPair(n, dtype=dtype)
        # 获取 MikotaPair 对象的 k 属性（对称矩阵 k）
        mik_k = mik.k
        # 获取 MikotaPair 对象的 m 属性（对角矩阵 m）
        mik_m = mik.m
        # 断言：验证 mik_k 的形状为 (n, n)
        assert mik_k.shape == (n, n)
        # 断言：验证 mik_k 的数据类型为 dtype
        assert mik_k.dtype == dtype
        # 断言：验证 mik_m 的形状为 (n, n)
        assert mik_m.shape == (n, n)
        # 断言：验证 mik_m 的数据类型为 dtype
        assert mik_m.dtype == dtype

        # 创建一个默认数据类型为 np.float64 的 MikotaPair 对象
        mik_default_dtype = MikotaPair(n)
        # 获取其 k 属性（对称矩阵 k）和 m 属性（对角矩阵 m）
        mikd_k = mik_default_dtype.k
        mikd_m = mik_default_dtype.m
        # 断言：验证 mikd_k 的形状为 (n, n)
        assert mikd_k.shape == (n, n)
        # 断言：验证 mikd_k 的数据类型为 np.float64
        assert mikd_k.dtype == np.float64
        # 断言：验证 mikd_m 的形状为 (n, n)
        assert mikd_m.shape == (n, n)
        # 断言：验证 mikd_m 的数据类型为 np.float64
        assert mikd_m.dtype == np.float64
        # 断言：验证 mik_k 转换为数组后与 mikd_k 转换为数组并转换数据类型为 dtype 后的值相等
        assert_array_equal(mik_k.toarray(),
                           mikd_k.toarray().astype(dtype))
        # 断言：验证 mik_k 转换为稀疏矩阵后再转换为数组与 mikd_k 转换为稀疏矩阵后再转换为数组并转换数据类型为 dtype 后的值相等
        assert_array_equal(mik_k.tosparse().toarray(),
                           mikd_k.tosparse().toarray().astype(dtype))

    # 使用 pytest 的多重参数化装饰器，测试 MikotaPair 对象的形状和数据类型
    @pytest.mark.parametrize('dtype', tested_types)
    @pytest.mark.parametrize('argument_dtype', ALLDTYPES)
    # 定义一个测试方法，用于测试点乘操作在类型保持和一致性方面的表现
    def test_dot(self, dtype, argument_dtype):
        """ Test the dot-product for type preservation and consistency.
        """
        # 根据输入的参数类型推广出结果的数据类型
        result_dtype = np.promote_types(argument_dtype, dtype)
        # 创建一个 MikotaPair 对象，使用给定的数据类型
        n = 5
        mik = MikotaPair(n, dtype=dtype)
        # 获取 MikotaPair 对象的 k 和 m 属性
        mik_k = mik.k
        mik_m = mik.m
        # 创建三个不同形状的输入数组
        x0 = np.arange(n)
        x1 = x0.reshape((-1, 1))
        x2 = np.arange(2 * n).reshape((n, 2))
        # 定义一个包含 mik_k 和 mik_m 的列表
        lo_set = [mik_k, mik_m]
        # 定义一个包含 x0, x1, x2 的列表作为输入数据集
        input_set = [x0, x1, x2]
        # 遍历 lo_set 中的每个对象和 input_set 中的每个数组进行测试
        for lo in lo_set:
            for x in input_set:
                # 计算 lo 和 x 的点乘结果，并将 x 转换为 argument_dtype 类型
                y = lo.dot(x.astype(argument_dtype))
                # 断言结果 y 的形状与 x 相同
                assert x.shape == y.shape
                # 断言能否将结果 y 的 dtype 转换为 result_dtype
                assert np.can_cast(y.dtype, result_dtype)
                # 如果 x 是二维数组
                if x.ndim == 2:
                    # 使用矩阵乘法计算 lo.toarray() 和 x.astype(argument_dtype) 的结果 ya
                    ya = lo.toarray() @ x.astype(argument_dtype)
                    # 断言 y 和 ya 在内容上相等
                    np.array_equal(y, ya)
                    # 断言能否将结果 ya 的 dtype 转换为 result_dtype
                    assert np.can_cast(ya.dtype, result_dtype)
                    # 使用稀疏矩阵乘法计算 lo.tosparse() 和 x.astype(argument_dtype) 的结果 ys
                    ys = lo.tosparse() @ x.astype(argument_dtype)
                    # 断言 y 和 ys 在内容上相等
                    assert np.array_equal(y, ys)
                    # 断言能否将结果 ys 的 dtype 转换为 result_dtype
                    assert np.can_cast(ys.dtype, result_dtype)
```