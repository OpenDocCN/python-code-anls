# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_construct.py`

```
"""test sparse matrix construction functions"""

# 导入必要的库和模块
import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
        assert_array_equal, assert_array_almost_equal_nulp)
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state

# 导入稀疏矩阵相关的类和函数
from scipy.sparse import (csr_matrix, coo_matrix,
                          csr_array, coo_array,
                          csc_array, bsr_array,
                          dia_array, dok_array,
                          lil_array, csc_matrix,
                          bsr_matrix, dia_matrix,
                          lil_matrix, sparray, spmatrix,
                          _construct as construct)
from scipy.sparse._construct import rand as sprand

# 支持的稀疏矩阵格式列表
sparse_formats = ['csr','csc','coo','bsr','dia','lil','dok']

# TODO 检查是否支持 format=XXX 的情况


def _sprandn(m, n, density=0.01, format="coo", dtype=None, random_state=None):
    # Helper function for testing.
    # 检查并设置随机数生成器
    random_state = check_random_state(random_state)
    # 使用标准正态分布生成随机数据
    data_rvs = random_state.standard_normal
    # 调用 construct 模块的 random 函数生成稀疏矩阵
    return construct.random(m, n, density, format, dtype,
                            random_state, data_rvs)


def _sprandn_array(m, n, density=0.01, format="coo", dtype=None, random_state=None):
    # Helper function for testing.
    # 检查并设置随机数生成器
    random_state = check_random_state(random_state)
    # 使用标准正态分布生成随机数据的采样函数
    data_sampler = random_state.standard_normal
    # 调用 construct 模块的 random_array 函数生成稀疏矩阵数组
    return construct.random_array((m, n), density=density, format=format, dtype=dtype,
                                  random_state=random_state, data_sampler=data_sampler)


class TestConstructUtils:

    @pytest.mark.parametrize("cls", [
        csc_array, csr_array, coo_array, bsr_array,
        dia_array, dok_array, lil_array
    ])
    def test_singleton_array_constructor(self, cls):
        # 测试稀疏矩阵的单例数组构造函数是否能正确抛出 ValueError 异常
        with pytest.raises(
            ValueError,
            match=(
                'scipy sparse array classes do not support '
                'instantiation from a scalar'
            )
        ):
            cls(0)
    
    @pytest.mark.parametrize("cls", [
        csc_matrix, csr_matrix, coo_matrix,
        bsr_matrix, dia_matrix, lil_matrix
    ])
    def test_singleton_matrix_constructor(self, cls):
        """
        This test is for backwards compatibility post scipy 1.13.
        The behavior observed here is what is to be expected
        with the older matrix classes. This test comes with the
        exception of dok_matrix, which was not working pre scipy1.12
        (unlike the rest of these).
        """
        # 测试稀疏矩阵的单例矩阵构造函数的形状是否正确
        assert cls(0).shape == (1, 1)
    # 定义一个测试函数 `test_spdiags`
    def test_spdiags(self):
        # 创建三个不同的对角线数组 `diags1`, `diags2`, `diags3`
        diags1 = array([[1, 2, 3, 4, 5]])
        diags2 = array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9,10]])
        diags3 = array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9,10],
                         [11,12,13,14,15]])

        # 初始化一个空列表 `cases`，用于存储测试用例
        cases = []
        # 向 `cases` 列表添加不同的测试用例，每个用例包括对角线数组 `d`、偏移 `o`、行数 `m`、列数 `n`、预期结果 `result`
        cases.append((diags1, 0, 1, 1, [[1]]))
        cases.append((diags1, [0], 1, 1, [[1]]))
        cases.append((diags1, [0], 2, 1, [[1],[0]]))
        cases.append((diags1, [0], 1, 2, [[1,0]]))
        cases.append((diags1, [1], 1, 2, [[0,2]]))
        cases.append((diags1,[-1], 1, 2, [[0,0]]))
        cases.append((diags1, [0], 2, 2, [[1,0],[0,2]]))
        cases.append((diags1,[-1], 2, 2, [[0,0],[1,0]]))
        cases.append((diags1, [3], 2, 2, [[0,0],[0,0]]))
        cases.append((diags1, [0], 3, 4, [[1,0,0,0],[0,2,0,0],[0,0,3,0]]))
        cases.append((diags1, [1], 3, 4, [[0,2,0,0],[0,0,3,0],[0,0,0,4]]))
        cases.append((diags1, [2], 3, 5, [[0,0,3,0,0],[0,0,0,4,0],[0,0,0,0,5]]))

        cases.append((diags2, [0,2], 3, 3, [[1,0,8],[0,2,0],[0,0,3]]))
        cases.append((diags2, [-1,0], 3, 4, [[6,0,0,0],[1,7,0,0],[0,2,8,0]]))
        cases.append((diags2, [2,-3], 6, 6, [[0,0,3,0,0,0],
                                              [0,0,0,4,0,0],
                                              [0,0,0,0,5,0],
                                              [6,0,0,0,0,0],
                                              [0,7,0,0,0,0],
                                              [0,0,8,0,0,0]]))

        cases.append((diags3, [-1,0,1], 6, 6, [[6,12, 0, 0, 0, 0],
                                                [1, 7,13, 0, 0, 0],
                                                [0, 2, 8,14, 0, 0],
                                                [0, 0, 3, 9,15, 0],
                                                [0, 0, 0, 4,10, 0],
                                                [0, 0, 0, 0, 5, 0]]))
        cases.append((diags3, [-4,2,-1], 6, 5, [[0, 0, 8, 0, 0],
                                                 [11, 0, 0, 9, 0],
                                                 [0,12, 0, 0,10],
                                                 [0, 0,13, 0, 0],
                                                 [1, 0, 0,14, 0],
                                                 [0, 2, 0, 0,15]]))
        cases.append((diags3, [-1, 1, 2], len(diags3[0]), len(diags3[0]),
                      [[0, 7, 13, 0, 0],
                       [1, 0, 8, 14, 0],
                       [0, 2, 0, 9, 15],
                       [0, 0, 3, 0, 10],
                       [0, 0, 0, 4, 0]]))

        # 对于每个测试用例，执行断言检查结果是否符合预期
        for d, o, m, n, result in cases:
            if len(d[0]) == m and m == n:
                assert_equal(construct.spdiags(d, o).toarray(), result)
            assert_equal(construct.spdiags(d, o, m, n).toarray(), result)
            assert_equal(construct.spdiags(d, o, (m, n)).toarray(), result)
    # 测试默认情况下 construct.diags 的功能
    def test_diags_default(self):
        # 创建一个包含整数数组的 NumPy 数组
        a = array([1, 2, 3, 4, 5])
        # 断言 construct.diags(a) 转换成稀疏数组后与 np.diag(a) 相等
        assert_equal(construct.diags(a).toarray(), np.diag(a))

    # 测试包含非方形数组时 construct.diags 的行为
    def test_diags_default_bad(self):
        # 创建一个包含整数数组的二维 NumPy 数组
        a = array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        # 断言 construct.diags(a) 抛出 ValueError 异常
        assert_raises(ValueError, construct.diags, a)

    # 测试在不同情况下 construct.diags 的行为
    def test_diags_bad(self):
        # 创建三个整数数组
        a = array([1, 2, 3, 4, 5])
        b = array([6, 7, 8, 9, 10])
        c = array([11, 12, 13, 14, 15])

        cases = []
        # 向 cases 列表添加元组，每个元组包含输入、偏移和期望的形状
        cases.append(([a[:0]], 0, (1, 1)))
        cases.append(([a[:4],b,c[:3]], [-1,0,1], (5, 5)))
        cases.append(([a[:2],c,b[:3]], [-4,2,-1], (6, 5)))
        cases.append(([a[:2],c,b[:3]], [-4,2,-1], None))
        cases.append(([], [-4,2,-1], None))
        cases.append(([1], [-5], (4, 4)))
        cases.append(([a], 0, None))

        # 对每个 case 进行迭代，断言 construct.diags 抛出 ValueError 异常
        for d, o, shape in cases:
            assert_raises(ValueError, construct.diags, d, offsets=o, shape=shape)

        # 断言 construct.diags 抛出 TypeError 异常
        assert_raises(TypeError, construct.diags, [[None]], offsets=[0])

    # 测试 construct.diags 与 np.diag 的比较
    def test_diags_vs_diag(self):
        # 设置随机种子
        np.random.seed(1234)

        # 对于不同数量的对角线进行迭代测试
        for n_diags in [1, 2, 3, 4, 5, 10]:
            # 根据 n_diags 生成一个随机大小的整数 n
            n = 1 + n_diags//2 + np.random.randint(0, 10)

            # 生成随机偏移并打乱顺序
            offsets = np.arange(-n+1, n-1)
            np.random.shuffle(offsets)
            offsets = offsets[:n_diags]

            # 生成随机对角线数组
            diagonals = [np.random.rand(n - abs(q)) for q in offsets]

            # 使用 construct.diags 创建稀疏矩阵 mat
            mat = construct.diags(diagonals, offsets=offsets)
            # 计算稠密矩阵 dense_mat
            dense_mat = sum([np.diag(x, j) for x, j in zip(diagonals, offsets)])

            # 断言稀疏矩阵与稠密矩阵的近似相等性
            assert_array_almost_equal_nulp(mat.toarray(), dense_mat)

            # 如果偏移长度为 1，则单独测试 construct.diags 的行为
            if len(offsets) == 1:
                mat = construct.diags(diagonals[0], offsets=offsets[0])
                dense_mat = np.diag(diagonals[0], offsets[0])
                assert_array_almost_equal_nulp(mat.toarray(), dense_mat)

    # 测试 construct.diags 的 dtype 参数
    def test_diags_dtype(self):
        # 使用 dtype 参数创建一个整数类型的稀疏矩阵 x
        x = construct.diags([2.2], offsets=[0], shape=(2, 2), dtype=int)
        # 断言稀疏矩阵 x 的数据类型为整数
        assert_equal(x.dtype, int)
        # 断言稀疏矩阵 x 的数组表示与预期相等
        assert_equal(x.toarray(), [[2, 0], [0, 2]])

    # 测试当只有一个对角线时 construct.diags 的行为
    def test_diags_one_diagonal(self):
        # 创建一个包含整数的列表 d
        d = list(range(5))
        # 对于不同的偏移 k 进行迭代测试
        for k in range(-5, 6):
            # 断言 construct.diags(d, offsets=k) 和 construct.diags([d], offsets=[k]) 的数组表示相等
            assert_equal(construct.diags(d, offsets=k).toarray(),
                         construct.diags([d], offsets=[k]).toarray())

    # 测试当输入为空时 construct.diags 的行为
    def test_diags_empty(self):
        # 创建一个空的稀疏矩阵 x
        x = construct.diags([])
        # 断言稀疏矩阵 x 的形状为 (0, 0)
        assert_equal(x.shape, (0, 0))

    # 使用 pytest.mark.parametrize 对 construct.identity 和 construct.eye_array 进行参数化测试
    @pytest.mark.parametrize("identity", [construct.identity, construct.eye_array])
    def test_identity(self, identity):
        # 断言 identity(1) 的数组表示为 [[1]]
        assert_equal(identity(1).toarray(), [[1]])
        # 断言 identity(2) 的数组表示为 [[1,0],[0,1]]
        assert_equal(identity(2).toarray(), [[1,0],[0,1]])

        # 使用 dtype 和 format 参数创建一个对角矩阵 I
        I = identity(3, dtype='int8', format='dia')
        # 断言矩阵 I 的数据类型为 int8
        assert_equal(I.dtype, np.dtype('int8'))
        # 断言矩阵 I 的格式为 'dia'
        assert_equal(I.format, 'dia')

        # 对于所有的稀疏格式进行迭代测试
        for fmt in sparse_formats:
            I = identity(3, format=fmt)
            # 断言矩阵 I 的格式与预期相等
            assert_equal(I.format, fmt)
            # 断言矩阵 I 的数组表示为 [[1,0,0],[0,1,0],[0,0,1]]
            assert_equal(I.toarray(), [[1,0,0],[0,1,0],[0,0,1]])
    # 使用 pytest 的参数化标记，指定测试函数 test_eye 和 test_eye_one 参数为 construct.eye 和 construct.eye_array
    @pytest.mark.parametrize("eye", [construct.eye, construct.eye_array])
    # 定义测试函数 test_eye
    def test_eye(self, eye):
        # 断言对角矩阵的生成结果与预期一致
        assert_equal(eye(1,1).toarray(), [[1]])
        assert_equal(eye(2,3).toarray(), [[1,0,0],[0,1,0]])
        assert_equal(eye(3,2).toarray(), [[1,0],[0,1],[0,0]])
        assert_equal(eye(3,3).toarray(), [[1,0,0],[0,1,0],[0,0,1]])

        # 断言生成的对角矩阵的数据类型与指定的 dtype 一致
        assert_equal(eye(3,3,dtype='int16').dtype, np.dtype('int16'))

        # 嵌套循环测试不同维度和偏移情况下的对角矩阵生成
        for m in [3, 5]:
            for n in [3, 5]:
                for k in range(-5,6):
                    # 当偏移 k 超出矩阵边界时，验证是否会引发 ValueError 异常
                    if (k > 0 and k > n) or (k < 0 and abs(k) > m):
                        with pytest.raises(
                            ValueError, match="Offset.*out of bounds"
                        ):
                            eye(m, n, k=k)
                    else:
                        # 断言生成的稀疏对角矩阵与 numpy 中的 np.eye 结果一致
                        assert_equal(
                            eye(m, n, k=k).toarray(),
                            np.eye(m, n, k=k)
                        )
                        # 当 m 等于 n 时，验证未指定 n 参数的情况下生成的对角矩阵与 numpy 中的 np.eye 结果一致
                        if m == n:
                            assert_equal(
                                eye(m, k=k).toarray(),
                                np.eye(m, n, k=k)
                            )

    # 使用 pytest 的参数化标记，指定测试函数 test_eye_one 参数为 construct.eye 和 construct.eye_array
    @pytest.mark.parametrize("eye", [construct.eye, construct.eye_array])
    # 定义测试函数 test_eye_one
    def test_eye_one(self, eye):
        # 断言对角矩阵的生成结果与预期一致
        assert_equal(eye(1).toarray(), [[1]])
        assert_equal(eye(2).toarray(), [[1,0],[0,1]])

        # 验证指定 dtype 和 format 后生成的对角矩阵的数据类型和格式正确
        I = eye(3, dtype='int8', format='dia')
        assert_equal(I.dtype, np.dtype('int8'))
        assert_equal(I.format, 'dia')

        # 遍历稀疏格式列表，验证生成的对角矩阵的格式和结果正确
        for fmt in sparse_formats:
            I = eye(3, format=fmt)
            assert_equal(I.format, fmt)
            assert_equal(I.toarray(), [[1,0,0],[0,1,0],[0,0,1]])

    # 定义测试函数 test_eye_array_vs_matrix，验证 construct.eye_array 生成的对象是 sparray 类型，而 construct.eye 生成的对象不是 sparray 类型
    def test_eye_array_vs_matrix(self):
        assert isinstance(construct.eye_array(3), sparray)
        assert not isinstance(construct.eye(3), sparray)
    # 定义一个测试方法 test_kron，用于测试矩阵的 Kronecker 乘积
    def test_kron(self):
        # 初始化测试用例列表
        cases = []

        # 向测试用例列表添加不同的矩阵
        cases.append(array([[0]]))
        cases.append(array([[-1]]))
        cases.append(array([[4]]))
        cases.append(array([[10]]))
        cases.append(array([[0],[0]]))
        cases.append(array([[0,0]]))
        cases.append(array([[1,2],[3,4]]))
        cases.append(array([[0,2],[5,0]]))
        cases.append(array([[0,2,-6],[8,0,14]]))
        cases.append(array([[5,4],[0,0],[6,0]]))
        cases.append(array([[5,4,4],[1,0,0],[6,0,8]]))
        cases.append(array([[0,1,0,2,0,5,8]]))
        cases.append(array([[0.5,0.125,0,3.25],[0,2.5,0,0]]))

        # 对所有测试用例进行迭代测试
        for a in cases:
            # 创建 CSR 格式的稀疏矩阵 ca
            ca = csr_array(a)
            for b in cases:
                # 创建 CSR 格式的稀疏矩阵 cb
                cb = csr_array(b)
                # 计算 numpy 的 Kronecker 乘积作为预期结果
                expected = np.kron(a, b)
                # 使用一些稀疏格式测试构造函数 kron
                for fmt in sparse_formats[1:4]:
                    # 调用 kron 方法计算结果
                    result = construct.kron(ca, cb, format=fmt)
                    # 断言结果的格式为 fmt
                    assert_equal(result.format, fmt)
                    # 断言结果转换为数组后与预期结果相等
                    assert_array_equal(result.toarray(), expected)
                    # 断言结果为 sparray（稀疏数组）类型
                    assert isinstance(result, sparray)

        # 对最后两个测试用例使用所有稀疏格式进行测试
        a = cases[-1]
        b = cases[-3]
        ca = csr_array(a)
        cb = csr_array(b)

        # 计算 numpy 的 Kronecker 乘积作为预期结果
        expected = np.kron(a, b)
        for fmt in sparse_formats:
            # 调用 kron 方法计算结果
            result = construct.kron(ca, cb, format=fmt)
            # 断言结果的格式为 fmt
            assert_equal(result.format, fmt)
            # 断言结果转换为数组后与预期结果相等
            assert_array_equal(result.toarray(), expected)
            # 断言结果为 sparray（稀疏数组）类型
            assert isinstance(result, sparray)

        # 检查当输入均为 spmatrix 时，kron 方法返回的是否为 spmatrix
        result = construct.kron(csr_matrix(a), csr_matrix(b), format=fmt)
        # 断言结果的格式为 fmt
        assert_equal(result.format, fmt)
        # 断言结果转换为数组后与预期结果相等
        assert_array_equal(result.toarray(), expected)
        # 断言结果为 spmatrix（稀疏矩阵）类型
        assert isinstance(result, spmatrix)

    # 定义一个测试方法 test_kron_large，用于测试大尺寸矩阵的 Kronecker 乘积
    def test_kron_large(self):
        # 定义矩阵的尺寸为 2 的 16 次方
        n = 2**16
        # 创建对角矩阵 a
        a = construct.diags_array([1], shape=(1, n), offsets=n-1)
        # 创建对角矩阵 b
        b = construct.diags_array([1], shape=(n, 1), offsets=1-n)

        # 调用 kron 方法计算 a 和 a 的 Kronecker 乘积
        construct.kron(a, a)
        # 调用 kron 方法计算 b 和 b 的 Kronecker 乘积
        construct.kron(b, b)

    # 定义一个测试方法 test_kronsum，用于测试矩阵的 Kronecker 和
    def test_kronsum(self):
        # 初始化测试用例列表
        cases = []

        # 向测试用例列表添加不同的矩阵
        cases.append(array([[0]]))
        cases.append(array([[-1]]))
        cases.append(array([[4]]))
        cases.append(array([[10]]))
        cases.append(array([[1,2],[3,4]]))
        cases.append(array([[0,2],[5,0]]))
        cases.append(array([[0,2,-6],[8,0,14],[0,3,0]]))
        cases.append(array([[1,0,0],[0,5,-1],[4,-2,8]]))

        # 对所有测试用例使用默认格式进行测试
        for a in cases:
            for b in cases:
                # 计算 np.kron(np.eye(b.shape[0]), a) + np.kron(b, np.eye(a.shape[0])) 作为预期结果
                expected = (np.kron(np.eye(b.shape[0]), a)
                            + np.kron(b, np.eye(a.shape[0])))
                # 调用 kronsum 方法计算结果并转换为数组
                result = construct.kronsum(csr_array(a), csr_array(b)).toarray()
                # 断言结果数组与预期数组相等
                assert_array_equal(result, expected)

        # 检查当输入均为 spmatrix 时，kronsum 方法返回的是否为预期结果数组
        result = construct.kronsum(csr_matrix(a), csr_matrix(b)).toarray()
        # 断言结果数组与预期数组相等
        assert_array_equal(result, expected)
    # 使用 pytest 的参数化装饰器，为 coo_matrix 和 coo_array 分别执行下列测试
    @pytest.mark.parametrize("coo_cls", [coo_matrix, coo_array])
    # 测试垂直堆叠操作 construct.vstack()
    def test_vstack(self, coo_cls):
        # 创建 COO 格式矩阵 A 和 B
        A = coo_cls([[1,2],[3,4]])
        B = coo_cls([[5,6]])

        # 期望的堆叠结果
        expected = array([[1, 2],
                          [3, 4],
                          [5, 6]])
        # 断言堆叠后的数组与期望的数组相等
        assert_equal(construct.vstack([A, B]).toarray(), expected)
        # 断言堆叠后的数组数据类型为 np.float32
        assert_equal(construct.vstack([A, B], dtype=np.float32).dtype,
                     np.float32)

        # 断言堆叠后的稀疏矩阵转化为数组后与期望的数组相等
        assert_equal(construct.vstack([A.todok(), B.todok()]).toarray(), expected)

        # 断言堆叠后的 CSR 格式矩阵转化为数组后与期望的数组相等
        assert_equal(construct.vstack([A.tocsr(), B.tocsr()]).toarray(),
                     expected)
        # 使用指定格式和数据类型进行堆叠操作后，断言结果数据类型为 np.float32
        result = construct.vstack([A.tocsr(), B.tocsr()],
                                  format="csr", dtype=np.float32)
        assert_equal(result.dtype, np.float32)
        # 断言结果的索引数据类型为 np.int32
        assert_equal(result.indices.dtype, np.int32)
        # 断言结果的指针数据类型为 np.int32
        assert_equal(result.indptr.dtype, np.int32)

        # 断言堆叠后的 CSC 格式矩阵转化为数组后与期望的数组相等
        assert_equal(construct.vstack([A.tocsc(), B.tocsc()]).toarray(),
                     expected)
        # 使用指定格式和数据类型进行堆叠操作后，断言结果数据类型为 np.float32
        result = construct.vstack([A.tocsc(), B.tocsc()],
                                  format="csc", dtype=np.float32)
        assert_equal(result.dtype, np.float32)
        # 断言结果的索引数据类型为 np.int32
        assert_equal(result.indices.dtype, np.int32)
        # 断言结果的指针数据类型为 np.int32

    # 测试不同类型矩阵或数组的垂直堆叠操作 construct.vstack()
    def test_vstack_matrix_or_array(self):
        # 定义矩阵 A 和 B
        A = [[1,2],[3,4]]
        B = [[5,6]]
        # 断言堆叠后的对象类型为 sparray
        assert isinstance(construct.vstack([coo_array(A), coo_array(B)]), sparray)
        assert isinstance(construct.vstack([coo_array(A), coo_matrix(B)]), sparray)
        assert isinstance(construct.vstack([coo_matrix(A), coo_array(B)]), sparray)
        # 断言堆叠后的对象类型为 spmatrix
        assert isinstance(construct.vstack([coo_matrix(A), coo_matrix(B)]), spmatrix)

    # 使用 pytest 的参数化装饰器，为 coo_matrix 和 coo_array 分别执行下列测试
    @pytest.mark.parametrize("coo_cls", [coo_matrix, coo_array])
    # 测试水平堆叠操作 construct.hstack()
    def test_hstack(self, coo_cls):
        # 创建 COO 格式矩阵 A 和 B
        A = coo_cls([[1,2],[3,4]])
        B = coo_cls([[5],[6]])

        # 期望的堆叠结果
        expected = array([[1, 2, 5],
                          [3, 4, 6]])
        # 断言堆叠后的数组与期望的数组相等
        assert_equal(construct.hstack([A, B]).toarray(), expected)
        # 断言堆叠后的数组数据类型为 np.float32
        assert_equal(construct.hstack([A, B], dtype=np.float32).dtype,
                     np.float32)

        # 断言堆叠后的稀疏矩阵转化为数组后与期望的数组相等
        assert_equal(construct.hstack([A.todok(), B.todok()]).toarray(), expected)

        # 断言堆叠后的 CSC 格式矩阵转化为数组后与期望的数组相等
        assert_equal(construct.hstack([A.tocsc(), B.tocsc()]).toarray(),
                     expected)
        # 使用指定格式和数据类型进行堆叠操作后，断言结果数据类型为 np.float32
        assert_equal(construct.hstack([A.tocsc(), B.tocsc()],
                                      dtype=np.float32).dtype,
                     np.float32)
        # 断言堆叠后的 CSR 格式矩阵转化为数组后与期望的数组相等
        assert_equal(construct.hstack([A.tocsr(), B.tocsr()]).toarray(),
                     expected)
        # 使用指定格式和数据类型进行堆叠操作后，断言结果数据类型为 np.float32
        assert_equal(construct.hstack([A.tocsr(), B.tocsr()],
                                      dtype=np.float32).dtype,
                     np.float32)
    # 定义一个测试函数，用于测试水平堆叠矩阵或数组的函数
    def test_hstack_matrix_or_array(self):
        # 创建两个二维列表 A 和 B
        A = [[1,2],[3,4]]
        B = [[5],[6]]
        # 断言水平堆叠 coo_array(A) 和 coo_array(B) 后的返回类型是 sparray
        assert isinstance(construct.hstack([coo_array(A), coo_array(B)]), sparray)
        # 断言水平堆叠 coo_array(A) 和 coo_matrix(B) 后的返回类型是 sparray
        assert isinstance(construct.hstack([coo_array(A), coo_matrix(B)]), sparray)
        # 断言水平堆叠 coo_matrix(A) 和 coo_array(B) 后的返回类型是 sparray
        assert isinstance(construct.hstack([coo_matrix(A), coo_array(B)]), sparray)
        # 断言水平堆叠 coo_matrix(A) 和 coo_matrix(B) 后的返回类型是 spmatrix
        assert isinstance(construct.hstack([coo_matrix(A), coo_matrix(B)]), spmatrix)

    @pytest.mark.parametrize("block_array", (construct.bmat, construct.block_array))
    # 定义一个测试函数，用于测试 block_array 的返回类型
    def test_block_return_type(self):
        # 将 block_array 赋值给变量 block
        block = construct.block_array

        # 使用 csr 格式确保使用 _compressed_sparse_stack
        # F,G 的形状确保使用 _stack_along_minor_axis
        # 使用列表版本确保不使用任何辅助函数的路径
        Fl, Gl = [[1, 2],[3, 4]], [[7], [5]]
        Fm, Gm = csr_matrix(Fl), csr_matrix(Gl)
        # 断言 block([[None, Fl], [Gl, None]], format="csr") 的返回类型是 sparray
        assert isinstance(block([[None, Fl], [Gl, None]], format="csr"), sparray)
        # 断言 block([[None, Fm], [Gm, None]], format="csr") 的返回类型是 sparray
        assert isinstance(block([[None, Fm], [Gm, None]], format="csr"), sparray)
        # 断言 block([[Fm, Gm]], format="csr") 的返回类型是 sparray
        assert isinstance(block([[Fm, Gm]], format="csr"), sparray)
    def test_bmat_return_type(self):
        """This can be removed after sparse matrix is removed"""
        # 获取函数 construct.bmat 的引用
        bmat = construct.bmat

        # 准备测试数据
        Fl, Gl = [[1, 2],[3, 4]], [[7], [5]]
        Fm, Gm = csr_matrix(Fl), csr_matrix(Gl)
        Fa, Ga = csr_array(Fl), csr_array(Gl)

        # 检查返回类型为 sparray，如果任何输入是 _is_array 输出数组，否则输出矩阵
        assert isinstance(bmat([[Fa, Ga]], format="csr"), sparray)
        assert isinstance(bmat([[Fm, Gm]], format="csr"), spmatrix)
        assert isinstance(bmat([[None, Fa], [Ga, None]], format="csr"), sparray)
        assert isinstance(bmat([[None, Fm], [Ga, None]], format="csr"), sparray)
        assert isinstance(bmat([[None, Fm], [Gm, None]], format="csr"), spmatrix)
        assert isinstance(bmat([[None, Fl], [Gl, None]], format="csr"), spmatrix)

        # 检查返回类型为 sparray，由 _compressed_sparse_stack 返回所有 csr
        assert isinstance(bmat([[Ga, Ga]], format="csr"), sparray)
        assert isinstance(bmat([[Gm, Ga]], format="csr"), sparray)
        assert isinstance(bmat([[Ga, Gm]], format="csr"), sparray)
        assert isinstance(bmat([[Gm, Gm]], format="csr"), spmatrix)
        
        # 形状为 2x2，因此不使用 _stack_along_minor_axis
        assert isinstance(bmat([[Fa, Fm]], format="csr"), sparray)
        assert isinstance(bmat([[Fm, Fm]], format="csr"), spmatrix)

        # 检查返回类型为 sparray，由 _compressed_sparse_stack 返回所有 csc
        assert isinstance(bmat([[Gm.tocsc(), Ga.tocsc()]], format="csc"), sparray)
        assert isinstance(bmat([[Gm.tocsc(), Gm.tocsc()]], format="csc"), spmatrix)
        
        # 形状为 2x2，因此不使用 _stack_along_minor_axis
        assert isinstance(bmat([[Fa.tocsc(), Fm.tocsc()]], format="csr"), sparray)
        assert isinstance(bmat([[Fm.tocsc(), Fm.tocsc()]], format="csr"), spmatrix)

        # 检查混合输入时的返回类型
        assert isinstance(bmat([[Gl, Ga]], format="csr"), sparray)
        assert isinstance(bmat([[Gm.tocsc(), Ga]], format="csr"), sparray)
        assert isinstance(bmat([[Gm.tocsc(), Gm]], format="csr"), spmatrix)
        assert isinstance(bmat([[Gm, Gm]], format="csc"), spmatrix)

    @pytest.mark.slow
    @pytest.mark.xfail_on_32bit("Can't create large array for test")
    def test_concatenate_int32_overflow(self):
        """ test for indptr overflow when concatenating matrices """
        # 检查系统内存是否足够进行测试
        check_free_memory(30000)

        # 设置矩阵大小为 n x n
        n = 33000
        A = csr_array(np.ones((n, n), dtype=bool))
        B = A.copy()

        # 使用 _compressed_sparse_stack 进行矩阵连接，axis=0 表示沿行堆叠
        C = construct._compressed_sparse_stack((A, B), axis=0,
                                               return_spmatrix=False)

        # 断言每个索引指针之间的差等于 n
        assert_(np.all(np.equal(np.diff(C.indptr), n)))
        
        # 断言 C.indices 的数据类型为 np.int64
        assert_equal(C.indices.dtype, np.int64)
        
        # 断言 C.indptr 的数据类型为 np.int64
        assert_equal(C.indptr.dtype, np.int64)
    def test_block_diag_basic(self):
        """ basic test for block_diag """
        # 创建稀疏矩阵 A
        A = coo_array([[1,2],[3,4]])
        # 创建稀疏矩阵 B
        B = coo_array([[5],[6]])
        # 创建稀疏矩阵 C
        C = coo_array([[7]])

        # 预期的稀疏矩阵结果
        expected = array([[1, 2, 0, 0],
                          [3, 4, 0, 0],
                          [0, 0, 5, 0],
                          [0, 0, 6, 0],
                          [0, 0, 0, 7]])

        # 断言 block_diag 函数的返回结果与预期结果相等
        assert_equal(construct.block_diag((A, B, C)).toarray(), expected)

    def test_block_diag_scalar_1d_args(self):
        """ block_diag with scalar and 1d arguments """
        # 测试包含一个 1D 矩阵和一个标量的情况
        assert_array_equal(construct.block_diag([[2,3], 4]).toarray(),
                           [[2, 3, 0], [0, 0, 4]])
        # 测试包含多个 1D 稀疏数组的情况
        A = coo_array([1,0,3])
        B = coo_array([0,4])
        assert_array_equal(construct.block_diag([A, B]).toarray(),
                           [[1, 0, 3, 0, 0], [0, 0, 0, 0, 4]])


    def test_block_diag_1(self):
        """ block_diag with one matrix """
        # 测试仅包含一个矩阵的情况
        assert_equal(construct.block_diag([[1, 0]]).toarray(),
                     array([[1, 0]]))
        assert_equal(construct.block_diag([[[1, 0]]]).toarray(),
                     array([[1, 0]]))
        assert_equal(construct.block_diag([[[1], [0]]]).toarray(),
                     array([[1], [0]]))
        # 测试仅包含一个标量的情况
        assert_equal(construct.block_diag([1]).toarray(),
                     array([[1]]))

    def test_block_diag_sparse_arrays(self):
        """ block_diag with sparse arrays """
        # 测试包含稀疏数组的情况
        A = coo_array([[1, 2, 3]], shape=(1, 3))
        B = coo_array([[4, 5]], shape=(1, 2))
        assert_equal(construct.block_diag([A, B]).toarray(),
                     array([[1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]))

        A = coo_array([[1], [2], [3]], shape=(3, 1))
        B = coo_array([[4], [5]], shape=(2, 1))
        assert_equal(construct.block_diag([A, B]).toarray(),
                     array([[1, 0], [2, 0], [3, 0], [0, 4], [0, 5]]))

    def test_block_diag_return_type(self):
        # 测试 block_diag 返回类型的断言
        A, B = coo_array([[1, 2, 3]]), coo_matrix([[2, 3, 4]])
        assert isinstance(construct.block_diag([A, A]), sparray)
        assert isinstance(construct.block_diag([A, B]), sparray)
        assert isinstance(construct.block_diag([B, A]), sparray)
        assert isinstance(construct.block_diag([B, B]), spmatrix)
    def test_random_sampling(self):
        # 对稀疏随机抽样进行简单的健全性检查
        for f in sprand, _sprandn:
            for t in [np.float32, np.float64, np.longdouble,
                      np.int32, np.int64, np.complex64, np.complex128]:
                # 创建稀疏矩阵 x，指定数据类型 t，形状为 (5, 10)，稠密度为 0.1
                x = f(5, 10, density=0.1, dtype=t)
                # 断言 x 的数据类型为 t
                assert_equal(x.dtype, t)
                # 断言 x 的形状为 (5, 10)
                assert_equal(x.shape, (5, 10))
                # 断言 x 的非零元素个数为 5
                assert_equal(x.nnz, 5)

            # 创建带有指定随机种子的稀疏矩阵 x1
            x1 = f(5, 10, density=0.1, random_state=4321)
            # 断言 x1 的数据类型为 np.float64
            assert_equal(x1.dtype, np.float64)

            # 创建带有指定随机状态的稀疏矩阵 x2
            x2 = f(5, 10, density=0.1,
                   random_state=np.random.RandomState(4321))

            # 断言 x1 和 x2 的数据数组相等
            assert_array_equal(x1.data, x2.data)
            # 断言 x1 和 x2 的行索引数组相等
            assert_array_equal(x1.row, x2.row)
            # 断言 x1 和 x2 的列索引数组相等
            assert_array_equal(x1.col, x2.col)

            # 对于不同的稠密度进行检查
            for density in [0.0, 0.1, 0.5, 1.0]:
                # 创建稀疏矩阵 x，形状为 (5, 10)，稠密度为 density
                x = f(5, 10, density=density)
                # 断言 x 的非零元素个数为 density 乘以 x 的元素总数
                assert_equal(x.nnz, int(density * np.prod(x.shape)))

            # 对于不同的格式进行检查
            for fmt in ['coo', 'csc', 'csr', 'lil']:
                # 创建指定格式的稀疏矩阵 x
                x = f(5, 10, format=fmt)
                # 断言 x 的格式为 fmt
                assert_equal(x.format, fmt)

            # 断言在非法参数输入时会触发 ValueError 异常
            assert_raises(ValueError, lambda: f(5, 10, 1.1))
            assert_raises(ValueError, lambda: f(5, 10, -0.1))

    def test_rand(self):
        # 对稀疏.rand进行简单的分布检查
        random_states = [None, 4321, np.random.RandomState()]
        try:
            gen = np.random.default_rng()
            random_states.append(gen)
        except AttributeError:
            pass

        # 遍历不同的随机状态进行检查
        for random_state in random_states:
            # 创建稀疏矩阵 x，形状为 (10, 20)，稠密度为 0.5，数据类型为 np.float64
            x = sprand(10, 20, density=0.5, dtype=np.float64,
                       random_state=random_state)
            # 断言 x 的所有数据元素大于等于 0
            assert_(np.all(np.less_equal(0, x.data)))
            # 断言 x 的所有数据元素小于等于 1
            assert_(np.all(np.less_equal(x.data, 1)))

    def test_randn(self):
        # 对稀疏.randn进行简单的分布检查
        # 根据统计学，部分数据应为负数，部分数据应大于 1
        random_states = [None, 4321, np.random.RandomState()]
        try:
            gen = np.random.default_rng()
            random_states.append(gen)
        except AttributeError:
            pass

        # 遍历不同的随机状态进行检查
        for rs in random_states:
            # 创建符合指定参数的稀疏矩阵 x
            x = _sprandn(10, 20, density=0.5, dtype=np.float64, random_state=rs)
            # 断言 x 的数据数组中至少有一个元素小于 0
            assert_(np.any(np.less(x.data, 0)))
            # 断言 x 的数据数组中至少有一个元素大于 1
            assert_(np.any(np.less(1, x.data)))
            # 创建符合指定参数的稀疏矩阵 x
            x = _sprandn_array(10, 20, density=0.5, dtype=np.float64, random_state=rs)
            # 断言 x 的数据数组中至少有一个元素小于 0
            assert_(np.any(np.less(x.data, 0)))
            # 断言 x 的数据数组中至少有一个元素大于 1
            assert_(np.any(np.less(1, x.data)))

    def test_random_accept_str_dtype(self):
        # 任何 np.dtype 可以转换为的数据类型都应该被接受
        # 作为 dtype 参数传入
        construct.random(10, 10, dtype='d')
        construct.random_array((10, 10), dtype='d')
    def test_random_sparse_matrix_returns_correct_number_of_non_zero_elements(self):
        # 测试随机稀疏矩阵生成函数，验证非零元素数量是否正确
        # 一个10 x 10的矩阵，密度为12.65%，应该有13个非零元素。
        # 10 x 10 x 0.1265 = 12.65，应向上取整为13，而不是12。
        
        # 生成一个10 x 10的随机稀疏矩阵，密度为0.1265
        sparse_matrix = construct.random(10, 10, density=0.1265)
        
        # 断言：验证矩阵中非零元素的数量是否为13
        assert_equal(sparse_matrix.count_nonzero(), 13)
        
        # 检查随机数组生成函数
        sparse_array = construct.random_array((10, 10), density=0.1265)
        
        # 断言：验证数组中非零元素的数量是否为13
        assert_equal(sparse_array.count_nonzero(), 13)
        
        # 断言：验证sparse_array对象是否为sparray类型的实例
        assert isinstance(sparse_array, sparray)
        
        # 检查大尺寸的情况
        shape = (2**33, 2**33)
        
        # 生成一个形状为(2**33, 2**33)的随机稀疏数组，密度为2.7105e-17
        sparse_array = construct.random_array(shape, density=2.7105e-17)
        
        # 断言：验证数组中非零元素的数量是否为2000
        assert_equal(sparse_array.count_nonzero(), 2000)
# 定义一个测试函数，用于测试不依赖于diags包装器的diags_array函数
def test_diags_array():
    # 创建一个包含1到4的一维NumPy数组作为对角线元素
    diag = np.arange(1, 5)

    # 断言调用construct.diags_array(diag)返回的稀疏矩阵与np.diag(diag)返回的对角矩阵相等
    assert_array_equal(construct.diags_array(diag).toarray(), np.diag(diag))

    # 断言调用construct.diags_array(diag, offsets=2)返回的稀疏矩阵与np.diag(diag, k=2)返回的主对角线偏移为2的对角矩阵相等
    assert_array_equal(
        construct.diags_array(diag, offsets=2).toarray(), np.diag(diag, k=2)
    )

    # 断言调用construct.diags_array(diag, offsets=2, shape=(4, 4))返回的稀疏矩阵与np.diag(diag, k=2)[:4, :4]返回的截取的4x4对角矩阵相等
    assert_array_equal(
        construct.diags_array(diag, offsets=2, shape=(4, 4)).toarray(),
        np.diag(diag, k=2)[:4, :4]
    )

    # 在指定了shape参数时，测试偏移超出边界的情况，预期引发ValueError异常，异常信息包含"out of bounds"
    with pytest.raises(ValueError, match=".*out of bounds"):
        construct.diags(np.arange(1, 5), 5, shape=(4, 4))
```