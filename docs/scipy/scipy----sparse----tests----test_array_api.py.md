# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_array_api.py`

```
import pytest  # 导入 pytest 库，用于编写和运行测试用例
import numpy as np  # 导入 NumPy 库，并用 np 别名引用
import numpy.testing as npt  # 导入 NumPy 测试模块，用 npt 别名引用
import scipy.sparse  # 导入 SciPy 稀疏矩阵模块
import scipy.sparse.linalg as spla  # 导入 SciPy 稀疏矩阵线性代数模块


sparray_types = ('bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil')  # 定义稀疏矩阵类型元组

sparray_classes = [  # 根据稀疏矩阵类型创建对应的类列表
    getattr(scipy.sparse, f'{T}_array') for T in sparray_types
]

A = np.array([  # 创建一个 NumPy 数组 A
    [0, 1, 2, 0],
    [2, 0, 0, 3],
    [1, 4, 0, 0]
])

B = np.array([  # 创建一个 NumPy 数组 B
    [0, 1],
    [2, 0]
])

X = np.array([  # 创建一个浮点型的 NumPy 数组 X
    [1, 0, 0, 1],
    [2, 1, 2, 0],
    [0, 2, 1, 0],
    [0, 0, 1, 2]
], dtype=float)


sparrays = [sparray(A) for sparray in sparray_classes]  # 使用 A 创建稀疏矩阵实例列表
square_sparrays = [sparray(B) for sparray in sparray_classes]  # 使用 B 创建稀疏矩阵实例列表
eig_sparrays = [sparray(X) for sparray in sparray_classes]  # 使用 X 创建稀疏矩阵实例列表

parametrize_sparrays = pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 创建参数化测试装饰器
    "A", sparrays, ids=sparray_types
)
parametrize_square_sparrays = pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 创建参数化测试装饰器
    "B", square_sparrays, ids=sparray_types
)
parametrize_eig_sparrays = pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 创建参数化测试装饰器
    "X", eig_sparrays, ids=sparray_types
)


@parametrize_sparrays  # 参数化测试：测试函数 test_sum 针对 sparrays 中的每个稀疏矩阵执行
def test_sum(A):
    assert not isinstance(A.sum(axis=0), np.matrix), \  # 断言：A 沿着 axis=0 的和不是 np.matrix 类型
        "Expected array, got matrix"
    assert A.sum(axis=0).shape == (4,)  # 断言：A 沿着 axis=0 的和的形状为 (4,)
    assert A.sum(axis=1).shape == (3,)  # 断言：A 沿着 axis=1 的和的形状为 (3,)


@parametrize_sparrays  # 参数化测试：测试函数 test_mean 针对 sparrays 中的每个稀疏矩阵执行
def test_mean(A):
    assert not isinstance(A.mean(axis=1), np.matrix), \  # 断言：A 沿着 axis=1 的均值不是 np.matrix 类型
        "Expected array, got matrix"


@parametrize_sparrays  # 参数化测试：测试函数 test_min_max 针对 sparrays 中的每个稀疏矩阵执行
def test_min_max(A):
    # 部分格式不支持 min/max 操作，因此在这里跳过
    if hasattr(A, 'min'):
        assert not isinstance(A.min(axis=1), np.matrix), \  # 断言：A 沿着 axis=1 的最小值不是 np.matrix 类型
            "Expected array, got matrix"
    if hasattr(A, 'max'):
        assert not isinstance(A.max(axis=1), np.matrix), \  # 断言：A 沿着 axis=1 的最大值不是 np.matrix 类型
            "Expected array, got matrix"
    if hasattr(A, 'argmin'):
        assert not isinstance(A.argmin(axis=1), np.matrix), \  # 断言：A 沿着 axis=1 的最小值索引不是 np.matrix 类型
            "Expected array, got matrix"
    if hasattr(A, 'argmax'):
        assert not isinstance(A.argmax(axis=1), np.matrix), \  # 断言：A 沿着 axis=1 的最大值索引不是 np.matrix 类型
            "Expected array, got matrix"


@parametrize_sparrays  # 参数化测试：测试函数 test_todense 针对 sparrays 中的每个稀疏矩阵执行
def test_todense(A):
    assert not isinstance(A.todense(), np.matrix), \  # 断言：A 转换为稠密数组不是 np.matrix 类型
        "Expected array, got matrix"


@parametrize_sparrays  # 参数化测试：测试函数 test_indexing 针对 sparrays 中的每个稀疏矩阵执行
def test_indexing(A):
    if A.__class__.__name__[:3] in ('dia', 'coo', 'bsr'):  # 如果稀疏矩阵类型的前缀是 'dia', 'coo', 'bsr'，则跳过测试
        return

    all_res = (  # 定义所有测试结果的元组
        A[1, :],
        A[:, 1],
        A[1, [1, 2]],
        A[[1, 2], 1],
        A[[0]],
        A[:, [1, 2]],
        A[[1, 2], :],
        A[1, [[1, 2]]],
        A[[[1, 2]], 1],
    )

    for res in all_res:  # 遍历所有测试结果
        assert isinstance(res, scipy.sparse.sparray), \  # 断言：每个结果是 scipy.sparse.sparray 类型
            f"Expected sparse array, got {res._class__.__name__}"


@parametrize_sparrays  # 参数化测试：测试函数 test_dense_addition 针对 sparrays 中的每个稀疏矩阵执行
def test_dense_addition(A):
    X = np.random.random(A.shape)  # 创建一个与 A 形状相同的随机 NumPy 数组 X
    assert not isinstance(A + X, np.matrix), \  # 断言：A 与 X 相加不是 np.matrix 类型
        "Expected array, got matrix"


@parametrize_sparrays  # 参数化测试：测试函数 test_sparse_addition 针对 sparrays 中的每个稀疏矩阵执行
def test_sparse_addition(A):
    assert isinstance((A + A), scipy.sparse.sparray), \  # 断言：A 与自身相加是 scipy.sparse.sparray 类型
        "Expected array, got matrix"


@parametrize_sparrays  # 参数化测试：测试函数 test_elementwise_mul 针对 sparrays 中的每个稀疏矩阵执行
def test_elementwise_mul(A):
    assert np.all((A * A).todense() == A.power(2).todense())  # 断言：A 的元素平方与 A 的平方等效


@parametrize_sparrays  # 参数化测试：测试函数
# 当 A 为 None 时，期望引发 TypeError 异常
with pytest.raises(TypeError):
    None * A

# 当 A 是稀疏矩阵时，期望引发 ValueError 异常
with pytest.raises(ValueError):
    np.eye(3) * scipy.sparse.csr_array(np.arange(6).reshape(2, 3))

# 断言 2*A 的结果与 A 的密集表示乘以 2 的结果相等
assert np.all((2 * A) == (A.todense() * 2))

# 断言 A 的密集表示乘以自身的结果与 A 的密集表示的平方相等
assert np.all((A.todense() * A) == (A.todense() ** 2))


@parametrize_sparrays
def test_matmul(A):
    # 断言 A 与其转置的乘积的密集表示等于 A 点乘其转置的密集表示
    assert np.all((A @ A.T).todense() == A.dot(A.T).todense())


@parametrize_sparrays
def test_power_operator(A):
    # 断言 A 的二次幂的类型为稀疏数组
    assert isinstance((A**2), scipy.sparse.sparray), "Expected array, got matrix"

    # 比较 A 的二次幂的密集表示与 A 的密集表示的平方
    npt.assert_equal((A**2).todense(), (A.todense())**2)

    # 对零次幂进行断言，期望引发 NotImplementedError 异常并匹配 "zero power"
    with pytest.raises(NotImplementedError, match="zero power"):
        A**0


@parametrize_sparrays
def test_sparse_divide(A):
    # 断言 A 除以自身的结果类型为 ndarray
    assert isinstance(A / A, np.ndarray)


@parametrize_sparrays
def test_sparse_dense_divide(A):
    # 在发出运行时警告的情况下，断言 A 除以其密集表示的结果类型为稀疏数组
    with pytest.warns(RuntimeWarning):
        assert isinstance((A / A.todense()), scipy.sparse.sparray)


@parametrize_sparrays
def test_dense_divide(A):
    # 断言 A 除以 2 的结果类型为稀疏数组
    assert isinstance((A / 2), scipy.sparse.sparray), "Expected array, got matrix"


@parametrize_sparrays
def test_no_A_attr(A):
    # 断言 A 没有属性 'A'，期望引发 AttributeError 异常
    with pytest.raises(AttributeError):
        A.A


@parametrize_sparrays
def test_no_H_attr(A):
    # 断言 A 没有属性 'H'，期望引发 AttributeError 异常
    with pytest.raises(AttributeError):
        A.H


@parametrize_sparrays
def test_getrow_getcol(A):
    # 断言 A 的列向量和行向量的类型为稀疏数组
    assert isinstance(A._getcol(0), scipy.sparse.sparray)
    assert isinstance(A._getrow(0), scipy.sparse.sparray)


# -- linalg --

@parametrize_sparrays
def test_as_linearoperator(A):
    # 将 A 转换为线性运算符，并断言其作用于向量 [1, 2, 3, 4] 的结果与 A 乘以向量 [1, 2, 3, 4] 的结果相近
    L = spla.aslinearoperator(A)
    npt.assert_allclose(L * [1, 2, 3, 4], A @ [1, 2, 3, 4])


@parametrize_square_sparrays
def test_inv(B):
    # 当 B 是非 csc 类型时，退出测试
    if B.__class__.__name__[:3] != 'csc':
        return

    # 计算 B 的逆 C，并断言其类型为稀疏数组，且其密集表示与 np.linalg.inv(B.todense()) 相近
    C = spla.inv(B)
    assert isinstance(C, scipy.sparse.sparray)
    npt.assert_allclose(C.todense(), np.linalg.inv(B.todense()))


@parametrize_square_sparrays
def test_expm(B):
    # 当 B 是非 csc 类型时，退出测试
    if B.__class__.__name__[:3] != 'csc':
        return

    # 将 B 转换为密集矩阵 Bmat，计算其指数 e^B 的稀疏数组 C，并断言其密集表示与 expm(Bmat) 的密集表示相近
    Bmat = scipy.sparse.csc_matrix(B)
    C = spla.expm(B)
    assert isinstance(C, scipy.sparse.sparray)
    npt.assert_allclose(
        C.todense(),
        spla.expm(Bmat).todense()
    )


@parametrize_square_sparrays
def test_expm_multiply(B):
    # 当 B 是非 csc 类型时，退出测试
    if B.__class__.__name__[:3] != 'csc':
        return

    # 断言 expm_multiply(B, [1, 2]) 的结果与 expm(B) @ [1, 2] 的结果相近
    npt.assert_allclose(
        spla.expm_multiply(B, np.array([1, 2])),
        spla.expm(B) @ [1, 2]
    )


@parametrize_sparrays
def test_norm(A):
    # 计算 A 的范数 C，并断言其与 A 的密集表示的范数相近
    C = spla.norm(A)
    npt.assert_allclose(C, np.linalg.norm(A.todense()))


@parametrize_square_sparrays
def test_onenormest(B):
    # 计算 B 的 1-范数 C，并断言其与 B 的密集表示的 1-范数相近
    C = spla.onenormest(B)
    npt.assert_allclose(C, np.linalg.norm(B.todense(), 1))


@parametrize_square_sparrays
def test_spsolve(B):
    # 当 B 的类型不是 'csc' 或 'csr' 时，退出测试
    if B.__class__.__name__[:3] not in ('csc', 'csr'):
        return

    # 断言用稀疏矩阵 B 解方程 [1, 2] 的结果与用其密集表示解方程 [1, 2] 的结果相近
    npt.assert_allclose(
        spla.spsolve(B, [1, 2]),
        np.linalg.solve(B.todense(), [1, 2])
    )
# 使用 pytest 的 parametrize 装饰器，为 test_spsolve_triangular 函数生成两个参数化的测试用例
@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_spsolve_triangular(fmt):
    # 定义稀疏矩阵 arr
    arr = [
        [1, 0, 0, 0],
        [2, 1, 0, 0],
        [3, 2, 1, 0],
        [4, 3, 2, 1],
    ]
    # 根据 fmt 参数选择创建 csr 或 csc 格式的稀疏矩阵 X
    if fmt == "csr":
        X = scipy.sparse.csr_array(arr)
    else:
        X = scipy.sparse.csc_array(arr)
    # 调用 spsolve_triangular 函数求解三角线性方程组
    spla.spsolve_triangular(X, [1, 2, 3, 4])


# 使用自定义的 parametrize_square_sparrays 装饰器为 test_factorized 函数注入稀疏矩阵参数 B
@parametrize_square_sparrays
def test_factorized(B):
    # 检查 B 是否为 csc 类型，如果不是则直接返回
    if B.__class__.__name__[:3] != 'csc':
        return

    # 对 B 进行 LU 分解
    LU = spla.factorized(B)
    # 断言 LU 分解的结果与使用 dense 矩阵求解的结果相近
    npt.assert_allclose(
        LU(np.array([1, 2])),
        np.linalg.solve(B.todense(), [1, 2])
    )


# 使用 parametrize_sparrays 和 pytest.mark.parametrize 装饰器为 test_solvers 函数注入稀疏矩阵参数 B 和 solver 参数
@parametrize_sparrays
@pytest.mark.parametrize(
    "solver",
    ["bicg", "bicgstab", "cg", "cgs", "gmres", "lgmres", "minres", "qmr",
     "gcrotmk", "tfqmr"]
)
def test_solvers(B, solver):
    # 根据不同的 solver 设置不同的关键字参数 kwargs
    if solver == "minres":
        kwargs = {}
    else:
        kwargs = {'atol': 1e-5}

    # 调用 sparse.linalg 模块中对应的 solver 函数求解线性方程组
    x, info = getattr(spla, solver)(B, np.array([1, 2]), **kwargs)
    # 断言求解过程中没有错误，并且解 x 与预期值 [1, 1] 相近
    assert info >= 0  # no errors, even if perhaps did not converge fully
    npt.assert_allclose(x, [1, 1], atol=1e-1)


# 使用 parametrize_eig_sparrays 装饰器为 test_eigs, test_eigsh, test_svds 函数注入稀疏矩阵参数 X
@parametrize_eig_sparrays
def test_eigs(X):
    # 调用 sparse.linalg.eigs 函数计算最大特征值和特征向量
    e, v = spla.eigs(X, k=1)
    # 断言特征值分解的精确性
    npt.assert_allclose(
        X @ v,
        e[0] * v
    )


@parametrize_eig_sparrays
def test_eigsh(X):
    # 对称化稀疏矩阵 X
    X = X + X.T
    # 调用 sparse.linalg.eigsh 函数计算最小特征值和特征向量
    e, v = spla.eigsh(X, k=1)
    # 断言特征值分解的精确性
    npt.assert_allclose(
        X @ v,
        e[0] * v
    )


@parametrize_eig_sparrays
def test_svds(X):
    # 使用 sparse.linalg.svds 函数进行奇异值分解，仅保留前三个奇异值
    u, s, vh = spla.svds(X, k=3)
    # 使用 numpy.linalg.svd 函数进行相同操作，比较结果的排序后的奇异值
    u2, s2, vh2 = np.linalg.svd(X.todense())
    s = np.sort(s)
    s2 = np.sort(s2[:3])
    # 断言奇异值的近似性
    npt.assert_allclose(s, s2, atol=1e-3)


# test_splu 函数测试 sparse.linalg.splu 的功能
def test_splu():
    # 创建 csc 格式的稀疏矩阵 X
    X = scipy.sparse.csc_array([
        [1, 0, 0, 0],
        [2, 1, 0, 0],
        [3, 2, 1, 0],
        [4, 3, 2, 1],
    ])
    # 对 X 进行 LU 分解
    LU = spla.splu(X)
    # 断言 LU 分解的结果与预期值的接近度
    npt.assert_allclose(
        LU.solve(np.array([1, 2, 3, 4])),
        np.asarray([1, 0, 0, 0], dtype=np.float64),
        rtol=1e-14, atol=3e-16
    )


# test_spilu 函数测试 sparse.linalg.spilu 的功能
def test_spilu():
    # 创建 csc 格式的稀疏矩阵 X
    X = scipy.sparse.csc_array([
        [1, 0, 0, 0],
        [2, 1, 0, 0],
        [3, 2, 1, 0],
        [4, 3, 2, 1],
    ])
    # 对 X 进行 incomplete LU 分解
    LU = spla.spilu(X)
    # 断言 LU 分解的结果与预期值的接近度
    npt.assert_allclose(
        LU.solve(np.array([1, 2, 3, 4])),
        np.asarray([1, 0, 0, 0], dtype=np.float64),
        rtol=1e-14, atol=3e-16
    )


# 使用两个嵌套的 parametrize 装饰器为 test_index_dtype_compressed 函数注入参数
@pytest.mark.parametrize(
    "cls,indices_attrs",
    [
        (
            scipy.sparse.csr_array,
            ["indices", "indptr"],
        ),
        (
            scipy.sparse.csc_array,
            ["indices", "indptr"],
        ),
        (
            scipy.sparse.coo_array,
            ["row", "col"],
        ),
    ]
)
@pytest.mark.parametrize("expected_dtype", [np.int64, np.int32])
def test_index_dtype_compressed(cls, indices_attrs, expected_dtype):
    # 创建一个稀疏矩阵对象
    input_array = scipy.sparse.coo_array(np.arange(9).reshape(3, 3))
    # 创建一个元组 coo_tuple，包含输入数组的数据和行列索引的元组
    coo_tuple = (
        input_array.data,  # 将输入数组的数据部分作为元组的第一个元素
        (
            input_array.row.astype(expected_dtype),  # 将输入数组的行索引转换为指定的数据类型
            input_array.col.astype(expected_dtype),  # 将输入数组的列索引转换为指定的数据类型
        )
    )

    # 使用 coo_tuple 创建一个新的稀疏矩阵对象，并进行属性检查
    result = cls(coo_tuple)
    for attr in indices_attrs:
        assert getattr(result, attr).dtype == expected_dtype  # 断言检查结果对象的属性数据类型是否符合预期

    # 使用 coo_tuple 和指定的形状创建一个新的稀疏矩阵对象，并进行属性检查
    result = cls(coo_tuple, shape=(3, 3))
    for attr in indices_attrs:
        assert getattr(result, attr).dtype == expected_dtype  # 断言检查结果对象的属性数据类型是否符合预期

    # 如果结果类是 scipy.sparse._compressed._cs_matrix 的子类
    if issubclass(cls, scipy.sparse._compressed._cs_matrix):
        # 将输入数组转换为 CSR 格式
        input_array_csr = input_array.tocsr()
        # 创建一个元组 csr_tuple，包含输入数组 CSR 格式的数据、列索引和行指针
        csr_tuple = (
            input_array_csr.data,  # 将 CSR 格式数组的数据部分作为元组的第一个元素
            input_array_csr.indices.astype(expected_dtype),  # 将 CSR 格式数组的列索引转换为指定的数据类型
            input_array_csr.indptr.astype(expected_dtype),  # 将 CSR 格式数组的行指针转换为指定的数据类型
        )

        # 使用 csr_tuple 创建一个新的 CSR 格式稀疏矩阵对象，并进行属性检查
        result = cls(csr_tuple)
        for attr in indices_attrs:
            assert getattr(result, attr).dtype == expected_dtype  # 断言检查结果对象的属性数据类型是否符合预期

        # 使用 csr_tuple 和指定的形状创建一个新的 CSR 格式稀疏矩阵对象，并进行属性检查
        result = cls(csr_tuple, shape=(3, 3))
        for attr in indices_attrs:
            assert getattr(result, attr).dtype == expected_dtype  # 断言检查结果对象的属性数据类型是否符合预期
# 定义测试函数，用于验证默认情况下生成的稀疏矩阵不是 sparray 类型
def test_default_is_matrix_diags():
    # 创建一个对角矩阵，对角线元素为 [0, 1, 2]
    m = scipy.sparse.diags([0, 1, 2])
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


def test_default_is_matrix_eye():
    # 创建一个单位矩阵，大小为 3x3
    m = scipy.sparse.eye(3)
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


def test_default_is_matrix_spdiags():
    # 使用给定的对角线数组创建一个稀疏矩阵，对角线元素为 [1, 2, 3]
    m = scipy.sparse.spdiags([1, 2, 3], 0, 3, 3)
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


def test_default_is_matrix_identity():
    # 创建一个单位矩阵，大小为 3x3
    m = scipy.sparse.identity(3)
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


def test_default_is_matrix_kron_dense():
    # 使用 Kronecker 乘积创建一个密集矩阵
    m = scipy.sparse.kron(
        np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]])
    )
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


def test_default_is_matrix_kron_sparse():
    # 使用 Kronecker 乘积创建一个稀疏矩阵
    m = scipy.sparse.kron(
        np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 0]])
    )
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


def test_default_is_matrix_kronsum():
    # 使用 Kronecker 和的形式创建一个矩阵
    m = scipy.sparse.kronsum(
        np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])
    )
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


def test_default_is_matrix_random():
    # 创建一个随机稀疏矩阵，大小为 3x3
    m = scipy.sparse.random(3, 3)
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


def test_default_is_matrix_rand():
    # 创建一个随机稀疏矩阵，大小为 3x3
    m = scipy.sparse.rand(3, 3)
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


@pytest.mark.parametrize("fn", (scipy.sparse.hstack, scipy.sparse.vstack))
def test_default_is_matrix_stacks(fn):
    """测试水平和垂直堆叠创建函数，与 `test_default_construction_fn_matrices` 相同的思路。"""
    # 创建两个 COO 稀疏矩阵 A 和 B
    A = scipy.sparse.coo_matrix(np.eye(2))
    B = scipy.sparse.coo_matrix([[0, 1], [1, 0]])
    # 使用给定的堆叠函数 fn 对 A 和 B 进行堆叠操作
    m = fn([A, B])
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


def test_blocks_default_construction_fn_matrices():
    """测试块创建函数，与 `test_default_construction_fn_matrices` 相同的思路。"""
    # 创建三个 COO 稀疏矩阵 A、B 和 C
    A = scipy.sparse.coo_matrix(np.eye(2))
    B = scipy.sparse.coo_matrix([[2], [0]])
    C = scipy.sparse.coo_matrix([[3]])

    # 创建块对角矩阵
    m = scipy.sparse.block_diag((A, B, C))
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)

    # 创建块矩阵
    m = scipy.sparse.bmat([[A, None], [None, C]])
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)


def test_format_property():
    # 遍历稀疏数组类型列表 sparray_types
    for fmt in sparray_types:
        # 获取对应格式的稀疏数组类 arr_cls
        arr_cls = getattr(scipy.sparse, f"{fmt}_array")
        # 使用 arr_cls 创建一个稀疏数组 M，内容为 [[1, 2]]
        M = arr_cls([[1, 2]])
        # 断言 M 的格式属性 format 等于当前格式 fmt
        assert M.format == fmt
        # 断言 M 的私有格式属性 _format 等于当前格式 fmt
        assert M._format == fmt
        # 尝试修改 M 的 format 属性，预期会触发 AttributeError 异常
        with pytest.raises(AttributeError):
            M.format = "qqq"


def test_issparse():
    # 创建一个单位矩阵 m，大小为 3x3
    m = scipy.sparse.eye(3)
    # 将 m 转换为 CSR 格式的稀疏数组 a
    a = scipy.sparse.csr_array(m)
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)
    # 断言 a 是 scipy.sparse.sparray 类型的实例
    assert isinstance(a, scipy.sparse.sparray)

    # 验证稀疏数组和稀疏矩阵都应该被判断为稀疏的
    assert scipy.sparse.issparse(a)
    assert scipy.sparse.issparse(m)

    # 验证 ndarray 和类似数组不是稀疏的
    assert not scipy.sparse.issparse(a.todense())
    assert not scipy.sparse.issparse(m.todense())
    # 创建一个稀疏的 3x3 单位矩阵
    m = scipy.sparse.eye(3)
    # 使用 csr 格式创建稀疏矩阵 a，基于 m 的结构
    a = scipy.sparse.csr_array(m)
    # 断言 m 不是 scipy 稀疏数组的实例
    assert not isinstance(m, scipy.sparse.sparray)
    # 断言 a 是 scipy 稀疏数组的实例
    assert isinstance(a, scipy.sparse.sparray)

    # 这个断言只对稀疏矩阵（sparse matrices）为真，不对稀疏数组（sparse arrays）为真
    assert not scipy.sparse.isspmatrix(a)
    # 断言 m 是稀疏矩阵
    assert scipy.sparse.isspmatrix(m)

    # ndarray 和 array_likes 不是稀疏的
    assert not scipy.sparse.isspmatrix(a.todense())
    assert not scipy.sparse.isspmatrix(m.todense())
# 使用 pytest 的 @pytest.mark.parametrize 装饰器定义参数化测试函数，用于测试稀疏矩阵格式识别函数
@pytest.mark.parametrize(
    # 定义参数 fmt 和 fn，分别对应稀疏矩阵格式字符串和对应的格式识别函数
    ("fmt", "fn"),
    (
        # 参数化元组，每个元组包含一个格式字符串和对应的 scipy.sparse 稀疏矩阵格式识别函数
        ("bsr", scipy.sparse.isspmatrix_bsr),
        ("coo", scipy.sparse.isspmatrix_coo),
        ("csc", scipy.sparse.isspmatrix_csc),
        ("csr", scipy.sparse.isspmatrix_csr),
        ("dia", scipy.sparse.isspmatrix_dia),
        ("dok", scipy.sparse.isspmatrix_dok),
        ("lil", scipy.sparse.isspmatrix_lil),
    ),
)
# 定义测试函数 test_isspmatrix_format(fmt, fn)，其中 fmt 和 fn 是参数化的输入
def test_isspmatrix_format(fmt, fn):
    # 创建一个 3x3 的单位稀疏矩阵 m，使用指定的格式 fmt
    m = scipy.sparse.eye(3, format=fmt)
    # 将 m 转换为 csr 格式，并转换为指定的 fmt 格式的稀疏矩阵 a
    a = scipy.sparse.csr_matrix(m).asformat(fmt)
    
    # 断言 m 不是 scipy.sparse.sparray 类型的实例
    assert not isinstance(m, scipy.sparse.sparray)
    # 断言 a 是 scipy.sparse.sparray 类型的实例
    assert isinstance(a, scipy.sparse.sparray)

    # 断言对于稀疏矩阵 a，fn(a) 应该为 False（因为 a 是稀疏矩阵，不是数组）
    assert not fn(a)
    # 断言对于稀疏矩阵 m，fn(m) 应该为 True
    assert fn(m)

    # ndarray 和 array_like 不是稀疏矩阵，因此 fn 应返回 False
    assert not fn(a.todense())
    assert not fn(m.todense())
```