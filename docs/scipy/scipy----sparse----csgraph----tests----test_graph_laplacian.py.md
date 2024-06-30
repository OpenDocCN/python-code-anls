# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\tests\test_graph_laplacian.py`

```
# 导入必要的库和模块
import pytest  # 导入pytest模块，用于编写和运行测试程序
import numpy as np  # 导入NumPy库，用于支持大量的数学函数和矩阵运算
from numpy.testing import assert_allclose  # 从NumPy的测试模块中导入函数，用于比较所有元素是否接近
from pytest import raises as assert_raises  # 从pytest模块中导入异常断言函数raises，重命名为assert_raises
from scipy import sparse  # 导入SciPy稀疏矩阵库，用于处理稀疏矩阵

from scipy.sparse import csgraph  # 从SciPy稀疏矩阵库中导入图论函数csgraph
from scipy._lib._util import np_long, np_ulong  # 从SciPy内部工具模块中导入特定类型

# 检查矩阵是否为整数类型（有符号整数或无符号长整数）
def check_int_type(mat):
    return np.issubdtype(mat.dtype, np.signedinteger) or np.issubdtype(
        mat.dtype, np_ulong
    )

# 测试拉普拉斯矩阵函数是否能正确处理值错误的情况
def test_laplacian_value_error():
    for t in int, float, complex:
        for m in ([1, 1],
                  [[[1]]],
                  [[1, 2, 3], [4, 5, 6]],
                  [[1, 2], [3, 4], [5, 5]]):
            A = np.array(m, dtype=t)
            assert_raises(ValueError, csgraph.laplacian, A)

# 计算显式的拉普拉斯矩阵
def _explicit_laplacian(x, normed=False):
    if sparse.issparse(x):
        x = x.toarray()  # 将稀疏矩阵转换为密集矩阵
    x = np.asarray(x)  # 将输入转换为NumPy数组
    y = -1.0 * x  # 初始化y为-x的矩阵
    for j in range(y.shape[0]):
        y[j,j] = x[j,j+1:].sum() + x[j,:j].sum()  # 计算非对角线元素的和并更新y
    if normed:
        d = np.diag(y).copy()  # 提取对角线元素并复制到d
        d[d == 0] = 1.0  # 将零元素置为1.0，避免除以零
        y /= d[:,None]**.5  # 根据行进行归一化
        y /= d[None,:]**.5  # 根据列进行归一化
    return y  # 返回计算得到的拉普拉斯矩阵

# 检查对称图拉普拉斯矩阵的计算是否正确
def _check_symmetric_graph_laplacian(mat, normed, copy=True):
    if not hasattr(mat, 'shape'):
        mat = eval(mat, dict(np=np, sparse=sparse))  # 使用给定的环境评估mat表达式

    if sparse.issparse(mat):
        sp_mat = mat
        mat = sp_mat.toarray()  # 将稀疏矩阵转换为密集矩阵
    else:
        sp_mat = sparse.csr_matrix(mat)  # 将密集矩阵转换为稀疏矩阵

    mat_copy = np.copy(mat)  # 复制密集矩阵mat
    sp_mat_copy = sparse.csr_matrix(sp_mat, copy=True)  # 复制稀疏矩阵sp_mat

    n_nodes = mat.shape[0]  # 获取节点数
    explicit_laplacian = _explicit_laplacian(mat, normed=normed)  # 计算显式的拉普拉斯矩阵
    laplacian = csgraph.laplacian(mat, normed=normed, copy=copy)  # 使用SciPy计算拉普拉斯矩阵
    sp_laplacian = csgraph.laplacian(sp_mat, normed=normed,
                                     copy=copy)  # 使用SciPy计算稀疏矩阵的拉普拉斯矩阵

    if copy:
        assert_allclose(mat, mat_copy)  # 断言mat和其复制品mat_copy的所有元素是否接近
        _assert_allclose_sparse(sp_mat, sp_mat_copy)  # 断言稀疏矩阵sp_mat和其复制品sp_mat_copy的所有元素是否接近
    else:
        if not (normed and check_int_type(mat)):
            assert_allclose(laplacian, mat)  # 断言拉普拉斯矩阵laplacian和mat的所有元素是否接近
            if sp_mat.format == 'coo':
                _assert_allclose_sparse(sp_laplacian, sp_mat)  # 断言稀疏拉普拉斯矩阵sp_laplacian和sp_mat的所有元素是否接近

    assert_allclose(laplacian, sp_laplacian.toarray())  # 断言拉普拉斯矩阵laplacian和稀疏拉普拉斯矩阵sp_laplacian的所有元素是否接近

    for tested in [laplacian, sp_laplacian.toarray()]:
        if not normed:
            assert_allclose(tested.sum(axis=0), np.zeros(n_nodes))  # 断言拉普拉斯矩阵tested每列的和是否接近零向量
        assert_allclose(tested.T, tested)  # 断言拉普拉斯矩阵tested是否对称
        assert_allclose(tested, explicit_laplacian)  # 断言拉普拉斯矩阵tested和显式计算的拉普拉斯矩阵是否接近

# 测试对称图拉普拉斯矩阵函数是否能正确处理
def test_symmetric_graph_laplacian():
    symmetric_mats = (
        'np.arange(10) * np.arange(10)[:, np.newaxis]',  # 生成一个对称矩阵
        'np.ones((7, 7))',  # 生成一个全1矩阵
        'np.eye(19)',  # 生成一个单位矩阵
        'sparse.diags([1, 1], [-1, 1], shape=(4, 4))',  # 生成一个稀疏对角矩阵
        'sparse.diags([1, 1], [-1, 1], shape=(4, 4)).toarray()',  # 转换稀疏对角矩阵为密集矩阵
        'sparse.diags([1, 1], [-1, 1], shape=(4, 4)).todense()',  # 转换稀疏对角矩阵为密集矩阵
        'np.vander(np.arange(4)) + np.vander(np.arange(4)).T'  # 生成一个对称矩阵
    )
    for mat in symmetric_mats:
        for normed in True, False:
            for copy in True, False:
                _check_symmetric_graph_laplacian(mat, normed, copy)

# 辅助函数，用于比较稀疏矩阵a和b的所有元素是否接近
def _assert_allclose_sparse(a, b, **kwargs):
    if sparse.issparse(a):
        a = a.toarray()  # 将稀疏矩阵转换为密集矩阵
    # 如果 b 是稀疏矩阵，则将其转换为密集数组表示
    if sparse.issparse(b):
        b = b.toarray()
    # 断言 a 和 b 在给定的误差范围内非常接近，使用传入的参数 kwargs 进行配置
    assert_allclose(a, b, **kwargs)
# 检查拉普拉斯矩阵的数据类型是否为 None，用于特定设置下的测试
def _check_laplacian_dtype_none(
    A, desired_L, desired_d, normed, use_out_degree, copy, dtype, arr_type
):
    # 将输入数组 A 转换为指定数据类型的数组
    mat = arr_type(A, dtype=dtype)
    # 计算拉普拉斯矩阵 L 和对角线数组 d
    L, d = csgraph.laplacian(
        mat,
        normed=normed,
        return_diag=True,
        use_out_degree=use_out_degree,
        copy=copy,
        dtype=None,
    )
    # 如果启用了 normed 并且输入矩阵类型为整数类型，则断言 L 和 d 的数据类型为 np.float64
    if normed and check_int_type(mat):
        assert L.dtype == np.float64
        assert d.dtype == np.float64
        # 断言计算的 L 和 d 与预期的 desired_L 和 desired_d 接近
        _assert_allclose_sparse(L, desired_L, atol=1e-12)
        _assert_allclose_sparse(d, desired_d, atol=1e-12)
    else:
        # 否则断言 L 和 d 的数据类型与指定的 dtype 相同，并将 desired_L 和 desired_d 转换为指定的 dtype
        assert L.dtype == dtype
        assert d.dtype == dtype
        desired_L = np.asarray(desired_L).astype(dtype)
        desired_d = np.asarray(desired_d).astype(dtype)
        # 断言计算的 L 和 d 与转换后的 desired_L 和 desired_d 接近
        _assert_allclose_sparse(L, desired_L, atol=1e-12)
        _assert_allclose_sparse(d, desired_d, atol=1e-12)

    # 如果不复制矩阵（copy=False）
    if not copy:
        # 如果未启用 normed 或输入矩阵类型不是整数类型
        if not (normed and check_int_type(mat)):
            # 根据输入矩阵的类型进行不同的断言
            if type(mat) is np.ndarray:
                assert_allclose(L, mat)
            elif mat.format == "coo":
                _assert_allclose_sparse(L, mat)


# 检查拉普拉斯矩阵的数据类型，用于通用设置下的测试
def _check_laplacian_dtype(
    A, desired_L, desired_d, normed, use_out_degree, copy, dtype, arr_type
):
    # 将输入数组 A 转换为指定数据类型的数组
    mat = arr_type(A, dtype=dtype)
    # 计算拉普拉斯矩阵 L 和对角线数组 d
    L, d = csgraph.laplacian(
        mat,
        normed=normed,
        return_diag=True,
        use_out_degree=use_out_degree,
        copy=copy,
        dtype=dtype,
    )
    # 断言 L 和 d 的数据类型与指定的 dtype 相同，并将 desired_L 和 desired_d 转换为指定的 dtype
    assert L.dtype == dtype
    assert d.dtype == dtype
    desired_L = np.asarray(desired_L).astype(dtype)
    desired_d = np.asarray(desired_d).astype(dtype)
    # 断言计算的 L 和 d 与转换后的 desired_L 和 desired_d 接近
    _assert_allclose_sparse(L, desired_L, atol=1e-12)
    _assert_allclose_sparse(d, desired_d, atol=1e-12)

    # 如果不复制矩阵（copy=False）
    if not copy:
        # 如果未启用 normed 或输入矩阵类型不是整数类型
        if not (normed and check_int_type(mat)):
            # 根据输入矩阵的类型进行不同的断言
            if type(mat) is np.ndarray:
                assert_allclose(L, mat)
            elif mat.format == 'coo':
                _assert_allclose_sparse(L, mat)


# 整数数据类型集合
INT_DTYPES = {np.intc, np_long, np.longlong}
# 实数数据类型集合
REAL_DTYPES = {np.float32, np.float64, np.longdouble}
# 复数数据类型集合
COMPLEX_DTYPES = {np.complex64, np.complex128, np.clongdouble}
# 使用排序列表确保测试的固定顺序
DTYPES = sorted(INT_DTYPES ^ REAL_DTYPES ^ COMPLEX_DTYPES, key=str)


# 参数化测试函数，测试不同设置下的拉普拉斯矩阵计算
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("arr_type", [np.array,
                                      sparse.csr_matrix,
                                      sparse.coo_matrix,
                                      sparse.csr_array,
                                      sparse.coo_array])
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("normed", [True, False])
@pytest.mark.parametrize("use_out_degree", [True, False])
def test_asymmetric_laplacian(use_out_degree, normed,
                              copy, dtype, arr_type):
    # 邻接矩阵 A 的定义
    A = [[0, 1, 0],
         [4, 2, 0],
         [0, 0, 0]]
    # 将 A 转换为指定数据类型的数组
    A = arr_type(np.array(A), dtype=dtype)
    # 复制 A 以备后用
    A_copy = A.copy()
    if not normed and use_out_degree:
        # 如果不进行归一化且使用出度，构造非归一化的拉普拉斯矩阵
        L = [[1, -1, 0],
             [-4, 4, 0],
             [0, 0, 0]]
        # 设置度数向量
        d = [1, 4, 0]

    if normed and use_out_degree:
        # 如果进行归一化且使用出度，构造归一化的拉普拉斯矩阵
        L = [[1, -0.5, 0],
             [-2, 1, 0],
             [0, 0, 0]]
        # 设置度数向量
        d = [1, 2, 1]

    if not normed and not use_out_degree:
        # 如果不进行归一化且不使用出度，构造非归一化的拉普拉斯矩阵（使用入度）
        L = [[4, -1, 0],
             [-4, 1, 0],
             [0, 0, 0]]
        # 设置度数向量
        d = [4, 1, 0]

    if normed and not use_out_degree:
        # 如果进行归一化且不使用出度，构造归一化的拉普拉斯矩阵（使用入度）
        L = [[1, -0.5, 0],
             [-2, 1, 0],
             [0, 0, 0]]
        # 设置度数向量
        d = [2, 1, 1]

    # 检查拉普拉斯矩阵的数据类型是否为None，并进行相关检查
    _check_laplacian_dtype_none(
        A,
        L,
        d,
        normed=normed,
        use_out_degree=use_out_degree,
        copy=copy,
        dtype=dtype,
        arr_type=arr_type,
    )

    # 检查拉普拉斯矩阵的数据类型，并进行相关检查
    _check_laplacian_dtype(
        A_copy,
        L,
        d,
        normed=normed,
        use_out_degree=use_out_degree,
        copy=copy,
        dtype=dtype,
        arr_type=arr_type,
    )
@pytest.mark.parametrize("fmt", ['csr', 'csc', 'coo', 'lil',
                                 'dok', 'dia', 'bsr'])
@pytest.mark.parametrize("normed", [True, False])
@pytest.mark.parametrize("copy", [True, False])
def test_sparse_formats(fmt, normed, copy):
    # 使用pytest的parametrize装饰器，对格式(fmt)、归一化(normed)、复制(copy)参数进行参数化测试

    # 创建稀疏矩阵(mat)并调用检查对称图拉普拉斯矩阵的函数进行测试
    mat = sparse.diags([1, 1], [-1, 1], shape=(4, 4), format=fmt)
    _check_symmetric_graph_laplacian(mat, normed, copy)


@pytest.mark.parametrize(
    "arr_type", [np.asarray,
                 sparse.csr_matrix,
                 sparse.coo_matrix,
                 sparse.csr_array,
                 sparse.coo_array]
)
@pytest.mark.parametrize("form", ["array", "function", "lo"])
def test_laplacian_symmetrized(arr_type, form):
    # 使用pytest的parametrize装饰器，对数组类型(arr_type)和形式(form)参数进行参数化测试

    # 创建邻接矩阵(mat)
    n = 3
    mat = arr_type(np.arange(n * n).reshape(n, n))

    # 计算拉普拉斯矩阵(L_in, d_in)，返回对角线(d_in)
    L_in, d_in = csgraph.laplacian(
        mat,
        return_diag=True,
        form=form,
    )

    # 计算拉普拉斯矩阵(L_out, d_out)，返回对角线(d_out)
    L_out, d_out = csgraph.laplacian(
        mat,
        return_diag=True,
        use_out_degree=True,
        form=form,
    )

    # 计算对称化拉普拉斯矩阵(Ls, ds)，返回对角线(ds)
    Ls, ds = csgraph.laplacian(
        mat,
        return_diag=True,
        symmetrized=True,
        form=form,
    )

    # 计算归一化对称化拉普拉斯矩阵(Ls_normed, ds_normed)，返回对角线(ds_normed)
    Ls_normed, ds_normed = csgraph.laplacian(
        mat,
        return_diag=True,
        symmetrized=True,
        normed=True,
        form=form,
    )

    # 将矩阵(mat)与其转置相加
    mat += mat.T

    # 计算对称化后的拉普拉斯矩阵(Lss, dss)，返回对角线(dss)
    Lss, dss = csgraph.laplacian(mat, return_diag=True, form=form)

    # 计算归一化后的对称化拉普拉斯矩阵(Lss_normed, dss_normed)，返回对角线(dss_normed)
    Lss_normed, dss_normed = csgraph.laplacian(
        mat,
        return_diag=True,
        normed=True,
        form=form,
    )

    # 断言检查结果是否接近
    assert_allclose(ds, d_in + d_out)
    assert_allclose(ds, dss)
    assert_allclose(ds_normed, dss_normed)

    # 创建字典(d)，存储各种形式的拉普拉斯矩阵和对角线
    d = {}
    for L in ["L_in", "L_out", "Ls", "Ls_normed", "Lss", "Lss_normed"]:
        if form == "array":
            d[L] = eval(L)
        else:
            d[L] = eval(L)(np.eye(n, dtype=mat.dtype))

    # 断言检查稀疏矩阵是否接近
    _assert_allclose_sparse(d["Ls"], d["L_in"] + d["L_out"].T)
    _assert_allclose_sparse(d["Ls"], d["Lss"])
    _assert_allclose_sparse(d["Ls_normed"], d["Lss_normed"])


@pytest.mark.parametrize(
    "arr_type", [np.asarray,
                 sparse.csr_matrix,
                 sparse.coo_matrix,
                 sparse.csr_array,
                 sparse.coo_array]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("normed", [True, False])
@pytest.mark.parametrize("symmetrized", [True, False])
@pytest.mark.parametrize("use_out_degree", [True, False])
@pytest.mark.parametrize("form", ["function", "lo"])
def test_format(dtype, arr_type, normed, symmetrized, use_out_degree, form):
    # 使用pytest的parametrize装饰器，对数据类型(dtype)、数组类型(arr_type)、归一化(normed)、
    # 对称化(symmetrized)、使用出度(use_out_degree)和形式(form)参数进行参数化测试

    # 创建邻接矩阵(mat)
    n = 3
    mat = [[0, 1, 0], [4, 2, 0], [0, 0, 0]]
    mat = arr_type(np.array(mat), dtype=dtype)

    # 计算拉普拉斯矩阵(Lo, do)，返回对角线(do)
    Lo, do = csgraph.laplacian(
        mat,
        return_diag=True,
        normed=normed,
        symmetrized=symmetrized,
        use_out_degree=use_out_degree,
        dtype=dtype,
    )
    # 计算输入矩阵的拉普拉斯矩阵和对角线，并返回结果
    La, da = csgraph.laplacian(
        mat,
        return_diag=True,         # 返回对角线信息
        normed=normed,            # 是否进行归一化处理
        symmetrized=symmetrized,  # 是否对称化处理
        use_out_degree=use_out_degree,  # 是否使用出度信息
        dtype=dtype,              # 指定数据类型
        form="array",             # 返回的形式为数组
    )
    
    # 断言两个数组的内容在数值上接近
    assert_allclose(do, da)
    
    # 断言稀疏矩阵 Lo 和 La 在数值上接近
    _assert_allclose_sparse(Lo, La)
    
    # 计算输入矩阵的拉普拉斯矩阵和对角线，并返回结果
    L, d = csgraph.laplacian(
        mat,
        return_diag=True,         # 返回对角线信息
        normed=normed,            # 是否进行归一化处理
        symmetrized=symmetrized,  # 是否对称化处理
        use_out_degree=use_out_degree,  # 是否使用出度信息
        dtype=dtype,              # 指定数据类型
        form=form,                # 返回的形式为指定的形式
    )
    
    # 断言两个对角线数组 d 在数值上接近
    assert_allclose(d, do)
    
    # 断言对角线数组 d 的数据类型符合指定的 dtype
    assert d.dtype == dtype
    
    # 使用 L 函数对单位矩阵进行操作，并将结果转换为指定的数据类型
    Lm = L(np.eye(n, dtype=mat.dtype)).astype(dtype)
    
    # 断言稀疏矩阵 Lm 和 Lo 在数值上接近，设定相对误差和绝对误差
    _assert_allclose_sparse(Lm, Lo, rtol=2e-7, atol=2e-7)
    
    # 创建一个 3x2 的数组 x
    x = np.arange(6).reshape(3, 2)
    
    # 如果不是归一化或者 dtype 是整数类型，断言 L(x) 等于 Lo @ x
    if not (normed and dtype in INT_DTYPES):
        assert_allclose(L(x), Lo @ x)
    else:
        # 如果归一化且 dtype 是整数类型，则跳过该断言
        pass
# 定义测试函数 test_format_error_message，用于测试格式错误消息
def test_format_error_message():
    # 使用 pytest 的上下文管理器，期望捕获 ValueError 异常，并匹配错误消息 "Invalid form: 'toto'"
    with pytest.raises(ValueError, match="Invalid form: 'toto'"):
        # 调用 csgraph.laplacian 函数，传入 np.eye(1) 作为参数，并指定 form='toto'
        _ = csgraph.laplacian(np.eye(1), form='toto')
```