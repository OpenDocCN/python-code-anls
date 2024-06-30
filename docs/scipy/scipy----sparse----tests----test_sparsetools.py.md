# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_sparsetools.py`

```
# 导入系统相关的模块
import sys
# 导入操作系统相关的模块
import os
# 导入垃圾回收模块
import gc
# 导入多线程处理模块
import threading

# 导入 numpy 库，并导入部分子模块和函数
import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
# 导入 scipy.sparse 库中的稀疏矩阵相关模块
from scipy.sparse import (_sparsetools, coo_matrix, csr_matrix, csc_matrix,
                          bsr_matrix, dia_matrix)
# 导入 scipy.sparse._sputils 模块中的 supported_dtypes 函数
from scipy.sparse._sputils import supported_dtypes
# 导入 scipy._lib._testutils 模块中的 check_free_memory 函数

import pytest
# 导入 pytest 中的 raises 函数并重命名为 assert_raises
from pytest import raises as assert_raises


def int_to_int8(n):
    """
    Wrap an integer to the interval [-128, 127].
    将整数 n 包装到区间 [-128, 127] 内。
    """
    return (n + 128) % 256 - 128


def test_exception():
    # 测试函数，验证是否会抛出 MemoryError 异常
    assert_raises(MemoryError, _sparsetools.test_throw_error)


def test_threads():
    # 并行线程执行的简单测试；不实际检查代码是否并行运行，
    # 只验证其产生了预期的结果。
    nthreads = 10  # 线程数
    niter = 100  # 迭代次数

    n = 20
    a = csr_matrix(np.ones([n, n]))  # 创建一个稀疏矩阵 a
    bres = []  # 存储每个线程的结果列表

    class Worker(threading.Thread):
        # 线程类定义
        def run(self):
            b = a.copy()  # 复制稀疏矩阵 a 到 b
            for j in range(niter):
                # 使用 sparsetools 中的 csr_plus_csr 函数进行稀疏矩阵相加
                _sparsetools.csr_plus_csr(n, n,
                                          a.indptr, a.indices, a.data,
                                          a.indptr, a.indices, a.data,
                                          b.indptr, b.indices, b.data)
            bres.append(b)  # 将结果 b 存入 bres 列表

    threads = [Worker() for _ in range(nthreads)]  # 创建 nthreads 个 Worker 线程
    for thread in threads:
        thread.start()  # 启动每个线程
    for thread in threads:
        thread.join()  # 等待所有线程执行完毕

    for b in bres:
        # 验证所有结果 b 是否符合预期，即所有元素是否均为 2
        assert_(np.all(b.toarray() == 2))


def test_regression_std_vector_dtypes():
    # 回归测试 gh-3780，验证 sparsetools.cxx 中的 std::vector typemaps 是否完整。
    for dtype in supported_dtypes:
        ad = np.array([[1, 2], [3, 4]]).astype(dtype)  # 创建指定 dtype 的数组 ad
        a = csr_matrix(ad, dtype=dtype)  # 使用指定 dtype 创建稀疏矩阵 a

        # 调用 getcol 方法使用 std::vector typemaps，验证不应该失败
        assert_equal(a.getcol(0).toarray(), ad[:, :1])


@pytest.mark.slow
@pytest.mark.xfail_on_32bit("Can't create large array for test")
def test_nnz_overflow():
    # 回归测试 gh-7230 / gh-7871，验证当 nnz > int32max 时 coo_toarray 是否会溢出。
    nnz = np.iinfo(np.int32).max + 1  # 设置 nnz 的值超过 int32 的最大值
    # 确保有足够的内存（约 20 GB）来运行此测试。
    check_free_memory((4 + 4 + 1) * nnz / 1e6 + 0.5)

    # 使用 nnz 个重复条目以保持稠密版本较小。
    row = np.zeros(nnz, dtype=np.int32)
    col = np.zeros(nnz, dtype=np.int32)
    data = np.zeros(nnz, dtype=np.int8)
    data[-1] = 4
    s = coo_matrix((data, (row, col)), shape=(1, 1), copy=False)
    # 求和 nnz 个重复项，生成包含 4 的 1x1 数组。
    d = s.toarray()

    assert_allclose(d, [[4]])


@pytest.mark.skipif(
    not (sys.platform.startswith('linux') and np.dtype(np.intp).itemsize >= 8),
    reason="test requires 64-bit Linux"
)
class TestInt32Overflow:
    """
    Some of the sparsetools routines use dense 2D matrices whose
    total size is not bounded by the nnz of the sparse matrix. These
    """
    pass
    routines`
    routines used to suffer from int32 wraparounds; here, we try to
    check that the wraparounds don't occur any more.
    """
    # 定义一个足够大的整数 n
    n = 50000

    def setup_method(self):
        # 断言 n 的平方大于 np.int32 的最大值，检查是否会发生整数溢出
        assert self.n**2 > np.iinfo(np.int32).max

        # 检查是否有足够的内存，即使所有内容同时运行
        try:
            parallel_count = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', '1'))
        except ValueError:
            parallel_count = np.inf

        check_free_memory(3000 * parallel_count)

    def teardown_method(self):
        # 执行垃圾回收
        gc.collect()

    def test_coo_todense(self):
        # 检查 *_todense 函数（参见 gh-2179）
        #
        # 最终所有这些函数调用 coo_matrix.todense

        n = self.n

        i = np.array([0, n-1])
        j = np.array([0, n-1])
        data = np.array([1, 2], dtype=np.int8)
        m = coo_matrix((data, (i, j)))

        r = m.todense()
        assert_equal(r[0,0], 1)
        assert_equal(r[-1,-1], 2)
        del r
        gc.collect()

    @pytest.mark.slow
    def test_matvecs(self):
        # 检查 *_matvecs 函数
        n = self.n

        i = np.array([0, n-1])
        j = np.array([0, n-1])
        data = np.array([1, 2], dtype=np.int8)
        m = coo_matrix((data, (i, j)))

        b = np.ones((n, n), dtype=np.int8)
        for sptype in (csr_matrix, csc_matrix, bsr_matrix):
            m2 = sptype(m)
            r = m2.dot(b)
            assert_equal(r[0,0], 1)
            assert_equal(r[-1,-1], 2)
            del r
            gc.collect()

        del b
        gc.collect()

    @pytest.mark.slow
    def test_dia_matvec(self):
        # 检查大型 dia_matrix 的 _matvec 操作
        n = self.n
        data = np.ones((n, n), dtype=np.int8)
        offsets = np.arange(n)
        m = dia_matrix((data, offsets), shape=(n, n))
        v = np.ones(m.shape[1], dtype=np.int8)
        r = m.dot(v)
        assert_equal(r[0], int_to_int8(n))
        del data, offsets, m, v, r
        gc.collect()

    _bsr_ops = [pytest.param("matmat", marks=pytest.mark.xslow),
                pytest.param("matvecs", marks=pytest.mark.xslow),
                "matvec",
                "diagonal",
                "sort_indices",
                pytest.param("transpose", marks=pytest.mark.xslow)]

    @pytest.mark.slow
    @pytest.mark.parametrize("op", _bsr_ops)
    def test_bsr_1_block(self, op):
        # Check: huge bsr_matrix (1-block)
        #
        # The point here is that indices inside a block may overflow.

        def get_matrix():
            # 获取一个矩阵对象，该对象是一个巨大的块压缩稀疏行矩阵 (1-block)
            n = self.n
            data = np.ones((1, n, n), dtype=np.int8)  # 创建全为1的数据数组
            indptr = np.array([0, 1], dtype=np.int32)  # 行指针数组
            indices = np.array([0], dtype=np.int32)    # 索引数组
            m = bsr_matrix((data, indices, indptr), blocksize=(n, n), copy=False)  # 创建块压缩稀疏行矩阵对象
            del data, indptr, indices  # 删除不再需要的数组，释放内存
            return m

        gc.collect()  # 手动触发垃圾回收
        try:
            getattr(self, "_check_bsr_" + op)(get_matrix)  # 调用特定的检查函数
        finally:
            gc.collect()  # 最终再次触发垃圾回收

    @pytest.mark.slow
    @pytest.mark.parametrize("op", _bsr_ops)
    def test_bsr_n_block(self, op):
        # Check: huge bsr_matrix (n-block)
        #
        # The point here is that while indices within a block don't
        # overflow, accumulators across many block may.

        def get_matrix():
            # 获取一个矩阵对象，该对象是一个巨大的块压缩稀疏行矩阵 (n-block)
            n = self.n
            data = np.ones((n, n, 1), dtype=np.int8)  # 创建全为1的数据数组
            indptr = np.array([0, n], dtype=np.int32)  # 行指针数组
            indices = np.arange(n, dtype=np.int32)    # 索引数组
            m = bsr_matrix((data, indices, indptr), blocksize=(n, 1), copy=False)  # 创建块压缩稀疏行矩阵对象
            del data, indptr, indices  # 删除不再需要的数组，释放内存
            return m

        gc.collect()  # 手动触发垃圾回收
        try:
            getattr(self, "_check_bsr_" + op)(get_matrix)  # 调用特定的检查函数
        finally:
            gc.collect()  # 最终再次触发垃圾回收

    def _check_bsr_matvecs(self, m):  # skip name check
        m = m()
        n = self.n

        # _matvecs
        r = m.dot(np.ones((n, 2), dtype=np.int8))  # 进行矩阵乘法运算
        assert_equal(r[0, 0], int_to_int8(n))  # 断言结果符合预期

    def _check_bsr_matvec(self, m):  # skip name check
        m = m()
        n = self.n

        # _matvec
        r = m.dot(np.ones((n,), dtype=np.int8))  # 进行矩阵乘法运算
        assert_equal(r[0], int_to_int8(n))  # 断言结果符合预期

    def _check_bsr_diagonal(self, m):  # skip name check
        m = m()
        n = self.n

        # _diagonal
        r = m.diagonal()  # 获取矩阵的对角线元素
        assert_equal(r, np.ones(n))  # 断言对角线元素为全1

    def _check_bsr_sort_indices(self, m):  # skip name check
        # _sort_indices
        m = m()
        m.sort_indices()  # 对矩阵的索引进行排序

    def _check_bsr_transpose(self, m):  # skip name check
        # _transpose
        m = m()
        m.transpose()  # 对矩阵进行转置操作

    def _check_bsr_matmat(self, m):  # skip name check
        m = m()
        n = self.n

        # _bsr_matmat
        m2 = bsr_matrix(np.ones((n, 2), dtype=np.int8), blocksize=(m.blocksize[1], 2))  # 创建块压缩稀疏行矩阵对象
        m.dot(m2)  # 矩阵乘法运算，不应导致 SIGSEGV 错误
        del m2  # 删除不再需要的对象，释放内存

        # _bsr_matmat
        m2 = bsr_matrix(np.ones((2, n), dtype=np.int8), blocksize=(2, m.blocksize[0]))  # 创建块压缩稀疏行矩阵对象
        m2.dot(m)  # 矩阵乘法运算，不应导致 SIGSEGV 错误
# 将此测试用例标记为跳过，原因是稀疏矩阵中的64位索引不可用
@pytest.mark.skip(reason="64-bit indices in sparse matrices not available")
def test_csr_matmat_int64_overflow():
    # 定义一个大数n，确保其平方大于np.int64的最大值
    n = 3037000500
    assert n**2 > np.iinfo(np.int64).max

    # 测试将消耗大量内存
    check_free_memory(n * (8*2 + 1) * 3 / 1e6)

    # 创建一个稀疏矩阵，数据为全1的np.int8数组
    data = np.ones((n,), dtype=np.int8)
    # 创建一个索引指针数组，从0到n+1，数据类型为np.int64
    indptr = np.arange(n+1, dtype=np.int64)
    # 创建一个索引数组，全为0，数据类型为np.int64
    indices = np.zeros(n, dtype=np.int64)
    # 使用以上数组创建一个CSR格式的稀疏矩阵a
    a = csr_matrix((data, indices, indptr))
    # 计算a的转置矩阵b
    b = a.T

    # 断言：执行a.dot(b)会引发RuntimeError异常
    assert_raises(RuntimeError, a.dot, b)


def test_upcast():
    # 创建一个复数类型的CSR稀疏矩阵a0
    a0 = csr_matrix([[np.pi, np.pi*1j], [3, 4]], dtype=complex)
    # 创建一个复数类型的数组b0
    b0 = np.array([256+1j, 2**32], dtype=complex)

    # 对于支持的数据类型中的每一对(a_dtype, b_dtype)
    for a_dtype in supported_dtypes:
        for b_dtype in supported_dtypes:
            msg = f"({a_dtype!r}, {b_dtype!r})"

            # 如果a_dtype是复数浮点类型，则使用复数部分创建a，否则使用实部创建a
            if np.issubdtype(a_dtype, np.complexfloating):
                a = a0.copy().astype(a_dtype)
            else:
                a = a0.real.copy().astype(a_dtype)

            # 如果b_dtype是复数浮点类型，则使用b0，否则将b0的实部转换为b_dtype类型，忽略警告
            if np.issubdtype(b_dtype, np.complexfloating):
                b = b0.copy().astype(b_dtype)
            else:
                with np.errstate(invalid="ignore"):
                    b = b0.real.copy().astype(b_dtype)

            # 如果a_dtype和b_dtype不同时为布尔类型，则创建一个布尔类型的空数组c，并断言调用_sparsetools.csr_matvec会引发ValueError异常
            if not (a_dtype == np.bool_ and b_dtype == np.bool_):
                c = np.zeros((2,), dtype=np.bool_)
                assert_raises(ValueError, _sparsetools.csr_matvec,
                              2, 2, a.indptr, a.indices, a.data, b, c)

            # 如果a_dtype是复数浮点类型而b_dtype不是，或者a_dtype不是复数浮点类型而b_dtype是，则创建一个双精度浮点类型的空数组c，并断言调用_sparsetools.csr_matvec不会引发异常
            if ((np.issubdtype(a_dtype, np.complexfloating) and
                 not np.issubdtype(b_dtype, np.complexfloating)) or
                (not np.issubdtype(a_dtype, np.complexfloating) and
                 np.issubdtype(b_dtype, np.complexfloating))):
                c = np.zeros((2,), dtype=np.float64)
                assert_raises(ValueError, _sparsetools.csr_matvec,
                              2, 2, a.indptr, a.indices, a.data, b, c)

            # 创建一个数组c，数据类型为a_dtype和b_dtype的结果类型，并调用_sparsetools.csr_matvec进行稀疏矩阵向量乘法操作，并断言其结果与稠密矩阵乘法的结果相近
            c = np.zeros((2,), dtype=np.result_type(a_dtype, b_dtype))
            _sparsetools.csr_matvec(2, 2, a.indptr, a.indices, a.data, b, c)
            assert_allclose(c, np.dot(a.toarray(), b), err_msg=msg)


def test_endianness():
    # 创建一个全为1的3x4数组d
    d = np.ones((3,4))
    # 偏移量数组offsets
    offsets = [-1,0,1]

    # 创建两个对角线稀疏矩阵a和b，分别使用小端和大端字节序
    a = dia_matrix((d.astype('<f8'), offsets), (4, 4))
    b = dia_matrix((d.astype('>f8'), offsets), (4, 4))
    # 创建一个长度为4的数组v
    v = np.arange(4)

    # 断言：a和b分别与向量v的乘积结果接近于预期结果
    assert_allclose(a.dot(v), [1, 3, 6, 5])
    assert_allclose(b.dot(v), [1, 3, 6, 5])
```