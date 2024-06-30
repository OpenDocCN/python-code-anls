# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\tests\test_lsmr.py`

```
"""
Copyright (C) 2010 David Fong and Michael Saunders
Distributed under the same license as SciPy

Testing Code for LSMR.

03 Jun 2010: First version release with lsmr.py

David Chin-lung Fong            clfong@stanford.edu
Institute for Computational and Mathematical Engineering
Stanford University

Michael Saunders                saunders@stanford.edu
Systems Optimization Laboratory
Dept of MS&E, Stanford University.

"""

# 从 numpy 库中导入所需的函数和类
from numpy import array, arange, eye, zeros, ones, transpose, hstack
# 从 numpy.linalg 库中导入 norm 函数
from numpy.linalg import norm
# 从 numpy.testing 库中导入 assert_allclose 函数
from numpy.testing import assert_allclose
# 导入 pytest 库
import pytest
# 从 scipy.sparse 库中导入 coo_matrix 类
from scipy.sparse import coo_matrix
# 从 scipy.sparse.linalg._interface 中导入 aslinearoperator 函数
from scipy.sparse.linalg._interface import aslinearoperator
# 从 scipy.sparse.linalg 库中导入 lsmr 函数
from scipy.sparse.linalg import lsmr
# 从当前目录中的 test_lsqr 模块中导入 G 和 b 变量
from .test_lsqr import G, b


# 定义 TestLSMR 类，用于测试 lsmr 函数
class TestLSMR:
    # 设置测试方法的初始化方法
    def setup_method(self):
        self.n = 10  # 设置测试用例中的 n 值为 10
        self.m = 10  # 设置测试用例中的 m 值为 10

    # 定义 assertCompatibleSystem 方法，用于验证 A 和 xtrue 是否兼容
    def assertCompatibleSystem(self, A, xtrue):
        Afun = aslinearoperator(A)  # 将 A 转换为线性操作符
        b = Afun.matvec(xtrue)  # 计算 A*xtrue
        x = lsmr(A, b)[0]  # 使用 lsmr 方法求解线性方程组 A*x = b
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-5)  # 断言解 x 与真实解 xtrue 的差异小于给定容差

    # 定义 testIdentityACase1 方法，测试单位矩阵 A 情况下的 lsmr 函数
    def testIdentityACase1(self):
        A = eye(self.n)  # 创建大小为 n*n 的单位矩阵 A
        xtrue = zeros((self.n, 1))  # 创建大小为 n*1 的零向量 xtrue
        self.assertCompatibleSystem(A, xtrue)  # 调用 assertCompatibleSystem 方法验证结果

    # 定义 testIdentityACase2 方法，测试单位矩阵 A 和全为 1 的 xtrue 情况下的 lsmr 函数
    def testIdentityACase2(self):
        A = eye(self.n)  # 创建大小为 n*n 的单位矩阵 A
        xtrue = ones((self.n,1))  # 创建大小为 n*1 的全为 1 的向量 xtrue
        self.assertCompatibleSystem(A, xtrue)  # 调用 assertCompatibleSystem 方法验证结果

    # 定义 testIdentityACase3 方法，测试单位矩阵 A 和逆序排列的 xtrue 情况下的 lsmr 函数
    def testIdentityACase3(self):
        A = eye(self.n)  # 创建大小为 n*n 的单位矩阵 A
        xtrue = transpose(arange(self.n,0,-1))  # 创建大小为 n*1 的逆序排列向量 xtrue
        self.assertCompatibleSystem(A, xtrue)  # 调用 assertCompatibleSystem 方法验证结果

    # 定义 testBidiagonalA 方法，测试下三角双对角矩阵 A 和逆序排列的 xtrue 情况下的 lsmr 函数
    def testBidiagonalA(self):
        A = lowerBidiagonalMatrix(20,self.n)  # 创建大小为 20*n 的下三角双对角矩阵 A
        xtrue = transpose(arange(self.n,0,-1))  # 创建大小为 n*1 的逆序排列向量 xtrue
        self.assertCompatibleSystem(A,xtrue)  # 调用 assertCompatibleSystem 方法验证结果

    # 定义 testScalarB 方法，测试标量矩阵 A 和标量 b 情况下的 lsmr 函数
    def testScalarB(self):
        A = array([[1.0, 2.0]])  # 创建大小为 1*2 的数组 A
        b = 3.0  # 创建标量 b
        x = lsmr(A, b)[0]  # 使用 lsmr 方法求解线性方程组 A*x = b
        assert norm(A.dot(x) - b) == pytest.approx(0)  # 断言解 x 满足 A*x = b

    # 定义 testComplexX 方法，测试单位矩阵 A 和复数类型 xtrue 情况下的 lsmr 函数
    def testComplexX(self):
        A = eye(self.n)  # 创建大小为 n*n 的单位矩阵 A
        xtrue = transpose(arange(self.n, 0, -1) * (1 + 1j))  # 创建大小为 n*1 的复数类型逆序排列向量 xtrue
        self.assertCompatibleSystem(A, xtrue)  # 调用 assertCompatibleSystem 方法验证结果

    # 定义 testComplexX0 方法，测试复数类型 A 和复数类型 b 情况下的 lsmr 函数
    def testComplexX0(self):
        A = 4 * eye(self.n) + ones((self.n, self.n))  # 创建复数类型的大小为 n*n 的矩阵 A
        xtrue = transpose(arange(self.n, 0, -1))  # 创建大小为 n*1 的逆序排列向量 xtrue
        b = aslinearoperator(A).matvec(xtrue)  # 计算 A*xtrue
        x0 = zeros(self.n, dtype=complex)  # 创建大小为 n 的复数类型零向量 x0
        x = lsmr(A, b, x0=x0)[0]  # 使用 lsmr 方法求解线性方程组 A*x = b，给定初始解 x0
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-5)  # 断言解 x 满足 A*x = b

    # 定义 testComplexA 方法，测试复数类型 A 情况下的 lsmr 函数
    def testComplexA(self):
        A = 4 * eye(self.n) + 1j * ones((self.n, self.n))  # 创建复数类型的大小为 n*n 的矩阵 A
        xtrue = transpose(arange(self.n, 0, -1).astype(complex))  # 创建大小为 n*1 的复数类型逆序排列向量 xtrue
        self.assertCompatibleSystem(A, xtrue)  # 调用 assertCompatibleSystem 方法验证结果

    # 定义 testComplexB 方法，测试复数类型 A 和复数类型 b 情况下的 lsmr 函数
    def testComplexB(self):
        A = 4 * eye(self.n) + ones((self.n, self.n))  # 创建复数类型的大小为 n*n 的矩阵 A
        xtrue = transpose(arange(self.n, 0, -1) * (1 + 1j))  # 创建大小为 n*1 的复数类型逆序排列向量 xtrue
        b = aslinearoperator(A).matvec(xtrue)  # 计算 A*xtrue
        x = lsmr(A, b)[0]  # 使用 lsmr 方法求解线性方程组 A*x = b
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-5)  # 断言解 x 满足 A*x = b

    # 定义 testColumnB 方法，测试单位矩阵 A 和列向量 b 情况下的 lsmr 函数
    def testColumnB(self):
        A = eye(self.n)  # 创建大小为 n*n 的单位矩阵 A
        b = ones((self.n, 1))  # 创建大小为 n*1 的全为 1 的列向量 b
        x
    # 定义测试初始化方法，用于验证默认设置未被修改
    def testInitialization(self):
        # 调用 lsmr 函数，获取结果元组中的第一个元素作为参考解 x_ref
        x_ref, _, itn_ref, normr_ref, *_ = lsmr(G, b)
        # 使用 assert_allclose 函数验证 b - G@x_ref 的范数是否接近 normr_ref，允许的误差为 1e-6
        assert_allclose(norm(b - G@x_ref), normr_ref, atol=1e-6)

        # 测试传入全零向量 x0 是否产生类似的结果
        x0 = zeros(b.shape)
        # 调用 lsmr 函数，传入 x0 参数，获取结果元组中的第一个元素作为 x
        x = lsmr(G, b, x0=x0)[0]
        # 使用 assert_allclose 函数验证 x 是否接近 x_ref
        assert_allclose(x, x_ref)

        # 测试使用 x0 进行热启动，并限制最大迭代次数为 1
        x0 = lsmr(G, b, maxiter=1)[0]

        # 再次调用 lsmr 函数，传入 x0 参数，获取结果元组中的前五个元素
        x, _, itn, normr, *_ = lsmr(G, b, x0=x0)
        # 使用 assert_allclose 函数验证 b - G@x 的范数是否接近 normr，允许的误差为 1e-6
        assert_allclose(norm(b - G@x), normr, atol=1e-6)

        # 提示信息：这里的收敛值不总是与参考值相同，因为误差估计会因从全零向量与从 x0 开始而有微小差异，因此仅比较范数和迭代次数 (itn)
        # x 通常比未使用 x0 的情况下收敛速度更快，因为它从 x0 开始。
        # itn == itn_ref 表示 lsmr(x0) 额外执行了一次迭代，参见上文。
        # -1 也是可能的，但很少见（10 万分之一的可能性），所以更可能是其他地方的错误。
        assert itn - itn_ref in (0, 1)

        # 如果执行了额外的迭代，normr 可能为 0，而 normr_ref 可能大得多。
        assert normr < normr_ref * (1 + 1e-6)
class TestLSMRReturns:
    # 设置测试方法的初始化
    def setup_method(self):
        # 设置变量 self.n 为 10
        self.n = 10
        # 创建一个 20x10 的下三角双对角矩阵，并赋值给 self.A
        self.A = lowerBidiagonalMatrix(20, self.n)
        # 创建一个逆序排列的长度为 self.n 的向量，并赋值给 self.xtrue
        self.xtrue = transpose(arange(self.n, 0, -1))
        # 将 self.A 转换为线性操作对象，并赋值给 self.Afun
        self.Afun = aslinearoperator(self.A)
        # 计算线性操作 self.Afun 作用于 self.xtrue 的结果，并赋值给 self.b
        self.b = self.Afun.matvec(self.xtrue)
        # 创建一个长度为 self.n，元素全为 1 的向量，并赋值给 self.x0
        self.x0 = ones(self.n)
        # 复制 self.x0 并赋值给 self.x00
        self.x00 = self.x0.copy()
        # 调用 lsmr 函数求解方程 self.A * x = self.b，并赋值给 self.returnValues
        self.returnValues = lsmr(self.A, self.b)
        # 调用 lsmr 函数求解方程 self.A * x = self.b，初始值为 self.x0，并赋值给 self.returnValuesX0
        self.returnValuesX0 = lsmr(self.A, self.b, x0=self.x0)

    # 测试 x0 是否未改变
    def test_unchanged_x0(self):
        # 从 self.returnValuesX0 解析出结果，并赋值给 x, istop, itn, normr, normar, normA, condA, normx
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValuesX0
        # 断言 self.x00 等于 self.x0
        assert_allclose(self.x00, self.x0)

    # 测试 normr 是否正确
    def testNormr(self):
        # 从 self.returnValues 解析出结果，并赋值给 x, istop, itn, normr, normar, normA, condA, normx
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        # 断言 self.b 减去 self.Afun.matvec(x) 的范数近似等于 normr
        assert norm(self.b - self.Afun.matvec(x)) == pytest.approx(normr)

    # 测试 normar 是否正确
    def testNormar(self):
        # 从 self.returnValues 解析出结果，并赋值给 x, istop, itn, normr, normar, normA, condA, normx
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        # 计算 self.Afun.rmatvec(self.b - self.Afun.matvec(x)) 的范数，与 normar 近似比较
        assert (norm(self.Afun.rmatvec(self.b - self.Afun.matvec(x)))
                == pytest.approx(normar))

    # 测试 normx 是否正确
    def testNormx(self):
        # 从 self.returnValues 解析出结果，并赋值给 x, istop, itn, normr, normar, normA, condA, normx
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        # 断言 x 的范数近似等于 normx
        assert norm(x) == pytest.approx(normx)


注释：
- `class TestLSMRReturns:`：定义一个测试类 `TestLSMRReturns`。
- `def setup_method(self):`：设置测试方法的初始化，在每个测试方法运行前执行。
- `self.n = 10`：初始化变量 `n` 为 10。
- `self.A = lowerBidiagonalMatrix(20, self.n)`：创建一个 20x10 的下三角双对角矩阵，并赋值给 `A`。
- `self.xtrue = transpose(arange(self.n, 0, -1))`：创建一个逆序排列的长度为 `n` 的向量，并赋值给 `xtrue`。
- `self.Afun = aslinearoperator(self.A)`：将 `A` 转换为线性操作对象，并赋值给 `Afun`。
- `self.b = self.Afun.matvec(self.xtrue)`：计算线性操作 `Afun` 作用于 `xtrue` 的结果，并赋值给 `b`。
- `self.x0 = ones(self.n)`：创建一个长度为 `n`，元素全为 1 的向量，并赋值给 `x0`。
- `self.x00 = self.x0.copy()`：复制 `x0` 并赋值给 `x00`。
- `self.returnValues = lsmr(self.A, self.b)`：调用 `lsmr` 函数求解方程 `A * x = b`，并赋值给 `returnValues`。
- `self.returnValuesX0 = lsmr(self.A, self.b, x0=self.x0)`：调用 `lsmr` 函数求解方程 `A * x = b`，初始值为 `x0`，并赋值给 `returnValuesX0`。
- `def test_unchanged_x0(self):`：定义测试方法 `test_unchanged_x0`，测试 `x0` 是否未改变。
- `x, istop, itn, normr, normar, normA, condA, normx = self.returnValuesX0`：从 `returnValuesX0` 解析出结果。
- `assert_allclose(self.x00, self.x0)`：断言 `x00` 等于 `x0`，检查 `x0` 是否未改变。
- `def testNormr(self):`、`def testNormar(self):`、`def testNormx(self):`：定义测试方法 `testNormr`、`testNormar`、`testNormx`，分别测试 `normr`、`normar`、`normx` 是否正确。
- `assert norm(...)`、`assert (...) == pytest.approx(...)`：使用断言检查计算结果是否符合预期。
- `lowerBidiagonalMatrix(m, n)`：定义函数 `lowerBidiagonalMatrix`，生成下三角双对角矩阵用于测试。
```