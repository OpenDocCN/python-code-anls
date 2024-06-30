# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_extract.py`

```
# 导入必要的库和模块
from numpy.testing import assert_equal
from scipy.sparse import csr_matrix, csr_array, sparray
import numpy as np
from scipy.sparse import _extract

# 定义测试类 TestExtract
class TestExtract:
    # 在每个测试方法执行前设置测试用例
    def setup_method(self):
        self.cases = [
            csr_array([[1,2]]),
            csr_array([[1,0]]),
            csr_array([[0,0]]),
            csr_array([[1],[2]]),
            csr_array([[1],[0]]),
            csr_array([[0],[0]]),
            csr_array([[1,2],[3,4]]),
            csr_array([[0,1],[0,0]]),
            csr_array([[0,0],[1,0]]),
            csr_array([[0,0],[0,0]]),
            csr_array([[1,2,0,0,3],[4,5,0,6,7],[0,0,8,9,0]]),
            csr_array([[1,2,0,0,3],[4,5,0,6,7],[0,0,8,9,0]]).T,
        ]

    # 测试 _extract.find 方法
    def test_find(self):
        # 对每个测试用例执行测试
        for A in self.cases:
            # 调用 _extract.find 方法，获取行索引 I，列索引 J，值 V
            I, J, V = _extract.find(A)
            # 使用行索引 I，列索引 J，和值 V 构建一个新的 csr_array B
            B = csr_array((V, (I, J)), shape=A.shape)
            # 断言 A 和 B 的稀疏矩阵表示是否相等
            assert_equal(A.toarray(), B.toarray())

    # 测试 _extract.tril 方法
    def test_tril(self):
        # 对每个测试用例执行测试
        for A in self.cases:
            # 将 A 转换为稠密数组 B
            B = A.toarray()
            # 对于不同的 k 值进行测试
            for k in [-3, -2, -1, 0, 1, 2, 3]:
                # 断言 _extract.tril(A, k=k) 和 np.tril(B, k=k) 的稀疏矩阵表示是否相等
                assert_equal(_extract.tril(A, k=k).toarray(), np.tril(B, k=k))

    # 测试 _extract.triu 方法
    def test_triu(self):
        # 对每个测试用例执行测试
        for A in self.cases:
            # 将 A 转换为稠密数组 B
            B = A.toarray()
            # 对于不同的 k 值进行测试
            for k in [-3, -2, -1, 0, 1, 2, 3]:
                # 断言 _extract.triu(A, k=k) 和 np.triu(B, k=k) 的稀疏矩阵表示是否相等
                assert_equal(_extract.triu(A, k=k).toarray(), np.triu(B, k=k))

    # 测试 _extract.tril 和 _extract.triu 返回的对象类型
    def test_array_vs_matrix(self):
        # 对每个测试用例执行测试
        for A in self.cases:
            # 断言 _extract.tril(A) 和 _extract.triu(A) 返回的对象类型是 sparray
            assert isinstance(_extract.tril(A), sparray)
            assert isinstance(_extract.triu(A), sparray)
            # 将 A 转换为 csr_matrix M
            M = csr_matrix(A)
            # 断言 _extract.tril(M) 和 _extract.triu(M) 返回的对象类型不是 sparray
            assert not isinstance(_extract.tril(M), sparray)
            assert not isinstance(_extract.triu(M), sparray)
```