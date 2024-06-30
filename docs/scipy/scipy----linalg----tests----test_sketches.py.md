# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_sketches.py`

```
"""Tests for _sketches.py."""

import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy.testing import assert_, assert_equal  # 导入 NumPy 测试模块中的断言函数
from scipy.linalg import clarkson_woodruff_transform  # 导入 Clarkson-Woodruff 变换函数
from scipy.linalg._sketches import cwt_matrix  # 导入特定于矩阵的 Clarkson-Woodruff 变换实现
from scipy.sparse import issparse, rand  # 导入稀疏矩阵和随机数生成函数
from scipy.sparse.linalg import norm  # 导入稀疏矩阵的范数计算函数


class TestClarksonWoodruffTransform:
    """
    Testing the Clarkson Woodruff Transform
    """
    rng = np.random.RandomState(seed=1179103485)  # 创建随机数生成器对象，并设置种子

    n_rows = 2000  # 矩阵的行数
    n_cols = 100   # 矩阵的列数
    density = 0.1  # 稀疏矩阵的密度

    n_sketch_rows = 200  # Sketch 矩阵的行数

    seeds = [1755490010, 934377150, 1391612830, 1752708722, 2008891431,
             1302443994, 1521083269, 1501189312, 1126232505, 1533465685]  # 测试用的种子列表

    A_dense = rng.randn(n_rows, n_cols)  # 创建一个密集矩阵
    A_csc = rand(
        n_rows, n_cols, density=density, format='csc', random_state=rng,
    )  # 创建一个稀疏的 CSC 格式的矩阵
    A_csr = rand(
        n_rows, n_cols, density=density, format='csr', random_state=rng,
    )  # 创建一个稀疏的 CSR 格式的矩阵
    A_coo = rand(
        n_rows, n_cols, density=density, format='coo', random_state=rng,
    )  # 创建一个稀疏的 COO 格式的矩阵

    test_matrices = [  # 收集所有的测试矩阵，包括密集和稀疏格式
        A_dense, A_csc, A_csr, A_coo,
    ]

    x = rng.randn(n_rows, 1) / np.sqrt(n_rows)  # 创建一个具有单位范数的测试向量

    def test_sketch_dimensions(self):
        """
        Test the dimensions of the sketch matrix
        """
        for A in self.test_matrices:
            for seed in self.seeds:
                sketch = clarkson_woodruff_transform(
                    A, self.n_sketch_rows, seed=seed
                )  # 应用 Clarkson-Woodruff 变换得到 Sketch 矩阵
                assert_(sketch.shape == (self.n_sketch_rows, self.n_cols))  # 断言 Sketch 矩阵的维度是否正确

    def test_seed_returns_identical_transform_matrix(self):
        """
        Test if the same seed returns identical transformation matrix
        """
        for A in self.test_matrices:
            for seed in self.seeds:
                S1 = cwt_matrix(
                    self.n_sketch_rows, self.n_rows, seed=seed
                ).toarray()  # 生成并转换成数组格式的 Clarkson-Woodruff 变换矩阵
                S2 = cwt_matrix(
                    self.n_sketch_rows, self.n_rows, seed=seed
                ).toarray()  # 再次生成相同种子下的 Clarkson-Woodruff 变换矩阵
                assert_equal(S1, S2)  # 断言两个变换矩阵是否相等

    def test_seed_returns_identically(self):
        """
        Test if the same seed returns identical sketches
        """
        for A in self.test_matrices:
            for seed in self.seeds:
                sketch1 = clarkson_woodruff_transform(
                    A, self.n_sketch_rows, seed=seed
                )  # 应用 Clarkson-Woodruff 变换得到第一个 Sketch 矩阵
                sketch2 = clarkson_woodruff_transform(
                    A, self.n_sketch_rows, seed=seed
                )  # 再次应用相同种子得到第二个 Sketch 矩阵
                if issparse(sketch1):
                    sketch1 = sketch1.toarray()  # 如果是稀疏矩阵，转换为数组
                if issparse(sketch2):
                    sketch2 = sketch2.toarray()  # 如果是稀疏矩阵，转换为数组
                assert_equal(sketch1, sketch2)  # 断言两个 Sketch 矩阵是否相等
    # 测试用例：验证特征值分解后保留弗罗贝尼乌斯范数的性质
    def test_sketch_preserves_frobenius_norm(self):
        # 由于草图的概率性质，多次运行测试并检查通过的尝试次数
        n_errors = 0
        # 遍历测试矩阵集合中的每个矩阵 A
        for A in self.test_matrices:
            # 根据 A 是否稀疏来计算真实范数
            if issparse(A):
                true_norm = norm(A)
            else:
                true_norm = np.linalg.norm(A)
            
            # 对于预定义的随机种子集合进行遍历
            for seed in self.seeds:
                # 使用 Clarkon-Woodruff 转换生成草图
                sketch = clarkson_woodruff_transform(
                    A, self.n_sketch_rows, seed=seed,
                )
                # 根据草图是否稀疏来计算其范数
                if issparse(sketch):
                    sketch_norm = norm(sketch)
                else:
                    sketch_norm = np.linalg.norm(sketch)

                # 检查草图的范数与真实范数之间的差异是否超过真实范数的 10%
                if np.abs(true_norm - sketch_norm) > 0.1 * true_norm:
                    n_errors += 1
        
        # 断言：未发现任何误差
        assert_(n_errors == 0)

    # 测试用例：验证向量范数在草图处理后的保留性质
    def test_sketch_preserves_vector_norm(self):
        n_errors = 0
        # 计算草图的行数，确保保留向量范数的精度
        n_sketch_rows = int(np.ceil(2. / (0.01 * 0.5**2)))
        # 计算真实向量的范数
        true_norm = np.linalg.norm(self.x)
        
        # 对于预定义的随机种子集合进行遍历
        for seed in self.seeds:
            # 使用 Clarkon-Woodruff 转换生成草图
            sketch = clarkson_woodruff_transform(
                self.x, n_sketch_rows, seed=seed,
            )
            # 计算草图的向量范数
            sketch_norm = np.linalg.norm(sketch)

            # 检查草图的向量范数与真实向量范数之间的差异是否超过真实范数的 50%
            if np.abs(true_norm - sketch_norm) > 0.5 * true_norm:
                n_errors += 1
        
        # 断言：未发现任何误差
        assert_(n_errors == 0)
```