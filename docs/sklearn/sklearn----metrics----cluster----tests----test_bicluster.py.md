# `D:\src\scipysrc\scikit-learn\sklearn\metrics\cluster\tests\test_bicluster.py`

```
# 导入需要的模块和函数
import numpy as np
from sklearn.metrics import consensus_score  # 导入共识评分函数
from sklearn.metrics.cluster._bicluster import _jaccard  # 导入 Jaccard 相似度计算函数
from sklearn.utils._testing import assert_almost_equal  # 导入近似相等断言函数


# 定义用于测试 Jaccard 相似度函数的函数
def test_jaccard():
    # 创建测试数据
    a1 = np.array([True, True, False, False])
    a2 = np.array([True, True, True, True])
    a3 = np.array([False, True, True, False])
    a4 = np.array([False, False, True, True])

    # 断言 Jaccard 相似度函数的结果
    assert _jaccard(a1, a1, a1, a1) == 1
    assert _jaccard(a1, a1, a2, a2) == 0.25
    assert _jaccard(a1, a1, a3, a3) == 1.0 / 7
    assert _jaccard(a1, a1, a4, a4) == 0


# 定义用于测试共识评分函数的函数
def test_consensus_score():
    # 创建测试数据
    a = [[True, True, False, False], [False, False, True, True]]
    b = a[::-1]

    # 断言共识评分函数的结果
    assert consensus_score((a, a), (a, a)) == 1
    assert consensus_score((a, a), (b, b)) == 1
    assert consensus_score((a, b), (a, b)) == 1
    assert consensus_score((a, b), (b, a)) == 1

    assert consensus_score((a, a), (b, a)) == 0
    assert consensus_score((a, a), (a, b)) == 0
    assert consensus_score((b, b), (a, b)) == 0
    assert consensus_score((b, b), (b, a)) == 0


# 定义用于测试共识评分函数处理不同数目的双聚类的函数
def test_consensus_score_issue2445():
    """处理 A 和 B 中双聚类数目不同的问题"""
    # 创建测试数据
    a_rows = np.array([
        [True, True, False, False],
        [False, False, True, True],
        [False, False, False, True],
    ])
    a_cols = np.array([
        [True, True, False, False],
        [False, False, True, True],
        [False, False, False, True],
    ])
    idx = [0, 2]
    
    # 计算并断言共识评分函数的结果
    s = consensus_score((a_rows, a_cols), (a_rows[idx], a_cols[idx]))
    # B 包含 A 中的 2 个 3 个双聚类，因此分数应为 2/3
    assert_almost_equal(s, 2.0 / 3.0)
```