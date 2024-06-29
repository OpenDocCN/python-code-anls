# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_lexsort.py`

```
# 导入 MultiIndex 类，用于多级索引操作
from pandas import MultiIndex

# 定义 TestIsLexsorted 类，用于测试 MultiIndex 的排序功能
class TestIsLexsorted:
    # 定义测试函数 test_is_lexsorted
    def test_is_lexsorted(self):
        # 定义多级索引的层级列表
        levels = [[0, 1], [0, 1, 2]]

        # 创建 MultiIndex 对象，包括层级列表 levels 和代码列表 codes
        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]]
        )
        # 断言索引是否按字典序排序
        assert index._is_lexsorted()

        # 创建另一个 MultiIndex 对象，包括层级列表 levels 和不同的代码列表 codes
        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]]
        )
        # 断言索引是否未按字典序排序
        assert not index._is_lexsorted()

        # 创建第三个 MultiIndex 对象，包括层级列表 levels 和另一组不同的代码列表 codes
        index = MultiIndex(
            levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]]
        )
        # 断言索引是否未按字典序排序
        assert not index._is_lexsorted()
        # 断言索引的字典序深度属性是否为 0
        assert index._lexsort_depth == 0


# 定义 TestLexsortDepth 类，用于测试 MultiIndex 的字典序深度属性
class TestLexsortDepth:
    # 定义测试函数 test_lexsort_depth
    def test_lexsort_depth(self):
        # 测试当指定 sortorder 参数时，_lexsort_depth 返回正确的排序深度
        # GH#28518

        # 定义多级索引的层级列表
        levels = [[0, 1], [0, 1, 2]]

        # 创建 MultiIndex 对象，包括层级列表 levels、代码列表 codes 和 sortorder 参数为 2
        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]], sortorder=2
        )
        # 断言索引的字典序深度属性是否为 2
        assert index._lexsort_depth == 2

        # 创建另一个 MultiIndex 对象，包括层级列表 levels、代码列表 codes 和 sortorder 参数为 1
        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]], sortorder=1
        )
        # 断言索引的字典序深度属性是否为 1
        assert index._lexsort_depth == 1

        # 创建第三个 MultiIndex 对象，包括层级列表 levels、代码列表 codes 和 sortorder 参数为 0
        index = MultiIndex(
            levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]], sortorder=0
        )
        # 断言索引的字典序深度属性是否为 0
        assert index._lexsort_depth == 0
```