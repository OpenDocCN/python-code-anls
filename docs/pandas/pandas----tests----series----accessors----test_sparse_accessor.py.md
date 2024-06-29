# `D:\src\scipysrc\pandas\pandas\tests\series\accessors\test_sparse_accessor.py`

```
# 从 pandas 库中导入 Series 类
from pandas import Series

# 定义一个测试类 TestSparseAccessor
class TestSparseAccessor:
    # 定义一个测试方法 test_sparse_accessor_updates_on_inplace
    def test_sparse_accessor_updates_on_inplace(self):
        # 创建一个稀疏 Series 对象，包含整数类型的稀疏数据
        ser = Series([1, 1, 2, 3], dtype="Sparse[int]")
        # 在 inplace=True 参数下执行 drop 操作，移除索引为 0 和 1 的元素
        return_value = ser.drop([0, 1], inplace=True)
        # 断言 drop 方法返回 None，表示原地修改了 Series 对象
        assert return_value is None
        # 断言稀疏 Series 的密度为 1.0，即全部非空元素的比例
        assert ser.sparse.density == 1.0
```