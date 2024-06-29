# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_keys.py`

```
# 导入必要的库
import numpy as np
import pytest

# 从 pandas 库中导入需要的模块
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    Series,
    date_range,
)

# 从 pandas 的测试模块中导入特定的函数和类
from pandas.tests.io.pytables.common import (
    ensure_clean_store,
    tables,
)

# 设置 pytest 标记为 single_cpu
pytestmark = pytest.mark.single_cpu


# 定义一个测试函数 test_keys，接受一个 setup_path 参数
def test_keys(setup_path):
    # 使用 ensure_clean_store 函数创建一个干净的 HDFStore 对象 store
    with ensure_clean_store(setup_path) as store:
        # 向 store 中添加三个 Series 对象，分别为 a、b、c
        store["a"] = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        store["b"] = Series(
            range(10), dtype="float64", index=[f"i_{i}" for i in range(10)]
        )
        store["c"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )

        # 断言 store 中对象的数量为 3
        assert len(store) == 3
        # 预期的 keys 列表
        expected = {"/a", "/b", "/c"}
        # 断言 store 的 keys 和预期的 keys 相同
        assert set(store.keys()) == expected
        # 断言 set(store) 和预期的 keys 相同
        assert set(store) == expected


# 定义一个测试函数 test_non_pandas_keys，接受 tmp_path 和 setup_path 参数
def test_non_pandas_keys(tmp_path, setup_path):
    # 定义三个表格类 Table1、Table2、Table3，都包含一个 Float32Col 的列
    class Table1(tables.IsDescription):
        value1 = tables.Float32Col()

    class Table2(tables.IsDescription):
        value2 = tables.Float32Col()

    class Table3(tables.IsDescription):
        value3 = tables.Float32Col()

    # 在临时路径 tmp_path 下创建 HDF 文件 path
    path = tmp_path / setup_path
    with tables.open_file(path, mode="w") as h5file:
        # 在 HDF 文件中创建一个名为 "group" 的组
        group = h5file.create_group("/", "group")
        # 在 "group" 组中创建三个表格：table1、table2、table3
        h5file.create_table(group, "table1", Table1, "Table 1")
        h5file.create_table(group, "table2", Table2, "Table 2")
        h5file.create_table(group, "table3", Table3, "Table 3")
    
    # 使用 HDFStore 打开路径 path
    with HDFStore(path) as store:
        # 断言 store 中的对象数量为 3
        assert len(store.keys(include="native")) == 3
        # 预期的 keys 列表
        expected = {"/group/table1", "/group/table2", "/group/table3"}
        # 断言 store 的 native 模式下的 keys 和预期的 keys 相同
        assert set(store.keys(include="native")) == expected
        # 断言 store 的 pandas 模式下的 keys 为空集合
        assert set(store.keys(include="pandas")) == set()
        # 遍历预期的 keys 列表
        for name in expected:
            # 获取名为 name 的 DataFrame
            df = store.get(name)
            # 断言 DataFrame df 的列数为 1
            assert len(df.columns) == 1


# 定义一个测试函数 test_keys_illegal_include_keyword_value，接受 setup_path 参数
def test_keys_illegal_include_keyword_value(setup_path):
    # 使用 ensure_clean_store 函数创建一个干净的 HDFStore 对象 store
    with ensure_clean_store(setup_path) as store:
        # 使用 pytest.raises 检查 ValueError 异常是否被抛出，且匹配指定的错误消息
        with pytest.raises(
            ValueError,
            match="`include` should be either 'pandas' or 'native' but is 'illegal'",
        ):
            # 调用 store.keys(include="illegal")，预期抛出 ValueError 异常
            store.keys(include="illegal")


# 定义一个测试函数 test_keys_ignore_hdf_softlink，接受 setup_path 参数
def test_keys_ignore_hdf_softlink(setup_path):
    # GH 20523
    # 向 ensure_clean_store 函数传入 setup_path，创建一个干净的 HDFStore 对象 store
    with ensure_clean_store(setup_path) as store:
        # 创建一个 DataFrame df，包含两列 "A" 和 "B"
        df = DataFrame({"A": range(5), "B": range(5)})
        # 将 df 存储到 store 中，键名为 "df"
        store.put("df", df)

        # 断言 store 的 keys 列表应该只包含 "/df"
        assert store.keys() == ["/df"]

        # 在 store 中创建一个软链接 "symlink"，指向 "df"
        store._handle.create_soft_link(store._handle.root, "symlink", "df")

        # 应该忽略软链接，所以仍然断言 store 的 keys 列表应该只包含 "/df"
        assert store.keys() == ["/df"]
```