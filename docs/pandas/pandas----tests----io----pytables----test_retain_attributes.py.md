# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_retain_attributes.py`

```
import pytest  # 导入 pytest 测试框架

from pandas import (  # 导入 pandas 库中的多个模块和函数
    DataFrame,
    DatetimeIndex,
    Series,
    _testing as tm,  # 导入 _testing 模块并重命名为 tm
    date_range,
    errors,  # 导入 errors 模块
    read_hdf,
)
from pandas.tests.io.pytables.common import (  # 导入 pytables.common 中的 _maybe_remove 和 ensure_clean_store 函数
    _maybe_remove,
    ensure_clean_store,
)

pytestmark = pytest.mark.single_cpu  # 在 pytest 中标记此测试用例为单 CPU 测试


def test_retain_index_attributes(setup_path, unit):
    # GH 3499, losing frequency info on index recreation
    dti = date_range("2000-1-1", periods=3, freq="h", unit=unit)  # 创建一个频率为每小时的日期时间索引对象
    df = DataFrame({"A": Series(range(3), index=dti)})  # 创建一个 DataFrame 对象，使用上述索引

    with ensure_clean_store(setup_path) as store:  # 创建一个空的存储区，并确保在退出时清除
        _maybe_remove(store, "data")  # 尝试移除名为 "data" 的存储数据
        store.put("data", df, format="table")  # 将 DataFrame 存储在名为 "data" 的存储区中，格式为表格形式

        result = store.get("data")  # 从存储中获取名为 "data" 的数据
        tm.assert_frame_equal(df, result)  # 使用测试框架中的函数验证存储中的数据与原始 DataFrame 是否相等

        for attr in ["freq", "tz", "name"]:  # 遍历需要验证的索引属性列表
            for idx in ["index", "columns"]:  # 遍历索引和列
                assert getattr(getattr(df, idx), attr, None) == getattr(  # 检查原始 DataFrame 和存储结果中的索引属性是否相等
                    getattr(result, idx), attr, None
                )

        dti2 = date_range("2002-1-1", periods=3, freq="D", unit=unit)  # 创建另一个每天频率的日期时间索引对象
        # 尝试追加一个具有不同频率的表格数据
        with tm.assert_produces_warning(errors.AttributeConflictWarning):  # 使用测试框架捕获属性冲突警告
            df2 = DataFrame({"A": Series(range(3), index=dti2)})  # 创建包含新索引的 DataFrame
            store.append("data", df2)  # 追加数据到名为 "data" 的存储区

        assert store.get_storer("data").info["index"]["freq"] is None  # 验证存储中名为 "data" 的索引频率为 None

        # 这是正常的操作
        _maybe_remove(store, "df2")  # 尝试移除名为 "df2" 的存储数据
        dti3 = DatetimeIndex(  # 创建一个自定义频率的日期时间索引对象
            ["2001-01-01", "2001-01-02", "2002-01-01"], dtype=f"M8[{unit}]"
        )
        df2 = DataFrame(  # 创建包含新索引的 DataFrame
            {
                "A": Series(
                    range(3),
                    index=dti3,
                )
            }
        )
        store.append("df2", df2)  # 追加数据到名为 "df2" 的存储区
        dti4 = date_range("2002-1-1", periods=3, freq="D", unit=unit)  # 创建另一个每天频率的日期时间索引对象
        df3 = DataFrame({"A": Series(range(3), index=dti4)})  # 创建包含新索引的 DataFrame
        store.append("df2", df3)  # 追加数据到名为 "df2" 的存储区


def test_retain_index_attributes2(tmp_path, setup_path):
    path = tmp_path / setup_path  # 构建临时路径和设置路径的组合路径对象

    with tm.assert_produces_warning(errors.AttributeConflictWarning):  # 使用测试框架捕获属性冲突警告
        df = DataFrame(  # 创建包含日期时间索引的 DataFrame 对象
            {"A": Series(range(3), index=date_range("2000-1-1", periods=3, freq="h"))}
        )
        df.to_hdf(path, key="data", mode="w", append=True)  # 将 DataFrame 写入 HDF 文件，追加模式

        df2 = DataFrame(  # 创建包含不同频率日期时间索引的 DataFrame 对象
            {"A": Series(range(3), index=date_range("2002-1-1", periods=3, freq="D"))}
        )
        df2.to_hdf(path, key="data", append=True)  # 追加数据到已有的 HDF 文件

        idx = date_range("2000-1-1", periods=3, freq="h")  # 创建一个每小时频率的日期时间索引对象
        idx.name = "foo"  # 设置索引名称为 "foo"
        df = DataFrame({"A": Series(range(3), index=idx)})  # 创建包含新索引的 DataFrame 对象
        df.to_hdf(path, key="data", mode="w", append=True)  # 将 DataFrame 写入 HDF 文件，追加模式

    assert read_hdf(path, key="data").index.name == "foo"  # 验证 HDF 文件中的索引名称为 "foo"

    with tm.assert_produces_warning(errors.AttributeConflictWarning):  # 使用测试框架捕获属性冲突警告
        idx2 = date_range("2001-1-1", periods=3, freq="h")  # 创建一个每小时频率的日期时间索引对象
        idx2.name = "bar"  # 设置索引名称为 "bar"
        df2 = DataFrame({"A": Series(range(3), index=idx2)})  # 创建包含新索引的 DataFrame 对象
        df2.to_hdf(path, key="data", append=True)  # 追加数据到已有的 HDF 文件

    assert read_hdf(path, "data").index.name is None  # 验证 HDF 文件中的索引名称为空
```