# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_categorical.py`

```
# 导入必要的库
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于测试框架

from pandas import (  # 从pandas库中导入以下模块和函数
    Categorical,  # 数据类型，用于表示分类数据
    DataFrame,  # 数据结构，用于表示二维数据表
    Series,  # 数据结构，用于表示一维数据序列
    _testing as tm,  # 测试辅助模块
    concat,  # 函数，用于连接数据结构
    read_hdf,  # 函数，用于读取HDF5格式数据
)
from pandas.tests.io.pytables.common import (  # 导入pytables.common中的以下函数
    _maybe_remove,  # 函数，用于可能的文件移除操作
    ensure_clean_store,  # 函数，用于确保存储空间清洁
)

# 将当前模块标记为仅适用于单CPU运行的测试
pytestmark = pytest.mark.single_cpu


def test_categorical(setup_path):
    # 定义测试函数test_categorical，接受setup_path参数
    pass  # 未定义测试内容，暂未实现


def test_categorical_conversion(tmp_path, setup_path):
    # 定义测试函数test_categorical_conversion，接受tmp_path和setup_path参数
    # GH13322
    # 检查read_hdf在具有分类列时，如果where条件未满足，则不返回行。
    
    # 定义观测ID、图像ID和数据列表
    obsids = ["ESP_012345_6789", "ESP_987654_3210"]
    imgids = ["APF00006np", "APF0001imm"]
    data = [4.3, 9.8]

    # 测试不使用分类时的情况
    df = DataFrame({"obsids": obsids, "imgids": imgids, "data": data})

    # 预期结果是一个与df类型相匹配的空DataFrame
    expected = df.iloc[[], :]
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table", data_columns=True)
    result = read_hdf(path, "df", where="obsids=B")
    tm.assert_frame_equal(result, expected)

    # 测试使用分类时的情况
    df.obsids = df.obsids.astype("category")
    df.imgids = df.imgids.astype("category")

    # 预期结果是一个与df类型相匹配的空DataFrame
    expected = df.iloc[[], :]
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table", data_columns=True)
    result = read_hdf(path, "df", where="obsids=B")
    tm.assert_frame_equal(result, expected)


def test_categorical_nan_only_columns(tmp_path, setup_path):
    # 定义测试函数test_categorical_nan_only_columns，接受tmp_path和setup_path参数
    # GH18413
    # 检查read_hdf在具有仅NaN值的分类列时是否能够正确读取。
    
    # 创建包含NaN值的分类数据框
    df = DataFrame(
        {
            "a": ["a", "b", "c", np.nan],
            "b": [np.nan, np.nan, np.nan, np.nan],
            "c": [1, 2, 3, 4],
            "d": Series([None] * 4, dtype=object),
        }
    )
    df["a"] = df.a.astype("category")
    df["b"] = df.b.astype("category")
    df["d"] = df.b.astype("category")
    expected = df
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table", data_columns=True)
    result = read_hdf(path, "df")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("where, expected", [["q", []], ["a", ["a"]]])
def test_convert_value(tmp_path, setup_path, where: str, expected):
    # 定义测试函数test_convert_value，接受tmp_path、setup_path和where参数，并使用参数化测试
    # GH39420
    # 检查read_hdf在具有分类列时是否能够按where条件过滤。
    
    # 创建包含列'a'的数据框
    df = DataFrame({"col": ["a", "b", "s"]})
    df.col = df.col.astype("category")
    max_widths = {"col": 1}
    categorical_values = sorted(df.col.unique())
    expected = DataFrame({"col": expected})
    expected.col = expected.col.astype("category")
    expected.col = expected.col.cat.set_categories(categorical_values)

    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table", min_itemsize=max_widths)
    result = read_hdf(path, where=f'col=="{where}"')
    tm.assert_frame_equal(result, expected)
```