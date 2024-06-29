# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_subclass.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库

from pandas import (  # 从 pandas 库中导入 DataFrame 和 Series 类
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入 pandas 测试模块

from pandas.io.pytables import (  # 从 pandas.io.pytables 模块中导入 HDFStore 和 read_hdf 函数
    HDFStore,
    read_hdf,
)

pytest.importorskip("tables")  # 如果 tables 模块不可导入，则跳过测试

# 定义一个测试类 TestHDFStoreSubclass
class TestHDFStoreSubclass:
    # 测试方法：验证 DataFrame 子类的支持性
    # GH 33748
    def test_supported_for_subclass_dataframe(self, tmp_path):
        data = {"a": [1, 2], "b": [3, 4]}
        sdf = tm.SubclassedDataFrame(data, dtype=np.intp)  # 创建 DataFrame 的子类对象

        expected = DataFrame(data, dtype=np.intp)  # 创建预期的 DataFrame 对象

        path = tmp_path / "temp.h5"  # 在临时路径下创建 HDF5 文件路径
        sdf.to_hdf(path, key="df")  # 将 DataFrame 的子类对象写入 HDF5 文件
        result = read_hdf(path, "df")  # 从 HDF5 文件中读取 DataFrame
        tm.assert_frame_equal(result, expected)  # 验证读取的 DataFrame 是否与预期相同

        path = tmp_path / "temp.h5"  # 再次使用相同的 HDF5 文件路径
        with HDFStore(path) as store:  # 使用 HDFStore 打开 HDF5 文件
            store.put("df", sdf)  # 将 DataFrame 的子类对象写入 HDF5 文件中
        result = read_hdf(path, "df")  # 从 HDF5 文件中读取 DataFrame
        tm.assert_frame_equal(result, expected)  # 验证读取的 DataFrame 是否与预期相同

    # 测试方法：验证 Series 子类的支持性
    def test_supported_for_subclass_series(self, tmp_path):
        data = [1, 2, 3]
        sser = tm.SubclassedSeries(data, dtype=np.intp)  # 创建 Series 的子类对象

        expected = Series(data, dtype=np.intp)  # 创建预期的 Series 对象

        path = tmp_path / "temp.h5"  # 在临时路径下创建 HDF5 文件路径
        sser.to_hdf(path, key="ser")  # 将 Series 的子类对象写入 HDF5 文件
        result = read_hdf(path, "ser")  # 从 HDF5 文件中读取 Series
        tm.assert_series_equal(result, expected)  # 验证读取的 Series 是否与预期相同

        path = tmp_path / "temp.h5"  # 再次使用相同的 HDF5 文件路径
        with HDFStore(path) as store:  # 使用 HDFStore 打开 HDF5 文件
            store.put("ser", sser)  # 将 Series 的子类对象写入 HDF5 文件中
        result = read_hdf(path, "ser")  # 从 HDF5 文件中读取 Series
        tm.assert_series_equal(result, expected)  # 验证读取的 Series 是否与预期相同
```