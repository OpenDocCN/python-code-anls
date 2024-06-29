# `D:\src\scipysrc\pandas\pandas\tests\io\test_spss.py`

```
# 导入必要的模块和库
import datetime  # 导入 datetime 模块
from pathlib import Path  # 导入 Path 类用于处理文件路径

import numpy as np  # 导入 numpy 库，用于科学计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas as pd  # 导入 pandas 库，用于数据分析
import pandas._testing as tm  # 导入 pandas 测试工具模块
from pandas.util.version import Version  # 导入 pandas 版本相关的工具类

pyreadstat = pytest.importorskip("pyreadstat")  # 导入并检查 pyreadstat 库是否可用


# TODO(CoW) - detection of chained assignment in cython
# https://github.com/pandas-dev/pandas/issues/51315
@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
@pytest.mark.parametrize("path_klass", [lambda p: p, Path])
def test_spss_labelled_num(path_klass, datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    # Licence at LICENSES/HAVEN_LICENSE, LICENSES/HAVEN_MIT
    fname = path_klass(datapath("io", "data", "spss", "labelled-num.sav"))

    # 读取 SPSS 格式的数据文件，转换分类数据为类别型
    df = pd.read_spss(fname, convert_categoricals=True)
    # 期望的数据框架，用于断言比较
    expected = pd.DataFrame({"VAR00002": "This is one"}, index=[0])
    expected["VAR00002"] = pd.Categorical(expected["VAR00002"])
    tm.assert_frame_equal(df, expected)

    # 再次读取 SPSS 数据文件，但不转换分类数据
    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"VAR00002": 1.0}, index=[0])
    tm.assert_frame_equal(df, expected)


@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
def test_spss_labelled_num_na(datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    # Licence at LICENSES/HAVEN_LICENSE, LICENSES/HAVEN_MIT
    fname = datapath("io", "data", "spss", "labelled-num-na.sav")

    # 读取 SPSS 格式的数据文件，转换分类数据为类别型
    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({"VAR00002": ["This is one", None]})
    expected["VAR00002"] = pd.Categorical(expected["VAR00002"])
    tm.assert_frame_equal(df, expected)

    # 再次读取 SPSS 数据文件，但不转换分类数据
    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"VAR00002": [1.0, np.nan]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
def test_spss_labelled_str(datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    # Licence at LICENSES/HAVEN_LICENSE, LICENSES/HAVEN_MIT
    fname = datapath("io", "data", "spss", "labelled-str.sav")

    # 读取 SPSS 格式的数据文件，转换分类数据为类别型
    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({"gender": ["Male", "Female"]})
    expected["gender"] = pd.Categorical(expected["gender"])
    tm.assert_frame_equal(df, expected)

    # 再次读取 SPSS 数据文件，但不转换分类数据
    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"gender": ["M", "F"]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
def test_spss_umlauts(datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    # 从 Haven 项目中获取的测试文件
    # 授权信息在 LICENSES/HAVEN_LICENSE 和 LICENSES/HAVEN_MIT 中
    pass  # 未实现的测试，暂时跳过
    # 指定文件路径，这里使用 datapath 函数来获取文件路径
    fname = datapath("io", "data", "spss", "umlauts.sav")

    # 使用 pandas 的 read_spss 函数读取 SPSS 文件，并将分类变量转换为 pandas 中的分类数据类型
    df = pd.read_spss(fname, convert_categoricals=True)

    # 预期的数据框，包含一个名为 var1 的列，其中有包含特殊字符的字符串作为分类数据
    expected = pd.DataFrame(
        {"var1": ["the ä umlaut", "the ü umlaut", "the ä umlaut", "the ö umlaut"]}
    )

    # 将预期的 var1 列转换为 pandas 的分类类型
    expected["var1"] = pd.Categorical(expected["var1"])

    # 使用 pandas 的 assert_frame_equal 函数比较读取的数据框 df 和预期的数据框 expected 是否相等
    tm.assert_frame_equal(df, expected)

    # 再次使用 read_spss 函数读取 SPSS 文件，这次不将分类变量转换为 pandas 的分类数据类型
    df = pd.read_spss(fname, convert_categoricals=False)

    # 另一个预期的数据框，包含一个名为 var1 的列，其中包含数值
    expected = pd.DataFrame({"var1": [1.0, 2.0, 1.0, 3.0]})

    # 再次使用 assert_frame_equal 函数比较读取的数据框 df 和新的预期数据框 expected 是否相等
    tm.assert_frame_equal(df, expected)
# 定义一个测试函数，用于测试读取 SPSS 数据时的 usecols 参数异常情况
def test_spss_usecols(datapath):
    # 构建文件路径
    fname = datapath("io", "data", "spss", "labelled-num.sav")

    # 使用 pytest 检查是否会抛出 TypeError，并验证错误消息中包含指定文本
    with pytest.raises(TypeError, match="usecols must be list-like."):
        # 调用 pd.read_spss 函数，尝试读取指定文件并指定一个非列表形式的 usecols 参数
        pd.read_spss(fname, usecols="VAR00002")


# 定义一个测试函数，测试在指定 dtype_backend 的情况下读取 SPSS 数据的行为
def test_spss_umlauts_dtype_backend(datapath, dtype_backend):
    # 指定数据文件路径，这是来自 Haven 项目的文件
    fname = datapath("io", "data", "spss", "umlauts.sav")

    # 调用 pd.read_spss 函数读取数据，关闭转换分类变量的选项，并根据 dtype_backend 指定后端处理方式
    df = pd.read_spss(fname, convert_categoricals=False, dtype_backend=dtype_backend)

    # 预期的 DataFrame 结果，包含指定的列和数据类型
    expected = pd.DataFrame({"var1": [1.0, 2.0, 1.0, 3.0]}, dtype="Int64")

    # 如果 dtype_backend 是 "pyarrow"，则需要进一步处理预期的 DataFrame
    if dtype_backend == "pyarrow":
        # 导入并检查 pyarrow 是否可用
        pa = pytest.importorskip("pyarrow")

        # 通过 ArrowExtensionArray 将预期的 DataFrame 转换为 Arrow 格式
        from pandas.arrays import ArrowExtensionArray

        expected = pd.DataFrame(
            {
                col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                for col in expected.columns
            }
        )

    # 使用 pandas.testing.assert_frame_equal 检查读取的 DataFrame 是否与预期的 DataFrame 相等
    tm.assert_frame_equal(df, expected)


# 定义一个测试函数，测试使用无效的 dtype_backend 参数时是否会引发 ValueError 异常
def test_invalid_dtype_backend():
    # 定义预期的错误消息
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )

    # 使用 pytest 检查是否会抛出 ValueError，并验证错误消息中包含指定文本
    with pytest.raises(ValueError, match=msg):
        # 调用 pd.read_spss 函数，尝试使用无效的 dtype_backend 参数
        pd.read_spss("test", dtype_backend="numpy")


# 定义一个测试函数，测试读取 SPSS 数据文件的元数据属性
@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
def test_spss_metadata(datapath):
    # 构建文件路径
    fname = datapath("io", "data", "spss", "labelled-num.sav")

    # 调用 pd.read_spss 函数读取数据，并获取其元数据属性
    df = pd.read_spss(fname)

    # 预期的元数据字典，包含各种数据文件的属性信息
    metadata = {
        "column_names": ["VAR00002"],
        "column_labels": [None],
        "column_names_to_labels": {"VAR00002": None},
        "file_encoding": "UTF-8",
        "number_columns": 1,
        "number_rows": 1,
        "variable_value_labels": {"VAR00002": {1.0: "This is one"}},
        "value_labels": {"labels0": {1.0: "This is one"}},
        "variable_to_label": {"VAR00002": "labels0"},
        "notes": [],
        "original_variable_types": {"VAR00002": "F8.0"},
        "readstat_variable_types": {"VAR00002": "double"},
        "table_name": None,
        "missing_ranges": {},
        "missing_user_values": {},
        "variable_storage_width": {"VAR00002": 8},
        "variable_display_width": {"VAR00002": 8},
        "variable_alignment": {"VAR00002": "unknown"},
        "variable_measure": {"VAR00002": "unknown"},
        "file_label": None,
        "file_format": "sav/zsav",
    }

    # 如果使用的 pyreadstat 版本 >= 1.2.4，则添加创建时间和修改时间属性到元数据中
    if Version(pyreadstat.__version__) >= Version("1.2.4"):
        metadata.update(
            {
                "creation_time": datetime.datetime(2015, 2, 6, 14, 33, 36),
                "modification_time": datetime.datetime(2015, 2, 6, 14, 33, 36),
            }
        )

    # 使用断言检查读取的 DataFrame 的属性是否等于预期的元数据字典
    assert df.attrs == metadata
```