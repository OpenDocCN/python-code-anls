# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_exceptions.py`

```
import pytest  # 导入 pytest 模块

jinja2 = pytest.importorskip("jinja2")  # 导入 jinja2 模块，如果导入失败则跳过

from pandas import (  # 从 pandas 库中导入以下模块：
    DataFrame,  # 数据框架
    MultiIndex,  # 多重索引
)

from pandas.io.formats.style import Styler  # 从 pandas 库中导入 Styler 类


@pytest.fixture
def df():
    return DataFrame(  # 创建一个数据框架对象
        data=[[0, -0.609], [1, -1.228]],  # 数据为二维列表
        columns=["A", "B"],  # 列名为 A 和 B
        index=["x", "y"],  # 索引为 x 和 y
    )


@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0)  # 创建 Styler 对象，设置 uuid_len 为 0


def test_concat_bad_columns(styler):
    msg = "`other.data` must have same columns as `Styler.data"  # 错误消息定义
    with pytest.raises(ValueError, match=msg):  # 检查是否会抛出 ValueError 异常，且异常消息与 msg 匹配
        styler.concat(DataFrame([[1, 2]]).style)  # 调用 Styler 对象的 concat 方法


def test_concat_bad_type(styler):
    msg = "`other` must be of type `Styler`"  # 错误消息定义
    with pytest.raises(TypeError, match=msg):  # 检查是否会抛出 TypeError 异常，且异常消息与 msg 匹配
        styler.concat(DataFrame([[1, 2]]))  # 调用 Styler 对象的 concat 方法


def test_concat_bad_index_levels(styler, df):
    df = df.copy()  # 复制数据框架 df
    df.index = MultiIndex.from_tuples([(0, 0), (1, 1)])  # 设置 df 的多重索引
    msg = "number of index levels must be same in `other`"  # 错误消息定义
    with pytest.raises(ValueError, match=msg):  # 检查是否会抛出 ValueError 异常，且异常消息与 msg 匹配
        styler.concat(df.style)  # 调用 Styler 对象的 concat 方法
```