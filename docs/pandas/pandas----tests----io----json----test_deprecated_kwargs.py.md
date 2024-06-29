# `D:\src\scipysrc\pandas\pandas\tests\io\json\test_deprecated_kwargs.py`

```
"""
Tests for the deprecated keyword arguments for `read_json`.
"""

# 导入需要的模块和库
from io import StringIO  # 导入StringIO类，用于创建内存中的文本I/O对象

import pandas as pd  # 导入pandas库，并使用pd作为别名
import pandas._testing as tm  # 导入pandas的测试模块，使用tm作为别名

from pandas.io.json import read_json  # 从pandas的JSON I/O模块中导入read_json函数


# 定义测试函数，用于测试read_json函数的不推荐关键字参数
def test_good_kwargs():
    # 创建一个测试用的DataFrame对象
    df = pd.DataFrame({"A": [2, 4, 6], "B": [3, 6, 9]}, index=[0, 1, 2])

    # 使用assert_produces_warning上下文管理器，验证没有产生警告
    with tm.assert_produces_warning(None):
        # 创建StringIO对象，将DataFrame转换为JSON格式并设定orient参数为"split"
        data1 = StringIO(df.to_json(orient="split"))
        # 调用read_json函数读取StringIO对象中的JSON数据，并验证与原始DataFrame相等
        tm.assert_frame_equal(df, read_json(data1, orient="split"))

        # 创建StringIO对象，将DataFrame转换为JSON格式并设定orient参数为"columns"
        data2 = StringIO(df.to_json(orient="columns"))
        # 调用read_json函数读取StringIO对象中的JSON数据，并验证与原始DataFrame相等
        tm.assert_frame_equal(df, read_json(data2, orient="columns"))

        # 创建StringIO对象，将DataFrame转换为JSON格式并设定orient参数为"index"
        data3 = StringIO(df.to_json(orient="index"))
        # 调用read_json函数读取StringIO对象中的JSON数据，并验证与原始DataFrame相等
        tm.assert_frame_equal(df, read_json(data3, orient="index"))
```