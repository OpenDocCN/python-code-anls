# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_nunique.py`

```
# 导入numpy库，用于生成随机数
import numpy as np

# 从pandas库中导入Categorical和Series类
from pandas import (
    Categorical,
    Series,
)


def test_nunique():
    # basics.rst 文档中的示例
    # 使用随机数生成器生成长度为500的标准正态分布随机数序列
    series = Series(np.random.default_rng(2).standard_normal(500))
    # 将序列的索引从20到499的位置设置为NaN（缺失值）
    series[20:500] = np.nan
    # 将序列的索引从10到19的位置设置为5000
    series[10:20] = 5000
    # 计算序列中不重复元素的数量
    result = series.nunique()
    # 断言结果是否等于11
    assert result == 11


def test_nunique_categorical():
    # GH#18051 GitHub issue编号18051
    # 创建一个空的Categorical类型的Series
    ser = Series(Categorical([]))
    # 断言空的Categorical类型的Series的不重复元素数量是否为0
    assert ser.nunique() == 0

    # 创建一个只包含NaN值的Categorical类型的Series
    ser = Series(Categorical([np.nan]))
    # 断言只包含NaN值的Categorical类型的Series的不重复元素数量是否为0
    assert ser.nunique() == 0
```