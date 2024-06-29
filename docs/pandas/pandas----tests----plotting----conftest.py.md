# `D:\src\scipysrc\pandas\pandas\tests\plotting\conftest.py`

```
# 导入所需的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库

from pandas import (  # 从 Pandas 库中导入以下模块
    DataFrame,  # 数据帧（DataFrame）
    to_datetime,  # 将数据转换为日期时间格式的函数
)


@pytest.fixture(autouse=True)
def autouse_mpl_cleanup(mpl_cleanup):
    # 定义自动使用的 pytest fixture，用于清理 Matplotlib
    pass


@pytest.fixture
def hist_df():
    # 定义一个名为 hist_df 的 pytest fixture，用于生成历史数据的数据帧

    n = 50  # 数据行数
    rng = np.random.default_rng(10)  # 使用种子值为 10 的随机数生成器

    # 随机生成 'Male' 和 'Female' 的性别数据，总数为 n
    gender = rng.choice(["Male", "Female"], size=n)

    # 随机生成 'A', 'B', 'C' 三个班级的数据，总数为 n
    classroom = rng.choice(["A", "B", "C"], size=n)

    # 使用正态分布随机生成身高数据，均值为 66，标准差为 4，总数为 n
    height = rng.normal(66, 4, size=n)

    # 使用正态分布随机生成体重数据，均值为 161，标准差为 32，总数为 n
    weight = rng.normal(161, 32, size=n)

    # 随机生成整数类别数据，范围为 [0, 4)，总数为 n
    category = rng.integers(4, size=n)

    # 随机生成日期时间数据，范围为 [812419200000000000, 819331200000000000)，总数为 n，数据类型为 np.int64
    datetime_data = rng.integers(812419200000000000, 819331200000000000, size=n, dtype=np.int64)
    datetime = to_datetime(datetime_data)  # 转换为日期时间格式

    # 创建 DataFrame 对象，存储以上生成的数据
    hist_df = DataFrame({
        "gender": gender,
        "classroom": classroom,
        "height": height,
        "weight": weight,
        "category": category,
        "datetime": datetime,
    })

    return hist_df  # 返回生成的历史数据 DataFrame 对象
```