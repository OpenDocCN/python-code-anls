# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_skew.py`

```
# 导入 NumPy 库，通常用于科学计算中的数组操作
import numpy as np

# 导入 Pandas 库，用于数据分析和处理
import pandas as pd
# 导入 Pandas 的测试模块
import pandas._testing as tm

# 定义一个测试函数，验证 groupby 的偏斜度方法的等效性
def test_groupby_skew_equivalence():
    # 设置数据行数
    nrows = 1000
    # 设置分组数目
    ngroups = 3
    # 设置列数
    ncols = 2
    # 设置 NaN 值的比例
    nan_frac = 0.05

    # 生成一个指定形状的随机数组，服从标准正态分布
    arr = np.random.default_rng(2).standard_normal((nrows, ncols))
    # 随机将数组中的元素设为 NaN，比例为 nan_frac
    arr[np.random.default_rng(2).random(nrows) < nan_frac] = np.nan

    # 创建一个 DataFrame 对象
    df = pd.DataFrame(arr)
    # 生成一个随机的分组数组
    grps = np.random.default_rng(2).integers(0, ngroups, size=nrows)
    # 对 DataFrame 按照分组进行分组操作
    gb = df.groupby(grps)

    # 调用 groupby 对象的 skew 方法，计算每个分组的偏斜度
    result = gb.skew()

    # 通过对每个分组单独计算偏斜度，生成预期的 DataFrame 结果
    grpwise = [grp.skew().to_frame(i).T for i, grp in gb]
    expected = pd.concat(grpwise, axis=0)
    # 将预期结果的索引类型转换为和结果相同的类型，用于32位编译
    expected.index = expected.index.astype(result.index.dtype)
    # 使用 Pandas 的测试工具检查两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
```