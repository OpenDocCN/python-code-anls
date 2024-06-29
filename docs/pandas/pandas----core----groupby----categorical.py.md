# `D:\src\scipysrc\pandas\pandas\core\groupby\categorical.py`

```
# 导入必要的模块
from __future__ import annotations  # 使用未来版本的类型注解特性

import numpy as np  # 导入 NumPy 库

from pandas.core.algorithms import unique1d  # 从 Pandas 中导入 unique1d 算法函数
from pandas.core.arrays.categorical import (  # 从 Pandas 中导入 Categorical 相关类和函数
    Categorical,
    CategoricalDtype,
    recode_for_categories,
)


def recode_for_groupby(c: Categorical, sort: bool, observed: bool) -> Categorical:
    """
    Code the categories to ensure we can groupby for categoricals.

    If observed=True, we return a new Categorical with the observed
    categories only.

    If sort=False, return a copy of self, coded with categories as
    returned by .unique(), followed by any categories not appearing in
    the data. If sort=True, return self.

    This method is needed solely to ensure the categorical index of the
    GroupBy result has categories in the order of appearance in the data
    (GH-8868).

    Parameters
    ----------
    c : Categorical
        输入的分类数据
    sort : bool
        表示 groupby 操作是否需要排序的布尔值参数
    observed : bool
        是否仅考虑观察到的值

    Returns
    -------
    Categorical
        如果 sort=False，则新的分类将按照代码中出现的顺序设置（除非 ordered=True，此时保留原始顺序），
        然后跟随任何在原始数据中未出现的类别。
    """
    # 只关心观察到的值
    if observed:
        # 在 c.ordered 为真时，相当于 return c.remove_unused_categories(), c

        unique_codes = unique1d(c.codes)  # 获取唯一的代码值，忽略类型检查

        take_codes = unique_codes[unique_codes != -1]  # 排除空值代码
        if sort:
            take_codes = np.sort(take_codes)  # 如果需要排序，则对代码进行排序

        # 根据唯一值重新编码
        categories = c.categories.take(take_codes)
        codes = recode_for_categories(c.codes, c.categories, categories)

        # 返回一个新的分类对象，映射了新的代码和类别
        dtype = CategoricalDtype(categories, ordered=c.ordered)
        return Categorical._simple_new(codes, dtype=dtype)

    # 如果已经按照 c.categories 排序，则直接返回
    if sort:
        return c

    # 当 sort=False 时，应按照出现的顺序排序组（GH-8868）

    # xref GH:46909: 重新排序代码比使用 (set|add|reorder)_categories 更快
    all_codes = np.arange(c.categories.nunique())
    # GH 38140: 在类别的索引器中排除 NaN
    unique_notnan_codes = unique1d(c.codes[c.codes != -1])  # 获取非 NaN 的唯一代码，忽略类型检查
    if sort:
        unique_notnan_codes = np.sort(unique_notnan_codes)
    if len(all_codes) > len(unique_notnan_codes):
        # GH 13179: 所有类别都需要存在，即使在数据中缺失
        missing_codes = np.setdiff1d(all_codes, unique_notnan_codes, assume_unique=True)
        take_codes = np.concatenate((unique_notnan_codes, missing_codes))
    else:
        take_codes = unique_notnan_codes

    return Categorical(c, c.unique().categories.take(take_codes))
```