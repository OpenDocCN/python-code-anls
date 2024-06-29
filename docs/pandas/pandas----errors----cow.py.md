# `D:\src\scipysrc\pandas\pandas\errors\cow.py`

```
# 定义警告消息，用于指示尝试通过链式赋值在 DataFrame 或 Series 的副本上设置值的情况
_chained_assignment_msg = (
    "A value is trying to be set on a copy of a DataFrame or Series "
    "through chained assignment.\n"
    "When using the Copy-on-Write mode, such chained assignment never works "
    "to update the original DataFrame or Series, because the intermediate "
    "object on which we are setting values always behaves as a copy.\n\n"
    "Try using '.loc[row_indexer, col_indexer] = value' instead, to perform "
    "the assignment in a single step.\n\n"
    "See the caveats in the documentation: "
    "https://pandas.pydata.org/pandas-docs/stable/user_guide/"
    "copy_on_write.html"
)

# 定义警告消息，用于指示尝试通过链式赋值和 inplace 方法在 DataFrame 或 Series 的副本上设置值的情况
_chained_assignment_method_msg = (
    "A value is trying to be set on a copy of a DataFrame or Series "
    "through chained assignment using an inplace method.\n"
    "When using the Copy-on-Write mode, such inplace method never works "
    "to update the original DataFrame or Series, because the intermediate "
    "object on which we are setting values always behaves as a copy.\n\n"
    "For example, when doing 'df[col].method(value, inplace=True)', try "
    "using 'df.method({col: value}, inplace=True)' instead, to perform "
    "the operation inplace on the original object.\n\n"
)
```