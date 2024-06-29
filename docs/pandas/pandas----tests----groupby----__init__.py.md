# `D:\src\scipysrc\pandas\pandas\tests\groupby\__init__.py`

```
# 定义函数，用于获取 groupby 方法所需的参数
def get_groupby_method_args(name, obj):
    """
    Get required arguments for a groupby method.

    When parametrizing a test over groupby methods (e.g. "sum", "mean", "fillna"),
    it is often the case that arguments are required for certain methods.

    Parameters
    ----------
    name: str
        Name of the method.
    obj: Series or DataFrame
        pandas object that is being grouped.

    Returns
    -------
    A tuple of required arguments for the method.
    """
    # 检查方法名是否在 ("nth", "fillna", "take") 中，若是则返回参数 (0,)
    if name in ("nth", "fillna", "take"):
        return (0,)
    # 检查方法名是否为 "quantile"，若是则返回参数 (0.5,)
    if name == "quantile":
        return (0.5,)
    # 检查方法名是否为 "corrwith"，若是则返回参数 (obj,)
    if name == "corrwith":
        return (obj,)
    # 默认情况下返回空元组，表示没有需要的额外参数
    return ()
```