# `D:\src\scipysrc\scikit-learn\sklearn\inspection\_pd_utils.py`

```
# 检查特征名称函数
def _check_feature_names(X, feature_names=None):
    """Check feature names.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    feature_names : None or array-like of shape (n_names,), dtype=str
        Feature names to check or `None`.

    Returns
    -------
    feature_names : list of str
        Feature names validated. If `feature_names` is `None`, then a list of
        feature names is provided, i.e. the column names of a pandas dataframe
        or a generic list of feature names (e.g. `["x0", "x1", ...]`) for a
        NumPy array.
    """
    # 如果 feature_names 为 None，则根据输入数据 X 的类型获取特征名称
    if feature_names is None:
        if hasattr(X, "columns") and hasattr(X.columns, "tolist"):
            # 获取 pandas dataframe 的列名
            feature_names = X.columns.tolist()
        else:
            # 为 NumPy 数组定义一组以数字索引命名的特征名
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    # 如果 feature_names 可以转换为列表，则转换之
    elif hasattr(feature_names, "tolist"):
        feature_names = feature_names.tolist()
    # 检查特征名列表中是否有重复项，若有则引发 ValueError 异常
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names should not contain duplicates.")

    return feature_names


# 获取特征索引函数
def _get_feature_index(fx, feature_names=None):
    """Get feature index.

    Parameters
    ----------
    fx : int or str
        Feature index or name.

    feature_names : list of str, default=None
        All feature names from which to search the indices.

    Returns
    -------
    idx : int
        Feature index.
    """
    # 如果 fx 是字符串，则查找它在 feature_names 中的索引
    if isinstance(fx, str):
        # 如果 feature_names 为 None，则无法进行操作，抛出 ValueError 异常
        if feature_names is None:
            raise ValueError(
                f"Cannot plot partial dependence for feature {fx!r} since "
                "the list of feature names was not provided, neither as "
                "column names of a pandas data-frame nor via the feature_names "
                "parameter."
            )
        try:
            # 返回特征名在 feature_names 中的索引
            return feature_names.index(fx)
        except ValueError as e:
            # 如果特征名不在 feature_names 中，则引发 ValueError 异常
            raise ValueError(f"Feature {fx!r} not in feature_names") from e
    # 如果 fx 不是字符串，则直接返回它作为特征索引
    return fx
```