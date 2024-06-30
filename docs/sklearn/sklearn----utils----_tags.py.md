# `D:\src\scipysrc\scikit-learn\sklearn\utils\_tags.py`

```
import numpy as np

# 默认的标签字典，用于描述估算器的属性
_DEFAULT_TAGS = {
    "array_api_support": False,
    "non_deterministic": False,
    "requires_positive_X": False,
    "requires_positive_y": False,
    "X_types": ["2darray"],
    "poor_score": False,
    "no_validation": False,
    "multioutput": False,
    "allow_nan": False,
    "stateless": False,
    "multilabel": False,
    "_skip_test": False,
    "_xfail_checks": False,
    "multioutput_only": False,
    "binary_only": False,
    "requires_fit": True,
    "preserves_dtype": [np.float64],
    "requires_y": False,
    "pairwise": False,
}


def _safe_tags(estimator, key=None):
    """安全获取估算器的标签信息。

    :class:`~sklearn.BaseEstimator` 提供了估算器标签的机制。
    但是，如果估算器没有继承这个基类，我们应该退回到默认的标签。

    对于 scikit-learn 内置的估算器，我们仍然应该依赖于 `self._get_tags()`。
    在不确定 `est` 来自哪里时，应该使用 `_safe_tags(self.base_estimator)`，
    其中 `self` 是一个元估算器，或者在常见的检查中使用。

    Parameters
    ----------
    estimator : estimator object
        要获取标签的估算器对象。

    key : str, default=None
        要获取的标签名称。默认为 `None`，返回所有标签。

    Returns
    -------
    tags : dict or tag value
        估算器的标签信息。如果指定了 `key`，则返回单个标签值。
    """
    # 如果估算器有 `_get_tags` 方法，则使用该方法获取标签
    if hasattr(estimator, "_get_tags"):
        tags_provider = "_get_tags()"
        tags = estimator._get_tags()
    # 如果估算器有 `_more_tags` 方法，则使用默认标签和该方法提供的标签合并
    elif hasattr(estimator, "_more_tags"):
        tags_provider = "_more_tags()"
        tags = {**_DEFAULT_TAGS, **estimator._more_tags()}
    # 否则，使用默认的标签字典
    else:
        tags_provider = "_DEFAULT_TAGS"
        tags = _DEFAULT_TAGS

    # 如果指定了 `key`，则返回对应的标签值；否则返回所有标签
    if key is not None:
        if key not in tags:
            raise ValueError(
                f"The key {key} is not defined in {tags_provider} for the "
                f"class {estimator.__class__.__name__}."
            )
        return tags[key]
    return tags
```