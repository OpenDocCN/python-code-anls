# `D:\src\scipysrc\scikit-learn\sklearn\impute\__init__.py`

```
# 导入需要的类型模块
import typing

# 导入基础的缺失值指示器和简单的填充器
from ._base import MissingIndicator, SimpleImputer
# 导入KNN填充器
from ._knn import KNNImputer

# 如果在类型检查时
if typing.TYPE_CHECKING:
    # 避免对实验性估算器（例如mypy）产生错误
    # TODO: 当估算器不再是实验性的时候，移除这个检查
    from ._iterative import IterativeImputer  # noqa

# 所有可以从该模块导出的类名列表
__all__ = ["MissingIndicator", "SimpleImputer", "KNNImputer"]


# TODO: 当估算器不再是实验性的时候，移除这个检查
def __getattr__(name):
    # 如果请求的属性是IterativeImputer，则引发导入错误
    if name == "IterativeImputer":
        raise ImportError(
            f"{name} is experimental and the API might change without any "
            "deprecation cycle. To use it, you need to explicitly import "
            "enable_iterative_imputer:\n"
            "from sklearn.experimental import enable_iterative_imputer"
        )
    # 对于其他未知的属性名称，引发属性错误
    raise AttributeError(f"module {__name__} has no attribute {name}")
```