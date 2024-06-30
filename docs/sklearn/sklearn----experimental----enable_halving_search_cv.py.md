# `D:\src\scipysrc\scikit-learn\sklearn\experimental\enable_halving_search_cv.py`

```
# 启用连续折半搜索估计器模块

# 导入此文件会动态设置 `model_selection` 模块的属性，包括
# :class:`~sklearn.model_selection.HalvingRandomSearchCV` 和
# :class:`~sklearn.model_selection.HalvingGridSearchCV`。

# 显式启用这个实验性功能
from sklearn.experimental import enable_halving_search_cv  # noqa

# 现在可以正常从 `model_selection` 导入
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import HalvingGridSearchCV

# `# noqa` 注释可以移除：它只是告诉像 flake8 这样的 linter 忽略未使用的导入。

# 使用 `setattr` 避免在 monkeypatching 时出现 mypy 的错误
setattr(model_selection, "HalvingRandomSearchCV", HalvingRandomSearchCV)
setattr(model_selection, "HalvingGridSearchCV", HalvingGridSearchCV)

# 更新 `model_selection.__all__`，使其包含新添加的类名
model_selection.__all__ += ["HalvingRandomSearchCV", "HalvingGridSearchCV"]
```