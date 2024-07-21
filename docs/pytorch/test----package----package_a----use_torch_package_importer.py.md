# `.\pytorch\test\package\package_a\use_torch_package_importer.py`

```py
# 尝试导入 torch_package_importer 模块，如果导入失败则不做任何操作
try:
    import torch_package_importer  # noqa: F401
except ImportError:
    pass
```