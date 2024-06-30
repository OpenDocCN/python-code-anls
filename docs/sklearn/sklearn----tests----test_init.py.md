# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_init.py`

```
# 模块顶层的基本单元测试，用于测试模块的顶层功能

# 模块作者信息
__author__ = "Yaroslav Halchenko"
# 模块使用的许可证信息
__license__ = "BSD"

# 尝试导入 sklearn 模块
try:
    # 导入 sklearn 模块的所有内容，忽略 PEP 8 的警告
    from sklearn import *  # noqa

    # 如果导入成功，设置顶层导入错误为 None
    _top_import_error = None
except Exception as e:
    # 如果导入过程中发生异常，记录异常对象到 _top_import_error
    _top_import_error = e

def test_import_skl():
    # 测试上述导入是否因某种原因失败
    # 在模块级别之外使用 "import *" 是不推荐的，因此我们依赖上面设置的变量来判断
    assert _top_import_error is None
```