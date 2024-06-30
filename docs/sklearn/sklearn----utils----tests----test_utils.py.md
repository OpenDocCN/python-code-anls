# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_utils.py`

```
# 导入 joblib 库，用于并行处理和持久化模型
import joblib
# 导入 pytest 库，用于编写和运行测试
import pytest
# 从 sklearn.utils 中导入 parallel_backend, register_parallel_backend, tosequence 函数
from sklearn.utils import parallel_backend, register_parallel_backend, tosequence


# TODO(1.7): remove
# 测试 IS_PYPY 是否已废弃
def test_is_pypy_deprecated():
    # 使用 pytest 来检测 FutureWarning，匹配 "IS_PYPY is deprecated" 字符串
    with pytest.warns(FutureWarning, match="IS_PYPY is deprecated"):
        # 从 sklearn.utils 中导入 IS_PYPY，忽略掉 Flake8 的检查
        from sklearn.utils import IS_PYPY  # noqa


# TODO(1.7): remove
# 测试 tosequence 是否已废弃
def test_tosequence_deprecated():
    # 使用 pytest 来检测 FutureWarning，匹配 "tosequence was deprecated in 1.5" 字符串
    with pytest.warns(FutureWarning, match="tosequence was deprecated in 1.5"):
        # 调用 tosequence 函数，传入一个列表参数
        tosequence([1, 2, 3])


# TODO(1.7): remove
# 测试 parallel_backend 和 register_parallel_backend 是否已废弃
def test_parallel_backend_deprecated():
    # 使用 pytest 来检测 FutureWarning，匹配 "parallel_backend is deprecated" 字符串
    with pytest.warns(FutureWarning, match="parallel_backend is deprecated"):
        # 调用 parallel_backend 函数，使用 "loky" 参数和 None 作为参数
        parallel_backend("loky", None)

    # 使用 pytest 来检测 FutureWarning，匹配 "register_parallel_backend is deprecated" 字符串
    with pytest.warns(FutureWarning, match="register_parallel_backend is deprecated"):
        # 调用 register_parallel_backend 函数，注册名为 "a_backend" 的后端，使用 None 作为参数
        register_parallel_backend("a_backend", None)

    # 从 joblib.parallel.BACKENDS 字典中删除 "a_backend" 键
    del joblib.parallel.BACKENDS["a_backend"]
```