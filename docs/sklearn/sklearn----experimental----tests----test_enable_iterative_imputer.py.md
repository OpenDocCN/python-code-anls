# `D:\src\scipysrc\scikit-learn\sklearn\experimental\tests\test_enable_iterative_imputer.py`

```
"""Tests for making sure experimental imports work as expected."""

# 导入所需的模块
import textwrap  # 文本包装，用于处理文本格式
import pytest  # 测试框架 pytest

# 从 sklearn.utils._testing 中导入 assert_run_python_script_without_output 函数
from sklearn.utils._testing import assert_run_python_script_without_output
# 从 sklearn.utils.fixes 中导入 _IS_WASM 常量
from sklearn.utils.fixes import _IS_WASM

# 使用 pytest.mark.xfail 标记测试用例，当 _IS_WASM 为 True 时，测试预期失败并添加原因说明
@pytest.mark.xfail(_IS_WASM, reason="cannot start subprocess")
def test_imports_strategies():
    # 确保不同的导入策略按预期工作或失败。

    # 由于 Python 缓存导入的模块，我们需要为每个测试用例运行一个子进程。
    # 否则，测试将不是独立的（手动从缓存中移除导入（sys.modules）是不推荐的，可能会导致许多复杂情况）。
    
    pattern = "IterativeImputer is experimental"
    
    # 好的导入测试：启用 IterativeImputer 和导入 IterativeImputer
    good_import = """
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    """
    assert_run_python_script_without_output(
        textwrap.dedent(good_import), pattern=pattern
    )

    # 先导入 sklearn.ensemble 然后测试：启用 IterativeImputer 和导入 IterativeImputer
    good_import_with_ensemble_first = """
    import sklearn.ensemble
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    """
    assert_run_python_script_without_output(
        textwrap.dedent(good_import_with_ensemble_first),
        pattern=pattern,
    )

    # 不良导入测试：尝试导入 IterativeImputer 应该触发 ImportError
    bad_imports = f"""
    import pytest

    with pytest.raises(ImportError, match={pattern!r}):
        from sklearn.impute import IterativeImputer

    import sklearn.experimental
    with pytest.raises(ImportError, match={pattern!r}):
        from sklearn.impute import IterativeImputer
    """
    assert_run_python_script_without_output(
        textwrap.dedent(bad_imports),
        pattern=pattern,
    )
```