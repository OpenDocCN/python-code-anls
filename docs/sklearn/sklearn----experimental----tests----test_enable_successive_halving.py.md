# `D:\src\scipysrc\scikit-learn\sklearn\experimental\tests\test_enable_successive_halving.py`

```
# 导入所需模块和库
"""Tests for making sure experimental imports work as expected."""

import textwrap  # 导入文本包装模块，用于格式化文本
import pytest  # 导入 pytest 模块，用于编写和运行测试用例

from sklearn.utils._testing import assert_run_python_script_without_output  # 导入辅助函数，用于在不产生输出的情况下运行 Python 脚本
from sklearn.utils.fixes import _IS_WASM  # 导入一个布尔值，表示是否在 WebAssembly 环境下

# 定义测试用例
@pytest.mark.xfail(_IS_WASM, reason="cannot start subprocess")
def test_imports_strategies():
    # Make sure different import strategies work or fail as expected.

    # 检查不同的导入策略是否按预期工作或失败

    # 定义检查的模式字符串
    pattern = "Halving(Grid|Random)SearchCV is experimental"

    # 好的导入方式1：从 sklearn.experimental 中启用半折算法搜索，并导入相关类
    good_import = """
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.model_selection import HalvingRandomSearchCV
    """
    assert_run_python_script_without_output(
        textwrap.dedent(good_import), pattern=pattern
    )

    # 好的导入方式2：先导入 sklearn.model_selection，然后从 sklearn.experimental 中启用半折算法搜索，并导入相关类
    good_import_with_model_selection_first = """
    import sklearn.model_selection
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.model_selection import HalvingRandomSearchCV
    """
    assert_run_python_script_without_output(
        textwrap.dedent(good_import_with_model_selection_first),
        pattern=pattern,
    )

    # 错误的导入方式：尝试导入未启用的类，应该抛出 ImportError 异常
    bad_imports = f"""
    import pytest

    with pytest.raises(ImportError, match={pattern!r}):
        from sklearn.model_selection import HalvingGridSearchCV

    import sklearn.experimental
    with pytest.raises(ImportError, match={pattern!r}):
        from sklearn.model_selection import HalvingRandomSearchCV
    """
    assert_run_python_script_without_output(
        textwrap.dedent(bad_imports),
        pattern=pattern,
    )
```