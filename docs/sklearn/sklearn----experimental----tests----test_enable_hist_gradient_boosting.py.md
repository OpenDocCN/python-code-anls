# `D:\src\scipysrc\scikit-learn\sklearn\experimental\tests\test_enable_hist_gradient_boosting.py`

```
"""Tests for making sure experimental imports work as expected."""

# 导入用于文本包装的模块
import textwrap

# 导入 Pytest 测试框架
import pytest

# 导入用于运行 Python 脚本并断言输出的函数
from sklearn.utils._testing import assert_run_python_script_without_output

# 导入一个标志，检查是否为 WebAssembly 环境
from sklearn.utils.fixes import _IS_WASM


# 标记为预期失败的测试用例，如果是 WebAssembly 环境则跳过
@pytest.mark.xfail(_IS_WASM, reason="cannot start subprocess")
def test_import_raises_warning():
    # 定义测试代码段，导入 sklearn.experimental 中的模块
    code = """
    import pytest
    # 确保在导入期间会发出 UserWarning，且警告信息匹配特定模式
    with pytest.warns(UserWarning, match="it is not needed to import"):
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    """
    # 期望的警告信息模式
    pattern = "it is not needed to import enable_hist_gradient_boosting anymore"
    # 运行 Python 脚本并验证输出
    assert_run_python_script_without_output(textwrap.dedent(code), pattern=pattern)
```