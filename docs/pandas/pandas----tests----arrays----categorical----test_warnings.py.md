# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_warnings.py`

```
import pytest  # 导入 pytest 库
import pandas._testing as tm  # 导入 pandas 的测试工具模块

class TestCategoricalWarnings:
    def test_tab_complete_warning(self, ip):
        # 导入 IPython 并检查最低版本要求
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter  # 导入 IPython 的自动完成模块

        code = "import pandas as pd; c = pd.Categorical([])"  # 定义一个字符串代码块
        ip.run_cell(code)  # 在 IPython 中运行代码块

        # 在 IPython 环境中，确保旧版 jedi 不会引发废弃警告
        with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
            with provisionalcompleter("ignore"):  # 使用 provisionalcompleter 忽略自动完成警告
                list(ip.Completer.completions("c.", 1))  # 获取在 "c." 上下文中的自动完成列表
```