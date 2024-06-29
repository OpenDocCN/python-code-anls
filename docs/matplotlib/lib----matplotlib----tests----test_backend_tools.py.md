# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_tools.py`

```py
# 导入 pytest 测试框架
import pytest

# 从 matplotlib.backend_tools 模块中导入 ToolHelpBase 类
from matplotlib.backend_tools import ToolHelpBase

# 使用 pytest 的 parametrize 装饰器，定义测试参数化数据
@pytest.mark.parametrize('rc_shortcut,expected', [
    ('home', 'Home'),                # 测试 'home' 快捷键转换为 'Home'
    ('backspace', 'Backspace'),      # 测试 'backspace' 快捷键转换为 'Backspace'
    ('f1', 'F1'),                    # 测试 'f1' 快捷键转换为 'F1'
    ('ctrl+a', 'Ctrl+A'),            # 测试 'ctrl+a' 快捷键转换为 'Ctrl+A'
    ('ctrl+A', 'Ctrl+Shift+A'),      # 测试 'ctrl+A' 快捷键转换为 'Ctrl+Shift+A'
    ('a', 'a'),                      # 测试 'a' 快捷键转换为 'a'
    ('A', 'A'),                      # 测试 'A' 快捷键转换为 'A'
    ('ctrl+shift+f1', 'Ctrl+Shift+F1'),  # 测试 'ctrl+shift+f1' 快捷键转换为 'Ctrl+Shift+F1'
    ('1', '1'),                      # 测试 '1' 快捷键转换为 '1'
    ('cmd+p', 'Cmd+P'),              # 测试 'cmd+p' 快捷键转换为 'Cmd+P'
    ('cmd+1', 'Cmd+1'),              # 测试 'cmd+1' 快捷键转换为 'Cmd+1'
])
# 定义测试函数 test_format_shortcut，验证 ToolHelpBase.format_shortcut 方法的输出是否符合预期
def test_format_shortcut(rc_shortcut, expected):
    assert ToolHelpBase.format_shortcut(rc_shortcut) == expected
```