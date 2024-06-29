# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_template.py`

```
"""
Backend-loading machinery tests, using variations on the template backend.
"""

# 导入必要的库和模块
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends import backend_template
from matplotlib.backends.backend_template import (
    FigureCanvasTemplate, FigureManagerTemplate)


# 测试加载模板
def test_load_template():
    # 设置使用模板作为后端
    mpl.use("template")
    # 断言确保创建的图形的画布类型是 FigureCanvasTemplate
    assert type(plt.figure().canvas) == FigureCanvasTemplate


# 测试加载旧的 API
def test_load_old_api(monkeypatch):
    # 创建一个模拟的简单命名空间，用于模拟 backend_template
    mpl_test_backend = SimpleNamespace(**vars(backend_template))
    # 定义一个新的图形管理器函数，返回 FigureManagerTemplate 对象
    mpl_test_backend.new_figure_manager = (
        lambda num, *args, FigureClass=mpl.figure.Figure, **kwargs:
        FigureManagerTemplate(
            FigureCanvasTemplate(FigureClass(*args, **kwargs)), num))
    # 使用 monkeypatch 替换模块中的 mpl_test_backend
    monkeypatch.setitem(sys.modules, "mpl_test_backend", mpl_test_backend)
    # 设置使用自定义的测试后端
    mpl.use("module://mpl_test_backend")
    # 断言确保创建的图形的画布类型是 FigureCanvasTemplate
    assert type(plt.figure().canvas) == FigureCanvasTemplate
    # 在交互式模式下绘制
    plt.draw_if_interactive()


# 测试显示功能
def test_show(monkeypatch):
    # 创建一个模拟的简单命名空间，用于模拟 backend_template
    mpl_test_backend = SimpleNamespace(**vars(backend_template))
    # 使用 MagicMock 创建一个 mock_show 函数
    mock_show = MagicMock()
    # 使用 monkeypatch 替换 FigureManagerTemplate 类中的 pyplot_show 方法
    monkeypatch.setattr(
        mpl_test_backend.FigureManagerTemplate, "pyplot_show", mock_show)
    # 使用 monkeypatch 替换模块中的 mpl_test_backend
    monkeypatch.setitem(sys.modules, "mpl_test_backend", mpl_test_backend)
    # 设置使用自定义的测试后端
    mpl.use("module://mpl_test_backend")
    # 调用 plt.show() 函数
    plt.show()
    # 断言确保 mock_show 方法被调用
    mock_show.assert_called_with()


# 测试显示旧的全局 API
def test_show_old_global_api(monkeypatch):
    # 创建一个模拟的简单命名空间，用于模拟 backend_template
    mpl_test_backend = SimpleNamespace(**vars(backend_template))
    # 使用 MagicMock 创建一个 mock_show 函数
    mock_show = MagicMock()
    # 使用 monkeypatch 替换 mpl_test_backend 中的 show 方法
    monkeypatch.setattr(mpl_test_backend, "show", mock_show, raising=False)
    # 使用 monkeypatch 替换模块中的 mpl_test_backend
    monkeypatch.setitem(sys.modules, "mpl_test_backend", mpl_test_backend)
    # 设置使用自定义的测试后端
    mpl.use("module://mpl_test_backend")
    # 调用 plt.show() 函数
    plt.show()
    # 断言确保 mock_show 方法被调用
    mock_show.assert_called_with()


# 测试加载对大小写敏感的后端
def test_load_case_sensitive(monkeypatch):
    # 创建一个模拟的简单命名空间，用于模拟 backend_template
    mpl_test_backend = SimpleNamespace(**vars(backend_template))
    # 使用 MagicMock 创建一个 mock_show 函数
    mock_show = MagicMock()
    # 使用 monkeypatch 替换 FigureManagerTemplate 类中的 pyplot_show 方法
    monkeypatch.setattr(
        mpl_test_backend.FigureManagerTemplate, "pyplot_show", mock_show)
    # 使用 monkeypatch 替换模块中的 mpl_Test_Backend，这里大小写敏感
    monkeypatch.setitem(sys.modules, "mpl_Test_Backend", mpl_test_backend)
    # 设置使用自定义的测试后端
    mpl.use("module://mpl_Test_Backend")
    # 调用 plt.show() 函数
    plt.show()
    # 断言确保 mock_show 方法被调用
    mock_show.assert_called_with()
```