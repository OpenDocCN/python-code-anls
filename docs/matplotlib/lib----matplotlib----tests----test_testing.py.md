# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_testing.py`

```py
import warnings  # 导入 warnings 模块，用于处理警告信息

import pytest  # 导入 pytest 测试框架

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
from matplotlib.testing.decorators import check_figures_equal  # 导入 matplotlib 测试装饰器，用于比较图形

@pytest.mark.xfail(  # 标记测试为预期失败，严格模式下会失败，原因是测试警告信息失败
    strict=True, reason="testing that warnings fail tests"
)
def test_warn_to_fail():
    warnings.warn("This should fail the test")  # 发出警告信息

@pytest.mark.parametrize("a", [1])  # 参数化测试，测试参数 a 的值为 1
@check_figures_equal(extensions=["png"])  # 使用 check_figures_equal 装饰器检查图形是否相等，比较的图形扩展名为 png
@pytest.mark.parametrize("b", [1])  # 参数化测试，测试参数 b 的值为 1
def test_parametrize_with_check_figure_equal(a, fig_ref, b, fig_test):
    assert a == b  # 断言 a 和 b 相等

def test_wrap_failure():
    with pytest.raises(ValueError, match="^The decorated function"):  # 测试捕获 ValueError 异常，异常信息匹配 "^The decorated function"
        @check_figures_equal()  # 使用 check_figures_equal 装饰器
        def should_fail(test, ref):
            pass  # 空函数体，用于测试失败情况

@pytest.mark.xfail(raises=RuntimeError, strict=True,
                   reason='Test for check_figures_equal test creating '
                          'new figures')
@check_figures_equal()  # 使用 check_figures_equal 装饰器
def test_check_figures_equal_extra_fig(fig_test, fig_ref):
    plt.figure()  # 创建一个新的图形

@check_figures_equal()  # 使用 check_figures_equal 装饰器
def test_check_figures_equal_closed_fig(fig_test, fig_ref):
    fig = plt.figure()  # 创建一个新的图形
    plt.close(fig)  # 关闭刚创建的图形
```