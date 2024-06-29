# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_style.py`

```py
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np
import pytest

import matplotlib as mpl
from matplotlib import pyplot as plt, style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION


PARAM = 'image.cmap'
VALUE = 'pink'
DUMMY_SETTINGS = {PARAM: VALUE}


@contextmanager
def temp_style(style_name, settings=None):
    """Context manager to create a style sheet in a temporary directory."""
    if not settings:
        settings = DUMMY_SETTINGS
    temp_file = f'{style_name}.{STYLE_EXTENSION}'
    try:
        with TemporaryDirectory() as tmpdir:
            # Write style settings to file in the tmpdir.
            Path(tmpdir, temp_file).write_text(
                "\n".join(f"{k}: {v}" for k, v in settings.items()),
                encoding="utf-8")
            # Add tmpdir to style path and reload so we can access this style.
            USER_LIBRARY_PATHS.append(tmpdir)
            style.reload_library()
            yield
    finally:
        style.reload_library()


def test_invalid_rc_warning_includes_filename(caplog):
    SETTINGS = {'foo': 'bar'}
    basename = 'basename'
    with temp_style(basename, SETTINGS):
        # style.reload_library() in temp_style() triggers the warning
        pass
    assert (len(caplog.records) == 1
            and basename in caplog.records[0].getMessage())


def test_available():
    with temp_style('_test_', DUMMY_SETTINGS):
        assert '_test_' in style.available


def test_use():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            assert mpl.rcParams[PARAM] == VALUE


def test_use_url(tmp_path):
    path = tmp_path / 'file'
    path.write_text('axes.facecolor: adeade', encoding='utf-8')
    with temp_style('test', DUMMY_SETTINGS):
        url = ('file:'
               + ('///' if sys.platform == 'win32' else '')
               + path.resolve().as_posix())
        with style.context(url):
            assert mpl.rcParams['axes.facecolor'] == "#adeade"


def test_single_path(tmp_path):
    mpl.rcParams[PARAM] = 'gray'
    path = tmp_path / f'text.{STYLE_EXTENSION}'
    path.write_text(f'{PARAM} : {VALUE}', encoding='utf-8')
    with style.context(path):
        assert mpl.rcParams[PARAM] == VALUE
    assert mpl.rcParams[PARAM] == 'gray'


def test_context():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            assert mpl.rcParams[PARAM] == VALUE
    # Check that this value is reset after the exiting the context.
    assert mpl.rcParams[PARAM] == 'gray'


def test_context_with_dict():
    original_value = 'gray'
    other_value = 'blue'
    mpl.rcParams[PARAM] = original_value
    with style.context({PARAM: other_value}):
        assert mpl.rcParams[PARAM] == other_value
    assert mpl.rcParams[PARAM] == original_value



# 注释：


@contextmanager
def temp_style(style_name, settings=None):
    """上下文管理器，用于在临时目录中创建样式表。"""
    if not settings:
        settings = DUMMY_SETTINGS
    temp_file = f'{style_name}.{STYLE_EXTENSION}'
    try:
        with TemporaryDirectory() as tmpdir:
            # 将样式设置写入临时目录中的文件。
            Path(tmpdir, temp_file).write_text(
                "\n".join(f"{k}: {v}" for k, v in settings.items()),
                encoding="utf-8")
            # 将临时目录添加到样式路径中并重新加载，以便可以访问这个样式。
            USER_LIBRARY_PATHS.append(tmpdir)
            style.reload_library()
            yield
    finally:
        # 重新加载样式库以确保干净状态。
        style.reload_library()


def test_invalid_rc_warning_includes_filename(caplog):
    SETTINGS = {'foo': 'bar'}
    basename = 'basename'
    with temp_style(basename, SETTINGS):
        # temp_style() 中的 style.reload_library() 触发警告。
        pass
    assert (len(caplog.records) == 1
            and basename in caplog.records[0].getMessage())


def test_available():
    with temp_style('_test_', DUMMY_SETTINGS):
        # 检查创建的样式是否在可用样式列表中。
        assert '_test_' in style.available


def test_use():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            # 检查在临时样式下设置的参数值。
            assert mpl.rcParams[PARAM] == VALUE


def test_use_url(tmp_path):
    path = tmp_path / 'file'
    path.write_text('axes.facecolor: adeade', encoding='utf-8')
    with temp_style('test', DUMMY_SETTINGS):
        url = ('file:'
               + ('///' if sys.platform == 'win32' else '')
               + path.resolve().as_posix())
        with style.context(url):
            # 检查通过 URL 载入样式后的参数设置。
            assert mpl.rcParams['axes.facecolor'] == "#adeade"


def test_single_path(tmp_path):
    mpl.rcParams[PARAM] = 'gray'
    path = tmp_path / f'text.{STYLE_EXTENSION}'
    path.write_text(f'{PARAM} : {VALUE}', encoding='utf-8')
    with style.context(path):
        # 检查在指定路径下的样式设置是否正确。
        assert mpl.rcParams[PARAM] == VALUE
    assert mpl.rcParams[PARAM] == 'gray'


def test_context():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            # 检查在上下文中设置的样式参数是否生效。
            assert mpl.rcParams[PARAM] == VALUE
    # 检查在退出上下文后参数是否被重置。
    assert mpl.rcParams[PARAM] == 'gray'


def test_context_with_dict():
    original_value = 'gray'
    other_value = 'blue'
    mpl.rcParams[PARAM] = original_value
    with style.context({PARAM: other_value}):
        # 检查在使用字典形式的样式设置时参数是否被正确修改。
        assert mpl.rcParams[PARAM] == other_value
    assert mpl.rcParams[PARAM] == original_value
# 测试带有字典在命名样式之后的情况，其中字典修改了相同的参数。
def test_context_with_dict_after_namedstyle():
    original_value = 'gray'  # 原始参数值
    other_value = 'blue'  # 替换后的参数值
    mpl.rcParams[PARAM] = original_value  # 设置参数值为原始值
    with temp_style('test', DUMMY_SETTINGS):  # 使用临时样式 'test'
        with style.context(['test', {PARAM: other_value}]):  # 应用样式 'test' 和参数修改字典
            assert mpl.rcParams[PARAM] == other_value  # 断言参数值已经被修改为新值
    assert mpl.rcParams[PARAM] == original_value  # 断言退出上下文后参数值恢复为原始值


# 测试带有字典在命名样式之前的情况，其中字典修改了相同的参数。
def test_context_with_dict_before_namedstyle():
    original_value = 'gray'  # 原始参数值
    other_value = 'blue'  # 替换后的参数值
    mpl.rcParams[PARAM] = original_value  # 设置参数值为原始值
    with temp_style('test', DUMMY_SETTINGS):  # 使用临时样式 'test'
        with style.context([{PARAM: other_value}, 'test']):  # 应用参数修改字典和样式 'test'
            assert mpl.rcParams[PARAM] == VALUE  # 断言参数值被正确设置为新值
    assert mpl.rcParams[PARAM] == original_value  # 断言退出上下文后参数值恢复为原始值


# 测试带有字典在命名样式之后的情况，其中字典修改了不同的参数。
def test_context_with_union_of_dict_and_namedstyle():
    original_value = 'gray'  # 原始参数值
    other_param = 'text.usetex'  # 另一个参数名称
    other_value = True  # 替换后的参数值
    d = {other_param: other_value}  # 参数修改字典
    mpl.rcParams[PARAM] = original_value  # 设置参数值为原始值
    mpl.rcParams[other_param] = (not other_value)  # 设置另一个参数的值
    with temp_style('test', DUMMY_SETTINGS):  # 使用临时样式 'test'
        with style.context(['test', d]):  # 应用样式 'test' 和参数修改字典
            assert mpl.rcParams[PARAM] == VALUE  # 断言参数值被正确设置为新值
            assert mpl.rcParams[other_param] == other_value  # 断言另一个参数值也被正确设置
    assert mpl.rcParams[PARAM] == original_value  # 断言退出上下文后参数值恢复为原始值
    assert mpl.rcParams[other_param] == (not other_value)  # 断言另一个参数值恢复为原始值


# 测试带有错误参数的情况。
def test_context_with_badparam():
    original_value = 'gray'  # 原始参数值
    other_value = 'blue'  # 替换后的参数值
    with style.context({PARAM: other_value}):  # 应用参数修改字典
        assert mpl.rcParams[PARAM] == other_value  # 断言参数值被正确设置为新值
        x = style.context({PARAM: original_value, 'badparam': None})  # 尝试添加不支持的参数
        with pytest.raises(KeyError):  # 检查是否抛出 KeyError 异常
            with x:
                pass
        assert mpl.rcParams[PARAM] == other_value  # 断言退出上下文后参数值恢复为原始值


# 参数化测试，用不同的等价样式测试参数值是否相等。
@pytest.mark.parametrize('equiv_styles',
                         [('mpl20', 'default'),
                          ('mpl15', 'classic')],
                         ids=['mpl20', 'mpl15'])
def test_alias(equiv_styles):
    rc_dicts = []
    for sty in equiv_styles:
        with style.context(sty):  # 应用等价样式
            rc_dicts.append(mpl.rcParams.copy())  # 复制当前参数配置

    rc_base = rc_dicts[0]  # 第一个样式的参数配置
    for nm, rc in zip(equiv_styles[1:], rc_dicts[1:]):
        assert rc_base == rc  # 断言所有样式的参数配置应该相等


# 测试不使用上下文管理器时 XKCD 风格的参数设置。
def test_xkcd_no_cm():
    assert mpl.rcParams["path.sketch"] is None  # 断言参数值为 None
    plt.xkcd()  # 开启 XKCD 风格
    assert mpl.rcParams["path.sketch"] == (1, 100, 2)  # 断言参数值被正确设置为指定元组
    np.testing.break_cycles()  # 手动释放循环引用
    assert mpl.rcParams["path.sketch"] == (1, 100, 2)  # 断言退出 XKCD 风格后参数值仍然保持


# 测试使用上下文管理器时 XKCD 风格的参数设置。
def test_xkcd_cm():
    assert mpl.rcParams["path.sketch"] is None  # 断言参数值为 None
    with plt.xkcd():  # 使用上下文管理器开启 XKCD 风格
        assert mpl.rcParams["path.sketch"] == (1, 100, 2)  # 断言参数值被正确设置为指定元组
    assert mpl.rcParams["path.sketch"] is None  # 断言退出 XKCD 风格后参数值恢复为 None


# 测试 STYLE_BLACKLIST 是否包含所有 mpl.rcsetup._validators 的内容。
def test_up_to_date_blacklist():
    assert mpl.style.core.STYLE_BLACKLIST <= {*mpl.rcsetup._validators}  # 断言 STYLE_BLACKLIST 是 _validators 的子集


# 测试从模块中加载样式。
def test_style_from_module(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)  # 添加临时路径到系统路径
    monkeypatch.chdir(tmp_path)  # 切换工作目录
    # 创建一个临时路径作为包的路径，用于存放测试样式包
    pkg_path = tmp_path / "mpl_test_style_pkg"
    
    # 在临时路径下创建一个目录作为包的路径
    pkg_path.mkdir()
    
    # 在包的路径下创建一个名为 test_style.mplstyle 的文件，并写入指定的内容
    (pkg_path / "test_style.mplstyle").write_text(
        "lines.linewidth: 42", encoding="utf-8")
    
    # 在包的路径下创建一个名为 .mplstyle 的文件（注意点号开头），并写入指定的内容
    pkg_path.with_suffix(".mplstyle").write_text(
        "lines.linewidth: 84", encoding="utf-8")
    
    # 使用 mpl_test_style_pkg.test_style 样式来设置 Matplotlib 的样式
    mpl.style.use("mpl_test_style_pkg.test_style")
    
    # 断言当前的 Matplotlib 参数中 lines.linewidth 是否为 42
    assert mpl.rcParams["lines.linewidth"] == 42
    
    # 使用 mpl_test_style_pkg.mplstyle 样式来设置 Matplotlib 的样式
    mpl.style.use("mpl_test_style_pkg.mplstyle")
    
    # 断言当前的 Matplotlib 参数中 lines.linewidth 是否为 84
    assert mpl.rcParams["lines.linewidth"] == 84
    
    # 使用 ./mpl_test_style_pkg.mplstyle 样式（相对路径）来设置 Matplotlib 的样式
    mpl.style.use("./mpl_test_style_pkg.mplstyle")
    
    # 断言当前的 Matplotlib 参数中 lines.linewidth 是否为 84
    assert mpl.rcParams["lines.linewidth"] == 84
```