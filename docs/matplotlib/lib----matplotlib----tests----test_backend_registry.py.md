# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_registry.py`

```
from collections.abc import Sequence
from typing import Any

import pytest

import matplotlib as mpl
from matplotlib.backends import BackendFilter, backend_registry


@pytest.fixture
def clear_backend_registry():
    # Fixture that clears the singleton backend_registry before and after use
    # so that the test state remains isolated.
    backend_registry._clear()  # 清除 backend_registry 的状态
    yield  # 暂停执行测试，在此期间运行测试函数
    backend_registry._clear()  # 再次清除 backend_registry 的状态，确保测试环境隔离


def has_duplicates(seq: Sequence[Any]) -> bool:
    return len(seq) > len(set(seq))  # 检查序列中是否有重复项


@pytest.mark.parametrize(
    'framework,expected',
    [
        ('qt', 'qtagg'),
        ('gtk3', 'gtk3agg'),
        ('gtk4', 'gtk4agg'),
        ('wx', 'wxagg'),
        ('tk', 'tkagg'),
        ('macosx', 'macosx'),
        ('headless', 'agg'),
        ('does not exist', None),
    ]
)
def test_backend_for_gui_framework(framework, expected):
    assert backend_registry.backend_for_gui_framework(framework) == expected
    # 测试不同 GUI 框架对应的后端是否符合预期


def test_list_builtin():
    backends = backend_registry.list_builtin()  # 获取内置后端列表
    assert not has_duplicates(backends)  # 确保内置后端列表中没有重复项
    # 使用集合比较，因为顺序不重要
    assert {*backends} == {
        'gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'macosx', 'nbagg', 'notebook',
        'qtagg', 'qtcairo', 'qt5agg', 'qt5cairo', 'tkagg',
        'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo', 'agg', 'cairo', 'pdf', 'pgf',
        'ps', 'svg', 'template',
    }


@pytest.mark.parametrize(
    'filter,expected',
    [
        (BackendFilter.INTERACTIVE,
         ['gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'macosx', 'nbagg', 'notebook',
          'qtagg', 'qtcairo', 'qt5agg', 'qt5cairo', 'tkagg',
          'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo']),
        (BackendFilter.NON_INTERACTIVE,
         ['agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']),
    ]
)
def test_list_builtin_with_filter(filter, expected):
    backends = backend_registry.list_builtin(filter)  # 使用指定过滤器获取内置后端列表
    assert not has_duplicates(backends)  # 确保内置后端列表中没有重复项
    # 使用集合比较，因为顺序不重要
    assert {*backends} == {*expected}


def test_list_gui_frameworks():
    frameworks = backend_registry.list_gui_frameworks()  # 获取支持的 GUI 框架列表
    assert not has_duplicates(frameworks)  # 确保 GUI 框架列表中没有重复项
    # 使用集合比较，因为顺序不重要
    assert {*frameworks} == {
        "gtk3", "gtk4", "macosx", "qt", "qt5", "qt6", "tk", "wx",
    }


@pytest.mark.parametrize("backend, is_valid", [
    ("agg", True),
    ("QtAgg", True),
    ("module://anything", True),
    ("made-up-name", False),
])
def test_is_valid_backend(backend, is_valid):
    assert backend_registry.is_valid_backend(backend) == is_valid
    # 测试验证后端名称的函数是否按预期工作


@pytest.mark.parametrize("backend, normalized", [
    ("agg", "matplotlib.backends.backend_agg"),
    ("QtAgg", "matplotlib.backends.backend_qtagg"),
    ("module://Anything", "Anything"),
])
def test_backend_normalization(backend, normalized):
    assert backend_registry._backend_module_name(backend) == normalized
    # 测试后端名称规范化函数是否按预期工作


def test_deprecated_rcsetup_attributes():
    match = "was deprecated in Matplotlib 3.9"
    # 此测试函数可能用于验证某些已弃用的配置属性是否按预期工作
    # 使用 pytest 模块来捕获特定警告类型（MatplotlibDeprecationWarning）的警告信息，并匹配指定的文本模式（match）
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match=match):
        # 访问 Matplotlib 的 rcsetup 模块中的 interactive_bk 属性，可能会触发警告
        mpl.rcsetup.interactive_bk
    
    # 使用 pytest 模块来捕获特定警告类型（MatplotlibDeprecationWarning）的警告信息，并匹配指定的文本模式（match）
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match=match):
        # 访问 Matplotlib 的 rcsetup 模块中的 non_interactive_bk 属性，可能会触发警告
        mpl.rcsetup.non_interactive_bk
    
    # 使用 pytest 模块来捕获特定警告类型（MatplotlibDeprecationWarning）的警告信息，并匹配指定的文本模式（match）
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match=match):
        # 访问 Matplotlib 的 rcsetup 模块中的 all_backends 属性，可能会触发警告
        mpl.rcsetup.all_backends
# 测试内联的 matplotlib_inline 是否可导入，如果不可导入则跳过这个测试
def test_entry_points_inline():
    pytest.importorskip('matplotlib_inline')
    # 获取所有可用的后端列表
    backends = backend_registry.list_all()
    # 断言 'inline' 后端是否在列表中
    assert 'inline' in backends


# 测试 ipympl 是否可导入，如果不可导入则跳过这个测试
def test_entry_points_ipympl():
    pytest.importorskip('ipympl')
    # 获取所有可用的后端列表
    backends = backend_registry.list_all()
    # 断言 'ipympl' 后端是否在列表中
    assert 'ipympl' in backends
    # 断言 'widget' 后端是否在列表中
    assert 'widget' in backends


# 测试在清空后端注册表的情况下，名称与内置名称冲突的情况
def test_entry_point_name_shadows_builtin(clear_backend_registry):
    with pytest.raises(RuntimeError):
        # 验证并存储给定的入口点，引发运行时错误如果名称冲突
        backend_registry._validate_and_store_entry_points([('qtagg', 'module1')])


# 测试具有重复名称的入口点的情况
def test_entry_point_name_duplicate(clear_backend_registry):
    with pytest.raises(RuntimeError):
        # 验证并存储给定的入口点，引发运行时错误如果名称重复
        backend_registry._validate_and_store_entry_points([('some_name', 'module1'), ('some_name', 'module2')])


# 测试具有相同名称和值的入口点是否被接受
def test_entry_point_identical(clear_backend_registry):
    # Issue https://github.com/matplotlib/matplotlib/issues/28367
    # 多个具有相同名称和值（值为模块）的入口点是可以接受的
    n = len(backend_registry._name_to_module)
    backend_registry._validate_and_store_entry_points([('some_name', 'some.module'), ('some_name', 'some.module')])
    # 断言注册表中的名称到模块映射数量增加了一个
    assert len(backend_registry._name_to_module) == n + 1
    # 断言 'some_name' 映射到的模块为 'module://some.module'
    assert backend_registry._name_to_module['some_name'] == 'module://some.module'


# 测试入口点名称是否为模块的情况
def test_entry_point_name_is_module(clear_backend_registry):
    with pytest.raises(RuntimeError):
        # 验证并存储给定的入口点，引发运行时错误如果名称为 'module://...' 形式
        backend_registry._validate_and_store_entry_points([('module://backend.something', 'module1')])


# 参数化测试，测试仅在需要时加载入口点的情况
@pytest.mark.parametrize('backend', [
    'agg',
    'module://matplotlib.backends.backend_agg',
])
def test_load_entry_points_only_if_needed(clear_backend_registry, backend):
    # 断言加载的入口点列表为空
    assert not backend_registry._loaded_entry_points
    # 解析并返回指定后端的结果
    check = backend_registry.resolve_backend(backend)
    # 断言解析结果为指定的后端和 None
    assert check == (backend, None)
    # 再次断言加载的入口点列表为空
    assert not backend_registry._loaded_entry_points
    # 强制加载所有入口点
    backend_registry.list_all()
    # 最后断言加载的入口点列表非空
    assert backend_registry._loaded_entry_points


# 参数化测试，测试解析 GUI 或后端的情况
@pytest.mark.parametrize(
    'gui_or_backend, expected_backend, expected_gui',
    [
        ('agg', 'agg', None),
        ('qt', 'qtagg', 'qt'),
        ('TkCairo', 'tkcairo', 'tk'),
    ]
)
def test_resolve_gui_or_backend(gui_or_backend, expected_backend, expected_gui):
    # 解析 GUI 或后端，返回解析结果
    backend, gui = backend_registry.resolve_gui_or_backend(gui_or_backend)
    # 断言解析后的后端与预期的后端相符
    assert backend == expected_backend
    # 断言解析后的 GUI 与预期的 GUI 相符
    assert gui == expected_gui


# 测试解析无效 GUI 或后端的情况
def test_resolve_gui_or_backend_invalid():
    match = "is not a recognised GUI loop or backend name"
    with pytest.raises(RuntimeError, match=match):
        # 尝试解析一个无效的 GUI 或后端，应该引发运行时错误并匹配指定消息
        backend_registry.resolve_gui_or_backend('no-such-name')
```