# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_pyplot.py`

```
# 导入 difflib 模块，用于生成文本差异比较的统一格式的 diff
import difflib

# 导入 numpy 库，用于科学计算
import numpy as np

# 导入 sys 模块，提供对 Python 解释器的访问
import sys

# 从 pathlib 模块中导入 Path 类，用于处理文件路径
from pathlib import Path

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 matplotlib 库，并从中导入 subprocess_run_for_testing 函数和 pyplot 模块
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt


# 定义测试函数 test_pyplot_up_to_date，测试 pyplot 是否为最新版本
def test_pyplot_up_to_date(tmp_path):
    # 导入 black，如果不存在则跳过测试
    pytest.importorskip("black")

    # 找到并获取 matplotlib 的 boilerplate.py 文件路径
    gen_script = Path(mpl.__file__).parents[2] / "tools/boilerplate.py"
    if not gen_script.exists():
        pytest.skip("boilerplate.py not found")

    # 读取原始 pyplot.py 的内容
    orig_contents = Path(plt.__file__).read_text()

    # 在临时路径下创建 pyplot.py 文件，并将原始内容写入
    plt_file = tmp_path / 'pyplot.py'
    plt_file.write_text(orig_contents, 'utf-8')

    # 运行 boilerplate.py 脚本来更新 pyplot.py
    subprocess_run_for_testing(
        [sys.executable, str(gen_script), str(plt_file)],
        check=True)

    # 读取更新后的 pyplot.py 的内容
    new_contents = plt_file.read_text('utf-8')

    # 比较原始内容和更新后内容，如果不一致，则输出差异信息
    if orig_contents != new_contents:
        diff_msg = '\n'.join(
            difflib.unified_diff(
                orig_contents.split('\n'), new_contents.split('\n'),
                fromfile='found pyplot.py',
                tofile='expected pyplot.py',
                n=0, lineterm=''))
        pytest.fail(
            "pyplot.py is not up-to-date. Please run "
            "'python tools/boilerplate.py' to update pyplot.py. "
            "This needs to be done from an environment where your "
            "current working copy is installed (e.g. 'pip install -e'd). "
            "Here is a diff of unexpected differences:\n%s" % diff_msg
        )


# 定义测试函数 test_copy_docstring_and_deprecators，测试复制文档字符串和废弃项
def test_copy_docstring_and_deprecators(recwarn):
    # 使用装饰器更名和关键字参数设置
    @mpl._api.rename_parameter(mpl.__version__, "old", "new")
    @mpl._api.make_keyword_only(mpl.__version__, "kwo")
    def func(new, kwo=None):
        pass

    # 复制文档字符串和废弃项到 wrapper_func
    @plt._copy_docstring_and_deprecators(func)
    def wrapper_func(new, kwo=None):
        pass

    # 执行 wrapper_func 函数的不同调用方式，验证是否触发废弃项警告
    wrapper_func(None)
    wrapper_func(new=None)
    wrapper_func(None, kwo=None)
    wrapper_func(new=None, kwo=None)
    assert not recwarn
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        wrapper_func(old=None)
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        wrapper_func(None, None)


# 定义测试函数 test_pyplot_box，测试 pyplot 的 box 方法
def test_pyplot_box():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    # 禁用坐标轴边框
    plt.box(False)
    assert not ax.get_frame_on()

    # 启用坐标轴边框
    plt.box(True)
    assert ax.get_frame_on()

    # 切换坐标轴边框状态（此时应该禁用）
    plt.box()
    assert not ax.get_frame_on()

    # 再次切换坐标轴边框状态（此时应该启用）
    plt.box()
    assert ax.get_frame_on()


# 定义测试函数 test_stackplot_smoke，对 stackplot 进行小规模的功能测试
def test_stackplot_smoke():
    # 对 stackplot 进行简单的功能测试（参见 issue #12405）
    plt.stackplot([1, 2, 3], [1, 2, 3])


# 定义测试函数 test_nrows_error，测试在错误情况下使用 subplot 方法
def test_nrows_error():
    # 断言在指定 nrows 参数时会触发 TypeError 异常
    with pytest.raises(TypeError):
        plt.subplot(nrows=1)
    with pytest.raises(TypeError):
        plt.subplot(ncols=1)


# 定义测试函数 test_ioff，测试 matplotlib 的交互模式关闭功能
def test_ioff():
    # 打开交互模式并验证是否生效
    plt.ion()
    assert mpl.is_interactive()

    # 使用 plt.ioff() 上下文管理器关闭交互模式，并验证是否生效
    with plt.ioff():
        assert not mpl.is_interactive()
    assert mpl.is_interactive()

    # 直接调用 plt.ioff() 方法关闭交互模式，并验证是否生效
    plt.ioff()
    assert not mpl.is_interactive()

    # 再次使用 plt.ioff() 上下文管理器验证是否保持关闭状态
    with plt.ioff():
        assert not mpl.is_interactive()

    # 最终确认交互模式确实处于关闭状态
    assert not mpl.is_interactive()


# 定义测试函数 test_ion，测试 matplotlib 的交互模式打开功能
def test_ion():
    # 先关闭交互模式，并验证是否生效
    plt.ioff()
    assert not mpl.is_interactive()

    # 使用 plt.ion() 上下文管理器打开交互模式，并验证是否生效
    with plt.ion():
        assert mpl.is_interactive()
    # 断言当前 Matplotlib 不处于交互模式
    assert not mpl.is_interactive()
    
    # 将 Matplotlib 切换为交互模式
    plt.ion()
    
    # 断言当前 Matplotlib 处于交互模式
    assert mpl.is_interactive()
    
    # 使用上下文管理器切换到交互模式，确保在退出时恢复到之前的状态
    with plt.ion():
        # 在上下文管理器内，仍然保持 Matplotlib 处于交互模式
        assert mpl.is_interactive()
    
    # 再次断言当前 Matplotlib 处于交互模式
    assert mpl.is_interactive()
def test_nested_ion_ioff():
    # 初始状态是交互模式
    plt.ion()

    # 混合使用 ioff/ion
    with plt.ioff():
        # 断言当前不是交互模式
        assert not mpl.is_interactive()
        with plt.ion():
            # 断言当前是交互模式
            assert mpl.is_interactive()
        # 再次断言不是交互模式
        assert not mpl.is_interactive()
    # 最终断言是交互模式
    assert mpl.is_interactive()

    # 多余的上下文管理器
    with plt.ioff():
        with plt.ioff():
            # 断言不是交互模式
            assert not mpl.is_interactive()
    # 最终断言是交互模式
    assert mpl.is_interactive()

    # 切换为非交互模式
    plt.ioff()

    # 再次混合使用 ioff/ion
    with plt.ion():
        # 断言当前是交互模式
        assert mpl.is_interactive()
        with plt.ioff():
            # 断言当前不是交互模式
            assert not mpl.is_interactive()
        # 再次断言当前是交互模式
        assert mpl.is_interactive()
    # 最终断言不是交互模式
    assert not mpl.is_interactive()

    # 多余的上下文管理器
    with plt.ion():
        with plt.ion():
            # 断言当前是交互模式
            assert mpl.is_interactive()
    # 最终断言不是交互模式
    assert not mpl.is_interactive()

    # 切换为非交互模式
    with plt.ioff():
        plt.ion()
    # 最终断言不是交互模式
    assert not mpl.is_interactive()


def test_close():
    try:
        # 尝试关闭一个无效类型的对象
        plt.close(1.1)
    except TypeError as e:
        # 断言捕获到的异常消息正确
        assert str(e) == "close() argument must be a Figure, an int, " \
                         "a string, or None, not <class 'float'>"


def test_subplot_reuse():
    # 创建子图并断言是当前活动子图
    ax1 = plt.subplot(121)
    assert ax1 is plt.gca()
    # 创建第二个子图并断言是当前活动子图
    ax2 = plt.subplot(122)
    assert ax2 is plt.gca()
    # 重复使用已存在的子图配置，断言是第一个子图仍然是当前活动子图
    ax3 = plt.subplot(121)
    assert ax1 is plt.gca()
    assert ax1 is ax3


def test_axes_kwargs():
    # plt.axes() 每次都创建新的坐标轴，即使参数不同
    plt.figure()
    ax = plt.axes()
    ax1 = plt.axes()
    assert ax is not None
    assert ax1 is not ax
    plt.close()

    plt.figure()
    ax = plt.axes(projection='polar')
    ax1 = plt.axes(projection='polar')
    assert ax is not None
    assert ax1 is not ax
    plt.close()

    plt.figure()
    ax = plt.axes(projection='polar')
    ax1 = plt.axes()
    assert ax is not None
    assert ax1.name == 'rectilinear'
    assert ax1 is not ax
    plt.close()


def test_subplot_replace_projection():
    # plt.subplot() 查找具有相同子图规范的坐标轴，如果存在并且参数匹配则返回它，否则创建新的
    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax3 = plt.subplot(1, 2, 1, projection='polar')
    ax4 = plt.subplot(1, 2, 1, projection='polar')
    assert ax is not None
    assert ax1 is ax
    assert ax2 is not ax
    assert ax3 is not ax
    assert ax3 is ax4

    assert ax in fig.axes
    assert ax2 in fig.axes
    assert ax3 in fig.axes

    assert ax.name == 'rectilinear'
    assert ax2.name == 'rectilinear'
    assert ax3.name == 'polar'


def test_subplot_kwarg_collision():
    # 创建极坐标子图，断言相同的参数将返回同一个坐标轴对象
    ax1 = plt.subplot(projection='polar', theta_offset=0)
    ax2 = plt.subplot(projection='polar', theta_offset=0)
    assert ax1 is ax2
    # 移除一个子图
    ax1.remove()
    # 再次创建极坐标子图，断言参数不同会创建新的坐标轴对象
    ax3 = plt.subplot(projection='polar', theta_offset=1)
    # 确保变量 ax1 不是变量 ax3
    assert ax1 is not ax3
    
    # 确保变量 ax1 不在当前图形中的所有轴列表中
    assert ax1 not in plt.gcf().axes
def test_gca():
    # 创建一个新的空图形
    plt.figure()
    # 获取当前的轴对象（如果不存在，则创建一个新的）
    ax = plt.gca()
    # 再次获取当前的轴对象，应与之前获取的相同
    ax1 = plt.gca()
    # 断言 ax 不为空
    assert ax is not None
    # 断言 ax1 和 ax 是同一个对象
    assert ax1 is ax
    # 关闭当前图形
    plt.close()


def test_subplot_projection_reuse():
    # 创建一个标准的坐标轴对象
    ax1 = plt.subplot(111)
    # 断言 ax1 是当前的轴对象
    assert ax1 is plt.gca()
    # 再次创建相同的坐标轴对象，应返回之前创建的对象
    assert ax1 is plt.subplot(111)
    # 移除 ax1 对象
    ax1.remove()
    # 创建一个极坐标图
    ax2 = plt.subplot(111, projection='polar')
    # 断言 ax2 是当前的轴对象
    assert ax2 is plt.gca()
    # 此操作应该已经删除了第一个坐标轴对象 ax1
    assert ax1 not in plt.gcf().axes
    # 如果没有额外的参数传递，应返回之前创建的 ax2 对象
    assert ax2 is plt.subplot(111)
    # 移除 ax2 对象
    ax2.remove()
    # 显式设置投影为 'rectilinear'，应创建一个新的坐标轴对象
    ax3 = plt.subplot(111, projection='rectilinear')
    # 断言 ax3 是当前的轴对象
    assert ax3 is plt.gca()
    # 断言 ax3 和 ax2 不是同一个对象
    assert ax3 is not ax2
    # 断言 ax2 不在当前图形的轴对象列表中
    assert ax2 not in plt.gcf().axes


def test_subplot_polar_normalization():
    # 创建一个极坐标图
    ax1 = plt.subplot(111, projection='polar')
    # 使用 polar=True 创建一个极坐标图
    ax2 = plt.subplot(111, polar=True)
    # 使用 polar=True 和 projection='polar' 创建一个极坐标图
    ax3 = plt.subplot(111, polar=True, projection='polar')
    # 断言 ax1、ax2、ax3 是同一个对象
    assert ax1 is ax2
    assert ax1 is ax3

    # 使用 pytest 检查异常情况：polar=True，但同时使用 projection='3d'
    with pytest.raises(ValueError,
                       match="polar=True, yet projection='3d'"):
        ax2 = plt.subplot(111, polar=True, projection='3d')


def test_subplot_change_projection():
    # 创建一个空的坐标轴集合
    created_axes = set()
    # 创建一个标准坐标轴对象
    ax = plt.subplot()
    # 将 ax 对象添加到创建的坐标轴集合中
    created_axes.add(ax)
    # 不同投影类型的列表
    projections = ('aitoff', 'hammer', 'lambert', 'mollweide',
                   'polar', 'rectilinear', '3d')
    # 遍历投影类型列表
    for proj in projections:
        # 移除当前的坐标轴对象
        ax.remove()
        # 创建指定投影类型的坐标轴对象
        ax = plt.subplot(projection=proj)
        # 断言新创建的 ax 是当前的轴对象
        assert ax is plt.subplot()
        # 断言 ax 的名称与当前投影类型相符
        assert ax.name == proj
        # 将新创建的 ax 对象添加到坐标轴集合中
        created_axes.add(ax)
    # 断言坐标轴集合中只有一个对象
    assert len(created_axes) == 1 + len(projections)


def test_polar_second_call():
    # 第一次调用创建一个极坐标图
    ln1, = plt.polar(0., 1., 'ro')
    # 第二次调用应该重用现有的坐标轴
    ln2, = plt.polar(1.57, .5, 'bo')
    # 断言 ln1 和 ln2 使用的是同一个坐标轴对象
    assert ln1.axes is ln2.axes
    

def test_fallback_position():
    # 检查 position 关键字参数在没有指定 rect 的情况下是否有效
    axref = plt.axes([0.2, 0.2, 0.5, 0.5])
    axtest = plt.axes(position=[0.2, 0.2, 0.5, 0.5])
    # 使用 numpy 测试确保两个坐标轴对象的边界框坐标一致
    np.testing.assert_allclose(axtest.bbox.get_points(),
                               axref.bbox.get_points())

    # 检查如果指定了 rect 参数，position 关键字参数是否被忽略
    axref = plt.axes([0.2, 0.2, 0.5, 0.5])
    axtest = plt.axes([0.2, 0.2, 0.5, 0.5], position=[0.1, 0.1, 0.8, 0.8])
    # 使用 numpy 测试确保两个坐标轴对象的边界框坐标一致
    np.testing.assert_allclose(axtest.bbox.get_points(),
                               axref.bbox.get_points())


def test_set_current_figure_via_subfigure():
    # 创建一个新的图形对象 fig1
    fig1 = plt.figure()
    # 创建 fig1 中的两个子图
    subfigs = fig1.subfigures(2)

    # 创建一个新的图形对象
    plt.figure()
    # 断言当前图形对象不是之前创建的 fig1
    assert plt.gcf() != fig1
    # 创建一个新的图形对象，并将其设置为子图列表的第二个子图
    current = plt.figure(subfigs[1])
    # 使用断言确保当前的图形对象是 fig1
    assert plt.gcf() == fig1
    # 使用断言确保变量 current 和 fig1 是同一个图形对象
    assert current == fig1
def test_set_current_axes_on_subfigure():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 创建包含两个子图的 subfigures 对象
    subfigs = fig.subfigures(2)

    # 在第一个子图中创建一个 subplot
    ax = subfigs[0].subplots(1, squeeze=True)
    # 在第二个子图中创建一个 subplot
    subfigs[1].subplots(1, squeeze=True)

    # 断言当前的坐标轴不是 ax
    assert plt.gca() != ax
    # 将当前的坐标轴设置为 ax
    plt.sca(ax)
    # 断言当前的坐标轴是 ax
    assert plt.gca() == ax


def test_pylab_integration():
    # 导入 IPython，如果不存在则跳过测试
    IPython = pytest.importorskip("IPython")
    # 运行 IPython 并启用 --pylab 模式，检查相关条件
    mpl.testing.subprocess_run_helper(
        IPython.start_ipython,
        "--pylab",
        "-c",
        ";".join((
            "import matplotlib.pyplot as plt",
            "assert plt._REPL_DISPLAYHOOK == plt._ReplDisplayHook.IPYTHON",
        )),
        timeout=60,
    )


def test_doc_pyplot_summary():
    """Test that pyplot_summary lists all the plot functions."""
    # 获取 pyplot_summary 的路径
    pyplot_docs = Path(__file__).parent / '../../../doc/api/pyplot_summary.rst'
    # 如果文件不存在，则跳过测试
    if not pyplot_docs.exists():
        pytest.skip("Documentation sources not available")

    def extract_documented_functions(lines):
        """
        Return a list of all the functions that are mentioned in the
        autosummary blocks contained in *lines*.

        An autosummary block looks like this::

            .. autosummary::
               :toctree: _as_gen
               :template: autosummary.rst
               :nosignatures:

               plot
               plot_date

        """
        functions = []
        in_autosummary = False
        for line in lines:
            if not in_autosummary:
                if line.startswith(".. autosummary::"):
                    in_autosummary = True
            else:
                if not line or line.startswith("   :"):
                    # 空行或 autosummary 参数
                    continue
                if not line[0].isspace():
                    # 不再缩进：autosummary 块的结束
                    in_autosummary = False
                    continue
                functions.append(line.strip())
        return functions

    # 读取文档的所有行
    lines = pyplot_docs.read_text().split("\n")
    # 提取文档中记录的函数列表
    doc_functions = set(extract_documented_functions(lines))
    # 获取当前 matplotlib 中所有的 pyplot 命令
    plot_commands = set(plt._get_pyplot_commands())
    # 检查文档中未记录的 pyplot 命令
    missing = plot_commands.difference(doc_functions)
    if missing:
        raise AssertionError(
            f"The following pyplot functions are not listed in the "
            f"documentation. Please add them to doc/api/pyplot_summary.rst: "
            f"{missing!r}")
    # 检查文档中多余的 pyplot 命令
    extra = doc_functions.difference(plot_commands)
    if extra:
        raise AssertionError(
            f"The following functions are listed in the pyplot documentation, "
            f"but they do not exist in pyplot. "
            f"Please remove them from doc/api/pyplot_summary.rst: {extra!r}")


def test_minor_ticks():
    # 创建一个新的图形对象
    plt.figure()
    # 绘制一条简单的折线图
    plt.plot(np.arange(1, 10))
    # 获取次要刻度的位置和标签
    tick_pos, tick_labels = plt.xticks(minor=True)
    # 断言次要刻度标签为空数组
    assert np.all(tick_labels == np.array([], dtype=np.float64))
    assert tick_labels == []

    # 设置 y 轴的次要刻度
    plt.yticks(ticks=[3.5, 6.5], labels=["a", "b"], minor=True)
    # 获取当前的坐标轴对象
    ax = plt.gca()
    # 获取 Y 轴上的次要刻度位置
    tick_pos = ax.get_yticks(minor=True)
    # 获取 Y 轴上的次要刻度标签
    tick_labels = ax.get_yticklabels(minor=True)
    # 断言：确保所有的次要刻度位置都等于 [3.5, 6.5]
    assert np.all(tick_pos == np.array([3.5, 6.5]))
    # 断言：确保所有的次要刻度标签文本为 ['a', 'b']
    assert [l.get_text() for l in tick_labels] == ['a', 'b']
# 定义测试函数，验证在不关闭后端的情况下切换 Matplotlib 后端
def test_switch_backend_no_close():
    # 切换 Matplotlib 后端到 'agg'
    plt.switch_backend('agg')
    # 创建一个新的图形对象
    fig = plt.figure()
    # 再次创建一个新的图形对象
    fig = plt.figure()
    # 断言当前已创建的图形对象数量为 2
    assert len(plt.get_fignums()) == 2
    # 再次切换 Matplotlib 后端到 'agg'
    plt.switch_backend('agg')
    # 断言当前已创建的图形对象数量为 2，确认切换后端不会影响现有图形数量
    assert len(plt.get_fignums()) == 2
    # 使用 pytest 来捕获 MatplotlibDeprecationWarning 警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        # 再次切换 Matplotlib 后端到 'svg'
        plt.switch_backend('svg')
    # 断言当前已创建的图形对象数量为 0，确认切换后端到 'svg' 会清空图形对象
    assert len(plt.get_fignums()) == 0


# 定义一个示例函数，用于在创建图形对象时设置一个特定的钩子函数
def figure_hook_example(figure):
    # 将 _test_was_here 属性设置为 True，表示钩子函数执行过
    figure._test_was_here = True


# 测试图形对象钩子函数的功能
def test_figure_hook():
    # 定义一个测试用的 rc 参数字典，设置 figure.hooks 为指定的钩子函数 example
    test_rc = {
        'figure.hooks': ['matplotlib.tests.test_pyplot:figure_hook_example']
    }
    # 在特定的 rc 上下文中运行测试
    with mpl.rc_context(test_rc):
        # 创建一个新的图形对象
        fig = plt.figure()
    # 断言在 rc 上下文中设置的钩子函数已经执行过，即 fig._test_was_here 应为 True
    assert fig._test_was_here


# 测试在多次调用相同 figure 编号时的行为
def test_multiple_same_figure_calls():
    # 创建编号为 1 的图形对象 fig，设置尺寸为 (1, 2)
    fig = mpl.pyplot.figure(1, figsize=(1, 2))
    # 使用 pytest 来捕获 UserWarning 警告，并匹配指定的警告信息
    with pytest.warns(UserWarning, match="Ignoring specified arguments in this call"):
        # 再次创建编号为 1 的图形对象 fig2，设置尺寸为 (3, 4)
        fig2 = mpl.pyplot.figure(1, figsize=(3, 4))
    # 使用 pytest 来捕获 UserWarning 警告，并匹配指定的警告信息
    with pytest.warns(UserWarning, match="Ignoring specified arguments in this call"):
        # 使用 fig 作为参数创建图形对象，设置尺寸为 (5, 6)
        mpl.pyplot.figure(fig, figsize=(5, 6))
    # 断言 fig 和 fig2 是同一个对象
    assert fig is fig2
    # 再次创建编号为 1 的图形对象 fig3，不应触发警告
    fig3 = mpl.pyplot.figure(1)  # Checks for false warnings
    # 断言 fig 和 fig3 是同一个对象
    assert fig is fig3


# 测试在传递参数 'all' 时关闭所有图形对象时的警告
def test_close_all_warning():
    # 创建一个新的图形对象 fig1
    fig1 = plt.figure()
    # 使用 pytest 来捕获 UserWarning 警告，并匹配指定的警告信息
    with pytest.warns(UserWarning, match="closes all existing figures"):
        # 创建一个新的图形对象 fig2，参数为 "all"，关闭所有已有的图形对象
        fig2 = plt.figure("all")
```