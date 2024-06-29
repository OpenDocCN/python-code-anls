# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_rcparams.py`

```
# 导入必要的库
import copy  # 导入copy模块，用于复制对象
import os  # 导入os模块，提供与操作系统交互的功能
import subprocess  # 导入subprocess模块，用于调用系统命令
import sys  # 导入sys模块，提供对Python解释器的访问
from unittest import mock  # 从unittest模块中导入mock，用于模拟对象

from cycler import cycler, Cycler  # 导入cycler模块中的cycler和Cycler类
import pytest  # 导入pytest测试框架

import matplotlib as mpl  # 导入matplotlib库的mpl别名
from matplotlib import _api, _c_internal_utils  # 从matplotlib库中导入内部工具
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块的plt别名
import matplotlib.colors as mcolors  # 导入matplotlib.colors模块的mcolors别名
import numpy as np  # 导入NumPy库的np别名
from matplotlib.rcsetup import (  # 从matplotlib.rcsetup模块中导入多个函数
    validate_bool, validate_color, validate_colorlist,
    _validate_color_or_linecolor, validate_cycler,
    validate_float, validate_fontstretch, validate_fontweight,
    validate_hatch, validate_hist_bins, validate_int,
    validate_markevery, validate_stringlist, validate_sketch,
    _validate_linestyle, _listify_validator)
from matplotlib.testing import subprocess_run_for_testing  # 导入matplotlib.testing模块中的函数


def test_rcparams(tmp_path):
    # 设置rc参数'text.usetex'为False
    mpl.rc('text', usetex=False)
    # 设置rc参数'lines.linewidth'为22
    mpl.rc('lines', linewidth=22)

    # 获取当前rc参数'text.usetex'的值
    usetex = mpl.rcParams['text.usetex']
    # 获取当前rc参数'lines.linewidth'的值
    linewidth = mpl.rcParams['lines.linewidth']

    # 在临时目录下创建测试用的rc文件
    rcpath = tmp_path / 'test_rcparams.rc'
    rcpath.write_text('lines.linewidth: 33', encoding='utf-8')

    # 使用字典形式测试rc_context
    with mpl.rc_context(rc={'text.usetex': not usetex}):
        # 断言当前rc参数'text.usetex'的值与预期相符
        assert mpl.rcParams['text.usetex'] == (not usetex)
    # 恢复rc参数'text.usetex'的值为之前的值
    assert mpl.rcParams['text.usetex'] == usetex

    # 使用文件名形式测试rc_context（mpl.rc设置linewidth为33）
    with mpl.rc_context(fname=rcpath):
        # 断言当前rc参数'lines.linewidth'的值与预期相符
        assert mpl.rcParams['lines.linewidth'] == 33
    # 恢复rc参数'lines.linewidth'的值为之前的值
    assert mpl.rcParams['lines.linewidth'] == linewidth

    # 同时使用文件名和字典形式测试rc_context
    with mpl.rc_context(fname=rcpath, rc={'lines.linewidth': 44}):
        # 断言当前rc参数'lines.linewidth'的值与预期相符
        assert mpl.rcParams['lines.linewidth'] == 44
    # 恢复rc参数'lines.linewidth'的值为之前的值
    assert mpl.rcParams['lines.linewidth'] == linewidth

    # 测试作为装饰器的rc_context功能（并测试可重复使用性，调用func两次）
    @mpl.rc_context({'lines.linewidth': 44})
    def func():
        # 断言当前rc参数'lines.linewidth'的值为预期值
        assert mpl.rcParams['lines.linewidth'] == 44

    # 调用func函数两次，验证装饰器的功能
    func()
    func()

    # 测试rc_file功能
    mpl.rc_file(rcpath)
    # 断言当前rc参数'lines.linewidth'的值为预期值
    assert mpl.rcParams['lines.linewidth'] == 33


def test_RcParams_class():
    # 创建一个RcParams对象，包含多个参数设置
    rc = mpl.RcParams({'font.cursive': ['Apple Chancery',
                                        'Textile',
                                        'Zapf Chancery',
                                        'cursive'],
                       'font.family': 'sans-serif',
                       'font.weight': 'normal',
                       'font.size': 12})

    # 预期的rc对象字符串表示形式
    expected_repr = """
RcParams({'font.cursive': ['Apple Chancery',
                           'Textile',
                           'Zapf Chancery',
                           'cursive'],
          'font.family': ['sans-serif'],
          'font.size': 12.0,
          'font.weight': 'normal'})""".lstrip()

    # 断言生成的字符串表示形式与预期相符
    assert expected_repr == repr(rc)

    # 预期的rc对象的str输出
    expected_str = """
font.cursive: ['Apple Chancery', 'Textile', 'Zapf Chancery', 'cursive']
font.family: ['sans-serif']
font.size: 12.0
font.weight: normal""".lstrip()

    # 断言生成的str输出与预期相符
    assert expected_str == str(rc)

    # 测试find_all功能
    # 使用断言验证 rc.find_all('i[vz]') 返回的列表应该是 ['font.cursive', 'font.size'] 并且按字母顺序排序
    assert ['font.cursive', 'font.size'] == sorted(rc.find_all('i[vz]'))
    
    # 使用断言验证 rc.find_all('family') 返回的列表应该是 ['font.family']
    assert ['font.family'] == list(rc.find_all('family'))
# 测试函数，用于验证 mpl.RcParams.update() 方法在输入数据不合法时是否会引发 ValueError 异常
def test_rcparams_update():
    # 创建一个 RcParams 对象 rc，包含一个有效的键值对 'figure.figsize': (3.5, 42)
    rc = mpl.RcParams({'figure.figsize': (3.5, 42)})
    # 创建一个包含无效数据的字典 bad_dict，'figure.figsize' 的值包含多余的元素 1
    bad_dict = {'figure.figsize': (3.5, 42, 1)}
    # 确保 update 方法在输入验证时会引发 ValueError 异常
    with pytest.raises(ValueError):
        rc.update(bad_dict)


# 测试函数，用于验证 mpl.RcParams 初始化时是否会对输入数据进行验证
def test_rcparams_init():
    # 确保 RcParams 的初始化在输入验证时会引发 ValueError 异常
    with pytest.raises(ValueError):
        mpl.RcParams({'figure.figsize': (3.5, 42, 1)})


# 测试函数，用于验证 cycler() 函数在输入参数数量不合法时是否会引发 TypeError 异常
def test_nargs_cycler():
    # 导入 cycler 函数
    from matplotlib.rcsetup import cycler as ccl
    # 确保 cycler() 函数在输入参数数量不合法时会引发 TypeError 异常，并且异常信息匹配 '3 were given'
    with pytest.raises(TypeError, match='3 were given'):
        ccl(ccl(color=list('rgb')), 2, 3)


# 测试函数，用于验证 Bug 2543 是否被修复
def test_Bug_2543():
    # 进行测试时，过滤掉由于已废弃的 rcparams 导致的警告信息
    with _api.suppress_matplotlib_deprecation_warning():
        # 在 mpl.rc_context() 下进行测试
        with mpl.rc_context():
            # 复制 mpl.rcParams 的当前配置
            _copy = mpl.rcParams.copy()
            # 遍历复制的配置，并将其赋值回 mpl.rcParams
            for key in _copy:
                mpl.rcParams[key] = _copy[key]
        # 使用 deepcopy 复制 mpl.rcParams
        with mpl.rc_context():
            copy.deepcopy(mpl.rcParams)
    # 确保在 validate_bool 函数中传入 None 时会引发 ValueError 异常
    with pytest.raises(ValueError):
        validate_bool(None)
    # 确保在 mpl.rc_context() 中设置 'svg.fonttype' 为 True 时会引发 ValueError 异常
    with pytest.raises(ValueError):
        with mpl.rc_context():
            mpl.rcParams['svg.fonttype'] = True


# 测试用例列表，用于验证 legend 的颜色设置是否正确
legend_color_tests = [
    ('face', {'color': 'r'}, mcolors.to_rgba('r')),
    ('face', {'color': 'inherit', 'axes.facecolor': 'r'},
     mcolors.to_rgba('r')),
    ('face', {'color': 'g', 'axes.facecolor': 'r'}, mcolors.to_rgba('g')),
    ('edge', {'color': 'r'}, mcolors.to_rgba('r')),
    ('edge', {'color': 'inherit', 'axes.edgecolor': 'r'},
     mcolors.to_rgba('r')),
    ('edge', {'color': 'g', 'axes.facecolor': 'r'}, mcolors.to_rgba('g'))
]
# 测试用例 ID 列表，对应每个测试用例的描述
legend_color_test_ids = [
    'same facecolor',
    'inherited facecolor',
    'different facecolor',
    'same edgecolor',
    'inherited edgecolor',
    'different facecolor',
]


# 参数化测试函数，用于验证 legend 的颜色设置是否正确
@pytest.mark.parametrize('color_type, param_dict, target', legend_color_tests,
                         ids=legend_color_test_ids)
def test_legend_colors(color_type, param_dict, target):
    # 将 param_dict 中的 'color' 键值对转换为 'legend.{color_type}color'
    param_dict[f'legend.{color_type}color'] = param_dict.pop('color')
    # 获取用于获取颜色的函数名称，例如 'get_facecolor' 或 'get_edgecolor'
    get_func = f'get_{color_type}color'

    # 在 mpl.rc_context(param_dict) 下进行测试
    with mpl.rc_context(param_dict):
        # 创建图表和轴对象
        _, ax = plt.subplots()
        # 绘制一条测试线
        ax.plot(range(3), label='test')
        # 添加图例
        leg = ax.legend()
        # 断言图例的颜色是否与预期目标一致
        assert getattr(leg.legendPatch, get_func)() == target


# 测试函数，用于验证设置 'lines.markerfacecolor' 的全局配置是否生效
def test_mfc_rcparams():
    # 设置 'lines.markerfacecolor' 为 'r'
    mpl.rcParams['lines.markerfacecolor'] = 'r'
    # 创建一条 Line2D 对象 ln
    ln = mpl.lines.Line2D([1, 2], [1, 2])
    # 断言 ln 的 markerfacecolor 是否为 'r'
    assert ln.get_markerfacecolor() == 'r'


# 测试函数，用于验证设置 'lines.markeredgecolor' 的全局配置是否生效
def test_mec_rcparams():
    # 设置 'lines.markeredgecolor' 为 'r'
    mpl.rcParams['lines.markeredgecolor'] = 'r'
    # 创建一条 Line2D 对象 ln
    ln = mpl.lines.Line2D([1, 2], [1, 2])
    # 断言 ln 的 markeredgecolor 是否为 'r'
    assert ln.get_markeredgecolor() == 'r'


# 测试函数，用于验证设置 'axes.titlecolor' 的全局配置是否生效
def test_axes_titlecolor_rcparams():
    # 设置 'axes.titlecolor' 为 'r'
    mpl.rcParams['axes.titlecolor'] = 'r'
    # 创建图表和轴对象
    _, ax = plt.subplots()
    # 设置轴标题为 "Title"
    title = ax.set_title("Title")
    # 断言标题的颜色是否为 'r'
    assert title.get_color() == 'r'
# 定义一个测试函数，用于测试处理 Issue 1713 的情况
def test_Issue_1713(tmp_path):
    # 创建临时文件路径 tmp_path/test_rcparams.rc，并写入内容 'timezone: UTC'
    rcpath = tmp_path / 'test_rcparams.rc'
    rcpath.write_text('timezone: UTC', encoding='utf-8')
    # 使用模拟的 locale.getpreferredencoding 返回值 'UTF-32-BE' 来执行测试
    with mock.patch('locale.getpreferredencoding', return_value='UTF-32-BE'):
        # 调用 mpl.rc_params_from_file 函数，从 rcpath 中加载参数，设置 True 和 False 两个标志
        rc = mpl.rc_params_from_file(rcpath, True, False)
    # 断言 rc 中的 'timezone' 参数是否等于 'UTC'
    assert rc.get('timezone') == 'UTC'


# 定义一个测试函数，用于测试动画帧格式设置的情况
def test_animation_frame_formats():
    # 设置动画帧格式参数 animation.frame_format，允许的格式包括 'png', 'jpeg', 'tiff', 'raw', 'rgba', 'ppm', 'sgi', 'bmp', 'pbm', 'svg'
    for fmt in ['png', 'jpeg', 'tiff', 'raw', 'rgba', 'ppm',
                'sgi', 'bmp', 'pbm', 'svg']:
        mpl.rcParams['animation.frame_format'] = fmt


# 生成验证器测试用例的生成器函数，根据 valid 参数决定返回验证成功或失败的情况
def generate_validator_testcases(valid):
    # 遍历验证器字典列表 validation_tests
    for validator_dict in validation_tests:
        validator = validator_dict['validator']
        if valid:
            # 遍历验证成功的参数及目标值，生成相应的测试用例
            for arg, target in validator_dict['success']:
                yield validator, arg, target
        else:
            # 遍历验证失败的参数及异常类型，生成相应的测试用例
            for arg, error_type in validator_dict['fail']:
                yield validator, arg, error_type


# 使用 pytest.mark.parametrize 注解，参数化测试验证器的有效情况
@pytest.mark.parametrize('validator, arg, target',
                         generate_validator_testcases(True))
def test_validator_valid(validator, arg, target):
    # 调用验证器 validator 处理参数 arg，得到结果 res
    res = validator(arg)
    if isinstance(target, np.ndarray):
        # 如果目标值 target 是 numpy 数组，则使用 np.testing.assert_equal 进行断言
        np.testing.assert_equal(res, target)
    elif not isinstance(target, Cycler):
        # 如果目标值 target 不是 Cycler 对象，则简单断言结果 res 是否等于 target
        assert res == target
    else:
        # 对于 Cycler 对象，需要逐一比较其列表表示是否相同
        assert list(res) == list(target)


# 使用 pytest.mark.parametrize 注解，参数化测试验证器的无效情况
@pytest.mark.parametrize('validator, arg, exception_type',
                         generate_validator_testcases(False))
def test_validator_invalid(validator, arg, exception_type):
    # 使用 pytest.raises 断言异常类型 exception_type 被抛出
    with pytest.raises(exception_type):
        validator(arg)


# 使用 pytest.mark.parametrize 注解，参数化测试验证字体粗细 weight 的有效和无效情况
@pytest.mark.parametrize('weight, parsed_weight', [
    ('bold', 'bold'),
    ('BOLD', ValueError),  # weight 是区分大小写的，这里应该抛出 ValueError
    (100, 100),
    ('100', 100),
    (np.array(100), 100),
    # 分数型的字体粗细不被定义，应该抛出 ValueError，但在历史上未必如此
    (20.6, 20),
    ('20.6', ValueError),
    ([100], ValueError),
])
def test_validate_fontweight(weight, parsed_weight):
    if parsed_weight is ValueError:
        # 断言调用 validate_fontweight(weight) 抛出 ValueError 异常
        with pytest.raises(ValueError):
            validate_fontweight(weight)
    else:
        # 断言调用 validate_fontweight(weight) 返回值等于 parsed_weight
        assert validate_fontweight(weight) == parsed_weight


# 使用 pytest.mark.parametrize 注解，参数化测试验证字体拉伸 stretch 的有效和无效情况
@pytest.mark.parametrize('stretch, parsed_stretch', [
    ('expanded', 'expanded'),
    ('EXPANDED', ValueError),  # stretch 是区分大小写的，这里应该抛出 ValueError
    (100, 100),
    ('100', 100),
    (np.array(100), 100),
    # 分数型的字体拉伸不被定义，应该抛出 ValueError，但在历史上未必如此
    (20.6, 20),
    ('20.6', ValueError),
    ([100], ValueError),
])
def test_validate_fontstretch(stretch, parsed_stretch):
    if parsed_stretch is ValueError:
        # 断言调用 validate_fontstretch(stretch) 抛出 ValueError 异常
        with pytest.raises(ValueError):
            validate_fontstretch(stretch)
    else:
        # 如果不是第一个条件满足，则断言字体拉伸是否有效，并与解析后的拉伸值进行比较
        assert validate_fontstretch(stretch) == parsed_stretch
def test_keymaps():
    # 获取所有包含 'keymap' 的配置键名列表
    key_list = [k for k in mpl.rcParams if 'keymap' in k]
    # 遍历键名列表，断言对应的配置值为列表类型
    for k in key_list:
        assert isinstance(mpl.rcParams[k], list)


def test_no_backend_reset_rccontext():
    # 断言默认后端不是 'module://aardvark'
    assert mpl.rcParams['backend'] != 'module://aardvark'
    # 在 rc_context 中修改后端为 'module://aardvark'，并断言修改成功
    with mpl.rc_context():
        mpl.rcParams['backend'] = 'module://aardvark'
    assert mpl.rcParams['backend'] == 'module://aardvark'


def test_rcparams_reset_after_fail():
    # 测试：修复之前的 bug，确保 rc_context 引发异常时不影响全局 rc 参数
    with mpl.rc_context(rc={'text.usetex': False}):
        assert mpl.rcParams['text.usetex'] is False
        # 断言在异常处理中尝试修改不存在的键将引发 KeyError
        with pytest.raises(KeyError):
            with mpl.rc_context(rc={'text.usetex': True, 'test.blah': True}):
                pass
        # 确保异常处理后 'text.usetex' 保持为 False
        assert mpl.rcParams['text.usetex'] is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
def test_backend_fallback_headless(tmp_path):
    # 准备用于 headless 模式的环境变量
    env = {**os.environ,
           "DISPLAY": "", "WAYLAND_DISPLAY": "",
           "MPLBACKEND": "", "MPLCONFIGDIR": str(tmp_path)}
    # 断言尝试运行使用 tkagg 后端的代码段会引发 subprocess.CalledProcessError
    with pytest.raises(subprocess.CalledProcessError):
        subprocess_run_for_testing(
            [sys.executable, "-c",
             "import matplotlib;"
             "matplotlib.use('tkagg');"
             "import matplotlib.pyplot;"
             "matplotlib.pyplot.plot(42);"
             ],
            env=env, check=True, stderr=subprocess.DEVNULL)


@pytest.mark.skipif(
    sys.platform == "linux" and not _c_internal_utils.display_is_valid(),
    reason="headless")
def test_backend_fallback_headful(tmp_path):
    # 在有显示器的环境中测试后备的 headful 模式
    pytest.importorskip("tkinter")
    env = {**os.environ, "MPLBACKEND": "", "MPLCONFIGDIR": str(tmp_path)}
    # 运行代码段检查 matplotlib 的后端设置
    backend = subprocess_run_for_testing(
        [sys.executable, "-c",
         "import matplotlib as mpl; "
         "sentinel = mpl.rcsetup._auto_backend_sentinel; "
         # 检查另一个实例上的后端设置是否与预期的 sentinel 一致
         "assert mpl.RcParams({'backend': sentinel})['backend'] == sentinel; "
         "assert mpl.rcParams._get('backend') == sentinel; "
         "import matplotlib.pyplot; "
         "print(matplotlib.get_backend())"],
        env=env, text=True, check=True, capture_output=True).stdout
    # 确保返回的后端不是 "agg"
    assert backend.strip().lower() != "agg"


def test_deprecation(monkeypatch):
    # 设置 patch.linewidth 的废弃映射，并进行废弃警告测试
    monkeypatch.setitem(
        mpl._deprecated_map, "patch.linewidth",
        ("0.0", "axes.linewidth", lambda old: 2 * old, lambda new: new / 2))
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        # 断言 patch.linewidth 的值与 axes.linewidth 的一半相等
        assert mpl.rcParams["patch.linewidth"] \
            == mpl.rcParams["axes.linewidth"] / 2
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        # 修改 patch.linewidth 的值为 1，并断言 axes.linewidth 的值为 2
        mpl.rcParams["patch.linewidth"] = 1
    assert mpl.rcParams["axes.linewidth"] == 2
    # 使用 monkeypatch 设置 matplotlib 中 _deprecated_ignore_map 的条目 "patch.edgecolor"，
    # 将其值设为 ("0.0", "axes.edgecolor")
    monkeypatch.setitem(
        mpl._deprecated_ignore_map, "patch.edgecolor",
        ("0.0", "axes.edgecolor"))

    # 断言当前 mpl.rcParams["patch.edgecolor"] 等于 mpl.rcParams["axes.edgecolor"]，
    # 在此期间会产生 MatplotlibDeprecationWarning 警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert mpl.rcParams["patch.edgecolor"] \
            == mpl.rcParams["axes.edgecolor"]

    # 使用 monkeypatch 设置 matplotlib 中 _deprecated_ignore_map 的条目 "patch.edgecolor"，
    # 将其值设为 "#abcd"
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        mpl.rcParams["patch.edgecolor"] = "#abcd"

    # 断言当前 mpl.rcParams["axes.edgecolor"] 不等于 "#abcd"
    assert mpl.rcParams["axes.edgecolor"] != "#abcd"

    # 使用 monkeypatch 设置 matplotlib 中 _deprecated_ignore_map 的条目 "patch.force_edgecolor"，
    # 将其值设为 ("0.0", None)
    monkeypatch.setitem(
        mpl._deprecated_ignore_map, "patch.force_edgecolor",
        ("0.0", None))

    # 断言当前 mpl.rcParams["patch.force_edgecolor"] 为 None，
    # 在此期间会产生 MatplotlibDeprecationWarning 警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert mpl.rcParams["patch.force_edgecolor"] is None

    # 使用 monkeypatch 设置 matplotlib 中 _deprecated_remain_as_none 的条目 "svg.hashsalt"，
    # 将其值设为 ("0.0",)
    monkeypatch.setitem(
        mpl._deprecated_remain_as_none, "svg.hashsalt",
        ("0.0",))

    # 将 mpl.rcParams["svg.hashsalt"] 设置为 "foobar"，
    # 断言其等于 "foobar"，这不会产生警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        mpl.rcParams["svg.hashsalt"] = "foobar"

    # 断言当前 mpl.rcParams["svg.hashsalt"] 确实等于 "foobar"，这不会产生警告
    assert mpl.rcParams["svg.hashsalt"] == "foobar"

    # 将 mpl.rcParams["svg.hashsalt"] 设置为 None，这不会产生警告
    mpl.rcParams["svg.hashsalt"] = None

    # 更新 mpl.rcParams，这不会产生警告
    mpl.rcParams.update(mpl.rcParams.copy())
    # 注意：警告被抑制是因为对 updater rcParams 的迭代受 suppress_matplotlib_deprecation_warning 保护，
    # 而不是因为有任何显式检查。
@pytest.mark.parametrize("value", [
    "best",             # 测试参数：字符串 "best"
    1,                  # 测试参数：整数 1
    "1",                # 测试参数：字符串 "1"
    (0.9, .7),          # 测试参数：元组 (0.9, 0.7)
    (-0.9, .7),         # 测试参数：元组 (-0.9, 0.7)
    "(0.9, .7)"         # 测试参数：字符串 "(0.9, .7)"
])
def test_rcparams_legend_loc(value):
    # 设置 matplotlib 的 rcParams 中的 'legend.loc' 为给定的值 value
    # rcParams['legend.loc'] 应允许任何上述格式的设置
    # 如果有任何不允许的格式，将会引发异常
    # 用于测试 GitHub 问题 #22338
    mpl.rcParams["legend.loc"] = value


@pytest.mark.parametrize("value", [
    "best",             # 测试参数：字符串 "best"
    1,                  # 测试参数：整数 1
    (0.9, .7),          # 测试参数：元组 (0.9, 0.7)
    (-0.9, .7),         # 测试参数：元组 (-0.9, 0.7)
])
def test_rcparams_legend_loc_from_file(tmp_path, value):
    # 应当能够从 matplotlibrc 文件中设置 rcParams['legend.loc']
    # rcParams['legend.loc'] 应当可以设置为 value 所示的任何值
    # 如果有任何不允许的格式，将会引发异常
    # 用于测试 GitHub 问题 #22338
    rc_path = tmp_path / "matplotlibrc"
    rc_path.write_text(f"legend.loc: {value}")

    # 使用 rc_context 从指定的 rc 文件名中设置上下文
    with mpl.rc_context(fname=rc_path):
        assert mpl.rcParams["legend.loc"] == value


@pytest.mark.parametrize("value", [(1, 2, 3), '1, 2, 3', '(1, 2, 3)'])
def test_validate_sketch(value):
    # 设置 matplotlib 的 rcParams 中的 'path.sketch' 为给定的值 value
    mpl.rcParams["path.sketch"] = value
    # 断言设置后的 'path.sketch' 应为 (1, 2, 3)
    assert mpl.rcParams["path.sketch"] == (1, 2, 3)
    # 调用 validate_sketch 函数，断言其返回结果应为 (1, 2, 3)
    assert validate_sketch(value) == (1, 2, 3)


@pytest.mark.parametrize("value", [1, '1', '1 2 3'])
def test_validate_sketch_error(value):
    # 使用 pytest 的 assertRaises 来捕获 ValueError 异常，断言异常消息包含指定内容
    with pytest.raises(ValueError, match="scale, length, randomness"):
        validate_sketch(value)
    # 设置 matplotlib 的 rcParams 中的 'path.sketch' 为给定的值 value
    with pytest.raises(ValueError, match="scale, length, randomness"):
        mpl.rcParams["path.sketch"] = value


@pytest.mark.parametrize("value", ['1, 2, 3', '(1,2,3)'])
def test_rcparams_path_sketch_from_file(tmp_path, value):
    # 创建 matplotlibrc 文件，设置 'path.sketch' 的值为 value
    rc_path = tmp_path / "matplotlibrc"
    rc_path.write_text(f"path.sketch: {value}")
    # 使用 rc_context 从指定的 rc 文件名中设置上下文
    with mpl.rc_context(fname=rc_path):
        assert mpl.rcParams["path.sketch"] == (1, 2, 3)
```