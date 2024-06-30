# `D:\src\scipysrc\seaborn\tests\test_rcmod.py`

```
import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.testing as npt
from seaborn import rcmod, palettes, utils

# 定义一个辅助函数，用于验证是否存在Verdana字体
def has_verdana():
    """Helper to verify if Verdana font is present"""
    # 导入matplotlib的字体管理器
    import matplotlib.font_manager as mplfm
    try:
        # 尝试查找Verdana字体，不使用默认字体
        verdana_font = mplfm.findfont('Verdana', fallback_to_default=False)
    except:  # noqa
        # 如果出现异常，表示Verdana字体不可用
        return False
    # 否则继续检查一个不存在的字体，用于验证默认字体的情况
    try:
        unlikely_font = mplfm.findfont("very_unlikely_to_exist1234",
                                       fallback_to_default=False)
    except:  # noqa
        # 如果找到了Verdana字体但未找到unlikely字体，则表示Verdana字体存在
        return True
    # 如果两者相同，则说明默认字体和Verdana相同
    return verdana_font != unlikely_font


class RCParamFixtures:
    
    # 重置参数的Fixture，在每个测试结束后将rc参数恢复为原始状态
    @pytest.fixture(autouse=True)
    def reset_params(self):
        yield
        rcmod.reset_orig()

    # 将原始列表展开为一维列表的方法
    def flatten_list(self, orig_list):
        iter_list = map(np.atleast_1d, orig_list)  # 对原始列表中的每个元素应用np.atleast_1d方法
        flat_list = [item for sublist in iter_list for item in sublist]  # 将所有元素展开成一维列表
        return flat_list

    # 断言rc参数与给定参数是否相等的方法
    def assert_rc_params(self, params):
        for k, v in params.items():
            # 由于matplotlib中的一些微妙问题会导致后端rcParam的意外值，这里不做验证
            if k == "backend":
                continue
            if isinstance(v, np.ndarray):
                npt.assert_array_equal(mpl.rcParams[k], v)
            else:
                assert mpl.rcParams[k] == v

    # 断言两组rc参数是否相等的方法
    def assert_rc_params_equal(self, params1, params2):
        for key, v1 in params1.items():
            # 由于matplotlib中的一些微妙问题会导致后端rcParam的意外值，这里不做验证
            if key == "backend":
                continue
            v2 = params2[key]
            if isinstance(v1, np.ndarray):
                npt.assert_array_equal(v1, v2)
            else:
                assert v1 == v2


class TestAxesStyle(RCParamFixtures):

    styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]

    # 测试默认样式返回值的方法
    def test_default_return(self):
        current = rcmod.axes_style()
        self.assert_rc_params(current)

    # 测试各种样式键的使用情况的方法
    def test_key_usage(self):
        _style_keys = set(rcmod._style_keys)
        for style in self.styles:
            assert not set(rcmod.axes_style(style)) ^ _style_keys

    # 测试非法样式的情况下是否会抛出异常的方法
    def test_bad_style(self):
        with pytest.raises(ValueError):
            rcmod.axes_style("i_am_not_a_style")
    def test_rc_override(self):
        # 定义一个字典 rc，用于设定参数覆盖样式
        rc = {"axes.facecolor": "blue", "foo.notaparam": "bar"}
        # 调用 rcmod.axes_style 函数，将参数 rc 应用到样式 "darkgrid" 上
        out = rcmod.axes_style("darkgrid", rc)
        # 断言设置后的输出中 "axes.facecolor" 应该为 "blue"
        assert out["axes.facecolor"] == "blue"
        # 断言设置后的输出中 "foo.notaparam" 不应存在
        assert "foo.notaparam" not in out

    def test_set_style(self):
        # 遍历 self.styles 中的样式列表
        for style in self.styles:
            # 调用 rcmod.axes_style 函数，应用样式并获取样式字典
            style_dict = rcmod.axes_style(style)
            # 设置当前样式为遍历得到的样式
            rcmod.set_style(style)
            # 断言当前样式参数是否与获取的样式字典一致
            self.assert_rc_params(style_dict)

    def test_style_context_manager(self):
        # 设置样式为 "darkgrid"
        rcmod.set_style("darkgrid")
        # 获取当前轴样式的原始参数
        orig_params = rcmod.axes_style()
        # 获取样式 "whitegrid" 的轴样式参数
        context_params = rcmod.axes_style("whitegrid")

        # 使用上下文管理器设置样式为 "whitegrid"
        with rcmod.axes_style("whitegrid"):
            # 断言当前轴样式参数与 "whitegrid" 样式参数一致
            self.assert_rc_params(context_params)
        # 断言恢复到上一次设置的样式后，参数与原始参数一致
        self.assert_rc_params(orig_params)

        @rcmod.axes_style("whitegrid")
        def func():
            # 在函数内部断言当前轴样式参数与 "whitegrid" 样式参数一致
            self.assert_rc_params(context_params)
        # 调用函数 func
        func()
        # 断言恢复到上一次设置的样式后，参数与原始参数一致
        self.assert_rc_params(orig_params)

    def test_style_context_independence(self):
        # 断言 rcmod._style_keys 和 rcmod._context_keys 的对称差为空集，即无交集
        assert set(rcmod._style_keys) ^ set(rcmod._context_keys)

    def test_set_rc(self):
        # 设置主题参数中的 "lines.linewidth" 为 4
        rcmod.set_theme(rc={"lines.linewidth": 4})
        # 断言当前主题参数中的 "lines.linewidth" 为 4
        assert mpl.rcParams["lines.linewidth"] == 4
        # 恢复默认主题参数
        rcmod.set_theme()

    def test_set_with_palette(self):
        # 重置为原始默认参数
        rcmod.reset_orig()

        # 设置主题为 "deep" 调色板
        rcmod.set_theme(palette="deep")
        # 断言获取的颜色循环与 "deep" 调色板的颜色循环一致
        assert utils.get_color_cycle() == palettes.color_palette("deep", 10)
        # 重置为原始默认参数
        rcmod.reset_orig()

        # 设置主题为 "deep" 调色板，并关闭颜色代码
        rcmod.set_theme(palette="deep", color_codes=False)
        # 断言获取的颜色循环与 "deep" 调色板的颜色循环一致
        assert utils.get_color_cycle() == palettes.color_palette("deep", 10)
        # 重置为原始默认参数
        rcmod.reset_orig()

        # 获取 "deep" 调色板的颜色列表
        pal = palettes.color_palette("deep")
        # 设置主题为指定的调色板 pal
        rcmod.set_theme(palette=pal)
        # 断言获取的颜色循环与 "deep" 调色板的颜色循环一致
        assert utils.get_color_cycle() == palettes.color_palette("deep", 10)
        # 重置为原始默认参数
        rcmod.reset_orig()

        # 设置主题为指定的调色板 pal，并关闭颜色代码
        rcmod.set_theme(palette=pal, color_codes=False)
        # 断言获取的颜色循环与 "deep" 调色板的颜色循环一致
        assert utils.get_color_cycle() == palettes.color_palette("deep", 10)
        # 重置为原始默认参数
        rcmod.reset_orig()

        # 恢复为原始默认参数
        rcmod.set_theme()

    def test_reset_defaults(self):
        # 重置所有默认参数
        rcmod.reset_defaults()
        # 断言当前参数与 mpl.rcParamsDefault 相同
        self.assert_rc_params(mpl.rcParamsDefault)
        # 恢复为原始默认参数
        rcmod.set_theme()

    def test_reset_orig(self):
        # 重置所有原始参数
        rcmod.reset_orig()
        # 断言当前参数与 mpl.rcParamsOrig 相同
        self.assert_rc_params(mpl.rcParamsOrig)
        # 恢复为原始默认参数
        rcmod.set_theme()

    def test_set_is_alias(self):
        # 设置主题为上下文 "paper" 和样式 "white"
        rcmod.set_theme(context="paper", style="white")
        # 复制当前 mpl.rcParams 的参数为 params1
        params1 = mpl.rcParams.copy()
        # 重置为原始默认参数
        rcmod.reset_orig()

        # 再次设置主题为上下文 "paper" 和样式 "white"
        rcmod.set_theme(context="paper", style="white")
        # 复制当前 mpl.rcParams 的参数为 params2
        params2 = mpl.rcParams.copy()

        # 断言两次设置后的参数 params1 和 params2 相等
        self.assert_rc_params_equal(params1, params2)

        # 恢复为原始默认参数
        rcmod.set_theme()
class TestPlottingContext(RCParamFixtures):
    # 定义测试类 TestPlottingContext，继承自 RCParamFixtures

    contexts = ["paper", "notebook", "talk", "poster"]
    # 设定上下文列表，包含了几种预定义的绘图上下文

    def test_default_return(self):
        # 测试默认返回情况的方法

        current = rcmod.plotting_context()
        # 调用 rcmod 模块的 plotting_context 方法，并获取当前的绘图上下文设置
        self.assert_rc_params(current)
        # 使用断言验证当前的绘图参数是否符合预期

    def test_key_usage(self):
        # 测试关键字使用情况的方法

        _context_keys = set(rcmod._context_keys)
        # 从 rcmod 模块中获取 _context_keys 的集合
        for context in self.contexts:
            # 遍历预定义的上下文列表
            missing = set(rcmod.plotting_context(context)) ^ _context_keys
            # 检查当前上下文的绘图参数与预定义的 _context_keys 是否一致
            assert not missing
            # 使用断言验证不存在参数缺失的情况

    def test_bad_context(self):
        # 测试错误上下文的方法

        with pytest.raises(ValueError):
            rcmod.plotting_context("i_am_not_a_context")
            # 使用断言确保当传入错误的上下文时会引发 ValueError 异常

    def test_font_scale(self):
        # 测试字体缩放功能的方法

        notebook_ref = rcmod.plotting_context("notebook")
        # 获取 notebook 上下文的绘图参数作为参考
        notebook_big = rcmod.plotting_context("notebook", 2)
        # 获取缩放倍数为 2 的 notebook 上下文的绘图参数

        font_keys = [
            "font.size",
            "axes.labelsize", "axes.titlesize",
            "xtick.labelsize", "ytick.labelsize",
            "legend.fontsize", "legend.title_fontsize",
        ]
        # 定义需要验证的字体参数列表

        for k in font_keys:
            assert notebook_ref[k] * 2 == notebook_big[k]
            # 使用断言验证缩放后的字体参数是否符合预期

    def test_rc_override(self):
        # 测试参数覆盖功能的方法

        key, val = "grid.linewidth", 5
        # 设定要修改的参数键和值
        rc = {key: val, "foo": "bar"}
        # 创建要应用的新参数字典
        out = rcmod.plotting_context("talk", rc=rc)
        # 调用 plotting_context 方法并传入新的参数字典
        assert out[key] == val
        # 使用断言验证修改后的参数值是否与预期相符
        assert "foo" not in out
        # 使用断言验证不应存在未定义的参数

    def test_set_context(self):
        # 测试设置上下文功能的方法

        for context in self.contexts:
            # 遍历预定义的上下文列表

            context_dict = rcmod.plotting_context(context)
            # 获取指定上下文的绘图参数字典
            rcmod.set_context(context)
            # 设置当前上下文为指定上下文
            self.assert_rc_params(context_dict)
            # 使用断言验证当前绘图参数是否符合预期

    def test_context_context_manager(self):
        # 测试上下文管理器功能的方法

        rcmod.set_context("notebook")
        # 设置当前上下文为 notebook
        orig_params = rcmod.plotting_context()
        # 获取当前绘图参数作为原始参数
        context_params = rcmod.plotting_context("paper")
        # 获取 paper 上下文的绘图参数

        with rcmod.plotting_context("paper"):
            # 使用上下文管理器设置上下文为 paper
            self.assert_rc_params(context_params)
            # 使用断言验证当前绘图参数是否符合 paper 上下文的预期
        self.assert_rc_params(orig_params)
        # 使用断言验证退出上下文管理器后恢复到原始参数状态

        @rcmod.plotting_context("paper")
        def func():
            # 使用装饰器设置上下文为 paper 的方法

            self.assert_rc_params(context_params)
            # 使用断言验证当前绘图参数是否符合 paper 上下文的预期
        func()
        # 调用方法
        self.assert_rc_params(orig_params)
        # 使用断言验证退出方法后恢复到原始参数状态


class TestPalette(RCParamFixtures):

    def test_set_palette(self):
        # 测试设置调色板功能的方法

        rcmod.set_palette("deep")
        # 设置调色板为 deep
        assert utils.get_color_cycle() == palettes.color_palette("deep", 10)
        # 使用断言验证获取的颜色循环是否与预期的 deep 调色板一致

        rcmod.set_palette("pastel6")
        # 设置调色板为 pastel6
        assert utils.get_color_cycle() == palettes.color_palette("pastel6", 6)
        # 使用断言验证获取的颜色循环是否与预期的 pastel6 调色板一致

        rcmod.set_palette("dark", 4)
        # 设置调色板为 dark，同时指定颜色数为 4
        assert utils.get_color_cycle() == palettes.color_palette("dark", 4)
        # 使用断言验证获取的颜色循环是否与预期的 dark 调色板一致

        rcmod.set_palette("Set2", color_codes=True)
        # 设置调色板为 Set2，并启用颜色代码
        assert utils.get_color_cycle() == palettes.color_palette("Set2", 8)
        # 使用断言验证获取的颜色循环是否与预期的 Set2 调色板一致

        assert mpl.colors.same_color(
            mpl.rcParams["patch.facecolor"], palettes.color_palette()[0]
        )
        # 使用断言验证 matplotlib 参数中的面颜色是否与默认调色板的第一个颜色一致


class TestFonts(RCParamFixtures):

    _no_verdana = not has_verdana()

    @pytest.mark.skipif(_no_verdana, reason="Verdana font is not present")
    # 根据条件跳过测试如果Verdana字体不可用
    # 定义测试函数，测试设置字体功能

    # 设置全局主题字体为Verdana
    rcmod.set_theme(font="Verdana")

    # 创建一个图表，获取图表对象ax，并设置x轴标签为"foo"
    _, ax = plt.subplots()
    ax.set_xlabel("foo")

    # 断言x轴标签的字体名称是否为"Verdana"
    assert ax.xaxis.label.get_fontname() == "Verdana"

    # 恢复默认主题设置
    rcmod.set_theme()




    # 定义测试函数，测试设置衬线字体功能

    # 设置全局主题字体为serif
    rcmod.set_theme(font="serif")

    # 创建一个图表，获取图表对象ax，并设置x轴标签为"foo"
    _, ax = plt.subplots()
    ax.set_xlabel("foo")

    # 断言x轴标签的字体名称是否在matplotlib默认字体设置的serif字体中
    assert ax.xaxis.label.get_fontname() in mpl.rcParams["font.serif"]

    # 恢复默认主题设置
    rcmod.set_theme()




    # 标记此测试，在Verdana字体不可用时跳过执行

    # 恢复默认主题设置
    rcmod.set_theme()

    # 设置字体无衬线字体为Verdana
    rcmod.set_style(rc={"font.sans-serif": ["Verdana"]})

    # 创建一个图表，获取图表对象ax，并设置x轴标签为"foo"
    _, ax = plt.subplots()
    ax.set_xlabel("foo")

    # 断言x轴标签的字体名称是否为"Verdana"
    assert ax.xaxis.label.get_fontname() == "Verdana"

    # 恢复默认主题设置
    rcmod.set_theme()
```