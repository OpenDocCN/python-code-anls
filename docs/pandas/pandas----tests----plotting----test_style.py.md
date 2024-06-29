# `D:\src\scipysrc\pandas\pandas\tests\plotting\test_style.py`

```
import pytest  # 导入 pytest 模块

from pandas import Series  # 从 pandas 库中导入 Series 类

mpl = pytest.importorskip("matplotlib")  # 导入 matplotlib，如果导入失败则跳过测试
plt = pytest.importorskip("matplotlib.pyplot")  # 导入 matplotlib.pyplot，如果导入失败则跳过测试
from pandas.plotting._matplotlib.style import get_standard_colors  # 从 pandas.plotting._matplotlib.style 中导入 get_standard_colors 函数


class TestGetStandardColors:
    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器定义多组测试参数
        "num_colors, expected",
        [
            (3, ["red", "green", "blue"]),  # 参数化测试用例：num_colors=3 时期望结果为 ["red", "green", "blue"]
            (5, ["red", "green", "blue", "red", "green"]),  # 参数化测试用例：num_colors=5 时期望结果为 ["red", "green", "blue", "red", "green"]
            (7, ["red", "green", "blue", "red", "green", "blue", "red"]),  # 参数化测试用例：num_colors=7 时期望结果为 ["red", "green", "blue", "red", "green", "blue", "red"]
            (2, ["red", "green"]),  # 参数化测试用例：num_colors=2 时期望结果为 ["red", "green"]
            (1, ["red"]),  # 参数化测试用例：num_colors=1 时期望结果为 ["red"]
        ],
    )
    def test_default_colors_named_from_prop_cycle(self, num_colors, expected):
        mpl_params = {
            "axes.prop_cycle": plt.cycler(color=["red", "green", "blue"]),  # 设置 mpl 参数，指定颜色循环为 ["red", "green", "blue"]
        }
        with mpl.rc_context(rc=mpl_params):  # 使用 mpl.rc_context 设置 matplotlib 的运行时环境
            result = get_standard_colors(num_colors=num_colors)  # 调用函数 get_standard_colors 获取结果
            assert result == expected  # 断言函数返回结果与期望值相等

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器定义多组测试参数
        "num_colors, expected",
        [
            (1, ["b"]),  # 参数化测试用例：num_colors=1 时期望结果为 ["b"]
            (3, ["b", "g", "r"]),  # 参数化测试用例：num_colors=3 时期望结果为 ["b", "g", "r"]
            (4, ["b", "g", "r", "y"]),  # 参数化测试用例：num_colors=4 时期望结果为 ["b", "g", "r", "y"]
            (5, ["b", "g", "r", "y", "b"]),  # 参数化测试用例：num_colors=5 时期望结果为 ["b", "g", "r", "y", "b"]
            (7, ["b", "g", "r", "y", "b", "g", "r"]),  # 参数化测试用例：num_colors=7 时期望结果为 ["b", "g", "r", "y", "b", "g", "r"]
        ],
    )
    def test_default_colors_named_from_prop_cycle_string(self, num_colors, expected):
        mpl_params = {
            "axes.prop_cycle": plt.cycler(color="bgry"),  # 设置 mpl 参数，指定颜色循环为 "bgry"
        }
        with mpl.rc_context(rc=mpl_params):  # 使用 mpl.rc_context 设置 matplotlib 的运行时环境
            result = get_standard_colors(num_colors=num_colors)  # 调用函数 get_standard_colors 获取结果
            assert result == expected  # 断言函数返回结果与期望值相等

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器定义多组测试参数
        "num_colors, expected_name",
        [
            (1, ["C0"]),  # 参数化测试用例：num_colors=1 时期望结果为 ["C0"]
            (3, ["C0", "C1", "C2"]),  # 参数化测试用例：num_colors=3 时期望结果为 ["C0", "C1", "C2"]
            (
                12,
                [
                    "C0",
                    "C1",
                    "C2",
                    "C3",
                    "C4",
                    "C5",
                    "C6",
                    "C7",
                    "C8",
                    "C9",
                    "C0",
                    "C1",
                ],
            ),  # 参数化测试用例：num_colors=12 时期望结果为 ["C0", "C1", ...]
        ],
    )
    def test_default_colors_named_undefined_prop_cycle(self, num_colors, expected_name):
        with mpl.rc_context(rc={}):  # 使用空字典作为 mpl 参数，重置 matplotlib 的运行时环境
            expected = [mpl.colors.to_hex(x) for x in expected_name]  # 使用 mpl.colors.to_hex 将颜色名转换为十六进制表示
            result = get_standard_colors(num_colors=num_colors)  # 调用函数 get_standard_colors 获取结果
            assert result == expected  # 断言函数返回结果与期望值相等

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器定义多组测试参数
        "num_colors, expected",
        [
            (1, ["red", "green", (0.1, 0.2, 0.3)]),  # 参数化测试用例：num_colors=1 时期望结果为 ["red", "green", (0.1, 0.2, 0.3)]
            (2, ["red", "green", (0.1, 0.2, 0.3)]),  # 参数化测试用例：num_colors=2 时期望结果为 ["red", "green", (0.1, 0.2, 0.3)]
            (3, ["red", "green", (0.1, 0.2, 0.3)]),  # 参数化测试用例：num_colors=3 时期望结果为 ["red", "green", (0.1, 0.2, 0.3)]
            (4, ["red", "green", (0.1, 0.2, 0.3), "red"]),  # 参数化测试用例：num_colors=4 时期望结果为 ["red", "green", (0.1, 0.2, 0.3), "red"]
        ],
    )
    def test_user_input_color_sequence(self, num_colors, expected):
        color = ["red", "green", (0.1, 0.2, 0.3)]  # 定义颜色序列
        result = get_standard_colors(color=color, num_colors=num_colors)  # 调用函数 get_standard_colors 获取结果
        assert result == expected  # 断言函数返回结果与期望值相等
    # 使用 pytest 的 parametrize 装饰器为 test_user_input_color_string 函数参数化多组输入
    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            # 当 num_colors 为 1 到 6 不同值时，期望的输出结果列表
            (1, ["r", "g", "b", "k"]),
            (2, ["r", "g", "b", "k"]),
            (3, ["r", "g", "b", "k"]),
            (4, ["r", "g", "b", "k"]),
            (5, ["r", "g", "b", "k", "r"]),
            (6, ["r", "g", "b", "k", "r", "g"]),
        ],
    )
    # 定义测试函数 test_user_input_color_string，用于测试颜色字符串输入情况下的 get_standard_colors 函数
    def test_user_input_color_string(self, num_colors, expected):
        # 定义颜色字符串
        color = "rgbk"
        # 调用 get_standard_colors 函数，获取结果
        result = get_standard_colors(color=color, num_colors=num_colors)
        # 断言结果是否与期望值一致
        assert result == expected

    # 使用 pytest 的 parametrize 装饰器为 test_user_input_color_floats 函数参数化多组输入
    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            # 当 num_colors 为 1 到 3 不同值时，期望的输出结果列表
            (1, [(0.1, 0.2, 0.3)]),
            (2, [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]),
            (3, [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]),
        ],
    )
    # 定义测试函数 test_user_input_color_floats，用于测试颜色浮点数元组输入情况下的 get_standard_colors 函数
    def test_user_input_color_floats(self, num_colors, expected):
        # 定义颜色浮点数元组
        color = (0.1, 0.2, 0.3)
        # 调用 get_standard_colors 函数，获取结果
        result = get_standard_colors(color=color, num_colors=num_colors)
        # 断言结果是否与期望值一致
        assert result == expected

    # 使用 pytest 的 parametrize 装饰器为 test_user_input_named_color_string 函数参数化多组输入
    @pytest.mark.parametrize(
        "color, num_colors, expected",
        [
            # 不同颜色名称及对应的 num_colors 值，期望的输出结果列表
            ("Crimson", 1, ["Crimson"]),
            ("DodgerBlue", 2, ["DodgerBlue", "DodgerBlue"]),
            ("firebrick", 3, ["firebrick", "firebrick", "firebrick"]),
        ],
    )
    # 定义测试函数 test_user_input_named_color_string，用于测试命名颜色字符串输入情况下的 get_standard_colors 函数
    def test_user_input_named_color_string(self, color, num_colors, expected):
        # 调用 get_standard_colors 函数，获取结果
        result = get_standard_colors(color=color, num_colors=num_colors)
        # 断言结果是否与期望值一致
        assert result == expected

    # 使用 pytest 的 parametrize 装饰器为 test_empty_color_raises 函数参数化多组输入
    @pytest.mark.parametrize("color", ["", [], (), Series([], dtype="object")])
    # 定义测试函数 test_empty_color_raises，测试空颜色参数时是否会引发 ValueError 异常
    def test_empty_color_raises(self, color):
        # 使用 pytest 的断言检查是否引发了预期的 ValueError 异常，并且异常消息匹配特定模式
        with pytest.raises(ValueError, match="Invalid color argument"):
            get_standard_colors(color=color, num_colors=1)

    # 使用 pytest 的 parametrize 装饰器为 test_bad_color_raises 函数参数化多组输入
    @pytest.mark.parametrize(
        "color",
        [
            "bad_color",  # 不合法的颜色字符串
            ("red", "green", "bad_color"),  # 包含不合法颜色的元组
            (0.1,),  # 不合法的浮点数元组
            (0.1, 0.2),  # 不合法的浮点数元组
            (0.1, 0.2, 0.3, 0.4, 0.5),  # 不合法的浮点数元组长度
        ],
    )
    # 定义测试函数 test_bad_color_raises，测试不合法颜色参数时是否会引发 ValueError 异常
    def test_bad_color_raises(self, color):
        # 使用 pytest 的断言检查是否引发了预期的 ValueError 异常，并且异常消息匹配特定模式
        with pytest.raises(ValueError, match="Invalid color"):
            get_standard_colors(color=color, num_colors=5)
```