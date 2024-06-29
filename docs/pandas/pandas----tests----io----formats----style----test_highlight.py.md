# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_highlight.py`

```
    {"axis": None, "color": "red"},  # 测试轴
    {"axis": 0, "subset": ["A"], "color": "red"},  # 测试子集并忽略 NaN 值
    {"axis": None, "props": "background-color: red"},  # 测试属性
    [
        {"left": 0, "right": 1},  # 测试基本范围，左边0，右边1
        {"left": 0, "right": 1, "props": "background-color: yellow"},  # 测试属性，左边0，右边1，设置属性为黄色背景
        {"left": -100, "right": 100, "subset": IndexSlice[[0, 1], :]},  # 测试子集，左边-100到100，子集为行0和1的所有列
        {"left": 0, "subset": IndexSlice[[0, 1], :]},  # 测试无右边界，左边0，子集为行0和1的所有列
        {"right": 1},  # 测试无左边界，右边1
        {"left": [0, 0, 11], "axis": 0},  # 测试左边作为序列，左边包含0、0、11，操作轴为0
        {"left": DataFrame({"A": [0, 0, 11], "B": [1, 1, 11]}), "axis": None},  # 测试DataFrame作为左边，列'A'包含0、0、11，列'B'包含1、1、11，操作轴为空
        {"left": 0, "right": [0, 1], "axis": 1},  # 测试右边作为序列，左边0，右边包含0、1，操作轴为1
    ],
def test_highlight_between(styler, kwargs):
    # 定义预期的高亮样式字典
    expected = {
        (0, 0): [("background-color", "yellow")],
        (0, 1): [("background-color", "yellow")],
    }
    # 执行高亮处理并获取计算上下文
    result = styler.highlight_between(**kwargs)._compute().ctx
    # 断言结果与预期相符
    assert result == expected


@pytest.mark.parametrize(
    "arg, map, axis",
    [
        ("left", [1, 2], 0),  # 0轴有3个元素而不是2个
        ("left", [1, 2, 3], 1),  # 1轴有2个元素而不是3个
        ("left", np.array([[1, 2], [1, 2]]), None),  # DataFrame是(2,3)而不是(2,2)
        ("right", [1, 2], 0),  # 'right'与'left'相同的测试
        ("right", [1, 2, 3], 1),  # 同上
        ("right", np.array([[1, 2], [1, 2]]), None),  # 同上
    ],
)
def test_highlight_between_raises(arg, styler, map, axis):
    # 设置错误消息
    msg = f"supplied '{arg}' is not correct shape"
    # 验证引发ValueError异常并匹配消息
    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(**{arg: map, "axis": axis})._compute()


def test_highlight_between_raises2(styler):
    # 设置错误消息
    msg = "values can be 'both', 'left', 'right', or 'neither'"
    # 验证引发ValueError异常并匹配消息，测试不正确的'inclusive'值
    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(inclusive="badstring")._compute()

    # 同上，测试不正确的'inclusive'值
    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(inclusive=1)._compute()


@pytest.mark.parametrize(
    "inclusive, expected",
    [
        (
            "both",
            {
                (0, 0): [("background-color", "yellow")],
                (0, 1): [("background-color", "yellow")],
            },
        ),
        ("neither", {}),
        ("left", {(0, 0): [("background-color", "yellow")]}),
        ("right", {(0, 1): [("background-color", "yellow")]}),
    ],
)
def test_highlight_between_inclusive(styler, inclusive, expected):
    # 设置测试参数
    kwargs = {"left": 0, "right": 1, "subset": IndexSlice[[0, 1], :]}
    # 执行高亮处理并获取计算上下文
    result = styler.highlight_between(**kwargs, inclusive=inclusive)._compute()
    # 断言结果与预期相符
    assert result.ctx == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q_left": 0.5, "q_right": 1, "axis": 0},  # 基本情况
        {"q_left": 0.5, "q_right": 1, "axis": None},  # 测试轴
        {"q_left": 0, "q_right": 1, "subset": IndexSlice[2, :]},  # 测试子集
        {"q_left": 0.5, "axis": 0},  # 测试无高位
        {"q_right": 1, "subset": IndexSlice[2, :], "axis": 1},  # 测试无低位
        {"q_left": 0.5, "axis": 0, "props": "background-color: yellow"},  # 测试属性
    ],
)
def test_highlight_quantile(styler, kwargs):
    # 定义预期的高亮样式字典
    expected = {
        (2, 0): [("background-color", "yellow")],
        (2, 1): [("background-color", "yellow")],
    }
    # 执行高亮处理并获取计算上下文
    result = styler.highlight_quantile(**kwargs)._compute().ctx
    # 断言结果与预期相符
    assert result == expected


@pytest.mark.parametrize(
    "f,kwargs",
    [
        ("highlight_min", {"axis": 1, "subset": IndexSlice[1, :]}),
        ("highlight_max", {"axis": 0, "subset": [0]}),
        ("highlight_quantile", {"axis": None, "q_left": 0.6, "q_right": 0.8}),
        ("highlight_between", {"subset": [0]}),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [  # 创建一个列表，包含不同的数据类型信息
        int,  # 整数类型
        float,  # 浮点数类型
        "datetime64[ns]",  # 表示日期时间类型，精确到纳秒
        str,  # 字符串类型
        "timedelta64[ns]",  # 表示时间间隔类型，精确到纳秒
    ],  # 结束数据类型列表
)
def test_all_highlight_dtypes(f, kwargs, dtype):
    # 创建一个包含两行两列数据的DataFrame，指定数据类型为dtype
    df = DataFrame([[0, 10], [20, 30]], dtype=dtype)
    # 如果函数f是"highlight_quantile"并且DataFrame的第一行第一列元素是字符串，返回None，因为quantile与字符串不兼容
    if f == "highlight_quantile" and isinstance(df.iloc[0, 0], (str)):
        return None  # quantile incompatible with str
    # 如果函数f是"highlight_between"，将kwargs中的"left"参数设为DataFrame第二行第一列元素，用于测试
    if f == "highlight_between":
        kwargs["left"] = df.iloc[1, 0]  # set the range low for testing

    # 预期的样式设定，以元组形式表示：位置(1, 0)的单元格背景色应为黄色
    expected = {(1, 0): [("background-color", "yellow")]}
    # 调用DataFrame样式对象的f方法，传入kwargs参数，计算样式并获取上下文
    result = getattr(df.style, f)(**kwargs)._compute().ctx
    # 断言计算得到的样式上下文与预期结果相等
    assert result == expected
```