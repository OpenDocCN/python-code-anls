# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_eng_formatting.py`

```
import numpy as np  # 导入 numpy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下模块：
    DataFrame,  # DataFrame 类，用于创建和操作数据帧
    reset_option,  # reset_option 函数，用于重置指定选项的默认值
    set_eng_float_format,  # set_eng_float_format 函数，设置工程浮点数格式
)

from pandas.io.formats.format import EngFormatter  # 从 pandas 库的格式化模块中导入 EngFormatter 类

@pytest.fixture(autouse=True)  # 定义一个 pytest fixture，自动应用于所有测试用例
def reset_float_format():
    yield  # fixture 的生成部分，无特定操作
    reset_option("display.float_format")  # 重置显示浮点数格式的选项为默认值

class TestEngFormatter:
    def test_eng_float_formatter2(self, float_frame):
        df = float_frame  # 使用传入的 float_frame 参数创建 DataFrame df
        df.loc[5] = 0  # 在 DataFrame 中的索引位置 5 处插入值 0

        set_eng_float_format()  # 设置工程浮点数格式，不带工程数前缀
        repr(df)  # 返回 DataFrame 的字符串表示形式

        set_eng_float_format(use_eng_prefix=True)  # 设置工程浮点数格式，带工程数前缀
        repr(df)  # 返回 DataFrame 的字符串表示形式

        set_eng_float_format(accuracy=0)  # 设置工程浮点数格式，精度为 0
        repr(df)  # 返回 DataFrame 的字符串表示形式

    def test_eng_float_formatter(self):
        df = DataFrame({"A": [1.41, 141.0, 14100, 1410000.0]})  # 创建包含浮点数列 A 的 DataFrame

        set_eng_float_format()  # 设置工程浮点数格式，不带工程数前缀
        result = df.to_string()  # 将 DataFrame 转换为字符串表示形式
        expected = (
            "             A\n"
            "0    1.410E+00\n"
            "1  141.000E+00\n"
            "2   14.100E+03\n"
            "3    1.410E+06"
        )
        assert result == expected  # 断言转换后的结果与预期的字符串表示相符

        set_eng_float_format(use_eng_prefix=True)  # 设置工程浮点数格式，带工程数前缀
        result = df.to_string()  # 将 DataFrame 转换为字符串表示形式
        expected = "         A\n0    1.410\n1  141.000\n2  14.100k\n3   1.410M"
        assert result == expected  # 断言转换后的结果与预期的字符串表示相符

        set_eng_float_format(accuracy=0)  # 设置工程浮点数格式，精度为 0
        result = df.to_string()  # 将 DataFrame 转换为字符串表示形式
        expected = "         A\n0    1E+00\n1  141E+00\n2   14E+03\n3    1E+06"
        assert result == expected  # 断言转换后的结果与预期的字符串表示相符

    def compare(self, formatter, input, output):
        formatted_input = formatter(input)  # 使用给定的 formatter 对输入进行格式化
        assert formatted_input == output  # 断言格式化后的结果与预期输出相等

    def compare_all(self, formatter, in_out):
        """
        Parameters:
        -----------
        formatter: EngFormatter under test
        in_out: list of tuples. Each tuple = (number, expected_formatting)

        It is tested if 'formatter(number) == expected_formatting'.
        *number* should be >= 0 because formatter(-number) == fmt is also
        tested. *fmt* is derived from *expected_formatting*
        """
        for input, output in in_out:
            self.compare(formatter, input, output)  # 对每个输入输出对调用 compare 方法进行断言
            self.compare(formatter, -input, "-" + output[1:])  # 对负数输入进行相应的断言
    # 定义一个测试方法，用于测试具有工程计数法前缀的指数格式化
    def test_exponents_with_eng_prefix(self):
        # 创建一个使用工程计数法前缀的工程格式化器，设置精度为3
        formatter = EngFormatter(accuracy=3, use_eng_prefix=True)
        # 计算根号2的值
        f = np.sqrt(2)
        # 定义输入输出对列表，每个元素包含一个值和其预期的格式化字符串
        in_out = [
            (f * 10**-24, " 1.414y"),
            (f * 10**-23, " 14.142y"),
            (f * 10**-22, " 141.421y"),
            (f * 10**-21, " 1.414z"),
            (f * 10**-20, " 14.142z"),
            (f * 10**-19, " 141.421z"),
            (f * 10**-18, " 1.414a"),
            (f * 10**-17, " 14.142a"),
            (f * 10**-16, " 141.421a"),
            (f * 10**-15, " 1.414f"),
            (f * 10**-14, " 14.142f"),
            (f * 10**-13, " 141.421f"),
            (f * 10**-12, " 1.414p"),
            (f * 10**-11, " 14.142p"),
            (f * 10**-10, " 141.421p"),
            (f * 10**-9, " 1.414n"),
            (f * 10**-8, " 14.142n"),
            (f * 10**-7, " 141.421n"),
            (f * 10**-6, " 1.414u"),
            (f * 10**-5, " 14.142u"),
            (f * 10**-4, " 141.421u"),
            (f * 10**-3, " 1.414m"),
            (f * 10**-2, " 14.142m"),
            (f * 10**-1, " 141.421m"),
            (f * 10**0, " 1.414"),
            (f * 10**1, " 14.142"),
            (f * 10**2, " 141.421"),
            (f * 10**3, " 1.414k"),
            (f * 10**4, " 14.142k"),
            (f * 10**5, " 141.421k"),
            (f * 10**6, " 1.414M"),
            (f * 10**7, " 14.142M"),
            (f * 10**8, " 141.421M"),
            (f * 10**9, " 1.414G"),
            (f * 10**10, " 14.142G"),
            (f * 10**11, " 141.421G"),
            (f * 10**12, " 1.414T"),
            (f * 10**13, " 14.142T"),
            (f * 10**14, " 141.421T"),
            (f * 10**15, " 1.414P"),
            (f * 10**16, " 14.142P"),
            (f * 10**17, " 141.421P"),
            (f * 10**18, " 1.414E"),
            (f * 10**19, " 14.142E"),
            (f * 10**20, " 141.421E"),
            (f * 10**21, " 1.414Z"),
            (f * 10**22, " 14.142Z"),
            (f * 10**23, " 141.421Z"),
            (f * 10**24, " 1.414Y"),
            (f * 10**25, " 14.142Y"),
            (f * 10**26, " 141.421Y"),
        ]
        # 调用自定义的比较方法，比较格式化后的结果是否符合预期
        self.compare_all(formatter, in_out)
    # 定义一个测试方法，用于测试不使用工程记数法前缀的 EngFormatter 类
    def test_exponents_without_eng_prefix(self):
        # 创建 EngFormatter 对象，设置精度为4，不使用工程记数法前缀
        formatter = EngFormatter(accuracy=4, use_eng_prefix=False)
        # 设置一个浮点数 f 为 π
        f = np.pi
        # 定义输入输出对，包含浮点数乘以不同数量级的字符串表示
        in_out = [
            (f * 10**-24, " 3.1416E-24"),
            (f * 10**-23, " 31.4159E-24"),
            (f * 10**-22, " 314.1593E-24"),
            (f * 10**-21, " 3.1416E-21"),
            (f * 10**-20, " 31.4159E-21"),
            (f * 10**-19, " 314.1593E-21"),
            (f * 10**-18, " 3.1416E-18"),
            (f * 10**-17, " 31.4159E-18"),
            (f * 10**-16, " 314.1593E-18"),
            (f * 10**-15, " 3.1416E-15"),
            (f * 10**-14, " 31.4159E-15"),
            (f * 10**-13, " 314.1593E-15"),
            (f * 10**-12, " 3.1416E-12"),
            (f * 10**-11, " 31.4159E-12"),
            (f * 10**-10, " 314.1593E-12"),
            (f * 10**-9, " 3.1416E-09"),
            (f * 10**-8, " 31.4159E-09"),
            (f * 10**-7, " 314.1593E-09"),
            (f * 10**-6, " 3.1416E-06"),
            (f * 10**-5, " 31.4159E-06"),
            (f * 10**-4, " 314.1593E-06"),
            (f * 10**-3, " 3.1416E-03"),
            (f * 10**-2, " 31.4159E-03"),
            (f * 10**-1, " 314.1593E-03"),
            (f * 10**0, " 3.1416E+00"),
            (f * 10**1, " 31.4159E+00"),
            (f * 10**2, " 314.1593E+00"),
            (f * 10**3, " 3.1416E+03"),
            (f * 10**4, " 31.4159E+03"),
            (f * 10**5, " 314.1593E+03"),
            (f * 10**6, " 3.1416E+06"),
            (f * 10**7, " 31.4159E+06"),
            (f * 10**8, " 314.1593E+06"),
            (f * 10**9, " 3.1416E+09"),
            (f * 10**10, " 31.4159E+09"),
            (f * 10**11, " 314.1593E+09"),
            (f * 10**12, " 3.1416E+12"),
            (f * 10**13, " 31.4159E+12"),
            (f * 10**14, " 314.1593E+12"),
            (f * 10**15, " 3.1416E+15"),
            (f * 10**16, " 31.4159E+15"),
            (f * 10**17, " 314.1593E+15"),
            (f * 10**18, " 3.1416E+18"),
            (f * 10**19, " 31.4159E+18"),
            (f * 10**20, " 314.1593E+18"),
            (f * 10**21, " 3.1416E+21"),
            (f * 10**22, " 31.4159E+21"),
            (f * 10**23, " 314.1593E+21"),
            (f * 10**24, " 3.1416E+24"),
            (f * 10**25, " 31.4159E+24"),
            (f * 10**26, " 314.1593E+24"),
        ]
        # 调用 self.compare_all 方法，比较 formatter 处理后的输出与预期输出是否一致
        self.compare_all(formatter, in_out)
    # 定义测试函数 test_rounding，用于测试 EngFormatter 类的舍入功能
    def test_rounding(self):
        # 创建 EngFormatter 对象，设置精度为 3，使用工程记数法前缀
        formatter = EngFormatter(accuracy=3, use_eng_prefix=True)
        # 定义输入输出对列表，每个元素为一个元组，包含输入浮点数和期望输出字符串
        in_out = [
            (5.55555, " 5.556"),
            (55.5555, " 55.556"),
            (555.555, " 555.555"),
            (5555.55, " 5.556k"),
            (55555.5, " 55.556k"),
            (555555, " 555.555k"),
        ]
        # 调用自定义方法 compare_all，比较 formatter 处理输入列表的输出与期望输出列表 in_out
        self.compare_all(formatter, in_out)

        # 更改 formatter 的精度为 1
        formatter = EngFormatter(accuracy=1, use_eng_prefix=True)
        # 更新输入输出对列表为对应精度下的期望结果
        in_out = [
            (5.55555, " 5.6"),
            (55.5555, " 55.6"),
            (555.555, " 555.6"),
            (5555.55, " 5.6k"),
            (55555.5, " 55.6k"),
            (555555, " 555.6k"),
        ]
        # 再次调用自定义方法 compare_all，比较 formatter 处理输入列表的输出与期望输出列表 in_out
        self.compare_all(formatter, in_out)

        # 更改 formatter 的精度为 0
        formatter = EngFormatter(accuracy=0, use_eng_prefix=True)
        # 更新输入输出对列表为对应精度下的期望结果
        in_out = [
            (5.55555, " 6"),
            (55.5555, " 56"),
            (555.555, " 556"),
            (5555.55, " 6k"),
            (55555.5, " 56k"),
            (555555, " 556k"),
        ]
        # 再次调用自定义方法 compare_all，比较 formatter 处理输入列表的输出与期望输出列表 in_out
        self.compare_all(formatter, in_out)

        # 再次将 formatter 的精度改回 3
        formatter = EngFormatter(accuracy=3, use_eng_prefix=True)
        # 调用 formatter 处理输入值 0，期望返回 " 0.000"
        result = formatter(0)
        # 使用断言确认结果与期望输出相符
        assert result == " 0.000"

    # 定义测试函数 test_nan，测试处理 NaN 的功能
    def test_nan(self):
        # 提出的问题 #11981

        # 创建 EngFormatter 对象，设置精度为 1，使用工程记数法前缀
        formatter = EngFormatter(accuracy=1, use_eng_prefix=True)
        # 使用 formatter 处理 NaN，期望返回 "NaN"
        result = formatter(np.nan)
        # 使用断言确认结果与期望输出相符
        assert result == "NaN"

        # 创建 DataFrame 对象 df，包含三列 "a", "b", "c"，每列对应列表中的值
        df = DataFrame(
            {
                "a": [1.5, 10.3, 20.5],
                "b": [50.3, 60.67, 70.12],
                "c": [100.2, 101.33, 120.33],
            }
        )
        # 使用 DataFrame 的 pivot_table 方法，基于列 "b" 和 "c"，聚合值为列 "a"，生成新的 DataFrame pt
        pt = df.pivot_table(values="a", index="b", columns="c")
        # 调用自定义方法 set_eng_float_format，设置全局浮点数显示格式精度为 1
        set_eng_float_format(accuracy=1)
        # 将 DataFrame pt 转换为字符串表示
        result = pt.to_string()
        # 使用断言确认结果中包含 "NaN"
        assert "NaN" in result

    # 定义测试函数 test_inf，测试处理正无穷大的功能
    def test_inf(self):
        # 提出的问题 #11981

        # 创建 EngFormatter 对象，设置精度为 1，使用工程记数法前缀
        formatter = EngFormatter(accuracy=1, use_eng_prefix=True)
        # 使用 formatter 处理正无穷大，期望返回 "inf"
        result = formatter(np.inf)
        # 使用断言确认结果与期望输出相符
        assert result == "inf"
```