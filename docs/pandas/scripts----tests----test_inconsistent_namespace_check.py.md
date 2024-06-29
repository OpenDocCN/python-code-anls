# `D:\src\scipysrc\pandas\scripts\tests\test_inconsistent_namespace_check.py`

```
import pytest  # 导入 pytest 库

from scripts.check_for_inconsistent_pandas_namespace import (  # 从指定脚本导入函数
    check_for_inconsistent_pandas_namespace,
)

BAD_FILE_0 = (  # 定义字符串常量 BAD_FILE_0，包含导入 pandas 模块后的代码
    "from pandas import Categorical\n"
    "cat_0 = Categorical()\n"
    "cat_1 = pd.Categorical()"
)
BAD_FILE_1 = (  # 定义字符串常量 BAD_FILE_1，包含导入 pandas 模块后的代码
    "from pandas import Categorical\n"
    "cat_0 = pd.Categorical()\n"
    "cat_1 = Categorical()"
)
BAD_FILE_2 = (  # 定义字符串常量 BAD_FILE_2，包含导入 pandas 模块后的代码
    "from pandas import Categorical\n"
    "cat_0 = pandas.Categorical()\n"
    "cat_1 = Categorical()"
)
GOOD_FILE_0 = (  # 定义字符串常量 GOOD_FILE_0，包含不涉及混用不同 pandas 引用的代码
    "from pandas import Categorical\ncat_0 = Categorical()\ncat_1 = Categorical()"
)
GOOD_FILE_1 = "cat_0 = pd.Categorical()\ncat_1 = pd.Categorical()"  # 定义字符串常量 GOOD_FILE_1，包含不涉及混用不同 pandas 引用的代码
GOOD_FILE_2 = "from array import array\nimport pandas as pd\narr = pd.array([])"  # 定义字符串常量 GOOD_FILE_2，包含导入 pandas 模块后的代码和其他操作

PATH = "t.py"  # 定义字符串常量 PATH，代表文件路径

@pytest.mark.parametrize(  # 使用 pytest 的参数化标记定义测试用例参数
    "content, expected",
    [
        (BAD_FILE_0, "t.py:3:8: Found both 'pd.Categorical' and 'Categorical' in t.py"),  # 测试不一致引用情况，期望输出指定的错误信息
        (BAD_FILE_1, "t.py:2:8: Found both 'pd.Categorical' and 'Categorical' in t.py"),  # 测试不一致引用情况，期望输出指定的错误信息
        (
            BAD_FILE_2,
            "t.py:2:8: Found both 'pandas.Categorical' and 'Categorical' in t.py",
        ),  # 测试不一致引用情况，期望输出指定的错误信息
    ],
)
def test_inconsistent_usage(content, expected, capsys):  # 定义测试函数，测试不一致的引用情况
    with pytest.raises(SystemExit):  # 检查是否引发 SystemExit 异常
        check_for_inconsistent_pandas_namespace(content, PATH, replace=False)  # 调用函数检查 pandas 引用的一致性
    result, _ = capsys.readouterr()  # 读取测试结果中的标准输出和标准错误
    assert result == expected  # 断言输出结果符合预期的错误信息


@pytest.mark.parametrize("content", [GOOD_FILE_0, GOOD_FILE_1, GOOD_FILE_2])  # 使用 pytest 的参数化标记定义测试用例参数
@pytest.mark.parametrize("replace", [True, False])  # 使用 pytest 的参数化标记定义测试用例参数
def test_consistent_usage(content, replace):  # 定义测试函数，测试一致的引用情况
    # 应该不会引发异常
    check_for_inconsistent_pandas_namespace(content, PATH, replace=replace)  # 调用函数检查 pandas 引用的一致性


@pytest.mark.parametrize("content", [BAD_FILE_0, BAD_FILE_1, BAD_FILE_2])  # 使用 pytest 的参数化标记定义测试用例参数
def test_inconsistent_usage_with_replace(content):  # 定义测试函数，测试替换后的不一致引用情况
    result = check_for_inconsistent_pandas_namespace(content, PATH, replace=True)  # 调用函数检查 pandas 引用的一致性，并进行替换
    expected = (  # 定义预期的替换后代码
        "from pandas import Categorical\ncat_0 = Categorical()\ncat_1 = Categorical()"
    )
    assert result == expected  # 断言函数返回的结果符合预期的替换后代码
```