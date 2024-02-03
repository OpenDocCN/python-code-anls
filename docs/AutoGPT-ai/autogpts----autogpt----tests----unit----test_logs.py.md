# `.\AutoGPT\autogpts\autogpt\tests\unit\test_logs.py`

```py
# 导入 pytest 模块
import pytest

# 从 autogpt.logs.utils 模块中导入 remove_color_codes 函数
from autogpt.logs.utils import remove_color_codes

# 使用 pytest.mark.parametrize 装饰器定义参数化测试
@pytest.mark.parametrize(
    "raw_text, clean_text",
    [
        (
            "COMMAND = \x1b[36mbrowse_website\x1b[0m  "
            "ARGUMENTS = \x1b[36m{'url': 'https://www.google.com',"
            " 'question': 'What is the capital of France?'}\x1b[0m",
            "COMMAND = browse_website  "
            "ARGUMENTS = {'url': 'https://www.google.com',"
            " 'question': 'What is the capital of France?'}",
        ),
        (
            "{'Schaue dir meine Projekte auf github () an, als auch meine Webseiten': "
            "'https://github.com/Significant-Gravitas/AutoGPT,"
            " https://discord.gg/autogpt und https://twitter.com/Auto_GPT'}",
            "{'Schaue dir meine Projekte auf github () an, als auch meine Webseiten': "
            "'https://github.com/Significant-Gravitas/AutoGPT,"
            " https://discord.gg/autogpt und https://twitter.com/Auto_GPT'}",
        ),
        ("", ""),
        ("hello", "hello"),
        ("hello\x1B[31m world", "hello world"),
        ("\x1B[36mHello,\x1B[32m World!", "Hello, World!"),
        (
            "\x1B[1m\x1B[31mError:\x1B[0m\x1B[31m file not found",
            "Error: file not found",
        ),
    ],
)
# 定义测试函数 test_remove_color_codes，用于测试 remove_color_codes 函数
def test_remove_color_codes(raw_text, clean_text):
    # 断言 remove_color_codes 函数处理后的结果与预期结果相等
    assert remove_color_codes(raw_text) == clean_text
```