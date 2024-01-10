# `MetaGPT\tests\metagpt\utils\test_text.py`

```

# 导入 pytest 模块
import pytest
# 从 metagpt.utils.text 模块中导入所需的函数
from metagpt.utils.text import (
    decode_unicode_escape,
    generate_prompt_chunk,
    reduce_message_length,
    split_paragraph,
)

# 生成消息的生成器函数
def _msgs():
    length = 20
    while length:
        yield "Hello," * 1000 * length
        length -= 1

# 生成段落的函数
def _paragraphs(n):
    return " ".join("Hello World." for _ in range(n))

# 测试函数 test_reduce_message_length
@pytest.mark.parametrize(
    "msgs, model_name, system_text, reserved, expected",
    [
        # 参数化测试数据
    ],
)
def test_reduce_message_length(msgs, model_name, system_text, reserved, expected):
    # 断言测试结果
    assert len(reduce_message_length(msgs, model_name, system_text, reserved)) / (len("Hello,")) / 1000 == expected

# 测试函数 test_generate_prompt_chunk
@pytest.mark.parametrize(
    "text, prompt_template, model_name, system_text, reserved, expected",
    [
        # 参数化测试数据
    ],
)
def test_generate_prompt_chunk(text, prompt_template, model_name, system_text, reserved, expected):
    # 断言测试结果
    ret = list(generate_prompt_chunk(text, prompt_template, model_name, system_text, reserved))
    assert len(ret) == expected

# 测试函数 test_split_paragraph
@pytest.mark.parametrize(
    "paragraph, sep, count, expected",
    [
        # 参数化测试数据
    ],
)
def test_split_paragraph(paragraph, sep, count, expected):
    # 断言测试结果
    ret = split_paragraph(paragraph, sep, count)
    assert ret == expected

# 测试函数 test_decode_unicode_escape
@pytest.mark.parametrize(
    "text, expected",
    [
        # 参数化测试数据
    ],
)
def test_decode_unicode_escape(text, expected):
    # 断言测试结果
    assert decode_unicode_escape(text) == expected

```