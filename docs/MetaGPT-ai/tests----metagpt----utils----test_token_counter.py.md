# `MetaGPT\tests\metagpt\utils\test_token_counter.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/24 17:54
@Author  : alexanderwu
@File    : test_token_counter.py
"""
# 导入 pytest 模块
import pytest

# 从 metagpt.utils.token_counter 模块中导入 count_message_tokens 和 count_string_tokens 函数
from metagpt.utils.token_counter import count_message_tokens, count_string_tokens

# 测试 count_message_tokens 函数
def test_count_message_tokens():
    # 定义消息列表
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    # 断言 count_message_tokens 函数返回的结果为 15
    assert count_message_tokens(messages) == 15

# 测试带有名称的 count_message_tokens 函数
def test_count_message_tokens_with_name():
    messages = [
        {"role": "user", "content": "Hello", "name": "John"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert count_message_tokens(messages) == 17

# 测试空输入的 count_message_tokens 函数
def test_count_message_tokens_empty_input():
    """Empty input should return 3 tokens"""
    assert count_message_tokens([]) == 3

# 测试 count_message_tokens 函数中的无效模型
def test_count_message_tokens_invalid_model():
    """Invalid model should raise a KeyError"""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    # 使用 pytest.raises 检查是否会引发 NotImplementedError
    with pytest.raises(NotImplementedError):
        count_message_tokens(messages, model="invalid_model")

# 测试 count_message_tokens 函数中的特定模型
def test_count_message_tokens_gpt_4():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert count_message_tokens(messages, model="gpt-4-0314") == 15

# 测试 count_string_tokens 函数
def test_count_string_tokens():
    """Test that the string tokens are counted correctly."""
    # 定义字符串
    string = "Hello, world!"
    # 断言 count_string_tokens 函数返回的结果为 4
    assert count_string_tokens(string, model_name="gpt-3.5-turbo-0301") == 4

# 测试空输入的 count_string_tokens 函数
def test_count_string_tokens_empty_input():
    """Test that the string tokens are counted correctly."""
    # 断言 count_string_tokens 函数返回的结果为 0
    assert count_string_tokens("", model_name="gpt-3.5-turbo-0301") == 0

# 测试 count_string_tokens 函数中的特定模型
def test_count_string_tokens_gpt_4():
    """Test that the string tokens are counted correctly."""
    string = "Hello, world!"
    assert count_string_tokens(string, model_name="gpt-4-0314") == 4

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```