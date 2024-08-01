# `.\DB-GPT-src\dbgpt\util\tests\test_function_utils.py`

```py
from typing import Any, Dict, List  # 导入类型提示

import pytest  # 导入 pytest 测试框架

from dbgpt.util.function_utils import rearrange_args_by_type  # 导入函数装饰器


class ChatPromptTemplate:  # 定义空的类 ChatPromptTemplate
    pass


class BaseMessage:  # 定义空的类 BaseMessage
    pass


class ModelMessage:  # 定义空的类 ModelMessage
    pass


class DummyClass:  # 定义 DummyClass 类
    @rearrange_args_by_type  # 使用函数装饰器 rearrange_args_by_type
    async def class_method(self, a: int, b: str, c: float):
        return a, b, c  # 返回参数 a, b, c 的元组

    @rearrange_args_by_type  # 使用函数装饰器 rearrange_args_by_type
    async def merge_history(
        self,
        prompt: ChatPromptTemplate,
        history: List[BaseMessage],
        prompt_dict: Dict[str, Any],
    ) -> List[ModelMessage]:
        return [type(prompt), type(history), type(prompt_dict)]  # 返回包含 prompt, history, prompt_dict 类型的列表

    @rearrange_args_by_type  # 使用函数装饰器 rearrange_args_by_type
    def sync_class_method(self, a: int, b: str, c: float):
        return a, b, c  # 返回参数 a, b, c 的元组


@rearrange_args_by_type  # 使用函数装饰器 rearrange_args_by_type
def sync_regular_function(a: int, b: str, c: float):
    return a, b, c  # 返回参数 a, b, c 的元组


@rearrange_args_by_type  # 使用函数装饰器 rearrange_args_by_type
async def regular_function(a: int, b: str, c: float):
    return a, b, c  # 返回参数 a, b, c 的元组


@pytest.mark.asyncio  # 使用 pytest 的 asyncio 标记
async def test_class_method_correct_order():
    instance = DummyClass()
    result = await instance.class_method(1, "b", 3.0)
    assert result == (1, "b", 3.0), "Class method failed with correct order"  # 断言结果符合预期


@pytest.mark.asyncio  # 使用 pytest 的 asyncio 标记
async def test_class_method_incorrect_order():
    instance = DummyClass()
    result = await instance.class_method("b", 3.0, 1)
    assert result == (1, "b", 3.0), "Class method failed with incorrect order"  # 断言结果符合预期


@pytest.mark.asyncio  # 使用 pytest 的 asyncio 标记
async def test_regular_function_correct_order():
    result = await regular_function(1, "b", 3.0)
    assert result == (1, "b", 3.0), "Regular function failed with correct order"  # 断言结果符合预期


@pytest.mark.asyncio  # 使用 pytest 的 asyncio 标记
async def test_regular_function_incorrect_order():
    result = await regular_function("b", 3.0, 1)
    assert result == (1, "b", 3.0), "Regular function failed with incorrect order"  # 断言结果符合预期


@pytest.mark.asyncio  # 使用 pytest 的 asyncio 标记
async def test_merge_history_correct_order():
    instance = DummyClass()
    result = await instance.merge_history(
        ChatPromptTemplate(), [BaseMessage()], {"key": "value"}
    )
    assert result == [ChatPromptTemplate, list, dict], "Failed with correct order"  # 断言结果符合预期


@pytest.mark.asyncio  # 使用 pytest 的 asyncio 标记
async def test_merge_history_incorrect_order_1():
    instance = DummyClass()
    result = await instance.merge_history(
        [BaseMessage()], ChatPromptTemplate(), {"key": "value"}
    )
    assert result == [ChatPromptTemplate, list, dict], "Failed with incorrect order 1"  # 断言结果符合预期


@pytest.mark.asyncio  # 使用 pytest 的 asyncio 标记
async def test_merge_history_incorrect_order_2():
    instance = DummyClass()
    result = await instance.merge_history(
        {"key": "value"}, [BaseMessage()], ChatPromptTemplate()
    )
    assert result == [ChatPromptTemplate, list, dict], "Failed with incorrect order 2"  # 断言结果符合预期


def test_sync_class_method_correct_order():
    instance = DummyClass()
    result = instance.sync_class_method(1, "b", 3.0)
    assert result == (1, "b", 3.0), "Sync class method failed with correct order"  # 断言结果符合预期


# 这里应该还有一个 test_sync_class_method_incorrect_order 的测试函数，但由于截断，未完全显示
    # 调用实例的同步类方法sync_class_method，并传入参数"b", 3.0, 1
    result = instance.sync_class_method("b", 3.0, 1)
    # 使用断言验证返回的结果是否符合预期：(1, "b", 3.0)，否则输出错误信息"Sync class method failed with incorrect order"
    assert result == (1, "b", 3.0), "Sync class method failed with incorrect order"
# 定义测试函数，测试同步常规函数的正确顺序
def test_sync_regular_function_correct_order():
    # 调用同步常规函数，传入参数 1, "b", 3.0，期望返回元组 (1, "b", 3.0)
    result = sync_regular_function(1, "b", 3.0)
    # 断言结果是否符合预期
    assert result == (1, "b", 3.0), "Sync regular function failed with correct order"

# 定义测试函数，测试同步常规函数的不正确顺序
def test_sync_regular_function_incorrect_order():
    # 调用同步常规函数，传入参数 "b", 3.0, 1，期望返回元组 (1, "b", 3.0)
    result = sync_regular_function("b", 3.0, 1)
    # 断言结果是否符合预期
    assert result == (1, "b", 3.0), "Sync regular function failed with incorrect order"
```