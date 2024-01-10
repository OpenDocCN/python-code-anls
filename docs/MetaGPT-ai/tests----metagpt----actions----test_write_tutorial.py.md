# `MetaGPT\tests\metagpt\actions\test_write_tutorial.py`

```

#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/6 21:41:34
@Author  : Stitch-z
@File    : test_write_tutorial.py
"""
# 导入所需的模块
from typing import Dict
import pytest
from metagpt.actions.write_tutorial import WriteContent, WriteDirectory

# 异步测试函数，测试写入目录功能
@pytest.mark.asyncio
@pytest.mark.parametrize(("language", "topic"), [("English", "Write a tutorial about Python")])
async def test_write_directory(language: str, topic: str):
    # 调用WriteDirectory类的run方法，传入语言和主题参数
    ret = await WriteDirectory(language=language).run(topic=topic)
    # 断言返回结果为字典类型
    assert isinstance(ret, dict)
    # 断言返回结果中包含"title"键
    assert "title" in ret
    # 断言返回结果中包含"directory"键
    assert "directory" in ret
    # 断言返回结果中"directory"值为列表类型
    assert isinstance(ret["directory"], list)
    # 断言返回结果中"directory"列表不为空
    assert len(ret["directory"])
    # 断言返回结果中"directory"列表第一个元素为字典类型
    assert isinstance(ret["directory"][0], dict)

# 异步测试函数，测试写入内容功能
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "topic", "directory"),
    [("English", "Write a tutorial about Python", {"Introduction": ["What is Python?", "Why learn Python?"]})],
)
async def test_write_content(language: str, topic: str, directory: Dict):
    # 调用WriteContent类的run方法，传入语言、主题和目录参数
    ret = await WriteContent(language=language, directory=directory).run(topic=topic)
    # 断言返回结果为字符串类型
    assert isinstance(ret, str)
    # 断言返回结果中包含目录的第一个键
    assert list(directory.keys())[0] in ret
    # 断言返回结果中包含目录的第一个值
    for value in list(directory.values())[0]:
        assert value in ret

```