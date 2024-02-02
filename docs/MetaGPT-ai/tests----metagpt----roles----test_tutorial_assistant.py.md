# `MetaGPT\tests\metagpt\roles\test_tutorial_assistant.py`

```py

#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/6 23:11:27
@Author  : Stitch-z
@File    : test_tutorial_assistant.py
"""

# 导入所需的模块
import aiofiles
import pytest

# 从metagpt.const模块中导入TUTORIAL_PATH常量
from metagpt.const import TUTORIAL_PATH
# 从metagpt.roles.tutorial_assistant模块中导入TutorialAssistant类
from metagpt.roles.tutorial_assistant import TutorialAssistant

# 使用pytest.mark.asyncio装饰器标记为异步测试
@pytest.mark.asyncio
# 使用pytest.mark.parametrize装饰器定义参数化测试
@pytest.mark.parametrize(("language", "topic"), [("Chinese", "Write a tutorial about pip")])
# 定义异步测试函数
async def test_tutorial_assistant(language: str, topic: str):
    # 创建TutorialAssistant对象
    role = TutorialAssistant(language=language)
    # 运行role对象的run方法，获取消息
    msg = await role.run(topic)
    # 断言TUTORIAL_PATH路径存在
    assert TUTORIAL_PATH.exists()
    # 获取消息内容作为文件名
    filename = msg.content
    # 异步打开文件
    async with aiofiles.open(filename, mode="r", encoding="utf-8") as reader:
        # 读取文件内容
        content = await reader.read()
        # 断言文件内容中包含"pip"
        assert "pip" in content

# 如果当前模块是主模块，则执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```