# `MetaGPT\tests\metagpt\roles\test_teacher.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/27 13:25
@Author  : mashenquan
@File    : test_teacher.py
"""
# 导入所需的模块
import os
from typing import Dict, Optional
# 导入 pytest 模块
import pytest
# 导入 pydantic 模块中的 BaseModel 类
from pydantic import BaseModel
# 导入自定义的模块
from metagpt.config import CONFIG, Config
from metagpt.roles.teacher import Teacher
from metagpt.schema import Message

# 标记为异步测试
@pytest.mark.asyncio
# 标记为跳过测试
@pytest.mark.skip
# 测试初始化函数
async def test_init():
    # 定义输入数据的数据模型
    class Inputs(BaseModel):
        name: str
        profile: str
        goal: str
        constraints: str
        desc: str
        kwargs: Optional[Dict] = None
        expect_name: str
        expect_profile: str
        expect_goal: str
        expect_constraints: str
        expect_desc: str
    # 定义输入数据
    inputs = [
        {
            "name": "Lily{language}",
            "expect_name": "Lily{language}",
            "profile": "X {teaching_language}",
            "expect_profile": "X {teaching_language}",
            "goal": "Do {something_big}, {language}",
            "expect_goal": "Do {something_big}, {language}",
            "constraints": "Do in {key1}, {language}",
            "expect_constraints": "Do in {key1}, {language}",
            "kwargs": {},
            "desc": "aaa{language}",
            "expect_desc": "aaa{language}",
        },
        {
            "name": "Lily{language}",
            "expect_name": "LilyCN",
            "profile": "X {teaching_language}",
            "expect_profile": "X EN",
            "goal": "Do {something_big}, {language}",
            "expect_goal": "Do sleep, CN",
            "constraints": "Do in {key1}, {language}",
            "expect_constraints": "Do in HaHa, CN",
            "kwargs": {"language": "CN", "key1": "HaHa", "something_big": "sleep", "teaching_language": "EN"},
            "desc": "aaa{language}",
            "expect_desc": "aaaCN",
        },
    ]
    # 复制环境变量
    env = os.environ.copy()
    # 遍历输入数据
    for i in inputs:
        # 根据数据模型创建输入实例
        seed = Inputs(**i)
        # 清空环境变量
        os.environ.clear()
        # 更新环境变量
        os.environ.update(env)
        # 创建配置对象
        CONFIG = Config()
        # 设置上下文
        CONFIG.set_context(seed.kwargs)
        print(CONFIG.options)
        # 断言语言是否在配置选项中
        assert bool("language" in seed.kwargs) == bool("language" in CONFIG.options)
        # 创建教师对象
        teacher = Teacher(
            name=seed.name,
            profile=seed.profile,
            goal=seed.goal,
            constraints=seed.constraints,
            desc=seed.desc,
        )
        # 断言教师对象的属性值
        assert teacher.name == seed.expect_name
        assert teacher.desc == seed.expect_desc
        assert teacher.profile == seed.expect_profile
        assert teacher.goal == seed.expect_goal
        assert teacher.constraints == seed.expect_constraints
        assert teacher.course_title == "teaching_plan"

# 标记为异步测试
@pytest.mark.asyncio
# 测试新文件名函数
async def test_new_file_name():
    # 定义输入数据的数据模型
    class Inputs(BaseModel):
        lesson_title: str
        ext: str
        expect: str
    # 定义输入数据
    inputs = [
        {"lesson_title": "# @344\n12", "ext": ".md", "expect": "_344_12.md"},
        {"lesson_title": "1#@$%!*&\\/:*?\"<>|\n\t '1", "ext": ".cc", "expect": "1_1.cc"},
    ]
    # 遍历输入数据
    for i in inputs:
        # 根据数据模型创建输入实例
        seed = Inputs(**i)
        # 调用新文件名函数
        result = Teacher.new_file_name(seed.lesson_title, seed.ext)
        # 断言结果
        assert result == seed.expect

# 标记为异步测试
@pytest.mark.asyncio
# 测试运行函数
async def test_run():
    # 设置上下文
    CONFIG.set_context({"language": "Chinese", "teaching_language": "English"})
    # 定义课程内容
    lesson = """
    UNIT 1 Making New Friends
    ...
    """
    # 创建教师对象
    teacher = Teacher()
    # 运行教师对象的 run 方法
    rsp = await teacher.run(Message(content=lesson))
    # 断言结果
    assert rsp

# 如果是主程序则运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```