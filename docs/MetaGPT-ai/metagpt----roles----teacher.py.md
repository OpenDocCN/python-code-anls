# `MetaGPT\metagpt\roles\teacher.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/27
@Author  : mashenquan
@File    : teacher.py
@Desc    : Used by Agent Store
@Modified By: mashenquan, 2023/8/22. A definition has been provided for the return value of _think: returning false indicates that further reasoning cannot continue.

"""

import re  # 导入正则表达式模块

import aiofiles  # 异步文件操作模块

from metagpt.actions import UserRequirement  # 导入用户需求类
from metagpt.actions.write_teaching_plan import TeachingPlanBlock, WriteTeachingPlanPart  # 导入教学计划块和写教学计划部分类
from metagpt.config import CONFIG  # 导入配置模块
from metagpt.logs import logger  # 导入日志模块
from metagpt.roles import Role  # 导入角色类
from metagpt.schema import Message  # 导入消息类
from metagpt.utils.common import any_to_str  # 导入通用工具函数


class Teacher(Role):
    """Support configurable teacher roles,
    with native and teaching languages being replaceable through configurations."""

    name: str = "Lily"  # 教师姓名
    profile: str = "{teaching_language} Teacher"  # 教师简介
    goal: str = "writing a {language} teaching plan part by part"  # 教学目标
    constraints: str = "writing in {language}"  # 约束条件
    desc: str = ""  # 描述

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = WriteTeachingPlanPart.format_value(self.name)  # 格式化教师姓名
        self.profile = WriteTeachingPlanPart.format_value(self.profile)  # 格式化教师简介
        self.goal = WriteTeachingPlanPart.format_value(self.goal)  # 格式化教学目标
        self.constraints = WriteTeachingPlanPart.format_value(self.constraints)  # 格式化约束条件
        self.desc = WriteTeachingPlanPart.format_value(self.desc)  # 格式化描述

    async def _think(self) -> bool:
        """Everything will be done part by part."""
        if not self.actions:  # 如果没有动作
            if not self.rc.news or self.rc.news[0].cause_by != any_to_str(UserRequirement):  # 如果没有新闻或者新闻不是由用户需求引起的
                raise ValueError("Lesson content invalid.")  # 抛出值错误异常，课程内容无效
            actions = []  # 初始化动作列表
            print(TeachingPlanBlock.TOPICS)  # 打印教学计划主题
            for topic in TeachingPlanBlock.TOPICS:  # 遍历教学计划主题
                act = WriteTeachingPlanPart(context=self.rc.news[0].content, topic=topic, llm=self.llm)  # 创建写教学计划部分的动作
                actions.append(act)  # 将动作添加到动作列表中
            self._init_actions(actions)  # 初始化动作

        if self.rc.todo is None:  # 如果没有待办事项
            self._set_state(0)  # 设置状态为0
            return True  # 返回True

        if self.rc.state + 1 < len(self.states):  # 如果状态加1小于状态列表的长度
            self._set_state(self.rc.state + 1)  # 设置状态为当前状态加1
            return True  # 返回True

        self.rc.todo = None  # 待办事项设为None
        return False  # 返回False

    async def _react(self) -> Message:
        ret = Message(content="")  # 初始化消息内容为空
        while True:  # 循环
            await self._think()  # 等待思考
            if self.rc.todo is None:  # 如果没有待办事项
                break  # 跳出循环
            logger.debug(f"{self._setting}: {self.rc.state=}, will do {self.rc.todo}")  # 记录调试信息
            msg = await self._act()  # 等待执行动作
            if ret.content != "":  # 如果消息内容不为空
                ret.content += "\n\n\n"  # 添加换行符
            ret.content += msg.content  # 添加消息内容
        logger.info(ret.content)  # 记录信息
        await self.save(ret.content)  # 保存消息内容
        return ret  # 返回消息

    async def save(self, content):
        """Save teaching plan"""
        filename = Teacher.new_file_name(self.course_title)  # 获取教学计划的文件名
        pathname = CONFIG.workspace_path / "teaching_plan"  # 教学计划路径
        pathname.mkdir(exist_ok=True)  # 创建教学计划路径
        pathname = pathname / filename  # 教学计划路径加上文件名
        try:
            async with aiofiles.open(str(pathname), mode="w", encoding="utf-8") as writer:  # 异步打开文件
                await writer.write(content)  # 写入内容
        except Exception as e:  # 捕获异常
            logger.error(f"Save failed：{e}")  # 记录错误信息
        logger.info(f"Save to:{pathname}")  # 记录信息

    @staticmethod
    def new_file_name(lesson_title, ext=".md"):
        """Create a related file name based on `lesson_title` and `ext`."""
        # Define the special characters that need to be replaced.
        illegal_chars = r'[#@$%!*&\\/:*?"<>|\n\t \']'  # 定义需要替换的特殊字符
        # Replace the special characters with underscores.
        filename = re.sub(illegal_chars, "_", lesson_title) + ext  # 用下划线替换特殊字符
        return re.sub(r"_+", "_", filename)  # 返回替换后的文件名

    @property
    def course_title(self):
        """Return course title of teaching plan"""
        default_title = "teaching_plan"  # 默认教学计划标题
        for act in self.actions:  # 遍历动作
            if act.topic != TeachingPlanBlock.COURSE_TITLE:  # 如果主题不是课程标题
                continue  # 继续下一次循环
            if act.rsp is None:  # 如果响应为空
                return default_title  # 返回默认标题
            title = act.rsp.lstrip("# \n")  # 去除标题前的空格和换行符
            if "\n" in title:  # 如果标题中包含换行符
                ix = title.index("\n")  # 获取换行符的索引
                title = title[0:ix]  # 截取标题
            return title  # 返回标题

        return default_title  # 返回默认标题

```