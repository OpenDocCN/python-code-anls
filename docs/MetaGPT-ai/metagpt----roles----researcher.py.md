# `MetaGPT\metagpt\roles\researcher.py`

```py

#!/usr/bin/env python
"""
@Modified By: mashenquan, 2023/8/22. A definition has been provided for the return value of _think: returning false indicates that further reasoning cannot continue.
@Modified By: mashenquan, 2023-11-1. According to Chapter 2.2.1 and 2.2.2 of RFC 116, change the data type of
        the `cause_by` value in the `Message` to a string to support the new message distribution feature.
"""

import asyncio  # 引入异步编程库
import re  # 引入正则表达式库

from pydantic import BaseModel  # 引入数据验证库

from metagpt.actions import Action, CollectLinks, ConductResearch, WebBrowseAndSummarize  # 引入自定义动作类
from metagpt.actions.research import get_research_system_text  # 引入研究系统文本获取函数
from metagpt.const import RESEARCH_PATH  # 引入研究路径常量
from metagpt.logs import logger  # 引入日志记录器
from metagpt.roles.role import Role, RoleReactMode  # 引入角色类和角色反应模式
from metagpt.schema import Message  # 引入消息类


class Report(BaseModel):  # 定义报告类，用于验证报告数据格式
    topic: str  # 报告主题
    links: dict[str, list[str]] = None  # 报告链接
    summaries: list[tuple[str, str]] = None  # 报告摘要
    content: str = ""  # 报告内容


class Researcher(Role):  # 定义研究员角色类，继承自角色类
    name: str = "David"  # 研究员姓名
    profile: str = "Researcher"  # 研究员角色
    goal: str = "Gather information and conduct research"  # 研究目标
    constraints: str = "Ensure accuracy and relevance of information"  # 约束条件
    language: str = "en-us"  # 语言设置

    def __init__(self, **kwargs):  # 初始化方法
        super().__init__(**kwargs)  # 调用父类初始化方法
        self._init_actions(  # 初始化动作列表
            [CollectLinks(name=self.name), WebBrowseAndSummarize(name=self.name), ConductResearch(name=self.name)]
        )
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)  # 设置反应模式
        if self.language not in ("en-us", "zh-cn"):  # 如果语言不在支持范围内
            logger.warning(f"The language `{self.language}` has not been tested, it may not work.")  # 记录警告日志

    async def _think(self) -> bool:  # 定义思考方法
        if self.rc.todo is None:  # 如果待办事项为空
            self._set_state(0)  # 设置状态为0
            return True  # 返回True

        if self.rc.state + 1 < len(self.states):  # 如果状态加1小于状态列表长度
            self._set_state(self.rc.state + 1)  # 设置状态为当前状态加1
        else:  # 否则
            self.rc.todo = None  # 待办事项设为None
            return False  # 返回False

    async def _act(self) -> Message:  # 定义行动方法
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")  # 记录日志
        todo = self.rc.todo  # 获取待办事项
        msg = self.rc.memory.get(k=1)[0]  # 从记忆中获取消息
        if isinstance(msg.instruct_content, Report):  # 如果消息内容是报告类型
            instruct_content = msg.instruct_content  # 获取指示内容
            topic = instruct_content.topic  # 获取主题
        else:  # 否则
            topic = msg.content  # 主题为消息内容

        research_system_text = self.research_system_text(topic, todo)  # 获取研究系统文本
        if isinstance(todo, CollectLinks):  # 如果待办事项是收集链接
            links = await todo.run(topic, 4, 4)  # 运行收集链接动作
            ret = Message(  # 创建消息对象
                content="", instruct_content=Report(topic=topic, links=links), role=self.profile, cause_by=todo
            )
        elif isinstance(todo, WebBrowseAndSummarize):  # 如果待办事项是浏览和总结
            links = instruct_content.links  # 获取链接
            todos = (todo.run(*url, query=query, system_text=research_system_text) for (query, url) in links.items())  # 执行动作
            summaries = await asyncio.gather(*todos)  # 并发执行动作
            summaries = list((url, summary) for i in summaries for (url, summary) in i.items() if summary)  # 整理摘要
            ret = Message(  # 创建消息对象
                content="", instruct_content=Report(topic=topic, summaries=summaries), role=self.profile, cause_by=todo
            )
        else:  # 否则
            summaries = instruct_content.summaries  # 获取摘要
            summary_text = "\n---\n".join(f"url: {url}\nsummary: {summary}" for (url, summary) in summaries)  # 摘要文本
            content = await self.rc.todo.run(topic, summary_text, system_text=research_system_text)  # 运行待办事项
            ret = Message(  # 创建消息对象
                content="",
                instruct_content=Report(topic=topic, content=content),
                role=self.profile,
                cause_by=self.rc.todo,
            )
        self.rc.memory.add(ret)  # 将消息对象添加到记忆中
        return ret  # 返回消息对象

    def research_system_text(self, topic, current_task: Action) -> str:  # 定义研究系统文本方法
        """BACKWARD compatible
        This allows sub-class able to define its own system prompt based on topic.
        return the previous implementation to have backward compatible
        Args:
            topic:
            language:

        Returns: str
        """
        return get_research_system_text(topic, self.language)  # 返回研究系统文本

    async def react(self) -> Message:  # 定义反应方法
        msg = await super().react()  # 调用父类的反应方法
        report = msg.instruct_content  # 获取指示内容
        self.write_report(report.topic, report.content)  # 写入报告
        return msg  # 返回消息对象

    def write_report(self, topic: str, content: str):  # 定义写入报告方法
        filename = re.sub(r'[\\/:"*?<>|]+', " ", topic)  # 替换文件名中的特殊字符
        filename = filename.replace("\n", "")  # 替换换行符
        if not RESEARCH_PATH.exists():  # 如果研究路径不存在
            RESEARCH_PATH.mkdir(parents=True)  # 创建研究路径
        filepath = RESEARCH_PATH / f"{filename}.md"  # 获取文件路径
        filepath.write_text(content)  # 写入文件内容


if __name__ == "__main__":  # 如果是主程序入口
    import fire  # 引入命令行工具库

    async def main(topic: str, language="en-us"):  # 定义主函数
        role = Researcher(language=language)  # 创建研究员角色对象
        await role.run(topic)  # 运行角色对象

    fire.Fire(main)  # 使用命令行工具执行主函数

```