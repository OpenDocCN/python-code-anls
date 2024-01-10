# `MetaGPT\metagpt\actions\write_teaching_plan.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/27
@Author  : mashenquan
@File    : write_teaching_plan.py
"""
# 导入必要的模块
from typing import Optional
from metagpt.actions import Action
from metagpt.config import CONFIG
from metagpt.logs import logger

# 定义一个名为WriteTeachingPlanPart的类，继承自Action类
class WriteTeachingPlanPart(Action):
    """Write Teaching Plan Part"""

    # 定义类的属性
    context: Optional[str] = None
    topic: str = ""
    language: str = "Chinese"
    rsp: Optional[str] = None

    # 异步方法，运行教学计划的写入
    async def run(self, with_message=None, **kwargs):
        # 获取教学计划的语句模板
        statement_patterns = TeachingPlanBlock.TOPIC_STATEMENTS.get(self.topic, [])
        statements = []
        # 格式化语句模板
        for p in statement_patterns:
            s = self.format_value(p)
            statements.append(s)
        # 根据不同的topic选择不同的格式化模板
        formatter = (
            TeachingPlanBlock.PROMPT_TITLE_TEMPLATE
            if self.topic == TeachingPlanBlock.COURSE_TITLE
            else TeachingPlanBlock.PROMPT_TEMPLATE
        )
        # 格式化提示语句
        prompt = formatter.format(
            formation=TeachingPlanBlock.FORMATION,
            role=self.prefix,
            statements="\n".join(statements),
            lesson=self.context,
            topic=self.topic,
            language=self.language,
        )

        logger.debug(prompt)
        # 发送提示语句并获取回复
        rsp = await self._aask(prompt=prompt)
        logger.debug(rsp)
        self._set_result(rsp)
        return self.rsp

    # 设置结果
    def _set_result(self, rsp):
        if TeachingPlanBlock.DATA_BEGIN_TAG in rsp:
            ix = rsp.index(TeachingPlanBlock.DATA_BEGIN_TAG)
            rsp = rsp[ix + len(TeachingPlanBlock.DATA_BEGIN_TAG) :]
        if TeachingPlanBlock.DATA_END_TAG in rsp:
            ix = rsp.index(TeachingPlanBlock.DATA_END_TAG)
            rsp = rsp[0:ix]
        self.rsp = rsp.strip()
        if self.topic != TeachingPlanBlock.COURSE_TITLE:
            return
        if "#" not in self.rsp or self.rsp.index("#") != 0:
            self.rsp = "# " + self.rsp

    # 返回topic值的字符串表示
    def __str__(self):
        """Return `topic` value when str()"""
        return self.topic

    # 在调试时显示topic值
    def __repr__(self):
        """Show `topic` value when debug"""
        return self.topic

    # 格式化value中的参数
    @staticmethod
    def format_value(value):
        """Fill parameters inside `value` with `options`."""
        if not isinstance(value, str):
            return value
        if "{" not in value:
            return value

        merged_opts = CONFIG.options or {}
        try:
            return value.format(**merged_opts)
        except KeyError as e:
            logger.warning(f"Parameter is missing:{e}")

        for k, v in merged_opts.items():
            value = value.replace("{" + f"{k}" + "}", str(v))
        return value

```