# `MetaGPT\tests\metagpt\management\test_skill_manager.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/6 12:38
@Author  : alexanderwu
@File    : test_skill_manager.py
"""
# 导入需要的模块
from metagpt.actions import WritePRD, WriteTest
from metagpt.logs import logger
from metagpt.management.skill_manager import SkillManager

# 定义测试函数
def test_skill_manager():
    # 创建技能管理器对象
    manager = SkillManager()
    # 打印技能管理器的存储内容
    logger.info(manager._store)

    # 创建 WritePRD 对象并设置描述
    write_prd = WritePRD(name="WritePRD")
    write_prd.desc = "基于老板或其他人的需求进行PRD的撰写，包括用户故事、需求分解等"
    # 创建 WriteTest 对象并设置描述
    write_test = WriteTest(name="WriteTest")
    write_test.desc = "进行测试用例的撰写"
    # 向技能管理器中添加技能
    manager.add_skill(write_prd)
    manager.add_skill(write_test)

    # 获取指定名称的技能
    skill = manager.get_skill("WriteTest")
    logger.info(skill)

    # 检索指定名称的技能
    rsp = manager.retrieve_skill("WritePRD")
    logger.info(rsp)
    assert rsp[0] == "WritePRD"

    # 检索指定名称的技能（中文描述）
    rsp = manager.retrieve_skill("写测试用例")
    logger.info(rsp)
    assert rsp[0] == "WriteTest"

    # 检索指定名称的技能及其得分
    rsp = manager.retrieve_skill_scored("写PRD")
    logger.info(rsp)

```