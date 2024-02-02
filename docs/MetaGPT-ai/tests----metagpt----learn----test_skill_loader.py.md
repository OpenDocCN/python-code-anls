# `MetaGPT\tests\metagpt\learn\test_skill_loader.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/19
@Author  : mashenquan
@File    : test_skill_loader.py
@Desc    : Unit tests.
"""
# 导入模块
from pathlib import Path
# 导入 pytest 模块
import pytest
# 从 metagpt.config 模块中导入 CONFIG 变量
from metagpt.config import CONFIG
# 从 metagpt.learn.skill_loader 模块中导入 SkillsDeclaration 类

# 标记为异步测试
@pytest.mark.asyncio
async def test_suite():
    # 设置 CONFIG.agent_skills 变量
    CONFIG.agent_skills = [
        {"id": 1, "name": "text_to_speech", "type": "builtin", "config": {}, "enabled": True},
        {"id": 2, "name": "text_to_image", "type": "builtin", "config": {}, "enabled": True},
        {"id": 3, "name": "ai_call", "type": "builtin", "config": {}, "enabled": True},
        {"id": 3, "name": "data_analysis", "type": "builtin", "config": {}, "enabled": True},
        {"id": 5, "name": "crawler", "type": "builtin", "config": {"engine": "ddg"}, "enabled": True},
        {"id": 6, "name": "knowledge", "type": "builtin", "config": {}, "enabled": True},
        {"id": 6, "name": "web_search", "type": "builtin", "config": {}, "enabled": True},
    ]
    # 设置 pathname 变量
    pathname = Path(__file__).parent / "../../../docs/.well-known/skills.yaml"
    # 加载 SkillsDeclaration 对象
    loader = await SkillsDeclaration.load(skill_yaml_file_name=pathname)
    # 获取技能列表
    skills = loader.get_skill_list()
    # 断言技能列表不为空
    assert skills
    # 断言技能列表长度大于等于3
    assert len(skills) >= 3
    # 遍历技能列表，断言描述和名称不为空
    for desc, name in skills.items():
        assert desc
        assert name

    # 获取实体对象
    entity = loader.entities.get("Assistant")
    # 断言实体对象不为空
    assert entity
    # 断言实体对象的技能列表不为空
    assert entity.skills
    # 遍历实体对象的技能列表，断言技能和参数不为空
    for sk in entity.skills:
        assert sk
        assert sk.arguments


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-s"])

```