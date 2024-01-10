# `MetaGPT\metagpt\management\skill_manager.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/5 01:44
@Author  : alexanderwu
@File    : skill_manager.py
@Modified By: mashenquan, 2023/8/20. Remove useless `llm`
"""
# 导入必要的模块和类
from metagpt.actions import Action
from metagpt.const import PROMPT_PATH
from metagpt.document_store.chromadb_store import ChromaStore
from metagpt.logs import logger

Skill = Action

# 定义技能管理类
class SkillManager:
    """Used to manage all skills"""

    def __init__(self):
        # 初始化技能存储
        self._store = ChromaStore("skill_manager")
        # 初始化技能字典
        self._skills: dict[str:Skill] = {}

    # 添加技能的方法
    def add_skill(self, skill: Skill):
        """
        Add a skill, add the skill to the skill pool and searchable storage
        :param skill: Skill
        :return:
        """
        # 将技能添加到技能字典和存储中
        self._skills[skill.name] = skill
        self._store.add(skill.desc, {"name": skill.name, "desc": skill.desc}, skill.name)

    # 删除技能的方法
    def del_skill(self, skill_name: str):
        """
        Delete a skill, remove the skill from the skill pool and searchable storage
        :param skill_name: Skill name
        :return:
        """
        # 从技能字典和存储中删除指定的技能
        self._skills.pop(skill_name)
        self._store.delete(skill_name)

    # 获取技能的方法
    def get_skill(self, skill_name: str) -> Skill:
        """
        Obtain a specific skill by skill name
        :param skill_name: Skill name
        :return: Skill
        """
        # 通过技能名获取对应的技能对象
        return self._skills.get(skill_name)

    # 通过搜索引擎检索技能的方法
    def retrieve_skill(self, desc: str, n_results: int = 2) -> list[Skill]:
        """
        Obtain skills through the search engine
        :param desc: Skill description
        :return: Multiple skills
        """
        # 通过描述信息使用搜索引擎检索技能
        return self._store.search(desc, n_results=n_results)["ids"][0]

    # 通过搜索引擎检索技能并返回得分的方法
    def retrieve_skill_scored(self, desc: str, n_results: int = 2) -> dict:
        """
        Obtain skills through the search engine
        :param desc: Skill description
        :return: Dictionary consisting of skills and scores
        """
        # 通过描述信息使用搜索引擎检索技能并返回得分
        return self._store.search(desc, n_results=n_results)

    # 生成技能描述的方法
    def generate_skill_desc(self, skill: Skill) -> str:
        """
        Generate descriptive text for each skill
        :param skill:
        :return:
        """
        # 生成每个技能的描述文本
        path = PROMPT_PATH / "generate_skill.md"
        text = path.read_text()
        logger.info(text)


if __name__ == "__main__":
    # 实例化技能管理类并生成技能描述
    manager = SkillManager()
    manager.generate_skill_desc(Action())

```