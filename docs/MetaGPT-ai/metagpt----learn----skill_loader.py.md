# `MetaGPT\metagpt\learn\skill_loader.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : skill_loader.py
@Desc    : Skill YAML Configuration Loader.
"""
# 导入所需的模块
from pathlib import Path
from typing import Dict, List, Optional
import aiofiles
import yaml
from pydantic import BaseModel, Field
from metagpt.config import CONFIG

# 定义 Example 模型
class Example(BaseModel):
    ask: str
    answer: str

# 定义 Returns 模型
class Returns(BaseModel):
    type: str
    format: Optional[str] = None

# 定义 Parameter 模型
class Parameter(BaseModel):
    type: str
    description: str = None

# 定义 Skill 模型
class Skill(BaseModel):
    name: str
    description: str = None
    id: str = None
    x_prerequisite: Dict = Field(default=None, alias="x-prerequisite")
    parameters: Dict[str, Parameter] = None
    examples: List[Example]
    returns: Returns

    @property
    def arguments(self) -> Dict:
        if not self.parameters:
            return {}
        ret = {}
        for k, v in self.parameters.items():
            ret[k] = v.description if v.description else ""
        return ret

# 定义 Entity 模型
class Entity(BaseModel):
    name: str = None
    skills: List[Skill]

# 定义 Components 模型
class Components(BaseModel):
    pass

# 定义 SkillsDeclaration 模型
class SkillsDeclaration(BaseModel):
    skillapi: str
    entities: Dict[str, Entity]
    components: Components = None

    @staticmethod
    async def load(skill_yaml_file_name: Path = None) -> "SkillsDeclaration":
        if not skill_yaml_file_name:
            skill_yaml_file_name = Path(__file__).parent.parent.parent / "docs/.well-known/skills.yaml"
        async with aiofiles.open(str(skill_yaml_file_name), mode="r") as reader:
            data = await reader.read(-1)
        skill_data = yaml.safe_load(data)
        return SkillsDeclaration(**skill_data)

    def get_skill_list(self, entity_name: str = "Assistant") -> Dict:
        """Return the skill name based on the skill description."""
        entity = self.entities.get(entity_name)
        if not entity:
            return {}

        # List of skills that the agent chooses to activate.
        agent_skills = CONFIG.agent_skills
        if not agent_skills:
            return {}

        class _AgentSkill(BaseModel):
            name: str

        names = [_AgentSkill(**i).name for i in agent_skills]
        return {s.description: s.name for s in entity.skills if s.name in names}

    def get_skill(self, name, entity_name: str = "Assistant") -> Skill:
        """Return a skill by name."""
        entity = self.entities.get(entity_name)
        if not entity:
            return None
        for sk in entity.skills:
            if sk.name == name:
                return sk

```