# `MetaGPT\tests\metagpt\test_team.py`

```py

#!/usr/bin/env python
# 指定解释器为 Python
# -*- coding: utf-8 -*-
# 指定编码格式为 UTF-8
# @Desc   : unittest of team
# 描述：团队的单元测试

from metagpt.roles.project_manager import ProjectManager
# 从 metagpt.roles.project_manager 模块导入 ProjectManager 类
from metagpt.team import Team
# 从 metagpt.team 模块导入 Team 类

def test_team():
    # 创建一个名为 company 的 Team 对象
    company = Team()
    # 向 company 团队中招聘一个 ProjectManager
    company.hire([ProjectManager()])

    # 断言 company 团队中的角色数量为 1
    assert len(company.env.roles) == 1

```