# `MetaGPT\metagpt\team.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 00:30
@Author  : alexanderwu
@File    : team.py
@Modified By: mashenquan, 2023/11/27. Add an archiving operation after completing the project, as specified in
        Section 2.2.3.3 of RFC 135.
"""

import warnings  # 导入警告模块
from pathlib import Path  # 导入路径模块
from typing import Any  # 导入类型提示模块

from pydantic import BaseModel, ConfigDict, Field  # 导入数据验证模块

from metagpt.actions import UserRequirement  # 导入用户需求模块
from metagpt.config import CONFIG  # 导入配置模块
from metagpt.const import MESSAGE_ROUTE_TO_ALL, SERDESER_PATH  # 导入常量
from metagpt.environment import Environment  # 导入环境模块
from metagpt.logs import logger  # 导入日志模块
from metagpt.roles import Role  # 导入角色模块
from metagpt.schema import Message  # 导入消息模块
from metagpt.utils.common import (  # 导入常用工具模块
    NoMoneyException,
    read_json_file,
    serialize_decorator,
    write_json_file,
)


class Team(BaseModel):
    """
    Team: Possesses one or more roles (agents), SOP (Standard Operating Procedures), and a env for instant messaging,
    dedicated to env any multi-agent activity, such as collaboratively writing executable code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 模型配置

    env: Environment = Field(default_factory=Environment)  # 环境
    investment: float = Field(default=10.0)  # 投资
    idea: str = Field(default="")  # 创意

    def __init__(self, **data: Any):
        super(Team, self).__init__(**data)
        if "roles" in data:
            self.hire(data["roles"])  # 雇佣角色
        if "env_desc" in data:
            self.env.desc = data["env_desc"]  # 设置环境描述

    def serialize(self, stg_path: Path = None):
        stg_path = SERDESER_PATH.joinpath("team") if stg_path is None else stg_path  # 序列化路径

        team_info_path = stg_path.joinpath("team_info.json")  # 团队信息路径
        write_json_file(team_info_path, self.model_dump(exclude={"env": True}))  # 写入团队信息

        self.env.serialize(stg_path.joinpath("environment"))  # 保存环境信息

    @classmethod
    def deserialize(cls, stg_path: Path) -> "Team":
        """stg_path = ./storage/team"""
        # recover team_info
        team_info_path = stg_path.joinpath("team_info.json")  # 团队信息路径
        if not team_info_path.exists():
            raise FileNotFoundError(
                "recover storage meta file `team_info.json` not exist, "
                "not to recover and please start a new project."
            )  # 文件不存在则抛出异常

        team_info: dict = read_json_file(team_info_path)  # 读取团队信息

        # recover environment
        environment = Environment.deserialize(stg_path=stg_path.joinpath("environment"))  # 恢复环境
        team_info.update({"env": environment})
        team = Team(**team_info)
        return team

    def hire(self, roles: list[Role]):
        """Hire roles to cooperate"""
        self.env.add_roles(roles)  # 雇佣角色合作

    def invest(self, investment: float):
        """Invest company. raise NoMoneyException when exceed max_budget."""
        self.investment = investment  # 投资
        CONFIG.max_budget = investment  # 设置最大预算
        logger.info(f"Investment: ${investment}.")  # 记录投资信息

    @staticmethod
    def _check_balance():
        if CONFIG.cost_manager.total_cost > CONFIG.cost_manager.max_budget:
            raise NoMoneyException(
                CONFIG.cost_manager.total_cost, f"Insufficient funds: {CONFIG.cost_manager.max_budget}"
            )  # 检查余额，如果超出最大预算则抛出异常

    def run_project(self, idea, send_to: str = ""):
        """Run a project from publishing user requirement."""
        self.idea = idea  # 运行项目
        self.env.publish_message(
            Message(role="Human", content=idea, cause_by=UserRequirement, send_to=send_to or MESSAGE_ROUTE_TO_ALL),
            peekable=False,
        )  # 发布消息

    def start_project(self, idea, send_to: str = ""):
        """
        Deprecated: This method will be removed in the future.
        Please use the `run_project` method instead.
        """
        warnings.warn(
            "The 'start_project' method is deprecated and will be removed in the future. "
            "Please use the 'run_project' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )  # 发出警告

        return self.run_project(idea=idea, send_to=send_to)  # 运行项目

    def _save(self):
        logger.info(self.model_dump_json())  # 保存信息

    @serialize_decorator
    async def run(self, n_round=3, idea="", send_to="", auto_archive=True):
        """Run company until target round or no money"""
        if idea:
            self.run_project(idea=idea, send_to=send_to)  # 运行项目

        while n_round > 0:
            n_round -= 1
            logger.debug(f"max {n_round=} left.")  # 记录剩余轮次
            self._check_balance()  # 检查余额

            await self.env.run()  # 运行环境
        self.env.archive(auto_archive)  # 存档环境
        return self.env.history  # 返回历史记录

```