# `.\AutoGPT\autogpts\autogpt\autogpt\config\ai_profile.py`

```py
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AIProfile(BaseModel):
    """
    Object to hold the AI's personality.

    Attributes:
        ai_name (str): The name of the AI.
        ai_role (str): The description of the AI's role.
        ai_goals (list): The list of objectives the AI is supposed to complete.
        api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
    """

    ai_name: str = ""
    ai_role: str = ""
    ai_goals: list[str] = Field(default_factory=list[str])
    api_budget: float = 0.0

    @staticmethod
    def load(ai_settings_file: str | Path) -> "AIProfile":
        """
        Returns class object with parameters (ai_name, ai_role, ai_goals, api_budget)
        loaded from yaml file if it exists, else returns class with no parameters.

        Parameters:
            ai_settings_file (Path): The path to the config yaml file.

        Returns:
            cls (object): An instance of given cls object
        """

        try:
            # 尝试打开配置文件，使用 utf-8 编码
            with open(ai_settings_file, encoding="utf-8") as file:
                # 加载 yaml 文件内容，如果文件不存在则返回空字典
                config_params = yaml.load(file, Loader=yaml.FullLoader) or {}
        except FileNotFoundError:
            # 如果文件不存在，则配置参数为空字典
            config_params = {}

        # 从配置参数中获取 AI 名称、角色、目标和 API 预算
        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        # 处理目标列表，去除特殊字符并转换为字符串
        ai_goals = [
            str(goal).strip("{}").replace("'", "").replace('"', "")
            if isinstance(goal, dict)
            else str(goal)
            for goal in config_params.get("ai_goals", [])
        ]
        api_budget = config_params.get("api_budget", 0.0)

        # 返回一个 AIProfile 实例，传入获取的参数
        return AIProfile(
            ai_name=ai_name, ai_role=ai_role, ai_goals=ai_goals, api_budget=api_budget
        )
    # 将类参数保存到指定的文件路径作为一个 yaml 文件
    def save(self, ai_settings_file: str | Path) -> None:
        """
        Saves the class parameters to the specified file yaml file path as a yaml file.

        Parameters:
            ai_settings_file (Path): The path to the config yaml file.

        Returns:
            None
        """

        # 以写入模式打开指定的文件路径，指定编码为 utf-8
        with open(ai_settings_file, "w", encoding="utf-8") as file:
            # 将类参数转换为字典形式，并使用 yaml.dump 写入到文件中，允许使用 Unicode
            yaml.dump(self.dict(), file, allow_unicode=True)
```