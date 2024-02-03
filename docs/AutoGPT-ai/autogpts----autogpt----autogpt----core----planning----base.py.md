# `.\AutoGPT\autogpts\autogpt\autogpt\core\planning\base.py`

```py
# 定义一个抽象基类 Planner，用于管理代理的规划和目标设定，通过构建语言模型提示。
class Planner(abc.ABC):
    """
    Manages the agent's planning and goal-setting
    by constructing language model prompts.
    """

    # 定义一个静态方法 decide_name_and_goals，用于从用户定义的目标决定代理的名称和目标
    @staticmethod
    @abc.abstractmethod
    async def decide_name_and_goals(
        user_objective: str,
    ) -> LanguageModelResponse:
        """Decide the name and goals of an Agent from a user-defined objective.

        Args:
            user_objective: The user-defined objective for the agent.

        Returns:
            The agent name and goals as a response from the language model.

        """
        ...

    # 定义一个抽象方法 plan，用于规划代理的下一个能力
    @abc.abstractmethod
    async def plan(self, context: PlanningContext) -> LanguageModelResponse:
        """Plan the next ability for the Agent.

        Args:
            context: A context object containing information about the agent's
                     progress, result, memories, and feedback.

        Returns:
            The next ability the agent should take along with thoughts and reasoning.

        """
        ...

    # 定义一个抽象方法 reflect，用于反思规划的能力并提供自我批评
    @abc.abstractmethod
    def reflect(
        self,
        context: ReflectionContext,
    ) -> LanguageModelResponse:
        """Reflect on a planned ability and provide self-criticism.

        Args:
            context: A context object containing information about the agent's
                     reasoning, plan, thoughts, and criticism.

        Returns:
            Self-criticism about the agent's plan.

        """
        ...
```