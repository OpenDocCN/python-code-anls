# `.\DB-GPT-src\dbgpt\agent\core\role.py`

```py
"""Role class for role-based conversation."""

# 引入抽象基类（ABC）用于定义角色类，继承自BaseModel
from abc import ABC
# 引入类型提示
from typing import Any, Dict, List, Optional

# 引入沙盒环境，用于模板处理
from jinja2.sandbox import SandboxedEnvironment

# 引入Pydantic相关模块
from dbgpt._private.pydantic import BaseModel, ConfigDict, Field

# 引入其他模块和类
from .action.base import ActionOutput
from .memory.agent_memory import AgentMemory, AgentMemoryFragment
from .memory.llm import LLMImportanceScorer, LLMInsightExtractor
from .profile import Profile, ProfileConfig


class Role(ABC, BaseModel):
    """Role class for role-based conversation."""

    # 定义模型配置字典，允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 角色的配置信息，使用ProfileConfig类型，必填项
    profile: ProfileConfig = Field(
        ...,
        description="The profile of the role.",
    )
    
    # 代理记忆，使用AgentMemory类型，默认工厂设置
    memory: AgentMemory = Field(default_factory=AgentMemory)

    # 固定子目标，可选字符串类型描述
    fixed_subgoal: Optional[str] = Field(None, description="Fixed subgoal")

    # 语言设定，默认为英语
    language: str = "en"
    
    # 是否为人类角色，默认为False
    is_human: bool = False
    
    # 是否为团队角色，默认为False
    is_team: bool = False

    # 模板环境，使用SandboxedEnvironment类型，默认工厂设置
    template_env: SandboxedEnvironment = Field(default_factory=SandboxedEnvironment)

    # 异步方法，构建角色的提示信息模板
    async def build_prompt(
        self,
        question: Optional[str] = None,
        is_system: bool = True,
        most_recent_memories: Optional[str] = None,
        **kwargs
    ) -> str:
        """Return the prompt template for the role.

        Args:
            question (Optional[str]): The question to include in the prompt.
            is_system (bool): Flag indicating if the prompt is for the system.
            most_recent_memories (Optional[str]): Recent memories relevant to the prompt.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The constructed prompt template.
        """
        # 生成资源变量
        resource_vars = await self.generate_resource_variables(question)

        # 根据is_system选择不同的方法构建提示模板
        if is_system:
            return self.current_profile.format_system_prompt(
                template_env=self.template_env,
                question=question,
                language=self.language,
                most_recent_memories=most_recent_memories,
                resource_vars=resource_vars,
                **kwargs,
            )
        else:
            return self.current_profile.format_user_prompt(
                template_env=self.template_env,
                question=question,
                language=self.language,
                most_recent_memories=most_recent_memories,
                resource_vars=resource_vars,
                **kwargs,
            )

    # 异步方法，生成资源变量字典
    async def generate_resource_variables(
        self, question: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate the resource variables.

        Args:
            question (Optional[str]): The question to include in the resource variables.

        Returns:
            Dict[str, Any]: The generated resource variables.
        """
        return {}

    # 校验角色身份的方法，仅声明不实现
    def identity_check(self) -> None:
        """Check the identity of the role."""
        pass

    # 获取角色名称的方法，调用当前配置的Profile对象的get_name方法
    def get_name(self) -> str:
        """Get the name of the role."""
        return self.current_profile.get_name()

    # 属性方法，返回当前配置的Profile对象
    @property
    def current_profile(self) -> Profile:
        """Return the current profile."""
        profile = self.profile.create_profile()
        return profile

    # 属性方法，返回角色名称，调用当前配置的Profile对象的get_name方法
    @property
    def name(self) -> str:
        """Return the name of the role."""
        return self.current_profile.get_name()

    # 属性方法，返回角色角色，调用当前配置的Profile对象的get_role方法
    @property
    def role(self) -> str:
        """Return the role of the role."""
        return self.current_profile.get_role()

    # 属性方法，返回角色
    # 返回当前角色的目标
    def goal(self) -> Optional[str]:
        """Return the goal of the role."""
        return self.current_profile.get_goal()

    # 返回当前角色的约束列表
    @property
    def constraints(self) -> Optional[List[str]]:
        """Return the constraints of the role."""
        return self.current_profile.get_constraints()

    # 返回当前角色的描述信息
    @property
    def desc(self) -> Optional[str]:
        """Return the description of the role."""
        return self.current_profile.get_description()

    # 返回当前角色的写入内存模板
    @property
    def write_memory_template(self) -> str:
        """Return the current save memory template."""
        return self.current_profile.get_write_memory_template()

    # 渲染给定模板并返回结果
    def _render_template(self, template: str, **kwargs):
        r_template = self.template_env.from_string(template)
        return r_template.render(**kwargs)

    # 返回内存重要性评分器实例（如果有）
    @property
    def memory_importance_scorer(self) -> Optional[LLMImportanceScorer]:
        """Create the memory importance scorer.

        The memory importance scorer is used to score the importance of a memory
        fragment.
        """
        return None

    # 返回内存洞察提取器实例（如果有）
    @property
    def memory_insight_extractor(self) -> Optional[LLMInsightExtractor]:
        """Create the memory insight extractor.

        The memory insight extractor is used to extract a high-level insight from a
        memory fragment.
        """
        return None

    # 异步方法：从内存中读取与给定问题相关的记忆内容
    async def read_memories(
        self,
        question: str,
    ) -> str:
        """Read the memories from the memory."""
        memories = await self.memory.read(question)
        recent_messages = [m.raw_observation for m in memories]
        return "".join(recent_messages)

    # 异步方法：向内存中写入给定问题的记忆内容及相关信息
    async def write_memories(
        self,
        question: str,
        ai_message: str,
        action_output: Optional[ActionOutput] = None,
        check_pass: bool = True,
        check_fail_reason: Optional[str] = None,
    ):
        # 省略部分...
    ) -> None:
        """
        Write the memories to the memory.

        We suggest you to override this method to save the conversation to memory
        according to your needs.

        Args:
            question(str): The question received.
            ai_message(str): The AI message, LLM output.
            action_output(ActionOutput): The action output.
            check_pass(bool): Whether the check pass.
            check_fail_reason(str): The check fail reason.
        """
        # 如果没有提供 action_output，则抛出数值错误
        if not action_output:
            raise ValueError("Action output is required to save to memory.")

        # 从 action_output 中获取思考内容，若为空则使用 ai_message
        mem_thoughts = action_output.thoughts or ai_message
        # 获取观察结果
        observation = action_output.observations
        # 如果检查未通过且存在观察结果和检查失败原因，则将失败原因添加到观察结果中
        if not check_pass and observation and check_fail_reason:
            observation += "\n" + check_fail_reason

        # 构建内存映射字典，将问题、思考内容、动作和观察结果保存其中
        memory_map = {
            "question": question,
            "thought": mem_thoughts,
            "action": action_output.action,
            "observation": observation,
        }
        # 获取写入内存的模板
        write_memory_template = self.write_memory_template
        # 使用模板渲染内存内容
        memory_content = self._render_template(write_memory_template, **memory_map)
        # 创建代理内存片段对象
        fragment = AgentMemoryFragment(memory_content)
        # 异步写入内存
        await self.memory.write(fragment)
```