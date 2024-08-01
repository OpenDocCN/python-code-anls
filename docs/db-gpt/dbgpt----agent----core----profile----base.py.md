# `.\DB-GPT-src\dbgpt\agent\core\profile\base.py`

```py
"""Profile module."""

# 导入必要的模块和类
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import cachetools
from jinja2.meta import find_undeclared_variables
from jinja2.sandbox import Environment, SandboxedEnvironment

# 导入特定于项目的模块和函数
from dbgpt._private.pydantic import BaseModel, ConfigDict, Field, model_validator
from dbgpt.util.configure import ConfigInfo, DynConfig

# 定义有效的模板关键字集合
VALID_TEMPLATE_KEYS = {
    "role",
    "name",
    "goal",
    "resource_prompt",
    "expand_prompt",
    "language",
    "constraints",
    "examples",
    "out_schema",
    "most_recent_memories",
    "question",
}

# 默认的系统提示模板
_DEFAULT_SYSTEM_TEMPLATE = """\
You are a {{ role }}, {% if name %}named {{ name }}, {% endif %}your goal is {{ goal }}.
Please think step by step to achieve the goal. You can use the resources given below. 
At the same time, please strictly abide by the constraints and specifications in IMPORTANT REMINDER.
{% if resource_prompt %}\
{{ resource_prompt }} 
{% endif %}\
{% if expand_prompt %}\
{{ expand_prompt }} 
{% endif %}\

*** IMPORTANT REMINDER ***
{% if language == 'zh' %}\
Please answer in simplified Chinese.
{% else %}\
Please answer in English.
{% endif %}\

{% if constraints %}\
{% for constraint in constraints %}\
{{ loop.index }}. {{ constraint }}
{% endfor %}\
{% endif %}\

{% if examples %}\
You can refer to the following examples:
{{ examples }}\
{% endif %}\

{% if out_schema %} {{ out_schema }} {% endif %}\
"""  # noqa

# 默认的用户提示模板
_DEFAULT_USER_TEMPLATE = """\
{% if most_recent_memories %}\
Most recent observations:
{{ most_recent_memories }}
{% endif %}\

{% if question %}\
Question: {{ question }}
{% endif %}
"""

# 默认的写入记忆提示模板
_DEFAULT_WRITE_MEMORY_TEMPLATE = """\
{% if question %}Question: {{ question }} {% endif %}
{% if thought %}Thought: {{ thought }} {% endif %}
{% if action %}Action: {{ action }} {% endif %}
"""


class Profile(ABC):
    """Profile interface."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of current agent."""

    @abstractmethod
    def get_role(self) -> str:
        """Return the role of current agent."""

    def get_goal(self) -> Optional[str]:
        """Return the goal of current agent."""
        return None

    def get_constraints(self) -> Optional[List[str]]:
        """Return the constraints of current agent."""
        return None

    def get_description(self) -> Optional[str]:
        """Return the description of current agent.

        It will not be used to generate prompt.
        """
        return None

    def get_expand_prompt(self) -> Optional[str]:
        """Return the expand prompt of current agent."""
        return None

    def get_examples(self) -> Optional[str]:
        """Return the examples of current agent."""
        return None

    @abstractmethod
    def get_system_prompt_template(self) -> str:
        """Return the prompt template of current agent."""

    @abstractmethod
    def get_user_prompt_template(self) -> str:
        """Return the user prompt template of current agent."""
        # 返回当前代理的用户提示模板字符串
        pass  # 这里使用 pass 表示函数体为空，暂不执行任何操作

    @abstractmethod
    def get_write_memory_template(self) -> str:
        """Return the save memory template of current agent."""
        # 抽象方法，子类需实现，返回当前代理的保存记忆模板字符串
        pass

    def format_system_prompt(
        self,
        template_env: Optional[Environment] = None,
        question: Optional[str] = None,
        language: str = "en",
        most_recent_memories: Optional[str] = None,
        resource_vars: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Format the system prompt.

        Args:
            template_env(Optional[Environment]): The template environment for jinja2.
            question(Optional[str]): The question.
            language(str): The language of current context.
            most_recent_memories(Optional[str]): The most recent memories, it reads
                from memory.
            resource_vars(Optional[Dict[str, Any]]): The resource variables.

        Returns:
            str: The formatted system prompt.
        """
        return self._format_prompt(
            self.get_system_prompt_template(),
            template_env=template_env,
            question=question,
            language=language,
            most_recent_memories=most_recent_memories,
            resource_vars=resource_vars,
            **kwargs
        )
        # 格式化系统提示信息，并返回格式化后的字符串

    def format_user_prompt(
        self,
        template_env: Optional[Environment] = None,
        question: Optional[str] = None,
        language: str = "en",
        most_recent_memories: Optional[str] = None,
        resource_vars: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Format the user prompt.

        Args:
            template_env(Optional[Environment]): The template environment for jinja2.
            question(Optional[str]): The question.
            language(str): The language of current context.
            most_recent_memories(Optional[str]): The most recent memories, it reads
                from memory.
            resource_vars(Optional[Dict[str, Any]]): The resource variables.

        Returns:
            str: The formatted user prompt.
        """
        return self._format_prompt(
            self.get_user_prompt_template(),
            template_env=template_env,
            question=question,
            language=language,
            most_recent_memories=most_recent_memories,
            resource_vars=resource_vars,
            **kwargs
        )
        # 格式化用户提示信息，并返回格式化后的字符串

    @property
    def _sub_render_keys(self) -> Set[str]:
        """Return the sub render keys.

        If the value is a string and the key is in the sub render keys, it will be
            rendered.

        Returns:
            Set[str]: The sub render keys.
        """
        return {"role", "name", "goal", "expand_prompt", "constraints"}
        # 返回子渲染键的集合，这些键在渲染时会被处理
    def _format_prompt(
        self,
        template: str,
        template_env: Optional[Environment] = None,
        question: Optional[str] = None,
        language: str = "en",
        most_recent_memories: Optional[str] = None,
        resource_vars: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Format the prompt."""
        # 如果未提供模板环境，使用默认的沙盒环境
        if not template_env:
            template_env = SandboxedEnvironment()
        
        # 准备要传递给模板的变量字典
        pass_vars = {
            "role": self.get_role(),
            "name": self.get_name(),
            "goal": self.get_goal(),
            "expand_prompt": self.get_expand_prompt(),
            "language": language,
            "constraints": self.get_constraints(),
            "most_recent_memories": (
                most_recent_memories if most_recent_memories else None
            ),
            "examples": self.get_examples(),
            "question": question,
        }
        
        # 如果提供了额外的资源变量，将其合并到 pass_vars 中
        if resource_vars:
            # 合并资源变量
            pass_vars.update(resource_vars)
        
        # 将 kwargs 中的变量也合并到 pass_vars 中
        pass_vars.update(kwargs)

        # 解析模板，查找模板中所有的变量
        template_parsed = template_env.parse(template)
        template_vars = find_undeclared_variables(template_parsed)
        
        # 保留在 pass_vars 中存在的有效模板变量
        filtered_data = {
            key: pass_vars[key] for key in template_vars if key in pass_vars
        }

        def _render_template(_template_env, _template: str, **_kwargs):
            # 使用模板环境 _template_env 渲染模板 _template，并传递关键字参数 _kwargs
            r_template = _template_env.from_string(_template)
            return r_template.render(**_kwargs)

        # 遍历过滤后的数据字典
        for key in filtered_data.keys():
            value = filtered_data[key]
            # 如果值需要进行子模板渲染并且值不为空
            if key in self._sub_render_keys and value:
                if isinstance(value, str):
                    # 渲染子模板
                    filtered_data[key] = _render_template(
                        template_env, value, **pass_vars
                    )
                elif isinstance(value, list):
                    # 如果值是列表，逐个渲染列表中的字符串元素
                    for i, item in enumerate(value):
                        if isinstance(item, str):
                            value[i] = _render_template(template_env, item, **pass_vars)
        
        # 最终使用模板环境渲染主模板 template，并传递过滤后的数据字典 filtered_data
        return _render_template(template_env, template, **filtered_data)
class DefaultProfile(BaseModel, Profile):
    """Default profile."""

    # 代理的名称
    name: str = Field("", description="The name of the agent.")
    # 代理的角色
    role: str = Field("", description="The role of the agent.")
    # 代理的目标（可选）
    goal: Optional[str] = Field(None, description="The goal of the agent.")
    # 代理的约束条件（可选，以列表形式）
    constraints: Optional[List[str]] = Field(
        None, description="The constraints of the agent."
    )

    # 代理的描述（可选，不用于生成提示）
    desc: Optional[str] = Field(
        None, description="The description of the agent, not used to generate prompt."
    )

    # 代理的扩展提示（可选）
    expand_prompt: Optional[str] = Field(
        None, description="The expand prompt of the agent."
    )

    # 代理的示例（可选）
    examples: Optional[str] = Field(
        None, description="The examples of the agent prompt."
    )

    # 代理的系统提示模板
    system_prompt_template: str = Field(
        _DEFAULT_SYSTEM_TEMPLATE, description="The system prompt template of the agent."
    )
    
    # 代理的用户提示模板
    user_prompt_template: str = Field(
        _DEFAULT_USER_TEMPLATE, description="The user prompt template of the agent."
    )

    # 代理的保存内存模板
    write_memory_template: str = Field(
        _DEFAULT_WRITE_MEMORY_TEMPLATE,
        description="The save memory template of the agent.",
    )

    def get_name(self) -> str:
        """Return the name of current agent."""
        return self.name

    def get_role(self) -> str:
        """Return the role of current agent."""
        return self.role

    def get_goal(self) -> Optional[str]:
        """Return the goal of current agent."""
        return self.goal

    def get_constraints(self) -> Optional[List[str]]:
        """Return the constraints of current agent."""
        return self.constraints

    def get_description(self) -> Optional[str]:
        """Return the description of current agent.

        It will not be used to generate prompt.
        """
        return self.desc

    def get_expand_prompt(self) -> Optional[str]:
        """Return the expand prompt of current agent."""
        return self.expand_prompt

    def get_examples(self) -> Optional[str]:
        """Return the examples of current agent."""
        return self.examples

    def get_system_prompt_template(self) -> str:
        """Return the prompt template of current agent."""
        return self.system_prompt_template

    def get_user_prompt_template(self) -> str:
        """Return the user prompt template of current agent."""
        return self.user_prompt_template

    def get_write_memory_template(self) -> str:
        """Return the save memory template of current agent."""
        return self.write_memory_template


class ProfileFactory:
    """Profile factory interface.

    It is used to create a profile.
    """

    @abstractmethod
    def create_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
    ) -> Optional[Profile]:
        """Create a profile."""
class LLMProfileFactory(ProfileFactory):
    """Create a profile by LLM.

    Based on LLM automatic generation, it usually specifies the rules of the generation
     configuration first, clarifies the composition and attributes of the agent
     configuration in the target population, and then gives a small number of samples,
    and finally LLM generates the configuration of all agents.
    """

    def create_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
    ) -> Optional[Profile]:
        """Create a profile by LLM.

        TODO: Implement this method.
        """
        pass


class DatasetProfileFactory(ProfileFactory):
    """Create a profile by dataset.

    Use existing data sets to generate agent configurations.

    In some cases, the data set contains a large amount of information about real people
    , first organize the information about real people in the data set into a natural
    language prompt, which is then used to generate the agent configuration.
    """

    def create_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
    ) -> Optional[Profile]:
        """Create a profile by dataset.

        TODO: Implement this method.
        """
        pass


class CompositeProfileFactory(ProfileFactory):
    """Create a profile by combining multiple profile factories."""

    def __init__(self, factories: List[ProfileFactory]):
        """Create a composite profile factory."""
        self.factories = factories

    def create_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
    ) -> Optional[Profile]:
        """Create a profile by combining multiple profile factories.

        TODO: Implement this method.
        """
        pass


class ProfileConfig(BaseModel):
    """Profile configuration.

    If factory is not specified, name and role must be specified.
    If factory is specified and name and role are also specified, the factory will be
    preferred.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile_id: int = Field(0, description="The profile ID.")
    name: str | ConfigInfo | None = DynConfig(..., description="The name of the agent.")
    role: str | ConfigInfo | None = DynConfig(..., description="The role of the agent.")
    goal: str | ConfigInfo | None = DynConfig(None, description="The goal.")
    constraints: List[str] | ConfigInfo | None = DynConfig(None, is_list=True)
    # 描述字段，可以是字符串、ConfigInfo 或 None，默认为 DynConfig 对象
    desc: str | ConfigInfo | None = DynConfig(
        None, description="The description of the agent."
    )
    # 扩展提示字段，可以是字符串、ConfigInfo 或 None，默认为 DynConfig 对象
    expand_prompt: str | ConfigInfo | None = DynConfig(
        None, description="The expand prompt."
    )
    # 示例字段，可以是字符串、ConfigInfo 或 None，默认为 DynConfig 对象
    examples: str | ConfigInfo | None = DynConfig(None, description="The examples.")

    # 系统提示模板字段，可以是字符串、ConfigInfo 或 None，默认为指定的系统提示模板
    system_prompt_template: str | ConfigInfo | None = DynConfig(
        _DEFAULT_SYSTEM_TEMPLATE, description="The prompt template."
    )
    # 用户提示模板字段，可以是字符串、ConfigInfo 或 None，默认为指定的用户提示模板
    user_prompt_template: str | ConfigInfo | None = DynConfig(
        _DEFAULT_USER_TEMPLATE, description="The user prompt template."
    )
    # 写入记忆模板字段，可以是字符串、ConfigInfo 或 None，默认为指定的写入记忆模板
    write_memory_template: str | ConfigInfo | None = DynConfig(
        _DEFAULT_WRITE_MEMORY_TEMPLATE, description="The save memory template."
    )
    # 配置文件工厂字段，可以是 ProfileFactory 对象或 None，默认为 None
    factory: ProfileFactory | None = Field(None, description="The profile factory.")

    # 类方法，用于在验证前检查值的有效性
    @model_validator(mode="before")
    @classmethod
    def check_before(cls, values):
        """Check before validation."""
        # 如果 values 是字典类型，则直接返回
        if isinstance(values, dict):
            return values
        # 如果 factory 字段为 None，则进行进一步的检查
        if values["factory"] is None:
            # 如果 name 字段也为 None，则抛出数值错误
            if values["name"] is None:
                raise ValueError("name must be specified if factory is not specified")
            # 如果 role 字段也为 None，则抛出数值错误
            if values["role"] is None:
                raise ValueError("role must be specified if factory is not specified")
        # 返回检查后的值
        return values

    # 使用 cachetools 库提供的 TTLCache 进行缓存，缓存有效期为 10 秒，最大缓存大小为 100
    @cachetools.cached(cachetools.TTLCache(maxsize=100, ttl=10))
    def create_profile(
        self,
        profile_id: Optional[int] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
    ) -> Profile:
        """Create a profile.

        If factory is specified, use the factory to create the profile.
        """
        # 初始化 factory_profile 变量为 None
        factory_profile = None
        # 如果未指定 profile_id，则使用 self.profile_id
        if profile_id is None:
            profile_id = self.profile_id
        # 将各个属性值赋给局部变量
        name = self.name
        role = self.role
        goal = self.goal
        constraints = self.constraints
        desc = self.desc
        expand_prompt = self.expand_prompt
        system_prompt_template = self.system_prompt_template
        user_prompt_template = self.user_prompt_template
        write_memory_template = self.write_memory_template
        examples = self.examples
        # 构建查询参数字典
        call_args = {
            "prefer_prompt_language": prefer_prompt_language,
            "prefer_model": prefer_model,
        }
        # 如果 name 是 ConfigInfo 实例，则调用 query 方法查询属性值
        if isinstance(name, ConfigInfo):
            name = name.query(**call_args)
        # 类似地处理其余属性
        if isinstance(role, ConfigInfo):
            role = role.query(**call_args)
        if isinstance(goal, ConfigInfo):
            goal = goal.query(**call_args)
        if isinstance(constraints, ConfigInfo):
            constraints = constraints.query(**call_args)
        if isinstance(desc, ConfigInfo):
            desc = desc.query(**call_args)
        if isinstance(expand_prompt, ConfigInfo):
            expand_prompt = expand_prompt.query(**call_args)
        if isinstance(examples, ConfigInfo):
            examples = examples.query(**call_args)
        if isinstance(system_prompt_template, ConfigInfo):
            system_prompt_template = system_prompt_template.query(**call_args)
        if isinstance(user_prompt_template, ConfigInfo):
            user_prompt_template = user_prompt_template.query(**call_args)
        if isinstance(write_memory_template, ConfigInfo):
            write_memory_template = write_memory_template.query(**call_args)

        # 如果存在 factory，则使用 factory 创建 profile
        if self.factory is not None:
            factory_profile = self.factory.create_profile(
                profile_id,
                name,
                role,
                goal,
                prefer_prompt_language,
                prefer_model,
            )

        # 如果 factory_profile 不为 None，则返回 factory_profile
        if factory_profile is not None:
            return factory_profile
        # 否则，创建并返回 DefaultProfile 对象
        return DefaultProfile(
            name=name,
            role=role,
            goal=goal,
            constraints=constraints,
            desc=desc,
            expand_prompt=expand_prompt,
            examples=examples,
            system_prompt_template=system_prompt_template,
            user_prompt_template=user_prompt_template,
            write_memory_template=write_memory_template,
        )

    def __hash__(self):
        """Return the hash value."""
        # 返回对象的哈希值
        return hash(self.profile_id)
```