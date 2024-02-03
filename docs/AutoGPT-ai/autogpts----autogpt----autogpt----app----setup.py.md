# `.\AutoGPT\autogpts\autogpt\autogpt\app\setup.py`

```py
"""Set up the AI and its goals"""
# 导入所需的模块和类型提示
import logging
from typing import Optional

# 导入自定义模块和配置
from autogpt.app.utils import clean_input
from autogpt.config import AIDirectives, AIProfile, Config
from autogpt.logs.helpers import print_attribute

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


def apply_overrides_to_ai_settings(
    ai_profile: AIProfile,
    directives: AIDirectives,
    override_name: Optional[str] = "",
    override_role: Optional[str] = "",
    replace_directives: bool = False,
    resources: Optional[list[str]] = None,
    constraints: Optional[list[str]] = None,
    best_practices: Optional[list[str]] = None,
):
    # 如果有覆盖的 AI 名称，则更新 AIProfile 中的 ai_name
    if override_name:
        ai_profile.ai_name = override_name
    # 如果有覆盖的 AI 角色，则更新 AIProfile 中的 ai_role
    if override_role:
        ai_profile.ai_role = override_role

    # 根据 replace_directives 的值来更新或追加 directives 中的资源、约束和最佳实践
    if replace_directives:
        if resources:
            directives.resources = resources
        if constraints:
            directives.constraints = constraints
        if best_practices:
            directives.best_practices = best_practices
    else:
        if resources:
            directives.resources += resources
        if constraints:
            directives.constraints += constraints
        if best_practices:
            directives.best_practices += best_practices


async def interactively_revise_ai_settings(
    ai_profile: AIProfile,
    directives: AIDirectives,
    app_config: Config,
):
    """Interactively revise the AI settings.

    Args:
        ai_profile (AIConfig): The current AI profile.
        ai_directives (AIDirectives): The current AI directives.
        app_config (Config): The application configuration.

    Returns:
        AIConfig: The revised AI settings.
    """
    # 获取特定的日志记录器
    logger = logging.getLogger("revise_ai_profile")

    # 初始化 revised 变量为 False
    revised = False

    # 返回更新后的 AI 设置
    return ai_profile, directives


def print_ai_settings(
    ai_profile: AIProfile,
    directives: AIDirectives,
    logger: logging.Logger,
    title: str = "AI Settings",
):
    # 打印 AI 设置的标题
    print_attribute(title, "")
    # 打印分隔线
    print_attribute("-" * len(title), "")
    # 打印 AI 名称属性
    print_attribute("Name :", ai_profile.ai_name)
    # 打印 AI 角色属性
    print_attribute("Role :", ai_profile.ai_role)

    # 打印约束属性，如果没有约束则打印 "(none)"
    print_attribute("Constraints:", "" if directives.constraints else "(none)")
    # 遍历并打印每个约束
    for constraint in directives.constraints:
        logger.info(f"- {constraint}")
    
    # 打印资源属性，如果没有资源则打印 "(none)"
    print_attribute("Resources:", "" if directives.resources else "(none)")
    # 遍历并打印每个资源
    for resource in directives.resources:
        logger.info(f"- {resource}")
    
    # 打印最佳实践属性，如果没有最佳实践则打印 "(none)"
    print_attribute("Best practices:", "" if directives.best_practices else "(none)")
    # 遍历并打印每个最佳实践
    for best_practice in directives.best_practices:
        logger.info(f"- {best_practice}")
```