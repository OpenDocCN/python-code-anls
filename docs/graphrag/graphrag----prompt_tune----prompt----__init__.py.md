# `.\graphrag\graphrag\prompt_tune\prompt\__init__.py`

```py
# 以下是一个用于生成提示和域的模块，用于特定实体类型、关系及其生成。

# 导入从其他模块引用的生成报告评分提示
from .community_report_rating import GENERATE_REPORT_RATING_PROMPT
# 导入从其他模块引用的生成社区报告者角色提示
from .community_reporter_role import GENERATE_COMMUNITY_REPORTER_ROLE_PROMPT
# 导入从其他模块引用的生成域提示
from .domain import GENERATE_DOMAIN_PROMPT
# 导入从其他模块引用的实体关系生成的 JSON 格式提示
from .entity_relationship import (
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT,
    # 导入从其他模块引用的实体关系生成提示
    ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
    # 导入从其他模块引用的未分类实体关系生成提示
    UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
)
# 导入从其他模块引用的实体类型生成的 JSON 格式提示
from .entity_types import (
    ENTITY_TYPE_GENERATION_JSON_PROMPT,
    # 导入从其他模块引用的实体类型生成提示
    ENTITY_TYPE_GENERATION_PROMPT,
)
# 导入从其他模块引用的语言检测提示
from .language import DETECT_LANGUAGE_PROMPT
# 导入从其他模块引用的生成个人形象提示
from .persona import GENERATE_PERSONA_PROMPT

# 模块中可以公开的所有符号列表
__all__ = [
    "DETECT_LANGUAGE_PROMPT",
    "ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT",
    "ENTITY_RELATIONSHIPS_GENERATION_PROMPT",
    "ENTITY_TYPE_GENERATION_JSON_PROMPT",
    "ENTITY_TYPE_GENERATION_PROMPT",
    "GENERATE_COMMUNITY_REPORTER_ROLE_PROMPT",
    "GENERATE_DOMAIN_PROMPT",
    "GENERATE_PERSONA_PROMPT",
    "GENERATE_REPORT_RATING_PROMPT",
    "UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT",
]
```