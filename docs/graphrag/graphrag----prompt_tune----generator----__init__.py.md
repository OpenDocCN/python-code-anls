# `.\graphrag\graphrag\prompt_tune\generator\__init__.py`

```py
# 引入模块：从当前包中导入各个功能模块和变量
from .community_report_rating import generate_community_report_rating
from .community_report_summarization import create_community_summarization_prompt
from .community_reporter_role import generate_community_reporter_role
from .defaults import MAX_TOKEN_COUNT
from .domain import generate_domain
from .entity_extraction_prompt import create_entity_extraction_prompt
from .entity_relationship import generate_entity_relationship_examples
from .entity_summarization_prompt import create_entity_summarization_prompt
from .entity_types import generate_entity_types
from .language import detect_language
from .persona import generate_persona

# 定义 __all__ 列表，指定了模块中可以被导入的公共接口
__all__ = [
    "MAX_TOKEN_COUNT",  # 最大令牌数量
    "create_community_summarization_prompt",  # 创建社区总结提示
    "create_entity_extraction_prompt",  # 创建实体提取提示
    "create_entity_summarization_prompt",  # 创建实体总结提示
    "detect_language",  # 检测语言
    "generate_community_report_rating",  # 生成社区报告评分
    "generate_community_reporter_role",  # 生成社区报告员角色
    "generate_domain",  # 生成领域
    "generate_entity_relationship_examples",  # 生成实体关系示例
    "generate_entity_types",  # 生成实体类型
    "generate_persona",  # 生成人物角色
]
```