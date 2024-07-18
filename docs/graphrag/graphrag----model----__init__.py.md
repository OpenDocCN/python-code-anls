# `.\graphrag\graphrag\model\__init__.py`

```py
# 导入模块中的特定类，这些类用于GraphRAG知识模型的数据表示和处理

# 导入Community类，用于表示社区
from .community import Community
# 导入CommunityReport类，用于表示社区报告
from .community_report import CommunityReport
# 导入Covariate类，用于表示协变量
from .covariate import Covariate
# 导入Document类，用于表示文档
from .document import Document
# 导入Entity类，用于表示实体
from .entity import Entity
# 导入Identified类，用于表示已识别的数据对象
from .identified import Identified
# 导入Named类，用于表示命名实体
from .named import Named
# 导入Relationship类，用于表示关系
from .relationship import Relationship
# 导入TextUnit类，用于表示文本单元
from .text_unit import TextUnit

# __all__列表定义了在使用"from module import *"时导入的符号，即限定了导出的公共接口
__all__ = [
    "Community",         # 社区类
    "CommunityReport",   # 社区报告类
    "Covariate",         # 协变量类
    "Document",          # 文档类
    "Entity",            # 实体类
    "Identified",        # 已识别数据对象类
    "Named",             # 命名实体类
    "Relationship",      # 关系类
    "TextUnit",          # 文本单元类
]
```