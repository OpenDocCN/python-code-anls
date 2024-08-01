# `.\DB-GPT-src\dbgpt\serve\core\__init__.py`

```py
# 从dbgpt.serve.core.config模块导入BaseServeConfig类
from dbgpt.serve.core.config import BaseServeConfig
# 从dbgpt.serve.core.schemas模块导入Result类和add_exception_handler函数
from dbgpt.serve.core.schemas import Result, add_exception_handler
# 从dbgpt.serve.core.serve模块导入BaseServe类
from dbgpt.serve.core.serve import BaseServe
# 从dbgpt.serve.core.service模块导入BaseService类
from dbgpt.serve.core.service import BaseService

# 定义一个包含导入类和函数名称的列表，用于指示模块的公共接口
__ALL__ = [
    "Result",  # 结果类
    "add_exception_handler",  # 添加异常处理函数
    "BaseServeConfig",  # 基础服务配置类
    "BaseService",  # 基础服务类
    "BaseServe",  # 基础服务类
]
```