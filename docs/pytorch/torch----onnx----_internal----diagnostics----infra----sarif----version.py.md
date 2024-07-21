# `.\pytorch\torch\onnx\_internal\diagnostics\infra\sarif\version.py`

```
# 引入 typing 模块中的 Final 类型提示符，用于声明常量
from typing import Final

# 声明并初始化 SARIF_VERSION 常量，表示 SARIF 格式的版本号为 "2.1.0"
SARIF_VERSION: Final = "2.1.0"

# 声明并初始化 SARIF_SCHEMA_LINK 常量，指向 SARIF 格式的 JSON Schema 文档链接
SARIF_SCHEMA_LINK: Final = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/cs01/schemas/sarif-schema-2.1.0.json"

# 添加 flake8: noqa 注释，用于告知 flake8 工具忽略该文件中的所有错误和警告
# 这通常用于跳过代码检查，例如避免因过长的行导致的警告
# 注意：这里的注释并不是 Python 代码的一部分，而是用于指导代码检查工具
```