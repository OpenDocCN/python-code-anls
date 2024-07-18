# `.\graphrag\graphrag\config\input_models\umap_config_input.py`

```py
# 版权声明，指出版权归 Microsoft Corporation 所有，遵循 MIT 许可证
# Licensed under the MIT License

# 导入必要的库：从 typing_extensions 中导入 NotRequired 和 TypedDict
from typing_extensions import NotRequired, TypedDict

# 定义一个新的类型 TypedDict，用于描述 UMAP 的配置参数
class UmapConfigInput(TypedDict):
    """Configuration section for UMAP."""
    
    # 定义一个字段 'enabled'，可以接受 bool、str 或 None 类型的值，且不是必需的
    enabled: NotRequired[bool | str | None]
```