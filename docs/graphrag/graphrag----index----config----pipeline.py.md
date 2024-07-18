# `.\graphrag\graphrag\index\config\pipeline.py`

```py
# 从未来导入注解功能，用于支持类型注解的提示
from __future__ import annotations

# 导入用于格式化输出的工具
from devtools import pformat
# 导入 Pydantic 中的基础模型
from pydantic import BaseModel
# 重命名 Pydantic 中的 Field 为 pydantic_Field
from pydantic import Field as pydantic_Field

# 导入其他配置类型的引用
from .cache import PipelineCacheConfigTypes
from .input import PipelineInputConfigTypes
from .reporting import PipelineReportingConfigTypes
from .storage import PipelineStorageConfigTypes
from .workflow import PipelineWorkflowReference

# 定义 PipelineConfig 类，继承自 Pydantic 的 BaseModel
class PipelineConfig(BaseModel):
    """Represent the configuration for a pipeline."""

    # 返回对象的字符串表示形式
    def __repr__(self) -> str:
        """Get a string representation."""
        return pformat(self, highlight=False)

    # 返回对象的字符串表示形式
    def __str__(self):
        """Get a string representation."""
        return str(self.model_dump_json(indent=4))

    # pipeline 配置可以继承另一个 pipeline 配置的列表或单个配置
    extends: list[str] | str | None = pydantic_Field(
        description="Extends another pipeline configuration", default=None
    )
    """Extends another pipeline configuration"""

    # pipeline 的输入配置，可以是多种输入类型之一，或者为空
    input: PipelineInputConfigTypes | None = pydantic_Field(
        default=None, discriminator="file_type"
    )
    """The input configuration for the pipeline."""

    # pipeline 的报告配置，可以是多种报告类型之一，或者为空
    reporting: PipelineReportingConfigTypes | None = pydantic_Field(
        default=None, discriminator="type"
    )
    """The reporting configuration for the pipeline."""

    # pipeline 的存储配置，可以是多种存储类型之一，或者为空
    storage: PipelineStorageConfigTypes | None = pydantic_Field(
        default=None, discriminator="type"
    )
    """The storage configuration for the pipeline."""

    # pipeline 的缓存配置，可以是多种缓存类型之一，或者为空
    cache: PipelineCacheConfigTypes | None = pydantic_Field(
        default=None, discriminator="type"
    )
    """The cache configuration for the pipeline."""

    # pipeline 的根目录，所有其他路径都以此为基础
    root_dir: str | None = pydantic_Field(
        description="The root directory for the pipeline. All other paths will be based on this root_dir.",
        default=None,
    )
    """The root directory for the pipeline."""

    # pipeline 的工作流配置列表，默认为空列表
    workflows: list[PipelineWorkflowReference] = pydantic_Field(
        description="The workflows for the pipeline.", default_factory=list
    )
    """The workflows for the pipeline."""
```