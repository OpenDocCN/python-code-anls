# `.\graphrag\graphrag\config\models\cluster_graph_config.py`

```py
# 版权声明，版权归 2024 年 Microsoft 公司所有，基于 MIT 许可证发布

"""默认配置的参数设置。"""

# 导入必要的模块和库
from pydantic import BaseModel, Field  # 导入 Pydantic 库中的基类和字段描述符
import graphrag.config.defaults as defs  # 导入 graphrag 库中的默认配置

# 定义用于聚类图的配置部分
class ClusterGraphConfig(BaseModel):
    """聚类图的配置部分。"""

    max_cluster_size: int = Field(
        description="要使用的最大聚类大小。", default=defs.MAX_CLUSTER_SIZE
    )
    strategy: dict | None = Field(
        description="要使用的聚类策略。", default=None
    )

    def resolved_strategy(self) -> dict:
        """获取已解析的聚类策略。"""
        from graphrag.index.verbs.graph.clustering import GraphCommunityStrategyType

        # 如果未指定策略，则使用默认的 Leiden 策略和指定的最大聚类大小
        return self.strategy or {
            "type": GraphCommunityStrategyType.leiden,
            "max_cluster_size": self.max_cluster_size,
        }
```