# `.\graphrag\graphrag\config\models\embed_graph_config.py`

```py
# 版权声明和许可证信息，指明此代码的版权归属及使用许可
# 来自默认配置的参数化设置。

# 引入 Pydantic 的 BaseModel 和 Field，用于定义数据模型和字段属性
from pydantic import BaseModel, Field

# 引入默认配置的定义
import graphrag.config.defaults as defs

# 定义 EmbedGraphConfig 类，继承自 Pydantic 的 BaseModel
class EmbedGraphConfig(BaseModel):
    """Node2Vec 的默认配置部分。"""

    # 是否启用 node2vec 的标志，默认值来自 defs.NODE2VEC_ENABLED
    enabled: bool = Field(
        description="A flag indicating whether to enable node2vec.",
        default=defs.NODE2VEC_ENABLED,
    )

    # node2vec 的步行次数，默认值来自 defs.NODE2VEC_NUM_WALKS
    num_walks: int = Field(
        description="The node2vec number of walks.", default=defs.NODE2VEC_NUM_WALKS
    )

    # node2vec 的步行长度，默认值来自 defs.NODE2VEC_WALK_LENGTH
    walk_length: int = Field(
        description="The node2vec walk length.", default=defs.NODE2VEC_WALK_LENGTH
    )

    # node2vec 的窗口大小，默认值来自 defs.NODE2VEC_WINDOW_SIZE
    window_size: int = Field(
        description="The node2vec window size.", default=defs.NODE2VEC_WINDOW_SIZE
    )

    # node2vec 的迭代次数，默认值来自 defs.NODE2VEC_ITERATIONS
    iterations: int = Field(
        description="The node2vec iterations.", default=defs.NODE2VEC_ITERATIONS
    )

    # node2vec 的随机种子，默认值来自 defs.NODE2VEC_RANDOM_SEED
    random_seed: int = Field(
        description="The node2vec random seed.", default=defs.NODE2VEC_RANDOM_SEED
    )

    # 图嵌入策略的覆盖，类型为 dict 或 None，默认值为 None
    strategy: dict | None = Field(
        description="The graph embedding strategy override.", default=None
    )

    # 解析并返回 node2vec 策略的函数
    def resolved_strategy(self) -> dict:
        """Get the resolved node2vec strategy."""
        # 从 graphrag.index.verbs.graph.embed 中引入 EmbedGraphStrategyType
        from graphrag.index.verbs.graph.embed import EmbedGraphStrategyType

        # 如果策略存在，则返回策略本身；否则返回一个包含 node2vec 默认参数的字典
        return self.strategy or {
            "type": EmbedGraphStrategyType.node2vec,
            "num_walks": self.num_walks,
            "walk_length": self.walk_length,
            "window_size": self.window_size,
            "iterations": self.iterations,
            "random_seed": self.iterations,  # 注意这里可能是一个错误，应为 self.random_seed
        }
```