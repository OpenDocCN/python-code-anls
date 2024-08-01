# `.\DB-GPT-src\dbgpt\app\initialization\db_model_initialization.py`

```py
"""
导入所有模型以确保它们已经在SQLAlchemy中注册。
"""

from dbgpt.app.knowledge.chunk_db import DocumentChunkEntity  # 导入文档分块实体模型
from dbgpt.app.knowledge.document_db import KnowledgeDocumentEntity  # 导入知识文档实体模型
from dbgpt.app.openapi.api_v1.feedback.feed_back_db import ChatFeedBackEntity  # 导入聊天反馈实体模型
from dbgpt.datasource.manages.connect_config_db import ConnectConfigEntity  # 导入连接配置实体模型
from dbgpt.model.cluster.registry_impl.db_storage import ModelInstanceEntity  # 导入模型实例实体模型
from dbgpt.serve.agent.db.my_plugin_db import MyPluginEntity  # 导入自定义插件实体模型
from dbgpt.serve.agent.db.plugin_hub_db import PluginHubEntity  # 导入插件中心实体模型
from dbgpt.serve.flow.models.models import ServeEntity as FlowServeEntity  # 导入流服务实体模型并命名为FlowServeEntity
from dbgpt.serve.prompt.models.models import ServeEntity as PromptManageEntity  # 导入提示管理实体模型并命名为PromptManageEntity
from dbgpt.serve.rag.models.models import KnowledgeSpaceEntity  # 导入知识空间实体模型
from dbgpt.storage.chat_history.chat_history_db import (  # 导入聊天历史和聊天历史消息实体模型
    ChatHistoryEntity,
    ChatHistoryMessageEntity,
)

_MODELS = [  # 创建模型列表
    PluginHubEntity,          # 插件中心实体模型
    MyPluginEntity,           # 自定义插件实体模型
    PromptManageEntity,       # 提示管理实体模型
    KnowledgeSpaceEntity,     # 知识空间实体模型
    KnowledgeDocumentEntity,  # 知识文档实体模型
    DocumentChunkEntity,      # 文档分块实体模型
    ChatFeedBackEntity,       # 聊天反馈实体模型
    ConnectConfigEntity,      # 连接配置实体模型
    ChatHistoryEntity,        # 聊天历史实体模型
    ChatHistoryMessageEntity, # 聊天历史消息实体模型
    ModelInstanceEntity,      # 模型实例实体模型
    FlowServeEntity,          # 流服务实体模型
]
```