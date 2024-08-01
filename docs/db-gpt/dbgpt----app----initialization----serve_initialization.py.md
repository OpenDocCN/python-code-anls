# `.\DB-GPT-src\dbgpt\app\initialization\serve_initialization.py`

```py
# 从 dbgpt._private.config 模块中导入 Config 类
# 从 dbgpt.component 模块中导入 SystemApp 类
from dbgpt._private.config import Config
from dbgpt.component import SystemApp


def register_serve_apps(system_app: SystemApp, cfg: Config):
    """Register serve apps"""
    # 设置系统应用的全局语言配置
    system_app.config.set("dbgpt.app.global.language", cfg.LANGUAGE)
    # 如果存在 API_KEYS，则设置系统应用的全局 API_KEYS 配置
    if cfg.API_KEYS:
        system_app.config.set("dbgpt.app.global.api_keys", cfg.API_KEYS)

    # ################################ Prompt Serve Register Begin ######################################
    # 从 dbgpt.serve.prompt.serve 模块中导入 SERVE_CONFIG_KEY_PREFIX 和 Serve 类
    from dbgpt.serve.prompt.serve import (
        SERVE_CONFIG_KEY_PREFIX as PROMPT_SERVE_CONFIG_KEY_PREFIX,
    )
    from dbgpt.serve.prompt.serve import Serve as PromptServe

    # 替换旧的 prompt serve
    # 设置配置信息
    system_app.config.set(f"{PROMPT_SERVE_CONFIG_KEY_PREFIX}default_user", "dbgpt")
    system_app.config.set(f"{PROMPT_SERVE_CONFIG_KEY_PREFIX}default_sys_code", "dbgpt")
    # 注册 prompt serve 应用，设置 API 前缀为 '/prompt'
    system_app.register(PromptServe, api_prefix="/prompt")
    # ################################ Prompt Serve Register End ########################################

    # ################################ Conversation Serve Register Begin ######################################
    # 从 dbgpt.serve.conversation.serve 模块中导入 SERVE_CONFIG_KEY_PREFIX 和 Serve 类
    from dbgpt.serve.conversation.serve import (
        SERVE_CONFIG_KEY_PREFIX as CONVERSATION_SERVE_CONFIG_KEY_PREFIX,
    )
    from dbgpt.serve.conversation.serve import Serve as ConversationServe

    # 设置默认的对话模型
    system_app.config.set(
        f"{CONVERSATION_SERVE_CONFIG_KEY_PREFIX}default_model", cfg.LLM_MODEL
    )
    # 注册对话 serve 应用，设置 API 前缀为 '/api/v1/chat/dialogue'
    system_app.register(ConversationServe, api_prefix="/api/v1/chat/dialogue")
    # ################################ Conversation Serve Register End ########################################

    # ################################ AWEL Flow Serve Register Begin ######################################
    # 从 dbgpt.serve.flow.serve 模块中导入 SERVE_CONFIG_KEY_PREFIX 和 Serve 类
    from dbgpt.serve.flow.serve import (
        SERVE_CONFIG_KEY_PREFIX as FLOW_SERVE_CONFIG_KEY_PREFIX,
    )
    from dbgpt.serve.flow.serve import Serve as FlowServe

    # 注册流程 serve 应用
    system_app.register(FlowServe)
    # ################################ AWEL Flow Serve Register End ########################################

    # ################################ Rag Serve Register Begin ######################################
    # 从 dbgpt.serve.rag.serve 模块中导入 SERVE_CONFIG_KEY_PREFIX 和 Serve 类
    from dbgpt.serve.rag.serve import (
        SERVE_CONFIG_KEY_PREFIX as RAG_SERVE_CONFIG_KEY_PREFIX,
    )
    from dbgpt.serve.rag.serve import Serve as RagServe

    # 注册 RAG serve 应用
    system_app.register(RagServe)
    # ################################ Rag Serve Register End ########################################

    # ################################ Datasource Serve Register Begin ######################################
    # 从 dbgpt.serve.datasource.serve 模块中导入 SERVE_CONFIG_KEY_PREFIX 和 Serve 类
    from dbgpt.serve.datasource.serve import (
        SERVE_CONFIG_KEY_PREFIX as DATASOURCE_SERVE_CONFIG_KEY_PREFIX,
    )
    from dbgpt.serve.datasource.serve import Serve as DatasourceServe

    # 注册数据源 serve 应用
    system_app.register(DatasourceServe)
    # ################################ Datasource Serve Register End ########################################
```