# `.\DB-GPT-src\dbgpt\app\llmserver.py`

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入标准库模块
import os
import sys

# 导入自定义模块和配置
from dbgpt._private.config import Config
from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG, LLM_MODEL_CONFIG
from dbgpt.model.cluster import run_worker_manager

# 设置根路径为当前脚本的上上上级目录
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

# 加载配置对象
CFG = Config()

# 获取语言模型的路径
model_path = LLM_MODEL_CONFIG.get(CFG.LLM_MODEL)

if __name__ == "__main__":
    """run llm server including controller, manager worker
    If you use gunicorn as a process manager, initialize_app can be invoke in `on_starting` hook.
    """
    # 运行集群管理器，包括控制器和工作节点
    run_worker_manager(
        model_name=CFG.LLM_MODEL,                    # 设置语言模型的名称
        model_path=model_path,                       # 设置语言模型的路径
        standalone=True,                             # 指示以独立模式运行
        port=CFG.MODEL_PORT,                         # 设置模型服务的端口号
        embedding_model_name=CFG.EMBEDDING_MODEL,    # 设置嵌入模型的名称
        embedding_model_path=EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL],  # 设置嵌入模型的路径
    )
```