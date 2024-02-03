# `stable-diffusion-webui\webui.py`

```
# 导入未来的注解特性
from __future__ import annotations

# 导入操作系统和时间模块
import os
import time

# 从模块中导入计时器
from modules import timer
# 从模块中导入初始化工具
from modules import initialize_util
# 从模块中导入初始化
from modules import initialize

# 启动计时器记录启动时间
startup_timer = timer.startup_timer
startup_timer.record("launcher")

# 导入模块
initialize.imports()

# 检查版本
initialize.check_versions()

# 创建 API
def create_api(app):
    # 从 API 模块中导入 Api 类
    from modules.api.api import Api
    # 从调用队列模块中导入队列锁
    from modules.call_queue import queue_lock

    # 创建 Api 实例
    api = Api(app, queue_lock)
    return api

# 仅 API
def api_only():
    # 从 FastAPI 模块中导入 FastAPI 类
    from fastapi import FastAPI
    # 从共享命令选项模块中导入命令选项
    from modules.shared_cmd_options import cmd_opts

    # 初始化
    initialize.initialize()

    # 创建 FastAPI 实例
    app = FastAPI()
    # 设置中间件
    initialize_util.setup_middleware(app)
    # 创建 API
    api = create_api(app)

    # 从脚本回调模块中导入回调函数
    from modules import script_callbacks
    # 在 UI 之前执行回调函数
    script_callbacks.before_ui_callback()
    # 应用启动回调函数
    script_callbacks.app_started_callback(None, app)

    # 打印启动时间
    print(f"Startup time: {startup_timer.summary()}.")
    # 启动 API
    api.launch(
        server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1",
        port=cmd_opts.port if cmd_opts.port else 7861,
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else ""
    )

# Web UI
def webui():
    # 从共享命令选项模块中导入命令选项
    from modules.shared_cmd_options import cmd_opts

    # 是否启动 API
    launch_api = cmd_opts.api
    # 初始化
    initialize.initialize()

    # 导入共享、UI 临时目录、脚本回调、UI、进度、额外网络模块

if __name__ == "__main__":
    # 从共享命令选项模块中导入命令选项
    from modules.shared_cmd_options import cmd_opts

    # 如果不启动 Web UI，则仅启动 API
    if cmd_opts.nowebui:
        api_only()
    else:
        webui()
```