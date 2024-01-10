# `MetaGPT\metagpt\tools\openapi_v3_hello.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/2 16:03
@Author  : mashenquan
@File    : openapi_v3_hello.py
@Desc    : Implement the OpenAPI Specification 3.0 demo and use the following command to test the HTTP service:

        curl -X 'POST' \
        'http://localhost:8082/openapi/greeting/dave' \
        -H 'accept: text/plain' \
        -H 'Content-Type: application/json' \
        -d '{}'
"""
# 导入 Path 模块
from pathlib import Path
# 导入 connexion 模块
import connexion

# 定义异步函数，实现 OpenAPI 规范的问候功能
async def post_greeting(name: str) -> str:
    return f"Hello {name}\n"

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 获取规范文件所在目录
    specification_dir = Path(__file__).parent.parent.parent / "docs/.well-known"
    # 创建异步应用对象
    app = connexion.AsyncApp(__name__, specification_dir=str(specification_dir))
    # 添加 OpenAPI 规范文件，并设置标题
    app.add_api("openapi.yaml", arguments={"title": "Hello World Example"})
    # 运行应用，监听 8082 端口
    app.run(port=8082)

```