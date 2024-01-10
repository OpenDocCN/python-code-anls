# `MetaGPT\metagpt\tools\metagpt_oas3_api_svc.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/17
@Author  : mashenquan
@File    : metagpt_oas3_api_svc.py
@Desc    : MetaGPT OpenAPI Specification 3.0 REST API service

        curl -X 'POST' \
        'http://localhost:8080/openapi/greeting/dave' \
        -H 'accept: text/plain' \
        -H 'Content-Type: application/json' \
        -d '{}'
"""

from pathlib import Path  # 导入 Path 模块

import connexion  # 导入 connexion 模块


def oas_http_svc():  # 定义函数 oas_http_svc
    """Start the OAS 3.0 OpenAPI HTTP service"""  # 函数说明
    print("http://localhost:8080/oas3/ui/")  # 打印服务地址
    specification_dir = Path(__file__).parent.parent.parent / "docs/.well-known"  # 获取规范文件目录
    app = connexion.AsyncApp(__name__, specification_dir=str(specification_dir))  # 创建异步应用对象
    app.add_api("metagpt_oas3_api.yaml")  # 添加 MetaGPT OpenAPI 规范文件
    app.add_api("openapi.yaml")  # 添加 openapi 规范文件
    app.run(port=8080)  # 运行应用，监听端口 8080


if __name__ == "__main__":  # 如果当前脚本被直接执行
    oas_http_svc()  # 调用 oas_http_svc 函数

```