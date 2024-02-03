# `.\AutoGPT\benchmark\agbenchmark\utils\get_data_from_helicone.py`

```py
# 导入所需的模块
import json
import logging
import os
from typing import Optional

import requests

# 从 agbenchmark.__main__ 模块中导入 BENCHMARK_START_TIME 变量
from agbenchmark.__main__ import BENCHMARK_START_TIME
# 从 agbenchmark.agent_interface 模块中导入 HELICONE_GRAPHQL_LOGS 变量
from agbenchmark.agent_interface import HELICONE_GRAPHQL_LOGS

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义从 Helicone 获取数据的函数，返回一个可选的浮点数
def get_data_from_helicone(challenge: str) -> Optional[float]:
    # 定义 GraphQL 服务器的端点
    url = "https://www.helicone.ai/api/graphql"

    # 设置请求头，通常需要设置内容类型和可能的授权令牌
    headers = {"authorization": f"Bearer {os.environ.get('HELICONE_API_KEY')}"}

    # 定义查询、变量和操作名称
    query = """
query ExampleQuery($properties: [PropertyFilter!]){
  aggregatedHeliconeRequest(properties: $properties) {
    costUSD
  }
}
"""

    variables = {
        "properties": [
            {
                "value": {"equals": os.environ.get("AGENT_NAME")},
                "name": "agent",
            },
            {
                "value": {"equals": BENCHMARK_START_TIME},
                "name": "benchmark_start_time",
            },
            {"value": {"equals": challenge}, "name": "challenge"},
        ]
    }

    # 如果 HELICONE_GRAPHQL_LOGS 为真，则记录调试信息
    if HELICONE_GRAPHQL_LOGS:
        logger.debug(f"Executing Helicone query:\n{query.strip()}")
        logger.debug(f"Query variables:\n{json.dumps(variables, indent=4)}")

    # 定义操作名称
    operation_name = "ExampleQuery"

    # 初始化数据字典和响应对象
    data = {}
    response = None

    try:
        # 发送 POST 请求到 Helicone 服务器
        response = requests.post(
            url,
            headers=headers,
            json={
                "query": query,
                "variables": variables,
                "operationName": operation_name,
            },
        )

        # 将响应内容解析为 JSON 格式
        data = response.json()
    except requests.HTTPError as http_err:
        # 如果请求出现 HTTP 错误，则记录错误信息并返回 None
        logger.error(f"Helicone returned an HTTP error: {http_err}")
        return None
    # 如果 JSON 解析出错，则记录原始响应内容并输出错误日志
    except json.JSONDecodeError:
        raw_response = response.text  # type: ignore
        logger.error(
            f"Helicone returned an invalid JSON response: '''{raw_response}'''"
        )
        # 返回空值
        return None
    
    # 如果出现其他异常，则记录错误信息并输出错误日志
    except Exception as err:
        logger.error(f"Error while trying to get data from Helicone: {err}")
        # 返回空值
        return None

    # 如果数据为空或者数据中没有"data"字段，则输出错误日志并返回空值
    if data is None or data.get("data") is None:
        logger.error("Invalid response received from Helicone: no data")
        logger.error(f"Offending response: {response}")
        # 返回空值
        return None
    
    # 从数据中获取"aggregatedHeliconeRequest"字段中的"costUSD"值，如果不存在则返回空值
    return (
        data.get("data", {}).get("aggregatedHeliconeRequest", {}).get("costUSD", None)
    )
```