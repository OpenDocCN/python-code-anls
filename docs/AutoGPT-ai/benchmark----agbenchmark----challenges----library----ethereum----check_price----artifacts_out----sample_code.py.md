# `.\AutoGPT\benchmark\agbenchmark\challenges\library\ethereum\check_price\artifacts_out\sample_code.py`

```py
# 导入 requests 模块
import requests

# 获取以太坊价格的函数，返回价格的浮点数
def get_ethereum_price() -> float:
    # 定义获取以太坊价格的 API 地址
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    # 发送 GET 请求获取 API 响应
    response = requests.get(url)

    # 如果响应状态码为 200，表示请求成功
    if response.status_code == 200:
        # 将响应内容解析为 JSON 格式
        data = response.json()
        # 返回以太坊价格（以太坊对美元的汇率）
        return data["ethereum"]["usd"]
    # 如果响应状态码不为 200，抛出异常
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")
```