# `.\agent\GenAINewsAgent\server\news.py`

```
# 导入必要的模块
import os  # 用于操作文件路径
import httpx  # 用于发起 HTTP 请求
from configs import NEWS_API_KEY, NEWS_BASE_URL  # 导入配置文件中的 API 密钥和基础 URL


# 定义一个异步函数，用于获取新闻信息
async def getNews(query: str, max_size: int = 8):
    # 使用异步 HTTP 客户端创建一个客户端对象，设置超时时间为 60 秒
    async with httpx.AsyncClient(timeout=60) as client:
        # 构建请求 URL，包括 API 密钥、查询词和最大返回数量
        response = await client.get(
            os.path.join(NEWS_BASE_URL, "news") +
            f"?apiKey={NEWS_API_KEY}&q={query}&size={max_size}")
        try:
            # 检查响应是否成功
            response.raise_for_status()
            # 返回 JSON 格式的响应内容
            return response.json()
        except httpx.HTTPStatusError as e:
            # 处理 HTTP 请求错误，打印错误信息
            print(
                f"Error resposne {e.response.status_code} while requesting {e.request.url!r}"
            )
            # 返回空值
            return None
```