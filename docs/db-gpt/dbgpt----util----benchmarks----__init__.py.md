# `.\DB-GPT-src\dbgpt\util\benchmarks\__init__.py`

```py
# 导入 requests 模块，用于发送 HTTP 请求
import requests

# 定义函数 get_random_user_agent，用于获取一个随机的用户代理字符串
def get_random_user_agent():
    # 用户代理字符串列表，模拟不同浏览器和操作系统的请求
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_4_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]
    # 返回随机选择的用户代理字符串
    return random.choice(user_agents)

# 定义函数 fetch_url，接收一个 URL 参数，并返回响应内容
def fetch_url(url):
    # 发送 GET 请求到指定 URL，使用随机选择的用户代理字符串
    response = requests.get(url, headers={'User-Agent': get_random_user_agent()})
    # 返回响应内容
    return response.content
```