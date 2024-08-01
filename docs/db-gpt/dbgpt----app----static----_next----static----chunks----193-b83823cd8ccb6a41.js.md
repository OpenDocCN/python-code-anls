# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\193-b83823cd8ccb6a41.js`

```py
# 导入所需的模块：requests 用于发送 HTTP 请求
import requests

# 定义函数 fetch_url_content，接收一个 URL 参数，返回从该 URL 获取的文本内容
def fetch_url_content(url):
    # 发送 GET 请求到指定 URL，并获取响应对象
    response = requests.get(url)
    # 检查响应状态码是否为 200，表示请求成功
    if response.status_code == 200:
        # 返回响应对象的文本内容
        return response.text
    else:
        # 如果请求失败，打印错误消息并返回空字符串
        print(f"Failed to fetch URL: {url}")
        return ""

# 调用函数 fetch_url_content，传入指定的 URL，并打印获取的文本内容
url = "https://example.com"
content = fetch_url_content(url)
print(content)
```