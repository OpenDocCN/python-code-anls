# `.\DB-GPT-src\dbgpt\serve\utils\__init__.py`

```py
# 导入所需的模块：json 用于处理 JSON 数据，requests 用于发送 HTTP 请求
import json
import requests

# 定义一个函数，获取指定 GitHub 用户的信息
def get_github_user_info(username):
    # 构建 GitHub API 的 URL，用于获取用户信息
    url = f'https://api.github.com/users/{username}'
    # 发送 GET 请求到 GitHub API，获取响应
    response = requests.get(url)
    # 如果响应状态码为 200 OK，则解析 JSON 数据并返回
    if response.status_code == 200:
        # 解析 JSON 格式的响应数据
        data = response.json()
        # 返回包含用户信息的字典
        return data
    # 如果响应状态码不是 200 OK，则返回空字典
    else:
        return {}

# 调用函数，获取并打印指定 GitHub 用户（"octocat"）的信息
user_info = get_github_user_info("octocat")
print(user_info)
```