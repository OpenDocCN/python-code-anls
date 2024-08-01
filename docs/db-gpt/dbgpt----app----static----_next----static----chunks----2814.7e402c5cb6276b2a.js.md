# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2814.7e402c5cb6276b2a.js`

```py
# 导入requests库，用于发送HTTP请求
import requests

# 定义一个名为get_random_user_agent的函数，用于获取随机的User-Agent头部信息
def get_random_user_agent():
    # 设置User-Agent头部信息的列表
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 OPR/77.0.4054.277',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 OPR/77.0.4054.277'
    ]
    # 返回随机选择的User-Agent头部信息
    return random.choice(user_agents)

# 定义一个URL字符串变量，存储待请求的网址
url = 'https://api.github.com'

# 使用requests库发送HTTP GET请求，传递headers参数设置User-Agent头部信息，并将响应对象赋值给response变量
response = requests.get(url, headers={'User-Agent': get_random_user_agent()})

# 输出响应状态码
print(response.status_code)
```