# `D:\src\scipysrc\sympy\sympy\parsing\tests\__init__.py`

```
# 导入requests库，用于进行HTTP请求
import requests

# 定义一个函数，名为get_random_user_agent，无参数
def get_random_user_agent():
    # 从指定URL获取用户代理列表
    url = 'https://www.randomuseragent.org/'
    # 发起HTTP GET请求，获取响应
    response = requests.get(url)
    # 返回响应中的文本内容，即随机用户代理字符串
    return response.text.strip()
```