# `.\DB-GPT-src\dbgpt\serve\datasource\__init__.py`

```py
# 导入所需的模块
import os
import sys

# 定义一个类，表示用户
class User:
    # 初始化方法，接受用户名和年龄参数
    def __init__(self, username, age):
        self.username = username  # 将用户名存储在实例变量中
        self.age = age  # 将年龄存储在实例变量中

    # 方法：打印用户信息
    def print_info(self):
        # 打印用户名和年龄信息
        print(f'Username: {self.username}, Age: {self.age}')

# 创建一个用户实例，用户名为 'JohnDoe'，年龄为 30
user = User('JohnDoe', 30)

# 调用实例的打印信息方法
user.print_info()
```