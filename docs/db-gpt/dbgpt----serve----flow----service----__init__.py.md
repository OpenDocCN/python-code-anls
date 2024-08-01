# `.\DB-GPT-src\dbgpt\serve\flow\service\__init__.py`

```py
# 导入所需模块
import os
import sys
from datetime import datetime

# 定义一个名为 `User` 的类
class User:
    # 初始化方法，设置用户的姓名和年龄
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    # 返回用户的详细信息字符串表示
    def get_details(self):
        return f"Name: {self.name}, Age: {self.age}"

# 创建一个名为 `john` 的 `User` 对象，姓名为 "John Doe"，年龄为 30
john = User("John Doe", 30)

# 输出 `john` 对象的详细信息
print(john.get_details())
```