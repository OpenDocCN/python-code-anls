# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\__init__.py`

```
# 导入所需模块
import os
from typing import List

# 定义一个数据类，用于存储姓名和年龄
class Person:
    def __init__(self, name: str, age: int):
        self.name = name  # 存储姓名
        self.age = age    # 存储年龄

# 函数：从文件中读取数据并返回包含所有人员对象的列表
def read_data(filename: str) -> List[Person]:
    # 创建一个空列表，用于存储所有的Person对象
    people = []

    # 打开文件进行读取
    with open(filename, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 分割每行数据，以逗号为分隔符
            data = line.strip().split(',')
            # 解析姓名和年龄数据
            name = data[0]
            age = int(data[1])
            # 创建Person对象并添加到列表中
            person = Person(name, age)
            people.append(person)

    # 返回所有Person对象的列表
    return people
```