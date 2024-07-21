# `.\pytorch\aten\src\ATen\cudnn\Exceptions.h`

```py
# 导入需要使用的模块
import os
import sys

# 定义一个类，表示一个简单的学生对象
class Student:
    # 初始化方法，用来初始化学生对象的属性
    def __init__(self, name, age):
        self.name = name  # 设置学生对象的名字属性
        self.age = age    # 设置学生对象的年龄属性

    # 定义一个方法，用来显示学生对象的详细信息
    def display(self):
        print("Name:", self.name)
        print("Age:", self.age)

# 创建一个学生对象
s = Student("John", 22)
# 调用显示方法，显示学生对象的详细信息
s.display()
```