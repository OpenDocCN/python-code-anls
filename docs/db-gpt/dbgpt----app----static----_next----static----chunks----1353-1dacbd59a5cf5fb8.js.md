# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\1353-1dacbd59a5cf5fb8.js`

```py
# 导入必要的模块：os（操作系统接口）、sys（系统特定的参数和功能）、re（正则表达式操作）、shutil（高级文件操作）、json（编码和解码 JSON 对象）
import os, sys, re, shutil, json

# 定义一个名为 MyException 的异常类，继承自内置的 Exception 类
class MyException(Exception):
    # 初始化方法，接受一个字符串 message 作为参数
    def __init__(self, message):
        # 调用父类 Exception 的初始化方法，传递 message 给父类处理
        super().__init__(message)
        # 将 message 存储在实例变量 self.message 中
        self.message = message

# 定义一个名为 MyClass 的类
class MyClass:
    # 类属性，一个类级别的变量，用于计数对象的数量
    count = 0

    # 初始化方法，接受一个参数 self（指向实例本身），以及可选的 name 和 age 参数
    def __init__(self, name=None, age=None):
        # 将类属性 count 的值增加 1
        MyClass.count += 1
        # 将参数 name 赋值给实例变量 self.name
        self.name = name
        # 将参数 age 赋值给实例变量 self.age
        self.age = age

    # 类方法，用于返回类属性 count 的当前值
    @classmethod
    def get_count(cls):
        # 返回类属性 count 的值
        return cls.count

    # 实例方法，用于返回实例的描述字符串
    def get_info(self):
        # 格式化输出实例的 name 和 age 属性
        return f"Name: {self.name}, Age: {self.age}"

# 创建一个名为 obj1 的 MyClass 类的实例，传入参数 'Alice' 和 30
obj1 = MyClass('Alice', 30)
# 创建一个名为 obj2 的 MyClass 类的实例，不传入 name 和 age 参数
obj2 = MyClass()

# 输出 obj1 和 obj2 的信息
print(obj1.get_info())
print(obj2.get_info())

# 输出 MyClass 类的对象数量
print(f"Number of instances created: {MyClass.get_count()}")

# 抛出自定义异常 MyException，并传入异常消息 "An error occurred"
raise MyException("An error occurred")
```