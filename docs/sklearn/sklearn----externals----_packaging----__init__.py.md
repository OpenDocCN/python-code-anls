# `D:\src\scipysrc\scikit-learn\sklearn\externals\_packaging\__init__.py`

```
# 定义一个类 Calculator，实现基本的数学运算功能
class Calculator:
    # 初始化方法，接受两个参数 num1 和 num2，分别表示操作数1和操作数2
    def __init__(self, num1, num2):
        # 使用 self 关键字将输入的 num1 赋值给实例变量 self.num1
        self.num1 = num1
        # 使用 self 关键字将输入的 num2 赋值给实例变量 self.num2
        self.num2 = num2

    # 方法 add，实现两个数的加法操作
    def add(self):
        # 返回 num1 和 num2 的和
        return self.num1 + self.num2

    # 方法 subtract，实现两个数的减法操作
    def subtract(self):
        # 返回 num1 和 num2 的差
        return self.num1 - self.num2

    # 方法 multiply，实现两个数的乘法操作
    def multiply(self):
        # 返回 num1 和 num2 的乘积
        return self.num1 * self.num2

    # 方法 divide，实现两个数的除法操作
    def divide(self):
        # 如果 num2 不为0，则返回 num1 除以 num2 的结果
        if self.num2 != 0:
            return self.num1 / self.num2
        else:
            # 如果 num2 为0，则打印错误消息并返回 None
            print("Error: Division by zero.")
            return None
```