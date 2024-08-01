# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\9343.b26f65cc33122b4a.js`

```py
# 定义一个类 Calculator，用于实现基本的数学运算
class Calculator:
    # 初始化方法，创建 Calculator 类的新实例
    def __init__(self):
        # 初始化实例变量 result 为 0
        self.result = 0
    
    # 加法方法，接受一个参数 num，将 num 加到 result 上
    def add(self, num):
        self.result += num
    
    # 减法方法，接受一个参数 num，将 num 从 result 中减去
    def subtract(self, num):
        self.result -= num
    
    # 乘法方法，接受一个参数 num，将 result 乘以 num
    def multiply(self, num):
        self.result *= num
    
    # 除法方法，接受一个参数 num，将 result 除以 num
    def divide(self, num):
        self.result /= num
    
    # 清零方法，将 result 重置为 0
    def clear(self):
        self.result = 0

# 创建一个 Calculator 的实例 calc
calc = Calculator()
# 对 calc 进行数学操作：加法 5
calc.add(5)
# 对 calc 进行数学操作：乘法 3
calc.multiply(3)
# 对 calc 进行数学操作：减法 2
calc.subtract(2)
# 对 calc 进行数学操作：除法 4
calc.divide(4)
# 打印出最终的计算结果
print(calc.result)
```