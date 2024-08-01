# `.\DB-GPT-src\dbgpt\core\awel\flow\tests\__init__.py`

```py
# 导入常用的时间处理模块
import time

# 定义一个函数，接收一个整数参数 x
def square(x):
    # 返回参数 x 的平方值
    return x * x

# 创建一个空列表
result = []

# 使用循环从 0 到 9
for i in range(10):
    # 将每个数的平方值添加到结果列表中
    result.append(square(i))

# 打印结果列表
print(result)
```