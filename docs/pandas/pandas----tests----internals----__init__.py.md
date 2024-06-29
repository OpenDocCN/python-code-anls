# `D:\src\scipysrc\pandas\pandas\tests\internals\__init__.py`

```
# 导入所需的模块：datetime（日期时间处理）和 re（正则表达式）
import datetime, re

# 定义一个常量 MAX_VAL，并赋值为 1000
MAX_VAL = 1000

# 定义一个函数 process_data，接受参数 data
def process_data(data):
    # 使用当前日期时间创建一个新的日期时间对象
    current_time = datetime.datetime.now()
    
    # 使用正则表达式查找 data 中的数字，并返回一个匹配对象列表
    matches = re.findall(r'\d+', data)
    
    # 如果匹配列表 matches 不为空
    if matches:
        # 将匹配到的第一个数字转换为整数，赋值给 num
        num = int(matches[0])
    else:
        # 如果没有找到数字，则将 num 设为 None
        num = None
    
    # 返回当前时间对象和找到的数字
    return current_time, num
```