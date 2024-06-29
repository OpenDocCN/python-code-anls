# `D:\src\scipysrc\pandas\pandas\tests\scalar\period\__init__.py`

```
# 导入所需模块：re 用于正则表达式的匹配操作
import re

# 定义一个字符串变量，包含待搜索的文本内容
text = '文本中包含的一些数字是 42 和 100。'

# 使用正则表达式查找文本中的所有数字，并存储在列表中
numbers = re.findall(r'\d+', text)

# 打印找到的所有数字列表
print(numbers)
```