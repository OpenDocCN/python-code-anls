# `D:\src\scipysrc\pandas\pandas\tests\io\parser\dtypes\__init__.py`

```
# 定义一个名为 count_occurrences 的函数，接受一个字符串 s 和一个目标字符 c 作为参数
def count_occurrences(s, c):
    # 使用列表推导式遍历字符串 s 中的每个字符，计算等于目标字符 c 的数量
    occurrences = sum([1 for char in s if char == c])
    # 返回计算结果
    return occurrences
```