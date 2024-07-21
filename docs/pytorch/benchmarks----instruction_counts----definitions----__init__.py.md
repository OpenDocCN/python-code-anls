# `.\pytorch\benchmarks\instruction_counts\definitions\__init__.py`

```
# 定义一个名为 extract_numbers 的函数，接收一个字符串参数 s
def extract_numbers(s):
    # 创建一个空列表，用于存储提取出来的数字
    numbers = []
    # 创建一个空字符串，用于暂时存储当前正在构建的数字
    current_num = ''
    
    # 遍历字符串 s 中的每个字符
    for char in s:
        # 如果当前字符是数字或者是小数点
        if char.isdigit() or char == '.':
            # 将当前字符添加到 current_num 中
            current_num += char
        # 如果当前字符不是数字或小数点，并且 current_num 非空
        elif current_num:
            # 将 current_num 转换为浮点数，并添加到 numbers 列表中
            numbers.append(float(current_num))
            # 重置 current_num 为空字符串，准备下一个数字的构建
            current_num = ''
    
    # 如果循环结束后 current_num 非空（最后一个字符可能是数字或小数点）
    if current_num:
        # 将最后的 current_num 转换为浮点数，并添加到 numbers 列表中
        numbers.append(float(current_num))
    
    # 返回提取出的数字列表
    return numbers
```