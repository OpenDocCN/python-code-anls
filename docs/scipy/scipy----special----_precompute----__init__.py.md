# `D:\src\scipysrc\scipy\scipy\special\_precompute\__init__.py`

```
# 导入需要的模块：os（操作系统接口）、re（正则表达式操作）、json（JSON编码和解码）、sys（系统特定的参数和函数）
import os
import re
import json
import sys

# 定义一个名为process_file的函数，接受一个参数filename
def process_file(filename):
    # 用指定的文件名打开文件，模式为只读
    with open(filename, 'r') as f:
        # 读取文件的全部内容，并将内容保存在变量content中
        content = f.read()
    
    # 使用正则表达式查找所有的数字，并将结果保存在变量matches中
    matches = re.findall(r'\d+', content)
    
    # 创建一个空的列表，用于存放解析后的数字
    numbers = []
    
    # 遍历所有匹配到的数字
    for match in matches:
        # 将匹配到的数字转换为整数，并添加到numbers列表中
        numbers.append(int(match))
    
    # 返回解析后的数字列表
    return numbers

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 如果命令行参数的数量不等于2，则打印错误信息并退出
    if len(sys.argv) != 2:
        print("Usage: python script.py filename")
        sys.exit(1)
    
    # 将命令行参数中的第一个参数（文件名）赋值给变量filename
    filename = sys.argv[1]
    
    # 如果指定的文件存在
    if os.path.exists(filename):
        # 调用process_file函数，处理指定的文件，并将结果保存在变量result中
        result = process_file(filename)
        # 打印解析后的结果
        print(result)
    else:
        # 如果文件不存在，则打印错误信息
        print(f"Error: {filename} not found.")
        # 退出程序并返回状态码1
        sys.exit(1)
```