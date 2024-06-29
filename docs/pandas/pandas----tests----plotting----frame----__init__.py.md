# `D:\src\scipysrc\pandas\pandas\tests\plotting\frame\__init__.py`

```
# 导入所需的模块
import os
import sys

# 定义一个名为 process_file 的函数，接收一个文件名作为参数
def process_file(filename):
    # 尝试打开指定文件，以只读方式
    try:
        with open(filename, 'r') as f:
            # 读取文件内容并存储在 content 变量中
            content = f.read()
    # 如果文件打开或读取过程中发生异常，则捕获异常并输出错误信息
    except IOError:
        print(f"Error: Could not open or read file '{filename}'")
        # 将错误码设为 1，表示发生了错误
        sys.exit(1)
    # 如果文件操作成功完成，则打印文件名和文件内容的长度
    else:
        print(f"File '{filename}' successfully processed. Length: {len(content)}")

# 调用 process_file 函数，传入文件名 'example.txt'
process_file('example.txt')
```