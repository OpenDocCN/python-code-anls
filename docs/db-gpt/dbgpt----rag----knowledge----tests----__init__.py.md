# `.\DB-GPT-src\dbgpt\rag\knowledge\tests\__init__.py`

```py
# 导入必要的模块：os（操作系统接口）、sys（系统特定的参数和功能）、json（编码和解码 JSON 数据）
import os
import sys
import json

# 定义一个函数，接收一个参数（文件名）
def process_file(filename):
    # 尝试打开指定文件，模式为只读
    try:
        with open(filename, 'r') as f:
            # 读取文件内容并解析为 JSON 格式的数据
            data = json.load(f)
            # 返回读取的数据
            return data
    # 如果文件操作出现异常（如文件不存在），捕获并处理该异常
    except FileNotFoundError:
        # 打印错误消息，指示文件未找到
        print(f"Error: File '{filename}' not found.")
        # 返回空字典作为默认值
        return {}
    # 捕获其他异常（如 JSON 解析错误）
    except json.JSONDecodeError as e:
        # 打印 JSON 解析错误的详细信息
        print(f"Error decoding JSON in file '{filename}': {e}")
        # 返回空字典作为默认值
        return {}

# 如果脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 检查命令行参数的数量
    if len(sys.argv) < 2:
        # 打印错误消息，指示缺少文件名参数
        print("Usage: python script.py <filename>")
        # 退出程序，并指示错误的退出状态码
        sys.exit(1)
    
    # 获取命令行参数中的文件名
    filename = sys.argv[1]
    # 调用处理文件的函数，并获取返回的数据
    result = process_file(filename)
    # 打印处理后的数据
    print(result)
```