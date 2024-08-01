# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2487-4522eeb3601ff54e.js`

```py
# 导入所需模块：os 模块用于操作系统相关功能，glob 模块用于文件路径名的模式匹配
import os
import glob

# 获取当前工作目录下所有扩展名为 .txt 的文件的列表
files = glob.glob(os.path.join(os.getcwd(), '*.txt'))

# 遍历文件列表
for file in files:
    # 打开文件进行读取操作
    with open(file, 'r') as f:
        # 读取文件内容并打印输出
        print(f.read())
```