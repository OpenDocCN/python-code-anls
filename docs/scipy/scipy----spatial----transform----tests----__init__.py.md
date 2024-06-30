# `D:\src\scipysrc\scipy\scipy\spatial\transform\tests\__init__.py`

```
# 导入所需的模块：os 模块用于操作文件系统，re 模块用于处理正则表达式
import os
import re

# 定义一个函数，参数为文件路径
def find_files(dir, pattern):
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(dir):
        # 遍历当前目录下的所有文件
        for basename in files:
            # 利用正则表达式检查文件名是否与模式匹配
            if re.match(pattern, basename):
                # 拼接当前文件的完整路径
                filename = os.path.join(root, basename)
                # 打印匹配的文件路径
                print(filename)

# 调用函数，查找指定目录下所有以 .py 结尾的文件
find_files('.', r'.*\.py$')
```