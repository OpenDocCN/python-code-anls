# `.\DB-GPT-src\dbgpt\serve\conversation\service\__init__.py`

```py
# 导入必要的模块：os 模块用于操作文件路径，shutil 模块用于文件操作
import os
import shutil

# 定义一个函数，用于复制指定目录及其子目录中的所有文件和文件夹到另一个目录
def copytree(src, dst):
    # 如果目标目录不存在，则创建它
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    # 遍历源目录中的所有文件和子目录
    for item in os.listdir(src):
        # 构建完整的源路径和目标路径
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        
        # 如果当前项目是一个子目录，则递归调用 copytree 函数
        if os.path.isdir(s):
            copytree(s, d)
        # 如果是文件，则使用 shutil 模块的 copy2 函数复制文件
        else:
            shutil.copy2(s, d)

# 示例用法：
# 将源目录 '/path/to/source' 复制到目标目录 '/path/to/destination'
copytree('/path/to/source', '/path/to/destination')
```