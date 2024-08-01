# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\1886.87ded856666fdf6b.js`

```py
# 导入所需模块：os 用于处理操作系统相关功能，shutil 用于高级文件操作，tempfile 用于创建临时文件和目录
import os
import shutil
import tempfile

# 定义一个函数 create_temporary_directory，用于创建一个临时目录并返回其路径
def create_temporary_directory():
    # 使用 tempfile 模块的 TemporaryDirectory 函数创建一个临时目录
    temp_dir = tempfile.TemporaryDirectory()
    # 返回临时目录的路径
    return temp_dir.name

# 创建一个名为 temp_dir 的临时目录，并获取其路径
temp_dir = create_temporary_directory()

# 打印出临时目录的路径，以便查看或进一步处理
print(f"Created temporary directory: {temp_dir}")

# 在临时目录中创建一个名为 'example.txt' 的空文件
with open(os.path.join(temp_dir, 'example.txt'), 'w') as f:
    pass

# 将创建的 'example.txt' 文件移动到当前工作目录下
shutil.move(os.path.join(temp_dir, 'example.txt'), './example.txt')

# 打印提示消息，表明文件已成功移动
print("Moved 'example.txt' to current directory")

# 清理临时目录，删除临时文件和目录
temp_dir.cleanup()

# 打印消息，表明临时目录已清理完成
print("Temporary directory cleaned up")
```