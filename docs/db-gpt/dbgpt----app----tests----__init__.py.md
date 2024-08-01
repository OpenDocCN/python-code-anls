# `.\DB-GPT-src\dbgpt\app\tests\__init__.py`

```py
# 导入必要的模块：os（操作系统接口）、shutil（高级文件操作）、tempfile（生成临时文件和目录）
import os
import shutil
import tempfile

# 定义一个函数：将指定目录下的所有文件（不包括子目录）复制到另一个目录中
def copy_files(src_dir, dest_dir):
    # 使用 tempfile 模块创建一个临时目录
    tmp_dir = tempfile.mkdtemp()

    try:
        # 枚举指定源目录下的所有文件和目录，仅遍历文件
        for item in os.listdir(src_dir):
            item_path = os.path.join(src_dir, item)
            # 如果是文件而非目录，则进行复制操作
            if os.path.isfile(item_path):
                shutil.copy(item_path, tmp_dir)

        # 将临时目录下的所有文件复制到目标目录中
        for item in os.listdir(tmp_dir):
            item_path = os.path.join(tmp_dir, item)
            shutil.copy(item_path, dest_dir)
    finally:
        # 最后，删除临时目录及其内容
        shutil.rmtree(tmp_dir)
```