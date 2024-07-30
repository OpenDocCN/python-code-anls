# `.\comic-translate\modules\ocr\pororo\pororo\tasks\utils\__init__.py`

```py
# 导入所需模块：os（操作系统相关功能）、shutil（高级文件操作功能）
import os
import shutil

# 定义函数 compress_files，接受两个参数：source_dir（源目录路径）、dest_zip（目标 ZIP 文件路径）
def compress_files(source_dir, dest_zip):
    # 切换当前工作目录到源目录路径
    os.chdir(source_dir)
    # 创建一个新的 ZIP 文件，以写入模式打开
    with zipfile.ZipFile(dest_zip, 'w') as zipf:
        # 遍历源目录下的所有文件和子目录
        for root, dirs, files in os.walk('.'):
            for file in files:
                # 构建当前文件的完整路径
                file_path = os.path.join(root, file)
                # 将当前文件添加到 ZIP 文件中，使用相对路径
                zipf.write(file_path, os.path.relpath(file_path, source_dir))
    
    # 复制生成的 ZIP 文件到当前工作目录之外的目标 ZIP 文件路径
    shutil.copy(dest_zip, dest_zip)
```