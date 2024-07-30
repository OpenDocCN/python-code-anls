# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\modules\__init__.py`

```py
# 导入所需的模块
import os
import shutil

# 定义函数，接收两个参数：源目录和目标目录
def copy_files(source_dir, dest_dir):
    # 如果目标目录不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 遍历源目录中的文件和子目录
    for item in os.listdir(source_dir):
        # 构建源文件或目录的完整路径
        source_item = os.path.join(source_dir, item)
        # 构建目标文件或目录的完整路径
        dest_item = os.path.join(dest_dir, item)
        
        # 如果是文件，则使用 shutil 库进行复制
        if os.path.isfile(source_item):
            shutil.copy(source_item, dest_item)
        # 如果是目录，则递归调用本函数，复制目录及其内容
        elif os.path.isdir(source_item):
            copy_files(source_item, dest_item)

# 示例调用函数，复制源目录到目标目录
copy_files('/path/to/source', '/path/to/destination')
```