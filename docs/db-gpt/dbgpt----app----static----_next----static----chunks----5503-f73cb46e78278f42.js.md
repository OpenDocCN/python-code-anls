# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5503-f73cb46e78278f42.js`

```py
# 导入必要的模块：os 模块用于操作文件路径，shutil 模块用于高级文件操作
import os
import shutil

# 定义一个函数，接收一个目标文件夹路径作为参数
def clean_folder(folder):
    # 确保目标文件夹存在，如果不存在则创建它
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        # 清空目标文件夹内所有文件和文件夹，保留文件夹本身
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                # 如果是文件则直接删除
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # 如果是文件夹则递归删除
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"无法删除 {file_path}：{e}")

# 调用函数清空指定路径下的文件夹
clean_folder('/path/to/target/folder')
```