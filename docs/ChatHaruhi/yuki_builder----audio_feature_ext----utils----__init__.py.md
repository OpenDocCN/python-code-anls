# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\utils\__init__.py`

```py
# 导入必要的模块：os（操作系统功能）、shutil（高级文件操作）、glob（全局文件搜索）
import os
import shutil
import glob

# 定义一个函数 find_files，接收两个参数：pattern（要搜索的文件模式）、path（搜索路径）
def find_files(pattern, path):
    # 使用 glob 模块在指定路径下查找匹配 pattern 的文件列表，返回结果
    files = glob.glob(os.path.join(path, pattern))
    # 返回找到的文件列表
    return files

# 设置要搜索的文件模式为 "*.txt"，搜索路径为当前目录下的 "data" 文件夹
pattern = "*.txt"
path = "./data"

# 调用 find_files 函数，查找所有以 .txt 结尾的文件列表
txt_files = find_files(pattern, path)

# 遍历找到的每一个 .txt 文件
for txt_file in txt_files:
    # 使用 shutil 模块的 copy 函数，将当前文件复制到 "backup" 文件夹下
    shutil.copy(txt_file, "./backup")
```