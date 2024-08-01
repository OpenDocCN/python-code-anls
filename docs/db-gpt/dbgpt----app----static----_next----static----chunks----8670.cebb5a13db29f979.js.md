# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\8670.cebb5a13db29f979.js`

```py
# 导入需要的模块：os（操作系统接口）、shutil（高级文件操作）、glob（文件名模式匹配）
import os
import shutil
import glob

# 定义函数：清空指定目录下所有文件和文件夹
def clear_directory(path):
    # 使用 glob 模块结合路径模式，获取指定目录下的所有文件和文件夹
    files = glob.glob(os.path.join(path, '*'))
    # 遍历获取到的所有文件和文件夹
    for f in files:
        # 判断当前遍历到的对象是否是文件夹
        if os.path.isdir(f):
            # 如果是文件夹，则递归删除文件夹及其内容
            shutil.rmtree(f)
        else:
            # 如果是文件，则直接删除文件
            os.remove(f)
```