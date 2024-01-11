# `ZeroNet\src\Debug\DebugMedia.py`

```
# 导入所需的模块
import os
import subprocess
import re
import logging
import time
import functools

# 从自定义的 Config 模块中导入 config 变量
from Config import config
# 从自定义的 util 模块中导入 helper 函数
from util import helper

# 定义函数 findfiles，用于在指定路径中查找特定扩展名的文件
def findfiles(path, find_ext):
    # 定义排序函数 sorter，用于对文件进行排序
    def sorter(f1, f2):
        f1 = f1[0].replace(path, "")
        f2 = f2[0].replace(path, "")
        if f1 == "":
            return 1
        elif f2 == "":
            return -1
        else:
            return helper.cmp(f1.lower(), f2.lower())

    # 遍历指定路径下的文件和文件夹
    for root, dirs, files in sorted(os.walk(path, topdown=False), key=functools.cmp_to_key(sorter)):
        # 遍历文件
        for file in sorted(files):
            file_path = root + "/" + file
            file_ext = file.split(".")[-1]
            # 如果文件扩展名在指定的扩展名列表中，并且文件名不以 "all." 开头，则返回文件路径
            if file_ext in find_ext and not file.startswith("all."):
                yield file_path.replace("\\", "/")

# 定义函数 findCoffeescriptCompiler，用于查找 coffeescript 编译器
def findCoffeescriptCompiler():
    coffeescript_compiler = None
    try:
        import distutils.spawn
        # 查找 coffee 可执行文件的路径，并设置为 coffeescript_compiler
        coffeescript_compiler = helper.shellquote(distutils.spawn.find_executable("coffee")) + " --no-header -p"
    except:
        pass
    # 如果找到了 coffeescript_compiler，则返回它，否则返回 False
    if coffeescript_compiler:
        return coffeescript_compiler
    else:
        return False

# 定义函数 merge，用于合并文件
def merge(merged_path):
    merged_path = merged_path.replace("\\", "/")
    merge_dir = os.path.dirname(merged_path)
    s = time.time()
    ext = merged_path.split(".")[-1]
    if ext == "js":  # 如果合并 .js 文件，则查找 .coffee 文件
        find_ext = ["js", "coffee"]
    else:
        find_ext = [ext]

    # 如果合并文件已存在，则获取其修改时间
    if os.path.isfile(merged_path):
        merged_mtime = os.path.getmtime(merged_path)
    else:
        merged_mtime = 0

    changed = {}
    # 遍历合并目录中指定扩展名的文件
    for file_path in findfiles(merge_dir, find_ext):
        # 如果文件的修改时间大于合并文件的修改时间加1秒，则将文件路径添加到 changed 字典中
        if os.path.getmtime(file_path) > merged_mtime + 1:
            changed[file_path] = True
    # 如果没有变化，则返回，不需要进行任何操作
    if not changed:
        return

    # 创建一个空字典，用于存储旧部分的数据
    old_parts = {}

    # 如果合并后的文件存在，则查找旧部分以避免不必要的重新编译
    if os.path.isfile(merged_path):
        # 读取合并后的文件内容
        merged_old = open(merged_path, "rb").read()
        # 使用正则表达式找到旧部分的内容，并存储到old_parts字典中
        for match in re.findall(rb"(/\* ---- (.*?) ---- \*/(.*?)(?=/\* ----|$))", merged_old, re.DOTALL):
            old_parts[match[1].decode()] = match[2].strip(b"\n\r")

    # 输出调试信息，记录合并的文件路径、是否有变化以及旧部分的数量
    logging.debug("Merging %s (changed: %s, old parts: %s)" % (merged_path, changed, len(old_parts)))

    # 初始化一个空列表，用于存储合并后的部分
    parts = []

    # 记录开始合并的时间
    s_total = time.time()

    # 将所有部分合并成一个字节流
    merged = b"\n".join(parts)

    # 如果文件类型是css，则进行厂商前缀处理
    if ext == "css":
        # 导入cssvendor模块，对合并后的内容进行厂商前缀处理
        from lib.cssvendor import cssvendor
        merged = cssvendor.prefix(merged)

    # 替换合并后的内容中的回车符
    merged = merged.replace(b"\r", b"")

    # 将合并后的内容写入到合并后的文件中
    open(merged_path, "wb").write(merged)

    # 输出调试信息，记录合并后的文件路径和合并所花费的时间
    logging.debug("Merged %s (%.2fs)" % (merged_path, time.time() - s_total))
# 如果当前模块是主程序，则执行以下代码
if __name__ == "__main__":
    # 设置日志记录器的日志级别为调试
    logging.getLogger().setLevel(logging.DEBUG)
    # 改变当前工作目录到上一级目录
    os.chdir("..")
    # 设置配置文件中的 coffeescript_compiler 变量为指定的字符串
    config.coffeescript_compiler = r'type "%s" | tools\coffee-node\bin\node.exe tools\coffee-node\bin\coffee --no-header -s -p'
    # 调用 merge 函数，合并指定路径下的 JavaScript 文件
    merge("data/12Hw8rTgzrNo4DSh2AkqwPRqDyTticwJyH/js/all.js")
```