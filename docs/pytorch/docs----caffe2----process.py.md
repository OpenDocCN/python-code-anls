# `.\pytorch\docs\caffe2\process.py`

```py
#!/usr/bin/env python3
## @package process
# Module doxygen.process
# Script to insert preamble for doxygen and regen API docs

# 导入标准库中的 os 模块
import os
# 导入 shutil 模块，用于高级文件操作
import shutil


# 定义函数 insert，用于向指定文件中插入文档描述
# 参数 originalfile：原始文件路径
# 参数 first_line：要插入的第一行内容
# 参数 description：要插入的模块描述
def insert(originalfile, first_line, description):
    # 打开原始文件
    with open(originalfile) as f:
        # 读取原始文件的第一行
        f1 = f.readline()
        # 如果原始文件的第一行不包含指定的 first_line
        if f1.find(first_line) < 0:
            # 组合需要写入的文档内容
            docs = first_line + description + f1
            # 打开新文件进行写入
            with open("newfile.txt", "w") as f2:
                f2.write(docs)  # 写入文档内容
                f2.write(f.read())  # 继续写入原始文件的其余部分
            # 将新文件重命名为原始文件名
            os.rename("newfile.txt", originalfile)
        else:
            print("already inserted")  # 如果已经插入过，则打印提示信息


# 移动到上级目录（从当前目录到父目录）
os.chdir("..")
# 使用系统命令执行 git checkout 操作，恢复指定目录下的文件
os.system("git checkout caffe2/contrib/.")
os.system("git checkout caffe2/distributed/.")
os.system("git checkout caffe2/experiments/.")
os.system("git checkout caffe2/python/.")

# 遍历当前目录及其子目录下的所有文件和文件夹
for root, dirs, files in os.walk("."):
    for file in files:
        # 对于以 .py 结尾但不以 _test.py 和 __.py 结尾的文件
        if (
            file.endswith(".py")
            and not file.endswith("_test.py")
            and not file.endswith("__.py")
        ):
            filepath = os.path.join(root, file)  # 构建文件的完整路径
            print("filepath: " + filepath)  # 打印文件路径
            directory = os.path.dirname(filepath)[2:]  # 获取文件所在目录，并去掉前面的 "./"
            directory = directory.replace("/", ".")  # 将路径中的 "/" 替换为 "."
            print("directory: " + directory)  # 打印处理后的目录名
            name = os.path.splitext(file)[0]  # 获取文件名（去掉扩展名）
            first_line = "## @package " + name  # 构建要插入的第一行内容
            description = "\n# Module " + directory + "." + name + "\n"  # 构建模块描述
            print(first_line, description)
            insert(filepath, first_line, description)  # 调用 insert 函数进行插入操作

# 如果路径 "doxygen/doxygen-python" 存在，则删除该目录及其内容
if os.path.exists("doxygen/doxygen-python"):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree("doxygen/doxygen-python")
else:
    os.makedirs("doxygen/doxygen-python")  # 否则，创建该路径

# 如果路径 "doxygen/doxygen-c" 存在，则删除该目录及其内容
if os.path.exists("doxygen/doxygen-c"):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree("doxygen/doxygen-c")
else:
    os.makedirs("doxygen/doxygen-c")  # 否则，创建该路径

# 使用系统命令执行 doxygen 命令，生成 Python 文档
os.system("doxygen .Doxyfile-python")
# 使用系统命令执行 doxygen 命令，生成 C 文档
os.system("doxygen .Doxyfile-c")
```