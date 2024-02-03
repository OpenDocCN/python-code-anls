# `stable-diffusion-webui\modules\util.py`

```py
# 导入 os 模块
import os
# 导入 re 模块
import re

# 从 modules 模块中导入 shared 对象
from modules import shared
# 从 modules.paths_internal 模块中导入 script_path 对象
from modules.paths_internal import script_path

# 定义一个函数，用于自然排序
def natural_sort_key(s, regex=re.compile('([0-9]+)')):
    # 返回一个列表，其中数字转换为整数，其它文本转换为小写
    return [int(text) if text.isdigit() else text.lower() for text in regex.split(s)]

# 定义一个函数，列出指定目录下的文件
def listfiles(dirname):
    # 获取目录下的所有文件名，并按自然排序排序
    filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname), key=natural_sort_key) if not x.startswith(".")]
    # 返回目录下的所有文件
    return [file for file in filenames if os.path.isfile(file)]

# 定义一个函数，返回指定 HTML 文件的路径
def html_path(filename):
    return os.path.join(script_path, "html", filename)

# 定义一个函数，读取指定 HTML 文件的内容
def html(filename):
    # 获取 HTML 文件的路径
    path = html_path(filename)

    # 如果文件存在，则读取文件内容并返回
    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()

    # 如果文件不存在，则返回空字符串
    return ""

# 定义一个函数，遍历指定路径下的文件
def walk_files(path, allowed_extensions=None):
    # 如果路径不存在，则返回空
    if not os.path.exists(path):
        return

    # 如果指定了允许的文件扩展名，则将其转换为集合
    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)

    # 获取路径下的所有文件和子目录
    items = list(os.walk(path, followlinks=True))
    # 根据路径名的自然排序对 items 进行排序
    items = sorted(items, key=lambda x: natural_sort_key(x[0]))

    # 遍历所有文件和子目录
    for root, _, files in items:
        # 对文件进行自然排序
        for filename in sorted(files, key=natural_sort_key):
            # 如果指定了允许的文件扩展名，并且当前文件的扩展名不在其中，则跳过
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext not in allowed_extensions:
                    continue

            # 如果不允许列出隐藏文件，并且当前文件路径中包含隐藏文件标记，则跳过
            if not shared.opts.list_hidden_files and ("/." in root or "\\." in root):
                continue

            # 返回文件的完整路径
            yield os.path.join(root, filename)

# 定义一个函数，用于打印信息
def ldm_print(*args, **kwargs):
    # 如果隐藏了 ldm_prints，则直接返回
    if shared.opts.hide_ldm_prints:
        return

    # 打印信息
    print(*args, **kwargs)
```