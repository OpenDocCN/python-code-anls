# `D:\src\scipysrc\scipy\tools\wheels\check_license.py`

```
#!/usr/bin/env python
"""
check_license.py [MODULE]

Check the presence of a LICENSE.txt in the installed module directory,
and that it appears to contain text prevalent for a SciPy binary
distribution.

"""
import os  # 导入操作系统相关的功能模块
import sys  # 导入系统相关的功能模块
import re   # 导入正则表达式模块
import argparse  # 导入命令行参数解析模块


def check_text(text):
    # 检查文本中是否包含 "Copyright (c)"，以及后续文本中是否包含特定字符串的正则表达式匹配
    ok = "Copyright (c)" in text and re.search(
        r"This binary distribution of \w+ also bundles the following software",
        text,
        re.IGNORECASE
    )
    return ok


def main():
    p = argparse.ArgumentParser(usage=__doc__.rstrip())  # 创建命令行参数解析器，使用脚本文档的用法说明
    p.add_argument("module", nargs="?", default="scipy")  # 添加一个位置参数 module，默认为 "scipy"
    args = p.parse_args()  # 解析命令行参数

    # Drop '' from sys.path
    sys.path.pop(0)  # 移除 sys.path 中的空字符串项

    # Find module path
    __import__(args.module)  # 动态导入指定名称的模块
    mod = sys.modules[args.module]  # 获取导入模块的引用

    # Check license text
    license_txt = os.path.join(os.path.dirname(mod.__file__), "LICENSE.txt")  # 获取模块的 LICENSE.txt 文件路径
    with open(license_txt, encoding="utf-8") as f:
        text = f.read()  # 读取 LICENSE.txt 文件内容

    ok = check_text(text)  # 调用函数检查读取的文本是否符合预期格式
    if not ok:
        print(
            f"ERROR: License text {license_txt} does not contain expected "
            "text fragments\n"
        )
        print(text)  # 输出未符合预期的授权文本内容
        sys.exit(1)  # 程序退出，返回非零状态表示错误

    sys.exit(0)  # 程序正常退出，返回零状态


if __name__ == "__main__":
    main()
```