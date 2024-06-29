# `.\numpy\tools\wheels\check_license.py`

```py
#!/usr/bin/env python
"""
check_license.py [MODULE]

Check the presence of a LICENSE.txt in the installed module directory,
and that it appears to contain text prevalent for a NumPy binary
distribution.

"""
import sys  # 导入 sys 模块，用于访问系统相关的功能
import re  # 导入 re 模块，用于正则表达式操作
import argparse  # 导入 argparse 模块，用于解析命令行参数
import pathlib  # 导入 pathlib 模块，用于处理文件路径

def check_text(text):
    # 检查文本是否包含 "Copyright (c)" 并且符合预期的二进制分发文本模式
    ok = "Copyright (c)" in text and re.search(
        r"This binary distribution of \w+ also bundles the following software",
        text,
    )
    return ok

def main():
    p = argparse.ArgumentParser(usage=__doc__.rstrip())
    p.add_argument("module", nargs="?", default="numpy")
    args = p.parse_args()

    # 从 sys.path 中移除空字符串项
    sys.path.pop(0)

    # 寻找指定模块的路径
    __import__(args.module)
    mod = sys.modules[args.module]

    # LICENSE.txt 通常安装在 .dist-info 目录中，因此在此查找它
    sitepkgs = pathlib.Path(mod.__file__).parent.parent
    distinfo_path = [s for s in sitepkgs.glob("numpy-*.dist-info")][0]

    # 检查许可证文本
    license_txt = distinfo_path / "LICENSE.txt"
    with open(license_txt, encoding="utf-8") as f:
        text = f.read()

    # 调用 check_text 函数检查许可证文本是否符合预期
    ok = check_text(text)
    if not ok:
        print(
            "ERROR: License text {} does not contain expected "
            "text fragments\n".format(license_txt)
        )
        print(text)
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
```