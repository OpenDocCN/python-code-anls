# `.\numpy\numpy\_build_utils\tempita.py`

```py
#!/usr/bin/env python3
import sys  # 导入 sys 模块，用于访问系统相关的功能
import os  # 导入 os 模块，用于操作系统功能
import argparse  # 导入 argparse 模块，用于解析命令行参数

from Cython import Tempita as tempita  # 从 Cython 模块导入 Tempita 类，并重命名为 tempita

# XXX: 如果这个导入失败会怎样（真的会失败吗？），为 Cython.tempita 进行供应商支持


def process_tempita(fromfile, outfile=None):
    """处理 tempita 模板文件并写入结果。

    模板文件的文件名预期以 `.c.in` 或 `.pyx.in` 结尾：
    例如，处理 `template.c.in` 会生成 `template.c`。

    """
    if outfile is None:
        # 我们正在处理 distutils 构建，在原地写入
        outfile = os.path.splitext(fromfile)[0]

    from_filename = tempita.Template.from_filename
    template = from_filename(fromfile, encoding=sys.getdefaultencoding())

    content = template.substitute()

    with open(outfile, 'w') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile",
        type=str,
        help="输入文件的路径"
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="输出文件的路径"
    )
    parser.add_argument(
        "-i",
        "--ignore",
        type=str,
        help="一个被忽略的输入 - 可能有助于在自定义目标之间添加依赖关系",
    )
    args = parser.parse_args()

    if not args.infile.endswith('.in'):
        raise ValueError(f"意外的扩展名: {args.infile}")

    outfile_abs = os.path.join(os.getcwd(), args.outfile)
    process_tempita(args.infile, outfile_abs)


if __name__ == "__main__":
    main()
```