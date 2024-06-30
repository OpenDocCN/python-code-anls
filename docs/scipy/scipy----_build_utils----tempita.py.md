# `D:\src\scipysrc\scipy\scipy\_build_utils\tempita.py`

```
#!/usr/bin/env python3
# 导入必要的模块
import sys  # 导入 sys 模块，用于获取默认编码等信息
import os   # 导入 os 模块，用于路径操作
import argparse  # 导入 argparse 模块，用于命令行参数解析

import tempita  # 导入 tempita 模块，用于处理模板文件


def process_tempita(fromfile, outfile=None):
    """处理 tempita 模板文件并写入结果。

    模板文件的扩展名应为 `.c.in` 或 `.pyx.in`：
    例如，处理 `template.c.in` 将生成 `template.c`。

    """
    # 从文件加载模板
    from_filename = tempita.Template.from_filename
    template = from_filename(fromfile,
                             encoding=sys.getdefaultencoding())

    # 替换模板中的变量，获取最终内容
    content = template.substitute()

    # 将最终内容写入输出文件
    with open(outfile, 'w') as f:
        f.write(content)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="输入文件的路径")
    parser.add_argument("-o", "--outdir", type=str,
                        help="输出目录的路径")
    parser.add_argument("--outfile", type=str,
                        help="输出文件的路径（使用此选项或 outdir）")
    parser.add_argument("-i", "--ignore", type=str,
                        help="被忽略的输入 - 可能有助于添加自定义目标之间的依赖关系")
    args = parser.parse_args()

    # 检查输入文件的扩展名
    if not args.infile.endswith('.in'):
        raise ValueError(f"意外的文件扩展名: {args.infile}")

    # 确保指定了输出路径
    if not (args.outdir or args.outfile):
        raise ValueError("缺少 `--outdir` 或 `--outfile` 参数来执行 tempita.py")

    # 确定输出文件的路径
    if args.outfile:
        outfile = args.outfile
    else:
        outdir_abs = os.path.join(os.getcwd(), args.outdir)
        outfile = os.path.join(outdir_abs,
                               os.path.splitext(os.path.split(args.infile)[1])[0])

    # 处理模板文件
    process_tempita(args.infile, outfile)


if __name__ == "__main__":
    main()
```