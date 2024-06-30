# `D:\src\scipysrc\sympy\release\build_docs.py`

```
#!/usr/bin/env python3
# 指定脚本的解释器为 Python 3

import os
from os.path import dirname, join, basename, normpath
from os import chdir
import shutil

from helpers import run  # 导入自定义的运行辅助函数


ROOTDIR = dirname(dirname(__file__))  # 获取当前脚本的父目录的父目录
DOCSDIR = join(ROOTDIR, 'doc')  # 设置文档目录路径为父目录下的 'doc' 子目录


def main(version, outputdir):
    os.makedirs(outputdir, exist_ok=True)  # 创建输出目录，如果不存在则创建，存在则忽略
    build_html(DOCSDIR, outputdir, version)  # 调用构建 HTML 文档的函数
    build_latex(DOCSDIR, outputdir, version)  # 调用构建 LaTeX PDF 文档的函数


def build_html(docsdir, outputdir, version):
    run('make', 'clean', cwd=docsdir)  # 在文档目录下执行清理操作
    run('make', 'html', cwd=docsdir)  # 在文档目录下执行生成 HTML 文档操作

    builddir = join(docsdir, '_build')  # 设置文档构建后的目录路径
    docsname = 'sympy-docs-html-%s' % (version,)  # 设置 HTML 文档的名称
    zipname = docsname + '.zip'  # 设置生成的 ZIP 文件名
    cwd = os.getcwd()  # 获取当前工作目录
    try:
        chdir(builddir)  # 切换到文档构建目录
        shutil.move('html', docsname)  # 将生成的 HTML 文件夹重命名为指定名称
        run('zip', '-9lr', zipname, docsname)  # 在当前目录下压缩指定文件夹为 ZIP 文件
    finally:
        chdir(cwd)  # 最终恢复到之前的工作目录
    shutil.move(join(builddir, zipname), join(outputdir, zipname))  # 将生成的 ZIP 文件移动到输出目录


def build_latex(docsdir, outputdir, version):
    run('make', 'clean', cwd=docsdir)  # 在文档目录下执行清理操作
    run('make', 'latexpdf', cwd=docsdir)  # 在文档目录下执行生成 LaTeX PDF 文档操作

    srcfilename = 'sympy-%s.pdf' % (version,)  # 设置源 LaTeX PDF 文件名
    dstfilename = 'sympy-docs-pdf-%s.pdf' % (version,)  # 设置目标 LaTeX PDF 文件名
    src = join('doc', '_build', 'latex', srcfilename)  # 设置源 LaTeX PDF 文件的完整路径
    dst = join(outputdir, dstfilename)  # 设置目标 LaTeX PDF 文件的完整路径
    shutil.copyfile(src, dst)  # 复制生成的 LaTeX PDF 文件到输出目录


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])  # 从命令行参数获取版本号和输出目录，并调用主函数进行处理
```