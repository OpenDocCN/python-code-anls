# `D:\src\scipysrc\sympy\release\update_docs.py`

```
#!/usr/bin/env python3

import json
import subprocess
import sys
from os.path import join, splitext, basename
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from shutil import copytree

def main(sympy_doc_git, doc_html_zip, version, dev_version, push=None):
    """Run this as ./update_docs.py SYMPY_DOC_GIT DOC_HTML_ZIP VERSION [--push]

    !!!!!!!!!!!!!!!!!
    NOTE: This is intended to be run as part of the release script.
    NOTE: This script will automatically push to the sympy_doc repo.
    !!!!!!!!!!!!!!!!!

    Args
    ====

    SYMPY_DOC_GIT: Path to the sympy_doc repo.
    DOC_HTML_ZIP: Path to the zip of the built html docs.
    VERSION: Version string of the release (e.g. "1.6")
    DEV_VERSION: Version string of the development version (e.g. "1.7.dev")
    --push (optional): Push the results (Warning this pushes direct to github)

    This script automates the "release docs" step described in the README of the
    sympy/sympy_doc repo:

    https://github.com/sympy/sympy_doc#release-docs
    """
    if push is None:
        push = False
    elif push == "--push":
        push = True
    else:
        raise ValueError("Invalid arguments")

    update_docs(sympy_doc_git, doc_html_zip, version, dev_version, push)


def update_docs(sympy_doc_git, doc_html_zip, version, dev_version, push):
    """Update documentation for a SymPy release.

    Args:
    - sympy_doc_git: Path to the sympy_doc repository.
    - doc_html_zip: Path to the zip file containing built HTML documentation.
    - version: Version string of the release.
    - dev_version: Version string of the development version.
    - push: Boolean indicating whether to push changes to the remote repository.

    This function executes the necessary steps to update the documentation and
    versions.json for a SymPy release.
    """

    # We started with a clean tree so restore it on error
    with git_rollback_on_error(sympy_doc_git, branch='gh-pages') as run:

        # Delete docs for the last version
        run('git', 'rm', '-rf', 'latest')

        # Extract new docs in replacement
        extract_docs(sympy_doc_git, doc_html_zip)

        # Commit new docs
        run('git', 'add', 'latest')
        run('git', 'commit', '-m', 'Add sympy %s docs' % version)

        # Update versions.json
        with open(join(sympy_doc_git, 'versions.json'), 'w') as f:
            json.dump({'latest': version, 'dev': dev_version}, f)
        run('git', 'diff')
        run('git', 'add', 'versions.json')
        run('git', 'commit', '-m', 'Update versions.json')

        if push:
            run('git', 'push')
        else:
            print('Results are committed but not pushed')


@contextmanager
def git_rollback_on_error(gitroot_path, branch='master'):
    """Context manager to handle git operations safely.

    Args:
    - gitroot_path: Path to the root of the git repository.
    - branch: Branch name to operate on (default is 'master').

    Yields:
    - run: Function to run git commands in the specified repository.

    Raises:
    - ValueError: If the repository is not in a clean state.
    """

    def run(*cmdline, **kwargs):
        """Run subprocess with cwd in sympy_doc"""
        print()
        print('Running: $ ' + ' '.join(cmdline))
        print()
        return subprocess.run(cmdline, cwd=gitroot_path, check=True, **kwargs)

    unclean_msg = "The git repo should be completely clean before running this"

    try:
        run('git', 'diff', '--exit-code') # Error if tree is unclean
    except subprocess.CalledProcessError:
        raise ValueError(unclean_msg)
    if run('git', 'clean', '-n', stdout=subprocess.PIPE).stdout:
        raise ValueError(unclean_msg)

    run('git', 'checkout', branch)
    run('git', 'pull')
    # 使用 subprocess 模块运行命令 'git rev-parse HEAD'，并将标准输出捕获到 bsha_start 变量
    bsha_start = run('git', 'rev-parse', 'HEAD', stdout=subprocess.PIPE).stdout
    
    # 将 bsha_start 的值去除首尾空白字符后，按 ASCII 解码为字符串，存入 sha_start 变量
    sha_start = bsha_start.strip().decode('ascii')

    try:
        # 尝试执行 yield run，yield 返回一个迭代器，可能用于生成器函数中
        yield run
    except Exception as e:
        # 如果出现异常，运行 'git reset --hard' 命令来回到之前的 Git 版本，使用 sha_start 变量存储的 SHA 标识
        run('git', 'reset', '--hard', sha_start)
        # 抛出捕获的异常，但将异常的原因（cause）设为 None
        raise e from None
# 定义一个函数，用于从 sympy 文档的 Git 仓库中提取文档
def extract_docs(sympy_doc_git, doc_html_zip):
    # 提取压缩文件的基本文件名（不包含扩展名）
    subdirname = splitext(basename(doc_html_zip))[0]

    # 使用临时目录来处理文件
    with TemporaryDirectory() as tempdir:
        # 打印信息：正在将文档解压至临时目录
        print()
        print('Extracting docs to ' + tempdir)
        print()
        
        # 解压 HTML 文档压缩文件到临时目录
        ZipFile(doc_html_zip).extractall(tempdir)

        # 打印信息：正在复制文档到 sympy_doc/latest
        print()
        print('Copying to sympy_doc/latest')
        print()

        # 源文件夹路径为解压后的子目录路径
        srcpath = join(tempdir, subdirname)
        # 目标文件夹路径为 sympy_doc_git 仓库中的 latest 文件夹路径
        dstpath = join(sympy_doc_git, 'latest')
        # 递归复制源文件夹内容到目标文件夹
        copytree(srcpath, dstpath)

# 如果当前脚本作为主程序运行，则调用 main 函数并传入命令行参数
if __name__ == "__main__":
    main(*sys.argv[1:])
```