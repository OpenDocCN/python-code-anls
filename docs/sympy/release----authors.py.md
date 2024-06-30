# `D:\src\scipysrc\sympy\release\authors.py`

```
#!/usr/bin/env python3

import os  # 导入操作系统模块
from pathlib import Path  # 导入路径操作模块
from subprocess import check_output  # 导入子进程模块中的输出检查函数
import unicodedata  # 导入Unicode数据模块


def main(version, prevversion, outdir):
    """
    Print authors text to put at the bottom of the release notes
    """
    outdir = Path(outdir)  # 将输出目录路径转换为Path对象
    authors, authorcount, newauthorcount = get_authors(version, prevversion)

    authors_text = f"""## Authors

The following people contributed at least one patch to this release (names are
given in alphabetical order by last name). A total of {authorcount} people
contributed to this release. People with a * by their names contributed a
patch for the first time for this release; {newauthorcount} people contributed
for the first time for this release.

Thanks to everyone who contributed to this release!
"""

    authors_lines = []
    for name in authors:
        authors_lines.append("- " + name)

    authors_text += '\n'.join(authors_lines)

    # Output to file and to screen
    with open(outdir / 'authors.txt', 'w') as authorsfile:
        authorsfile.write(authors_text)

    print()
    print(blue("Here are the authors to put at the bottom of the release notes."))
    print()
    print(authors_text)


def blue(text):
    return "\033[34m%s\033[0m" % text  # 返回带有蓝色样式的文本


def get_authors(version, prevversion):
    """
    Get the list of authors since the previous release

    Returns the list in alphabetical order by last name.  Authors who
    contributed for the first time for this release will have a star appended
    to the end of their names.

    Note: it's a good idea to use ./bin/mailmap_update.py (from the base sympy
    directory) to make AUTHORS and .mailmap up-to-date first before using
    this. fab vagrant release does this automatically.
    """
    def lastnamekey(name):
        """
        Sort key to sort by last name

        Note, we decided to sort based on the last name, because that way is
        fair. We used to sort by commit count or line number count, but that
        bumps up people who made lots of maintenance changes like updating
        mpmath or moving some files around.
        """
        # Note, this will do the wrong thing for people who have multi-word
        # last names, but there are also people with middle initials. I don't
        # know of a perfect way to handle everyone. Feel free to fix up the
        # list by hand.

        text = name.strip().split()[-1].lower()  # 获取姓名的最后一个单词，转换为小写
        # Convert things like Čertík to Certik
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')  # 规范化Unicode文本

    # The get_previous_version function can be flakey so we require the
    # previous version to be provided explicitly by the caller.
    #
    #old_release_tag = get_previous_version_tag(version)
    old_release_tag = 'sympy-' + prevversion  # 构造旧版本的标签字符串

    out = check_output(['git', '--no-pager', 'log', old_release_tag + '..', '--format=%aN'])
    releaseauthors = set(out.decode('utf-8').strip().split('\n'))  # 获取自旧版本标签以来的作者列表
    # 使用 git 命令获取指定发布标签之前的提交作者列表，输出结果作为字节流
    out = check_output(['git', '--no-pager', 'log', old_release_tag, '--format=%aN'])
    # 将字节流解码成 UTF-8 格式的字符串，去除首尾空白符后，按换行符分割成集合
    priorauthors = set(out.decode('utf-8').strip().split('\n'))

    # 去除 releaseauthors 中的空白字符串并转为集合
    releaseauthors = {name.strip() for name in releaseauthors if name.strip()}
    # 去除 priorauthors 中的空白字符串并转为集合
    priorauthors = {name.strip() for name in priorauthors if name.strip()}
    # 计算新作者集合，即在 releaseauthors 中但不在 priorauthors 中的作者
    newauthors = releaseauthors - priorauthors
    # 对新作者集合中的每个作者名后面加上 '*' 构成新的集合
    starred_newauthors = {name + "*" for name in newauthors}
    # 合并 releaseauthors 中不在 newauthors 中的作者和 starred_newauthors 构成最终的作者集合
    authors = releaseauthors - newauthors | starred_newauthors
    # 返回按作者姓氏排序后的作者列表、releaseauthors 的长度、newauthors 的长度的元组
    return (sorted(authors, key=lastnamekey), len(releaseauthors), len(newauthors))
# 获取指定版本的 SymPy 简短版本号，不包括任何rc标签（例如0.7.3）
def get_sympy_short_version(version):
    # 将版本号按点号分隔成部分
    parts = version.split('.')
    # 如果末尾部分不是数字，表示可能包含rc标签
    if not parts[-1].isdigit():
        # 如果末尾部分的第一个字符是数字，则只保留第一个字符
        if parts[-1][0].isdigit():
            parts[-1] = parts[-1][0]
        else:
            # 否则移除末尾部分
            parts.pop(-1)
    # 组合成简短版本号并返回
    return '.'.join(parts)


# 获取前一版本的标签
def get_previous_version_tag(version):
    """
    Get the version of the previous release
    """
    # 我们尝试尽可能可移植地获取 SymPy 的前一个发布的版本号。
    # 我们的策略是查看 git 标签。对于 git 标签，我们做以下假设：
    # - 标签仅用于发布版本
    # - 标签的命名遵循一致的规则:
    #   sympy-major.minor.micro[.rcnumber]
    #   （例如，sympy-0.7.2 或 sympy-0.7.2.rc1）
    # 具体来说，它会回溯标签历史，找到不包含当前短版本号作为子字符串的最近标签。
    
    # 获取 SymPy 的简短版本号
    shortversion = get_sympy_short_version(version)
    # 当前提交的指针
    curcommit = "HEAD"
    while True:
        # 构造 git 命令行，获取最近的标签
        cmdline = f'git describe --abbrev=0 --tags {curcommit}'
        print(cmdline)
        # 执行 git 命令，并解码输出结果
        curtag = check_output(cmdline.split()).decode('utf-8').strip()
        # 如果当前标签包含简短版本号，则继续向前找上一个标签
        if shortversion in curtag:
            # 如果标记的提交是一个合并提交，我们不能确保它会朝正确的方向前进。
            # 这种情况几乎不会发生，所以直接报错退出。
            cmdline = f'git rev-list --parents -n 1 {curtag}'
            print(cmdline)
            # 获取当前提交及其父提交的列表
            parents = check_output(cmdline.split()).decode('utf-8').strip().split()
            # 如果标记的提交是一个合并提交，需要手动确保 `get_previous_version_tag` 正确
            # assert len(parents) == 2, curtag
            # 当前提交更新为标记的提交的父提交
            curcommit = curtag + "^"  # 标记提交的父提交
        else:
            # 打印信息，使用当前标签作为前一个发布版本的标签
            print(blue("Using {tag} as the tag for the previous "
                       "release.".format(tag=curtag)))
            return curtag
    # 如果找不到前一个发布版本的标签，则报错退出
    sys.exit(red("Could not find the tag for the previous release."))


if __name__ == "__main__":
    import sys
    # 从命令行参数调用 main 函数并退出
    sys.exit(main(*sys.argv[1:]))
```