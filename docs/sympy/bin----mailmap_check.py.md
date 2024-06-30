# `D:\src\scipysrc\sympy\bin\mailmap_check.py`

```
# 指定Python解释器路径和字符编码
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 多行字符串，包含了关于脚本用途的详细说明和链接
"""
A tool to generate AUTHORS. We started tracking authors before moving to git,
so we have to do some manual rearrangement of the git history authors in order
to get the order in AUTHORS. bin/mailmap_check.py should be run before
committing the results.

See here for instructions on using this script:
https://docs.sympy.org/dev/contributing/new-contributors-guide/workflow-process.html#mailmap-instructions
"""

# 导入必要的未来模块
from __future__ import unicode_literals
from __future__ import print_function

# 导入系统模块
import sys

# 如果Python版本低于3.8，打印错误信息并退出脚本
if sys.version_info < (3, 8):
    sys.exit("This script requires Python 3.8 or newer")

# 导入路径操作模块、子进程模块、有序字典模块、默认字典模块和命令行参数解析模块
from pathlib import Path
from subprocess import run, PIPE
from collections import OrderedDict, defaultdict
from argparse import ArgumentParser

# 定义函数：返回当前脚本的父目录的父目录路径
def sympy_dir():
    return Path(__file__).resolve().parent.parent

# 将sympy库添加到系统路径中
sys.path.insert(0, str(sympy_dir()))

# 导入sympy库及其内部工具
import sympy
from sympy.utilities.misc import filldedent
from sympy.external.importtools import version_tuple

# 主函数定义，接受可变长度参数，用于更新.mailmap文件
def main(*args):

    # 创建参数解析器对象，描述为更新.mailmap文件
    parser = ArgumentParser(description='Update the .mailmap file')

    # 添加命令行选项：更新AUTHORS文件，不建议在拉取请求中使用此选项
    parser.add_argument('--update-authors', action='store_true',
            help=filldedent("""
            Also updates the AUTHORS file. DO NOT use this option as part of a
            pull request. The AUTHORS file will be updated later at the time a
            new version of SymPy is released."""))
    
    # 解析命令行参数
    args = parser.parse_args(args)

    # 检查git版本是否符合要求，若不符合则返回错误码1
    if not check_git_version():
        return 1

    # 尝试从git中获取作者信息，若出现断言错误则打印错误信息并返回错误码1
    try:
        git_people = get_authors_from_git()
    except AssertionError as msg:
        print(red(msg))
        return 1

    # 读取.mailmap文件的所有行内容
    lines_mailmap = read_lines(mailmap_path())

    # 定义用于排序的关键字函数key
    def key(line):
        # 如果行中包含'#'，则忽略'#'及其后内容
        if '#' in line:
            line = line.split('#')[0]
        
        # 统计'<'和'>'的数量，断言确保它们成对出现且只有一个或两个
        L, R = line.count("<"), line.count(">")
        assert L == R and L in (1, 2)
        
        # 返回小写的第一个电子邮件地址作为关键字
        return line.split(">", 1)[0].split("<")[1].lower()

    # 使用有序字典who存储按关键字排序的.mailmap文件行内容
    who = OrderedDict()
    for i, line in enumerate(lines_mailmap):
        try:
            who.setdefault(key(line), []).append(line)
        except AssertionError:
            who[i] = [line]

    # 初始化标志变量：问题、缺失、模糊和重复
    problems = False
    missing = False
    ambiguous = False
    dups = defaultdict(list)

    #
    # Here we use the git people with the most recent commit skipped. This
    # means we don't need to add .mailmap entries for the temporary merge
    # commit created in CI on a PR.
    #
    # 遍历 git_people 列表中的每个人员
    for person in git_people:
        # 获取该人员的电子邮件地址
        email = key(person)
        # 将当前人员添加到 dups 字典中的对应电子邮件地址的列表中
        dups[email].append(person)
        # 如果该电子邮件地址不在 who 字典中
        if email not in who:
            # 输出警告信息，指出该作者在 .mailmap 文件中未包含
            print(red("This author is not included in the .mailmap file:"))
            # 输出当前人员信息
            print(person)
            # 标记有遗漏的情况为真
            missing = True
        # 否则，如果该电子邮件地址在 who 字典中的条目中没有以当前人员名字开头的
        elif not any(p.startswith(person) for p in who[email]):
            # 输出警告信息，指出.mailmap 文件中存在歧义的名字
            print(red("Ambiguous names in .mailmap"))
            # 输出警告信息，指出该电子邮件地址出现在多个条目中
            print(red("This email address appears for multiple entries:"))
            # 输出当前人员信息
            print('Person:', person)
            # 输出 mailmap 条目信息
            print('Mailmap entries:')
            # 遍历该电子邮件地址在 who 字典中的所有条目
            for line in who[email]:
                # 输出每一条 mailmap 条目
                print(line)
            # 标记存在歧义的情况为真
            ambiguous = True

    # 如果存在遗漏的情况
    if missing:
        # 输出警告信息，提示需要更新 .mailmap 文件，因为存在未识别作者/电子邮件元数据的提交
        print(red(filldedent("""
        The .mailmap file needs to be updated because there are commits with
        unrecognised author/email metadata.
        """)))
        # 标记存在问题为真
        problems = True

    # 如果存在歧义的情况
    if ambiguous:
        # 输出警告信息，提示需要向 .mailmap 文件添加条目以指示所有提交的正确姓名和电子邮件别名
        print(red(filldedent("""
        Lines should be added to .mailmap to indicate the correct name and
        email aliases for all commits.
        """)))
        # 标记存在问题为真
        problems = True

    # 遍历 dups 字典中的每个条目
    for email, commitauthors in dups.items():
        # 如果同一电子邮件地址对应的提交作者数量大于2
        if len(commitauthors) > 2:
            # 输出警告信息，提示同一/歧义电子邮件地址记录了不同的元数据，需要更新 .mailmap 文件
            print(red(filldedent("""
            The following commits are recorded with different metadata but the
            same/ambiguous email address. The .mailmap file will need to be
            updated.""")))
            # 遍历输出所有涉及的提交作者
            for author in commitauthors:
                print(author)
            # 标记存在问题为真
            problems = True

    # 对 lines_mailmap 进行排序并赋值给 lines_mailmap_sorted
    lines_mailmap_sorted = sort_lines_mailmap(lines_mailmap)
    # 将排序后的 lines_mailmap_sorted 写入到 mailmap 文件中
    write_lines(mailmap_path(), lines_mailmap_sorted)

    # 如果排序后的 lines_mailmap_sorted 与原始 lines_mailmap 不同
    if lines_mailmap_sorted != lines_mailmap:
        # 标记存在问题为真
        problems = True
        # 输出警告信息，指示 mailmap 文件已重新排序
        print(red("The mailmap file was reordered"))

    # 检查是否需要更新 AUTHORS 文件
    #
    # 在这里，我们不跳过最后一次提交。如果更新了 AUTHORS 文件，我们需要最近提交的作者
    # 生成 AUTHORS 文件中的行列表
    lines_authors = make_authors_file_lines(git_people)
    # 读取当前 AUTHORS 文件的行列表
    old_lines_authors = read_lines(authors_path())

    # 遍历当前 AUTHORS 文件中的每个人员，从第9行开始
    for person in old_lines_authors[8:]:
        # 如果该人员不在 git_people 列表中
        if person not in git_people:
            # 输出警告信息，指出该作者在 AUTHORS 文件中存在，但不在 .mailmap 中
            print(red("This author is in the AUTHORS file but not .mailmap:"))
            # 输出当前人员信息
            print(person)
            # 标记存在问题为真
            problems = True

    # 如果存在问题
    if problems:
        # 输出警告信息，提醒查看如何更新 .mailmap 文件的说明
        print(red(filldedent("""
        For instructions on updating the .mailmap file see:
        """)))
def update_mailmap(args):
    # 更新 .mailmap 文件中的映射关系
    if args.mailmap:
        # 如果指定了 mailmap 参数，则按指定的文件路径更新 .mailmap 文件
        write_lines(mailmap_path(), args.mailmap)
        print(red("Changes were made in the .mailmap file"))
    else:
        # 否则，输出未发现需要更新的 .mailmap 文件
        print(green("No changes needed in .mailmap"))

    # 实际更新 AUTHORS 文件（如果传入了 --update-authors 参数）
    authors_changed = update_authors_file(lines_authors, old_lines_authors, args.update_authors)

    # 返回问题数量和作者变更数量的总和
    return int(problems) + int(authors_changed)


def update_authors_file(lines, old_lines, update_yesno):
    # 检查是否 AUTHORS 文件无需更新
    if old_lines == lines:
        print(green('No changes needed in AUTHORS.'))
        return 0

    # 是否要实际写入文件变更？
    if update_yesno:
        write_lines(authors_path(), lines)
        print(red("Changes were made in the authors file"))

    # 检查是否有新的作者被添加
    new_authors = []
    for i in sorted(set(lines) - set(old_lines)):
        try:
            author_name(i)
            new_authors.append(i)
        except AssertionError:
            continue

    # 如果有新作者被添加，则显示相关信息
    if new_authors:
        if update_yesno:
            print(yellow("The following authors were added to AUTHORS."))
        else:
            print(green(filldedent("""
                The following authors will be added to the AUTHORS file at the
                time of the next SymPy release.""")))
        print()
        for i in sorted(new_authors, key=lambda x: x.lower()):
            print('\t%s' % i)

    # 根据是否有新作者被添加和是否要更新，返回对应的值
    if new_authors and update_yesno:
        return 1
    else:
        return 0


def check_git_version():
    # 检查 Git 的版本是否符合要求
    minimal = '1.8.4.2'
    git_ver = run(['git', '--version'], stdout=PIPE, encoding='utf-8').stdout[12:]
    if version_tuple(git_ver) < version_tuple(minimal):
        print(yellow("Please use a git version >= %s" % minimal))
        return False
    else:
        return True


def authors_path():
    # 返回 AUTHORS 文件的路径
    return sympy_dir() / 'AUTHORS'


def mailmap_path():
    # 返回 .mailmap 文件的路径
    return sympy_dir() / '.mailmap'


def red(text):
    # 返回红色格式化后的文本
    return "\033[31m%s\033[0m" % text


def yellow(text):
    # 返回黄色格式化后的文本
    return "\033[33m%s\033[0m" % text


def green(text):
    # 返回绿色格式化后的文本
    return "\033[32m%s\033[0m" % text


def author_name(line):
    # 从给定行中解析作者姓名
    assert line.count("<") == line.count(">") == 1
    assert line.endswith(">")
    return line.split("<", 1)[0].strip()


def get_authors_from_git():
    # 从 Git 日志中获取作者列表
    git_command = ["git", "log", "--topo-order", "--reverse", "--format=%aN <%aE>"]

    # 获取父提交数量
    parents = run(["git", "rev-list", "--no-walk", "--count", "HEAD^@"],
                  stdout=PIPE, encoding='utf-8').stdout.strip()
    if parents != '1':
        # 跳过最近的提交，用于在 CI 中忽略合并提交时创建的冲突合并提交
        git_command.append("HEAD^"+parents)
    # 运行指定的 git 命令，获取输出，并按行划分成列表
    git_people = run(git_command, stdout=PIPE, encoding='utf-8').stdout.strip().split("\n")

    # 去除重复项，保持原始顺序
    git_people = list(OrderedDict.fromkeys(git_people))

    # 定义函数 move，用于移动列表中的元素以生成 AUTHORS 文件
    def move(l, i1, i2, who):
        # 移除指定索引处的元素，并在另一个位置插入
        x = l.pop(i1)
        # 检查移动的正确性，需确认 .mailmap 文件设置正确
        assert who == author_name(x), \
            '%s was not found at line %i' % (who, i1)
        l.insert(i2, x)

    # 根据 AUTHORS 文件的生成规则调整列表元素顺序和内容
    move(git_people, 2, 0, 'Ondřej Čertík')
    move(git_people, 42, 1, 'Fabian Pedregosa')
    move(git_people, 22, 2, 'Jurjen N.E. Bos')
    git_people.insert(4, "*Marc-Etienne M.Leveille <protonyc@gmail.com>")
    move(git_people, 10, 5, 'Brian Jorgensen')
    git_people.insert(11, "*Ulrich Hecht <ulrich.hecht@gmail.com>")
    # 再次确认 .mailmap 文件设置正确，确保移除的元素正确
    assert 'Kirill Smelkov' == author_name(git_people.pop(12)), 'Kirill Smelkov was not found at line 12'
    move(git_people, 12, 32, 'Sebastian Krämer')
    move(git_people, 227, 35, 'Case Van Horsen')
    git_people.insert(43, "*Dan <coolg49964@gmail.com>")
    move(git_people, 57, 59, 'Aaron Meurer')
    move(git_people, 58, 57, 'Andrew Docherty')
    move(git_people, 67, 66, 'Chris Smith')
    move(git_people, 79, 76, 'Kevin Goodsell')
    git_people.insert(84, "*Chu-Ching Huang <cchuang@mail.cgu.edu.tw>")
    move(git_people, 93, 92, 'James Pearson')
    # 再次确认 .mailmap 文件设置正确，确保移除的元素正确
    assert 'Sergey B Kirpichev' == author_name(git_people.pop(226)), 'Sergey B Kirpichev was not found at line 226.'

    # 查找并移除指定的机器人作者条目
    index = git_people.index("azure-pipelines[bot] <azure-pipelines[bot]@users.noreply.github.com>")
    git_people.pop(index)
    index = git_people.index("whitesource-bolt-for-github[bot] <whitesource-bolt-for-github[bot]@users.noreply.github.com>")
    git_people.pop(index)

    # 返回生成的 git 作者列表
    return git_people
# 定义一个函数，生成用于作者文件的文本行列表
def make_authors_file_lines(git_people):
    # 定义文件的头部信息，包括关于贡献者的说明和自动生成的声明
    header = filldedent("""
        All people who contributed to SymPy by sending at least a patch or
        more (in the order of the date of their first contribution), except
        those who explicitly didn't want to be mentioned. People with a * next
        to their names are not found in the metadata of the git history. This
        file is generated automatically by running `./bin/authors_update.py`.
        """).lstrip()
    # 添加关于作者总数的附加信息
    header_extra = "There are a total of %d authors."  % len(git_people)
    # 将头部信息按行分割为列表
    lines = header.splitlines()
    # 添加空行
    lines.append('')
    # 添加附加信息行
    lines.append(header_extra)
    # 添加空行
    lines.append('')
    # 将所有的贡献者信息追加到行列表中
    lines.extend(git_people)
    # 返回生成的文本行列表
    return lines


# 定义一个函数，按照 mailmap 文件的规则排序文本行
def sort_lines_mailmap(lines):
    # 查找非注释行的起始位置
    for n, line in enumerate(lines):
        if not line.startswith('#'):
            header_end = n
            break
    # 分割出头部信息和 mailmap 文件内容
    header = lines[:header_end]
    mailmap_lines = lines[header_end:]
    # 对 mailmap 文件内容进行排序，并将头部信息和排序后的内容合并返回
    return header + sorted(mailmap_lines)


# 定义一个函数，读取指定路径文件的所有行并返回
def read_lines(path):
    # 使用 UTF-8 编码打开文件，并返回去除换行符的每行内容列表
    with open(path, 'r', encoding='utf-8') as fin:
        return [line.strip() for line in fin.readlines()]


# 定义一个函数，将给定的文本行列表写入指定路径的文件中
def write_lines(path, lines):
    # 使用 UTF-8 编码以写模式打开文件
    with open(path, 'w', encoding='utf-8', newline='') as fout:
        # 将文本行列表写入文件，每行后添加换行符
        fout.write('\n'.join(lines))
        fout.write('\n')


# 如果当前脚本被作为主程序执行，则调用 main 函数并传入命令行参数
if __name__ == "__main__":
    import sys
    # 导入 sys 模块并使用命令行参数调用 main 函数，然后退出程序
    sys.exit(main(*sys.argv[1:]))
```