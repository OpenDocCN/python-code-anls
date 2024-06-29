# `D:\src\scipysrc\pandas\doc\sphinxext\announce.py`

```
"""
Script to generate contributor and pull request lists

This script generates contributor and pull request lists for release
announcements using GitHub v3 protocol. Use requires an authentication token in
order to have sufficient bandwidth, you can get one following the directions at
`<https://help.github.com/articles/creating-an-access-token-for-command-line-use/>_
Don't add any scope, as the default is read access to public information. The
token may be stored in an environment variable as you only get one chance to
see it.

Usage::

    $ ./scripts/announce.py <token> <revision range>

The output is utf8 rst.

Dependencies
------------

- gitpython
- pygithub

Some code was copied from scipy `tools/gh_lists.py` and `tools/authors.py`.

Examples
--------

From the bash command line with $GITHUB token.

    $ ./scripts/announce.py $GITHUB v1.11.0..v1.11.1 > announce.rst

"""

import codecs
import os
import re
import textwrap

from git import Repo

# Contributors to be renamed.
CONTRIBUTOR_MAPPING = {"znkjnffrezna": "znetbgcubravk"}

UTF8Writer = codecs.getwriter("utf8")
this_repo = Repo(os.path.join(os.path.dirname(__file__), "..", ".."))

author_msg = """\
A total of %d people contributed patches to this release.  People with a
"+" by their names contributed a patch for the first time.
"""

pull_request_msg = """\
A total of %d pull requests were merged for this release.
"""


def get_authors(revision_range):
    # 正则表达式模式，用于匹配 Git 短日志输出的作者信息
    pat = "^.*\\t(.*)$"
    # 分割出最后版本和当前版本
    lst_release, cur_release = (r.strip() for r in revision_range.split(".."))

    if "|" in cur_release:
        # 处理特殊情况，例如 v1.0.1|HEAD
        maybe_tag, head = cur_release.split("|")
        assert head == "HEAD"
        if maybe_tag in this_repo.tags:
            cur_release = maybe_tag
        else:
            cur_release = head
        revision_range = f"{lst_release}..{cur_release}"

    # 获取当前版本和之前版本的作者列表
    # 第一个 pass 用于获取“Co-authored by”提交的作者信息，这些来自机器人的后移
    xpr = re.compile(r"Co-authored-by: (?P<name>[^<]+) ")
    cur = set(
        xpr.findall(
            this_repo.git.log("--grep=Co-authored", "--pretty=%b", revision_range)
        )
    )
    # 添加普通提交的作者信息
    cur |= set(re.findall(pat, this_repo.git.shortlog("-s", revision_range), re.M))

    # 获取之前版本的作者列表
    pre = set(
        xpr.findall(this_repo.git.log("--grep=Co-authored", "--pretty=%b", lst_release))
    )
    pre |= set(re.findall(pat, this_repo.git.shortlog("-s", lst_release), re.M))

    # 清理自动合并的作者
    cur.discard("Homu")
    pre.discard("Homu")

    # 根据映射重命名贡献者
    # 遍历 CONTRIBUTOR_MAPPING 字典中的每个键值对，键为原作者名，值为新作者名
    for old_name, new_name in CONTRIBUTOR_MAPPING.items():
        # 对原作者名进行 ROT13 解码
        old_name_decoded = codecs.decode(old_name, "rot13")
        # 对新作者名进行 ROT13 解码
        new_name_decoded = codecs.decode(new_name, "rot13")
        
        # 如果解码后的原作者名存在于 pre 集合中
        if old_name_decoded in pre:
            # 从 pre 集合中移除解码后的原作者名
            pre.discard(old_name_decoded)
            # 将解码后的新作者名添加到 pre 集合中
            pre.add(new_name_decoded)
        
        # 如果解码后的原作者名存在于 cur 集合中
        if old_name_decoded in cur:
            # 从 cur 集合中移除解码后的原作者名
            cur.discard(old_name_decoded)
            # 将解码后的新作者名添加到 cur 集合中
            cur.add(new_name_decoded)

    # 为当前作者列表中的新作者名加上 '+' 符号
    authors = [s + " +" for s in cur - pre] + list(cur & pre)
    # 对作者列表按字母顺序进行排序
    authors.sort()
    # 返回排序后的作者列表
    return authors
# 定义一个函数，用于获取给定版本范围内的所有拉取请求对象列表
def get_pull_requests(repo, revision_range):
    prnums = []  # 初始化一个空列表，用于存放拉取请求的编号

    # 从正常的合并记录中查找拉取请求
    merges = this_repo.git.log("--oneline", "--merges", revision_range)
    issues = re.findall("Merge pull request \\#(\\d*)", merges)
    prnums.extend(int(s) for s in issues)  # 将找到的拉取请求编号转换为整数并加入列表中

    # 从 Homu 自动合并记录中查找拉取请求
    issues = re.findall("Auto merge of \\#(\\d*)", merges)
    prnums.extend(int(s) for s in issues)  # 将找到的拉取请求编号转换为整数并加入列表中

    # 从快进合并记录中查找拉取请求
    commits = this_repo.git.log(
        "--oneline", "--no-merges", "--first-parent", revision_range
    )
    issues = re.findall("^.*\\(\\#(\\d+)\\)$", commits, re.M)
    prnums.extend(int(s) for s in issues)  # 将找到的拉取请求编号转换为整数并加入列表中

    # 拉取请求编号排序
    prnums.sort()

    # 根据编号获取 GitHub 仓库中的拉取请求对象列表
    prs = [repo.get_pull(n) for n in prnums]

    return prs


# 构建版本组件信息
def build_components(revision_range, heading="Contributors"):
    lst_release, cur_release = (r.strip() for r in revision_range.split(".."))
    authors = get_authors(revision_range)  # 调用函数获取作者列表

    return {
        "heading": heading,
        "author_message": author_msg % len(authors),  # 格式化作者消息
        "authors": authors,  # 作者列表
    }


# 构建输出字符串
def build_string(revision_range, heading="Contributors"):
    components = build_components(revision_range, heading=heading)
    components["uline"] = "=" * len(components["heading"])  # 根据标题长度生成下划线
    components["authors"] = "* " + "\n* ".join(components["authors"])  # 生成作者列表的格式化字符串

    # 不要将这里改为 f-string，否则会破坏格式
    tpl = textwrap.dedent(
        """\
    {heading}
    {uline}

    {author_message}
    {authors}"""
    ).format(**components)  # 使用组件填充模板

    return tpl


# 主函数，生成版本发布的作者列表文档并打印输出
def main(revision_range):
    text = build_string(revision_range)  # 调用函数生成作者列表文档
    print(text)  # 打印输出文档内容


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate author lists for release")
    parser.add_argument("revision_range", help="<revision>..<revision>")  # 解析命令行参数
    args = parser.parse_args()
    main(args.revision_range)  # 调用主函数处理参数
```