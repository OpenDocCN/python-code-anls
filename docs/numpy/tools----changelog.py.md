# `.\numpy\tools\changelog.py`

```py
#!/usr/bin/env python3
"""
Script to generate contributor and pull request lists

This script generates contributor and pull request lists for release
changelogs using Github v3 protocol. Use requires an authentication token in
order to have sufficient bandwidth, you can get one following the directions at
`<https://help.github.com/articles/creating-an-access-token-for-command-line-use/>_
Don't add any scope, as the default is read access to public information. The
token may be stored in an environment variable as you only get one chance to
see it.

Usage::

    $ ./tools/announce.py <token> <revision range>

The output is utf8 rst.

Dependencies
------------

- gitpython
- pygithub
- git >= 2.29.0

Some code was copied from scipy `tools/gh_list.py` and `tools/authors.py`.

Examples
--------

From the bash command line with $GITHUB token::

    $ ./tools/announce $GITHUB v1.13.0..v1.14.0 > 1.14.0-changelog.rst

"""

import os
import sys
import re
from git import Repo
from github import Github

# Initialize the GitPython Repo object for the current repository directory
this_repo = Repo(os.path.join(os.path.dirname(__file__), ".."))

# Message template for authors in the release
author_msg =\
"""
A total of %d people contributed to this release.  People with a "+" by their
names contributed a patch for the first time.
"""

# Message template for pull requests in the release
pull_request_msg =\
"""
A total of %d pull requests were merged for this release.
"""


def get_authors(revision_range):
    # Split the revision range into last and current release versions
    lst_release, cur_release = [r.strip() for r in revision_range.split('..')]
    
    # Regular expression pattern to extract authors' names from shortlog output
    authors_pat = r'^.*\t(.*)$'

    # Define git shortlog groupings for current and previous releases
    grp1 = '--group=author'
    grp2 = '--group=trailer:co-authored-by'
    
    # Get shortlog output for current release and previous release
    cur = this_repo.git.shortlog('-s', grp1, grp2, revision_range)
    pre = this_repo.git.shortlog('-s', grp1, grp2, lst_release)
    
    # Extract authors' names using the defined pattern
    authors_cur = set(re.findall(authors_pat, cur, re.M))
    authors_pre = set(re.findall(authors_pat, pre, re.M))

    # Remove specific bot contributors from the sets
    authors_cur.discard('Homu')
    authors_pre.discard('Homu')
    authors_cur.discard('dependabot-preview')
    authors_pre.discard('dependabot-preview')

    # Identify new authors by comparing current and previous sets
    authors_new = [s + ' +' for s in authors_cur - authors_pre]
    authors_old = [s for s in authors_cur & authors_pre]
    
    # Combine new and old authors, then sort alphabetically
    authors = authors_new + authors_old
    authors.sort()
    
    return authors


def get_pull_requests(repo, revision_range):
    prnums = []

    # Extract pull request numbers from regular merges
    merges = this_repo.git.log('--oneline', '--merges', revision_range)
    issues = re.findall(r"Merge pull request \#(\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # Extract pull request numbers from Homu merges (Auto merges)
    issues = re.findall(r"Auto merge of \#(\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # Extract pull request numbers from fast forward squash-merges
    commits = this_repo.git.log('--oneline', '--no-merges', '--first-parent', revision_range)
    issues = re.findall(r'^.*\((\#|gh-|gh-\#)(\d+)\)$', commits, re.M)
    prnums.extend(int(s[1]) for s in issues)
    # 对 Pull Request 编号列表进行排序
    prnums.sort()
    # 使用列表推导式获取指定仓库中每个编号对应的 Pull Request 对象
    prs = [repo.get_pull(n) for n in prnums]
    # 返回获取到的 Pull Request 对象列表
    return prs
# 主函数，用于生成指定版本范围内的作者和Pull请求列表文档
def main(token, revision_range):
    # 将输入的版本范围按照 '..' 分割并去除首尾空格，得到最旧版本和当前版本
    lst_release, cur_release = [r.strip() for r in revision_range.split('..')]

    # 使用提供的 GitHub 访问令牌创建 GitHub 对象
    github = Github(token)
    # 获取指定仓库（numpy/numpy）的 GitHub 仓库对象
    github_repo = github.get_repo('numpy/numpy')

    # 获取版本范围内的作者列表
    authors = get_authors(revision_range)
    # 设置标题
    heading = "Contributors"
    # 输出标题和分隔线
    print()
    print(heading)
    print("="*len(heading))
    # 输出作者数量信息
    print(author_msg % len(authors))

    # 遍历并输出每位作者的名称
    for s in authors:
        print('* ' + s)

    # 获取版本范围内合并的 Pull 请求列表
    pull_requests = get_pull_requests(github_repo, revision_range)
    # 设置标题
    heading = "Pull requests merged"
    # 设置格式化输出模板
    pull_msg = "* `#{0} <{1}>`__: {2}"

    # 输出标题和分隔线
    print()
    print(heading)
    print("="*len(heading))
    # 输出 Pull 请求数量信息
    print(pull_request_msg % len(pull_requests))

    # 遍历并输出每个 Pull 请求的编号、链接和标题
    for pull in pull_requests:
        title = re.sub(r"\s+", " ", pull.title.strip())
        # 如果标题超过60个字符，则截断并添加省略号
        if len(title) > 60:
            remainder = re.sub(r"\s.*$", "...", title[60:])
            if len(remainder) > 20:
                remainder = title[:80] + "..."
            else:
                title = title[:60] + remainder
        # 输出格式化后的 Pull 请求信息
        print(pull_msg.format(pull.number, pull.html_url, title))


if __name__ == "__main__":
    from argparse import ArgumentParser

    # 创建参数解析器
    parser = ArgumentParser(description="Generate author/pr lists for release")
    # 添加必需的参数：GitHub 访问令牌和版本范围
    parser.add_argument('token', help='github access token')
    parser.add_argument('revision_range', help='<revision>..<revision>')
    # 解析命令行参数
    args = parser.parse_args()
    # 调用主函数，传入 GitHub 访问令牌和版本范围参数
    main(args.token, args.revision_range)
```