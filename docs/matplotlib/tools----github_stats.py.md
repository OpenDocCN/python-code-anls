# `D:\src\scipysrc\matplotlib\tools\github_stats.py`

```py
#!/usr/bin/env python
"""
Simple tools to query github.com and gather stats about issues.

To generate a report for Matplotlib 3.0.0, run:

    python github_stats.py --milestone 3.0.0 --since-tag v2.0.0
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import sys

from argparse import ArgumentParser
from datetime import datetime, timedelta
from subprocess import check_output

from gh_api import (
    get_paged_request, make_auth_header, get_pull_request, is_pull_request,
    get_milestone_id, get_issues_list, get_authors,
)
# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
PER_PAGE = 100

REPORT_TEMPLATE = """\
.. _github-stats:

{title}
{title_underline}

GitHub statistics for {since_day} (tag: {tag}) - {today}

These lists are automatically generated, and may be incomplete or contain duplicates.

We closed {n_issues} issues and merged {n_pulls} pull requests.
{milestone}
The following {nauthors} authors contributed {ncommits} commits.

{unique_authors}
{links}

Previous GitHub statistics
--------------------------

.. toctree::
    :maxdepth: 1
    :glob:
    :reversed:

    prev_whats_new/github_stats_*"""
MILESTONE_TEMPLATE = (
    'The full list can be seen `on GitHub '
    '<https://github.com/{project}/milestone/{milestone_id}?closed=1>`__\n')
LINKS_TEMPLATE = """
GitHub issues and pull requests:

Pull Requests ({n_pulls}):

{pull_request_report}

Issues ({n_issues}):

{issue_report}
"""

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def round_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)


def _parse_datetime(s):
    """Parse dates in the format returned by the GitHub API."""
    return datetime.strptime(s, ISO8601) if s else datetime.fromtimestamp(0)


def issues2dict(issues):
    """Convert a list of issues to a dict, keyed by issue number."""
    return {i['number']: i for i in issues}


def split_pulls(all_issues, project="matplotlib/matplotlib"):
    """Split a list of closed issues into non-PR Issues and Pull Requests."""
    pulls = []
    issues = []
    for i in all_issues:
        if is_pull_request(i):
            pull = get_pull_request(project, i['number'], auth=True)
            pulls.append(pull)
        else:
            issues.append(i)
    return issues, pulls


def issues_closed_since(period=timedelta(days=365),
                        project='matplotlib/matplotlib', pulls=False):
    """
    Get all issues closed since a particular point in time.

    *period* can either be a datetime object, or a timedelta object. In the
    latter case, it is used as a time before the present.
    """
    # 根据布尔值 pulls 决定要查询的 GitHub API 资源类型
    which = 'pulls' if pulls else 'issues'

    # 根据 period 的类型确定起始时间 since
    if isinstance(period, timedelta):
        # 如果 period 是 timedelta 类型，计算当前时间减去 period 后的整点时间
        since = round_hour(datetime.utcnow() - period)
    else:
        # 否则直接使用给定的 period 作为 since
        since = period
    
    # 构建 GitHub API 请求的 URL，包括项目名、资源类型、筛选条件和每页返回数量
    url = (
        f'https://api.github.com/repos/{project}/{which}'
        f'?state=closed'
        f'&sort=updated'
        f'&since={since.strftime(ISO8601)}'
        f'&per_page={PER_PAGE}')
    
    # 发起分页请求以获取所有已关闭的 issue 或 pull request 数据
    allclosed = get_paged_request(url, headers=make_auth_header())

    # 使用生成器表达式过滤出所有关闭时间晚于 since 的记录
    filtered = (i for i in allclosed
                if _parse_datetime(i['closed_at']) > since)
    
    if pulls:
        # 如果查询的是 pull requests，进一步过滤出合并时间晚于 since 的记录
        filtered = (i for i in filtered
                    if _parse_datetime(i['merged_at']) > since)
        
        # 过滤掉不是指向主分支的 pull requests（即 backports）
        filtered = (i for i in filtered if i['base']['ref'] == 'main')
    else:
        # 如果查询的是 issues，过滤掉所有 pull requests
        filtered = (i for i in filtered if not is_pull_request(i))

    # 将过滤后的结果转换为列表并返回
    return list(filtered)
# 定义一个函数，根据指定字段对问题列表进行排序，默认按照关闭日期降序排列
def sorted_by_field(issues, field='closed_at', reverse=False):
    # 使用 Python 内置的 sorted 函数进行排序，key 参数指定排序的依据为每个问题字典的指定字段值
    return sorted(issues, key=lambda i: i[field], reverse=reverse)


# 定义一个函数，生成关于问题列表的摘要报告，可选择是否显示问题的URL链接
def report(issues, show_urls=False):
    # 初始化空列表，用于存储报告的每一行
    lines = []
    
    # 根据参数 show_urls 决定是否显示问题的URL链接
    if show_urls:
        # 遍历问题列表
        for i in issues:
            # 根据 'merged_at' 字段是否存在来确定每个问题的角色类型
            role = 'ghpull' if 'merged_at' in i else 'ghissue'
            number = i['number']  # 获取问题编号
            # 处理问题标题，替换反引号为双反引号，去除首尾空白字符
            title = i['title'].replace('`', '``').strip()
            # 构建带有角色和编号的标题行，并添加到 lines 列表中
            lines.append(f'* :{role}:`{number}`: {title}')
    else:
        # 如果不显示URL链接，则简单输出问题的编号和标题
        for i in issues:
            number = i['number']  # 获取问题编号
            # 处理问题标题，替换反引号为双反引号，去除首尾空白字符
            title = i['title'].replace('`', '``').strip()
            # 构建简化的标题行，并添加到 lines 列表中
            lines.append('* {number}: {title}')
    
    # 将列表 lines 中的所有行连接成一个字符串，以换行符分隔，并返回该字符串
    return '\n'.join(lines)

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 是否在输出中添加所有问题的 reST 链接
    show_urls = True
    
    # 创建命令行参数解析器
    parser = ArgumentParser()
    
    # 添加命令行参数，用于指定起始点的 git 标签
    parser.add_argument(
        '--since-tag', type=str,
        help='The git tag to use for the starting point '
             '(typically the last macro release).')
    
    # 添加命令行参数，用于指定 GitHub 里程碑以过滤问题（可选）
    parser.add_argument(
        '--milestone', type=str,
        help='The GitHub milestone to use for filtering issues [optional].')
    
    # 添加命令行参数，用于指定要汇总的数据天数（使用此参数或 --since-tag）
    parser.add_argument(
        '--days', type=int,
        help='The number of days of data to summarize (use this or --since-tag).')
    
    # 添加命令行参数，默认为 'matplotlib/matplotlib'，指定要汇总的项目
    parser.add_argument(
        '--project', type=str, default='matplotlib/matplotlib',
        help='The project to summarize.')
    
    # 添加命令行参数，布尔类型，默认为 False，指定是否在输出中包含所有已关闭的问题和PR的链接
    parser.add_argument(
        '--links', action='store_true', default=False,
        help='Include links to all closed Issues and PRs in the output.')
    
    # 解析命令行参数
    opts = parser.parse_args()
    
    # 获取 --since-tag 参数的值
    tag = opts.since_tag
    
    # 根据 --days 参数或 git 标签设置 'since' 变量
    if opts.days:
        since = datetime.utcnow() - timedelta(days=opts.days)
    else:
        if not tag:
            tag = check_output(['git', 'describe', '--abbrev=0'],
                               encoding='utf8').strip()
        cmd = ['git', 'log', '-1', '--format=%ai', tag]
        # 获取 git 标签的日期和时区信息，并将其转换为 datetime 对象
        tagday, tz = check_output(cmd, encoding='utf8').strip().rsplit(' ', 1)
        since = datetime.strptime(tagday, "%Y-%m-%d %H:%M:%S")
        h = int(tz[1:3])
        m = int(tz[3:])
        td = timedelta(hours=h, minutes=m)
        if tz[0] == '-':
            since += td
        else:
            since -= td
    
    # 调用 round_hour 函数，将 'since' 变量舍入到最近的整点时间
    since = round_hour(since)
    
    # 获取 --milestone 参数的值
    milestone = opts.milestone
    # 获取 --project 参数的值，默认为 'matplotlib/matplotlib'
    project = opts.project
    
    # 输出获取 GitHub 统计信息的起始时间、标签和里程碑信息
    print(f'fetching GitHub stats since {since} (tag: {tag}, milestone: {milestone})',
          file=sys.stderr)
    # 如果存在里程碑，获取里程碑的唯一标识符
    milestone_id = get_milestone_id(project=project, milestone=milestone,
                                    auth=True)
    # 根据里程碑标识符获取关闭状态的与该里程碑相关的问题和拉取请求列表
    issues_and_pulls = get_issues_list(project=project, milestone=milestone_id,
                                       state='closed', auth=True)
    # 将获取的问题和拉取请求列表分割成独立的问题和拉取请求列表
    issues, pulls = split_pulls(issues_and_pulls, project=project)

else:
    # 如果不存在里程碑，获取自指定日期以来关闭的问题列表（不包括拉取请求）
    issues = issues_closed_since(since, project=project, pulls=False)
    # 获取自指定日期以来关闭的拉取请求列表
    pulls = issues_closed_since(since, project=project, pulls=True)

# 对问题列表和拉取请求列表按照字段排序，以逆序展示（按时间排序）
issues = sorted_by_field(issues, reverse=True)
pulls = sorted_by_field(pulls, reverse=True)

# 计算问题和拉取请求的数量
n_issues, n_pulls = map(len, (issues, pulls))
# 计算关闭的总数
n_total = n_issues + n_pulls
# 将起始日期格式化为字符串
since_day = since.strftime("%Y/%m/%d")
# 获取当前日期
today = datetime.today()

# 根据给定的里程碑（去除开头的 'v'）和当前日期生成标题
title = (f'GitHub statistics for {milestone.lstrip("v")} '
         f'{today.strftime("(%b %d, %Y)")}')

# 初始化提交数量为 0，所有作者列表为空
ncommits = 0
all_authors = []

if tag:
    # 如果有标签，打印与 GitHub 信息一起的 git 信息:
    since_tag = f'{tag}..'
    # 执行 git log 命令获取自标签以来的提交数
    cmd = ['git', 'log', '--oneline', since_tag]
    ncommits += len(check_output(cmd).splitlines())

    # 执行 git log 命令获取自标签以来的作者列表
    author_cmd = ['git', 'log', '--use-mailmap', '--format=* %aN', since_tag]
    # 将作者列表添加到所有作者列表中
    all_authors.extend(
        check_output(author_cmd, encoding='utf-8', errors='replace').splitlines())

# 初始化拉取请求作者列表为空
pr_authors = []
# 遍历每个拉取请求，获取其作者列表
for pr in pulls:
    pr_authors.extend(get_authors(pr))
# 计算总的提交数量，包括拉取请求的作者数量
ncommits = len(pr_authors) + ncommits - len(pulls)
# 构建 git check-mailmap 命令用于获取拉取请求作者的邮箱
author_cmd = ['git', 'check-mailmap'] + pr_authors
# 执行 git check-mailmap 命令获取带有邮箱的作者列表
with_email = check_output(author_cmd,
                          encoding='utf-8', errors='replace').splitlines()
# 将带有邮箱的作者列表添加到所有作者列表中，并将每个条目格式化为 '* name' 形式
all_authors.extend(['* ' + a.split(' <')[0] for a in with_email])
# 对所有作者列表进行去重和排序，按照不区分大小写的方式排序
unique_authors = sorted(set(all_authors), key=lambda s: s.lower())

if milestone:
    # 如果存在里程碑，生成里程碑的字符串表示
    milestone_str = MILESTONE_TEMPLATE.format(project=project,
                                              milestone_id=milestone_id)
else:
    milestone_str = ''

if opts.links:
    # 如果选项中包含链接，则生成包含链接的模板
    links = LINKS_TEMPLATE.format(n_pulls=n_pulls,
                                  pull_request_report=report(pulls, show_urls),
                                  n_issues=n_issues,
                                  issue_report=report(issues, show_urls))
else:
    links = ''

# 打印包含摘要报告的模板，可以直接包含到发布说明中
print(REPORT_TEMPLATE.format(title=title, title_underline='=' * len(title),
                             since_day=since_day, tag=tag,
                             today=today.strftime('%Y/%m/%d'),
                             n_issues=n_issues, n_pulls=n_pulls,
                             milestone=milestone_str,
                             nauthors=len(unique_authors), ncommits=ncommits,
                             unique_authors='\n'.join(unique_authors), links=links))
```