# `D:\src\scipysrc\scipy\tools\gh_lists.py`

```
#!/usr/bin/env python3
"""
gh_lists.py MILESTONE

Functions for Github API requests.
"""
import os
import re
import sys
import json
import collections
import argparse
import datetime
import time

from urllib.request import urlopen, Request, HTTPError

# 定义一个命名元组 Issue，用于表示 GitHub 的 issue，包含 id、title 和 url 三个字段
Issue = collections.namedtuple('Issue', ('id', 'title', 'url'))

# 主函数，负责处理命令行参数，并调用其他函数获取并打印 GitHub 项目中的 issue 和 pull request 列表
def main():
    # 创建参数解析器
    p = argparse.ArgumentParser(usage=__doc__.lstrip())
    # 添加命令行参数选项，--project 用于指定 GitHub 项目，默认为 'scipy/scipy'
    # 'milestone' 为必选参数，表示要查询的里程碑
    p.add_argument('--project', default='scipy/scipy')
    p.add_argument('milestone')
    args = p.parse_args()

    # 创建一个 CachedGet 对象，用于缓存 GitHub API 请求结果，实际请求由 GithubGet 对象处理
    getter = CachedGet('gh_cache.json', GithubGet())

    try:
        # 获取指定项目的所有里程碑
        milestones = get_milestones(getter, args.project)
        # 如果指定的里程碑不在返回的里程碑列表中，抛出错误信息
        if args.milestone not in milestones:
            msg = "Milestone {0} not available. Available milestones: {1}"
            msg = msg.format(args.milestone, ", ".join(sorted(milestones)))
            p.error(msg)

        # 获取指定里程碑下的所有 issue 列表，并按照 issue 的 id 排序
        issues = get_issues(getter, args.project, args.milestone)
        issues.sort()
    finally:
        # 保存缓存的 API 请求结果
        getter.save()

    # 分离出 pull request 和普通 issue 的列表
    prs = [x for x in issues if '/pull/' in x.url]
    issues = [x for x in issues if x not in prs]

    # 定义打印列表的函数，包括处理标题内容的逻辑
    def print_list(title, items):
        print()
        print(title)
        print("-"*len(title))
        print()

        # 定义一个函数用于替换标题中的特殊字符，保证格式正确
        def backtick_repl(matchobj):
            if matchobj.group(2) != ' ':
                post = '\ ' + matchobj.group(2)
            else:
                post = matchobj.group(2)
            return '``' + matchobj.group(1) + '``' + post

        # 遍历输出每个 issue 或 pull request 的信息
        for issue in items:
            # 格式化输出信息，确保标题格式正确并截断过长的标题
            msg = "* `#{0} <{1}>`__: {2}"
            title = re.sub("\\s+", " ", issue.title.strip())
            title = re.sub("([^`]|^)`([^`]|$)", "\g<1>``\g<2>", title)
            title = re.sub("``(.*?)``(.)", backtick_repl, title)
            title = title.replace('*', '\\*')
            if len(title) > 60:
                remainder = re.sub("\\s.*$", "...", title[60:])
                if len(remainder) > 20:
                    title = title[:80] + "..."
                else:
                    title = title[:60] + remainder
                if title.count('`') % 4 != 0:
                    title = title[:-3] + '``...'
            msg = msg.format(issue.id, issue.url, title)
            print(msg)
        print()

    # 打印普通 issue 列表
    msg = f"Issues closed for {args.milestone}"
    print_list(msg, issues)

    # 打印 pull request 列表
    msg = f"Pull requests for {args.milestone}"
    print_list(msg, prs)

    return 0

# 获取指定项目的所有里程碑，并返回一个字典，键为里程碑的标题，值为里程碑的编号
def get_milestones(getter, project):
    url = f"https://api.github.com/repos/{project}/milestones"
    # 调用 getter 对象的 get 方法获取 API 返回的 JSON 数据
    data = getter.get(url)

    # 解析 JSON 数据，构建里程碑标题到编号的映射字典
    milestones = {}
    for ms in data:
        milestones[ms['title']] = ms['number']
    return milestones

# 获取指定项目指定里程碑下的所有 issue 列表，并返回一个 Issue 命名元组的列表
def get_issues(getter, project, milestone):
    # 获取指定项目的所有里程碑
    milestones = get_milestones(getter, project)
    # 获取指定里程碑的编号
    mid = milestones[milestone]
    # 构建 GitHub API 的 URL，用于获取特定里程碑下已关闭的所有问题和合并请求
    url = "https://api.github.com/repos/{project}/issues?milestone={mid}&state=closed&sort=created&direction=asc"
    url = url.format(project=project, mid=mid)
    
    # 使用 getter 对象发起 GET 请求，获取数据
    data = getter.get(url)
    
    # 初始化一个空列表来存储所有的问题对象
    issues = []
    
    # 遍历获取的数据列表中的每一项，这些数据既包括问题（issues）也包括合并请求（pull requests）
    for issue_data in data:
        # 如果是合并请求（PR），检查其合并状态
        if "pull" in issue_data['html_url']:
            merge_status = issue_data['pull_request']['merged_at']
            # 如果合并状态为 None（即未被合并），跳过该合并请求
            if merge_status is None:
                continue
        # 将问题或已合并的合并请求的信息添加到 issues 列表中作为 Issue 对象
        issues.append(Issue(issue_data['number'],
                            issue_data['title'],
                            issue_data['html_url']))
    
    # 返回包含所有符合条件的 Issue 对象的列表
    return issues
# 定义一个缓存 GET 请求结果的类 CachedGet
class CachedGet:
    # 初始化方法，接受文件名和一个 getter 函数作为参数
    def __init__(self, filename, getter):
        self._getter = getter  # 缓存数据获取函数

        self.filename = filename  # 缓存文件名
        # 如果文件存在，加载缓存数据
        if os.path.isfile(filename):
            print(f"[gh_lists] using {filename} as cache "
                  f"(remove it if you want fresh data)",
                  file=sys.stderr)
            with open(filename, encoding='utf-8') as f:
                self.cache = json.load(f)  # 加载缓存数据到 self.cache
        else:
            self.cache = {}  # 如果文件不存在，则初始化为空字典

    # 获取数据的方法，参数为 URL
    def get(self, url):
        # 如果 URL 不在缓存中
        if url not in self.cache:
            # 调用 self._getter 的 get_multipage 方法获取数据
            data = self._getter.get_multipage(url)
            self.cache[url] = data  # 将获取的数据存入缓存
            return data  # 返回获取的数据
        else:
            # 如果 URL 在缓存中，输出缓存命中信息
            print("[gh_lists] (cached):", url, file=sys.stderr, flush=True)
            return self.cache[url]  # 直接返回缓存中的数据

    # 将缓存保存到文件的方法
    def save(self):
        tmp = self.filename + ".new"  # 创建临时文件名
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f)  # 将缓存数据以 JSON 格式保存到临时文件
        os.rename(tmp, self.filename)  # 将临时文件重命名为正式的缓存文件名


# 定义一个获取 GitHub 数据的类 GithubGet
class GithubGet:
    # 初始化方法，接受一个是否需要认证的参数
    def __init__(self, auth=False):
        self.headers = {'User-Agent': 'gh_lists.py',  # 设置 HTTP 请求头中的 User-Agent
                        'Accept': 'application/vnd.github.v3+json'}  # 设置接受的数据类型

        if auth:
            self.authenticate()  # 如果需要认证，则调用 authenticate 方法进行认证

        # 发送请求获取 GitHub API 的限制信息
        req = self.urlopen('https://api.github.com/rate_limit')
        try:
            if req.getcode() != 200:
                raise RuntimeError()  # 如果请求不成功，抛出运行时错误
            info = json.loads(req.read().decode('utf-8'))  # 解析并加载响应数据
        finally:
            req.close()  # 关闭请求

        self.ratelimit_remaining = int(info['rate']['remaining'])  # 设置剩余请求次数
        self.ratelimit_reset = float(info['rate']['reset'])  # 设置限制重置时间戳

    # 认证方法，用于输入 GitHub API 访问令牌
    def authenticate(self):
        print("Input a Github API access token.\n"
              "Personal tokens can be created at https://github.com/settings/tokens\n"
              "This script does not require any permissions (so don't give it any).",
              file=sys.stderr, flush=True)  # 提示用户输入 GitHub API 访问令牌的信息
        print("Access token: ", file=sys.stderr, end='', flush=True)  # 提示用户输入令牌
        token = input()  # 读取用户输入的令牌
        self.headers['Authorization'] = f'token {token.strip()}'  # 将令牌添加到 HTTP 请求头中

    # 发送 HTTP 请求的方法，接受 URL 和可选的认证参数
    def urlopen(self, url, auth=None):
        assert url.startswith('https://')  # 断言 URL 必须以 https:// 开头
        req = Request(url, headers=self.headers)  # 创建带有自定义 headers 的请求对象
        return urlopen(req, timeout=60)  # 发送请求并返回响应对象

    # 获取多页数据的方法，接受 URL 参数
    def get_multipage(self, url):
        data = []  # 初始化空列表用于存储数据
        while url:
            page_data, info, next_url = self.get(url)  # 调用 get 方法获取数据
            data += page_data  # 将获取的数据添加到 data 列表中
            url = next_url  # 更新 URL 为下一页的 URL
        return data  # 返回所有获取到的数据
    def get(self, url):
        # 循环直到成功获取数据或处理完异常情况
        while True:
            # 等待直到速率限制解除
            while self.ratelimit_remaining == 0 and self.ratelimit_reset > time.time():
                # 计算距离速率限制重置的剩余时间
                s = self.ratelimit_reset + 5 - time.time()
                if s <= 0:
                    break
                # 打印等待信息，包括重置时间和剩余秒数
                print(
                    "[gh_lists] rate limit exceeded: waiting until {} ({} s remaining)"
                    .format(datetime.datetime.fromtimestamp(self.ratelimit_reset)
                            .strftime('%Y-%m-%d %H:%M:%S'),
                            int(s)),
                    file=sys.stderr, flush=True
                )
                # 等待一段时间后再重试
                time.sleep(min(5*60, s))

            # 发起请求获取页面数据
            print("[gh_lists] get:", url, file=sys.stderr, flush=True)
            try:
                # 打开并读取 URL 对应的请求
                req = self.urlopen(url)
                try:
                    # 获取响应状态码、头部信息和 JSON 数据
                    code = req.getcode()
                    info = req.info()
                    data = json.loads(req.read().decode('utf-8'))
                finally:
                    # 关闭请求
                    req.close()
            except HTTPError as err:
                # 处理 HTTP 错误，获取状态码和头部信息，数据设为 None
                code = err.getcode()
                info = err.info()
                data = None

            # 如果状态码不是 200 或 403，则抛出运行时错误
            if code not in (200, 403):
                raise RuntimeError()

            # 解析响应中的链接头部信息，提取下一页的 URL
            next_url = None
            if 'Link' in info:
                m = re.search('<([^<>]*)>; rel="next"', info['Link'])
                if m:
                    next_url = m.group(1)

            # 更新速率限制信息
            if 'X-RateLimit-Remaining' in info:
                self.ratelimit_remaining = int(info['X-RateLimit-Remaining'])
            if 'X-RateLimit-Reset' in info:
                self.ratelimit_reset = float(info['X-RateLimit-Reset'])

            # 处理速率限制超出的情况
            if code != 200 or data is None:
                if self.ratelimit_remaining == 0:
                    continue  # 继续等待直到速率限制解除
                else:
                    raise RuntimeError()  # 其他情况下抛出运行时错误

            # 成功获取数据后返回数据、头部信息和下一页的 URL
            return data, info, next_url
# 如果当前脚本作为主程序执行（而非被导入为模块），则执行以下操作
if __name__ == "__main__":
    # 调用 main 函数，并退出程序，返回 main 函数的返回码
    sys.exit(main())
```