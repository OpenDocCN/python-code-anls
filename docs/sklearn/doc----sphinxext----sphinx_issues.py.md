# `D:\src\scipysrc\scikit-learn\doc\sphinxext\sphinx_issues.py`

```
"""A Sphinx extension for linking to your project's issue tracker.

Copyright 2014 Steven Loria

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import re  # 导入正则表达式模块

from docutils import nodes, utils  # 导入docutils中的节点和实用工具
from sphinx.util.nodes import split_explicit_title  # 从sphinx工具中导入分割显式标题的函数

__version__ = "1.2.0"  # 定义当前扩展的版本号
__author__ = "Steven Loria"  # 定义扩展的作者
__license__ = "MIT"  # 定义扩展的许可证


def user_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Sphinx role for linking to a user profile. Defaults to linking to
    Github profiles, but the profile URIS can be configured via the
    ``issues_user_uri`` config value.
    Examples: ::
        :user:`sloria`
    Anchor text also works: ::
        :user:`Steven Loria <sloria>`
    """
    options = options or {}  # 如果options为None，设置为空字典
    content = content or []  # 如果content为None，设置为空列表
    has_explicit_title, title, target = split_explicit_title(text)  # 使用split_explicit_title函数解析文本

    target = utils.unescape(target).strip()  # 对目标进行反转义和去除首尾空白
    title = utils.unescape(title).strip()  # 对标题进行反转义和去除首尾空白
    config = inliner.document.settings.env.app.config  # 获取当前应用程序的配置
    if config.issues_user_uri:  # 如果配置中存在issues_user_uri
        ref = config.issues_user_uri.format(user=target)  # 格式化用户URI
    else:
        ref = "https://github.com/{0}".format(target)  # 否则使用默认的Github URI
    if has_explicit_title:
        text = title  # 如果存在显式标题，则使用显式标题作为文本
    else:
        text = "@{0}".format(target)  # 否则使用目标作为文本

    link = nodes.reference(text=text, refuri=ref, **options)  # 创建一个文本为text、链接为ref的引用节点
    return [link], []  # 返回链接节点的列表和空列表


def cve_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Sphinx role for linking to a CVE on https://cve.mitre.org.
    Examples: ::
        :cve:`CVE-2018-17175`
    """
    options = options or {}  # 如果options为None，设置为空字典
    content = content or []  # 如果content为None，设置为空列表
    has_explicit_title, title, target = split_explicit_title(text)  # 使用split_explicit_title函数解析文本

    target = utils.unescape(target).strip()  # 对目标进行反转义和去除首尾空白
    title = utils.unescape(title).strip()  # 对标题进行反转义和去除首尾空白
    ref = "https://cve.mitre.org/cgi-bin/cvename.cgi?name={0}".format(target)  # 根据CVE名称创建链接URI
    text = title if has_explicit_title else target  # 如果存在显式标题，则使用标题作为文本，否则使用目标作为文本
    link = nodes.reference(text=text, refuri=ref, **options)  # 创建一个文本为text、链接为ref的引用节点
    return [link], []  # 返回链接节点的列表和空列表


class IssueRole(object):
    EXTERNAL_REPO_REGEX = re.compile(r"^(\w+)/(.+)([#@])([\w]+)$")
    # 初始化方法，接受多个参数用于设置实例变量
    def __init__(
        self, uri_config_option, format_kwarg, github_uri_template, format_text=None
    ):
        # 设置实例变量 uri_config_option 为传入的参数 uri_config_option
        self.uri_config_option = uri_config_option
        # 设置实例变量 format_kwarg 为传入的参数 format_kwarg
        self.format_kwarg = format_kwarg
        # 设置实例变量 github_uri_template 为传入的参数 github_uri_template
        self.github_uri_template = github_uri_template
        # 如果传入了 format_text 参数，则设置实例变量 format_text 为传入的参数值，否则使用默认的 self.default_format_text 方法返回值
        self.format_text = format_text or self.default_format_text

    # 静态方法，用于生成默认格式的文本表示问题编号
    @staticmethod
    def default_format_text(issue_no):
        return "#{0}".format(issue_no)

    # 创建一个节点，用于表示链接到问题或拉取请求的文本
    def make_node(self, name, issue_no, config, options=None):
        # 映射用于 GitHub 问题、拉取请求和提交的不同名称
        name_map = {"pr": "pull", "issue": "issues", "commit": "commit"}
        options = options or {}
        # 使用正则表达式检查是否是外部仓库链接
        repo_match = self.EXTERNAL_REPO_REGEX.match(issue_no)
        if repo_match:  # 如果是外部仓库
            # 解析用户名、仓库名、符号和问题编号
            username, repo, symbol, issue = repo_match.groups()
            # 如果传入的 name 不在映射字典中，则抛出错误
            if name not in name_map:
                raise ValueError(
                    "External repo linking not supported for :{}:".format(name)
                )
            # 获取正确的路径名（例如 issues 或 pull），构建完整的 GitHub 链接
            path = name_map.get(name)
            ref = "https://github.com/{issues_github_path}/{path}/{n}".format(
                issues_github_path="{}/{}".format(username, repo), path=path, n=issue
            )
            # 格式化问题编号文本，并移除可能存在的 "#" 字符
            formatted_issue = self.format_text(issue).lstrip("#")
            # 根据用户名、仓库名、符号和格式化后的问题编号构建显示文本
            text = "{username}/{repo}{symbol}{formatted_issue}".format(**locals())
            # 创建一个指向 ref 的链接节点，并设置文本和其他选项
            link = nodes.reference(text=text, refuri=ref, **options)
            return link

        # 如果 issue_no 不是 "-" 或 "0"
        if issue_no not in ("-", "0"):
            # 获取配置中的 URI 模板
            uri_template = getattr(config, self.uri_config_option, None)
            if uri_template:
                # 使用 URI 模板格式化问题编号生成完整的引用链接
                ref = uri_template.format(**{self.format_kwarg: issue_no})
            elif config.issues_github_path:
                # 如果未设置 URI 模板，则使用 github_uri_template 格式化 GitHub 的链接模板
                ref = self.github_uri_template.format(
                    issues_github_path=config.issues_github_path, n=issue_no
                )
            else:
                # 如果既没有设置 URI 模板也没有设置 issues_github_path，则抛出 ValueError 异常
                raise ValueError(
                    "Neither {} nor issues_github_path is set".format(
                        self.uri_config_option
                    )
                )
            # 格式化问题编号文本
            issue_text = self.format_text(issue_no)
            # 创建一个指向 ref 的链接节点，并设置文本和其他选项
            link = nodes.reference(text=issue_text, refuri=ref, **options)
        else:
            # 如果 issue_no 是 "-" 或 "0"，则返回 None
            link = None
        return link

    # 实现调用对象的方法，用于处理文本中的多个问题编号并生成相应的链接节点列表
    def __call__(
        self, name, rawtext, text, lineno, inliner, options=None, content=None
    ):
        options = options or {}
        content = content or []
        # 解析文本中的问题编号列表，去除可能存在的空格并解析转义字符
        issue_nos = [each.strip() for each in utils.unescape(text).split(",")]
        # 获取当前文档的配置信息
        config = inliner.document.settings.env.app.config
        ret = []
        # 遍历问题编号列表
        for i, issue_no in enumerate(issue_nos):
            # 调用 make_node 方法生成对应的链接节点
            node = self.make_node(name, issue_no, config, options=options)
            # 将生成的节点添加到返回结果列表中
            ret.append(node)
            # 如果当前节点不是列表中的最后一个节点，则添加一个分隔符节点（逗号）
            if i != len(issue_nos) - 1:
                sep = nodes.raw(text=", ", format="html")
                ret.append(sep)
        return ret, []
# 创建用于处理 issue 的 Sphinx 角色对象
issue_role = IssueRole(
    uri_config_option="issues_uri",  # 使用配置中的 issues_uri 作为 URI 的配置选项
    format_kwarg="issue",            # 指定格式化参数为 'issue'
    github_uri_template="https://github.com/{issues_github_path}/issues/{n}",  # GitHub 上 issue 的模板 URI
)

# 创建用于处理 pull request 的 Sphinx 角色对象
pr_role = IssueRole(
    uri_config_option="issues_pr_uri",  # 使用配置中的 issues_pr_uri 作为 URI 的配置选项
    format_kwarg="pr",                 # 指定格式化参数为 'pr'
    github_uri_template="https://github.com/{issues_github_path}/pull/{n}",  # GitHub 上 PR 的模板 URI
)

# 创建用于处理 commit 的 Sphinx 角色对象
commit_role = IssueRole(
    uri_config_option="issues_commit_uri",  # 使用配置中的 issues_commit_uri 作为 URI 的配置选项
    format_kwarg="commit",                 # 指定格式化参数为 'commit'
    github_uri_template="https://github.com/{issues_github_path}/commit/{n}",  # GitHub 上 commit 的模板 URI
    format_text=format_commit_text,        # 指定格式化文本函数为 format_commit_text
)

def format_commit_text(sha):
    return sha[:7]  # 返回给定 SHA 的前七个字符作为格式化的 commit 文本

def setup(app):
    # 添加配置值 issues_uri 到 Sphinx 应用，用于配置 issues 的 URI 模板
    app.add_config_value("issues_uri", default=None, rebuild="html")
    # 添加配置值 issues_pr_uri 到 Sphinx 应用，用于配置 PR 的 URI 模板
    app.add_config_value("issues_pr_uri", default=None, rebuild="html")
    # 添加配置值 issues_commit_uri 到 Sphinx 应用，用于配置 commit 的 URI 模板
    app.add_config_value("issues_commit_uri", default=None, rebuild="html")
    # 添加配置值 issues_github_path 到 Sphinx 应用，用于配置 GitHub 路径的快捷方式
    app.add_config_value("issues_github_path", default=None, rebuild="html")
    # 添加配置值 issues_user_uri 到 Sphinx 应用，用于配置用户 profile 的 URI 模板
    app.add_config_value("issues_user_uri", default=None, rebuild="html")
    
    # 添加自定义的 Sphinx 角色，用于处理 issue
    app.add_role("issue", issue_role)
    # 添加自定义的 Sphinx 角色，用于处理 PR
    app.add_role("pr", pr_role)
    # 添加自定义的 Sphinx 角色，用于处理用户 profile
    app.add_role("user", user_role)
    # 添加自定义的 Sphinx 角色，用于处理 commit
    app.add_role("commit", commit_role)
    # 添加自定义的 Sphinx 角色，用于处理 CVE
    app.add_role("cve", cve_role)
    
    # 返回字典，指定插件版本和并发安全信息
    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
```