# `D:\src\scipysrc\matplotlib\doc\sphinxext\github.py`

```
"""
Define text roles for GitHub.

* ghissue - Issue
* ghpull - Pull Request
* ghuser - User

Adapted from bitbucket example here:
https://bitbucket.org/birkenfeld/sphinx-contrib/src/tip/bitbucket/sphinxcontrib/bitbucket.py

Authors
-------

* Doug Hellmann
* Min RK
"""
#
# Original Copyright (c) 2010 Doug Hellmann.  All rights reserved.
#

from docutils import nodes, utils                   # 导入需要的模块：nodes和utils
from docutils.parsers.rst.roles import set_classes  # 导入函数set_classes


def make_link_node(rawtext, app, type, slug, options):
    """
    Create a link to a github resource.

    :param rawtext: Text being replaced with link node.
    :param app: Sphinx application context
    :param type: Link type (issues, changeset, etc.)
    :param slug: ID of the thing to link to
    :param options: Options dictionary passed to role func.
    """
    
    try:
        base = app.config.github_project_url          # 获取配置中的GitHub项目URL
        if not base:
            raise AttributeError                      # 若未设置GitHub项目URL则抛出异常
        if not base.endswith('/'):
            base += '/'                               # 确保URL以斜杠结尾
    except AttributeError as err:
        raise ValueError(
            f'github_project_url configuration value is not set '
            f'({err})') from err                       # 若出现AttributeError，则抛出包含原始异常信息的ValueError

    ref = base + type + '/' + slug + '/'             # 构建完整的GitHub资源URL
    set_classes(options)                             # 设置节点的CSS类
    prefix = "#"                                     
    if type == 'pull':
        prefix = "PR " + prefix                       # 如果类型是'pull'，则前缀为"PR #"
    node = nodes.reference(rawtext, prefix + utils.unescape(slug), refuri=ref,
                           **options)                # 创建并返回链接节点
    return node


def ghissue_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Link to a GitHub issue.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    
    try:
        issue_num = int(text)                        # 尝试将text转换为整数，即GitHub issue的编号
        if issue_num <= 0:
            raise ValueError                         # 如果编号小于等于0，则抛出ValueError
    except ValueError:
        msg = inliner.reporter.error(
            'GitHub issue number must be a number greater than or equal to 1; '
            '"%s" is invalid.' % text, line=lineno)  # 报告错误，GitHub issue编号必须大于等于1
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]                          # 返回错误消息

    app = inliner.document.settings.env.app          # 获取应用程序上下文
    if 'pull' in name.lower():
        category = 'pull'                            # 如果role名称中包含'pull'，则category为'pull'
    elif 'issue' in name.lower():
        category = 'issues'                          # 如果role名称中包含'issue'，则category为'issues'
    else:
        msg = inliner.reporter.error(
            'GitHub roles include "ghpull" and "ghissue", '
            '"%s" is invalid.' % name, line=lineno)  # 报告错误，无效的GitHub角色名称
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]                          # 返回错误消息

    node = make_link_node(rawtext, app, category, str(issue_num), options)  # 创建GitHub资源链接节点
    return [node], []                                # 返回链接节点和空消息列表
def ghuser_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Link to a GitHub user.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role (GitHub username).
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    # 构建 GitHub 用户的链接地址
    ref = 'https://www.github.com/' + text
    # 创建一个文档节点，表示 GitHub 用户的链接
    node = nodes.reference(rawtext, text, refuri=ref, **options)
    # 返回一个包含文档节点的列表和空列表（系统消息）
    return [node], []


def ghcommit_role(
        name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Link to a GitHub commit.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role (GitHub commit hash).
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    # 获取 Sphinx 应用程序上下文
    app = inliner.document.settings.env.app
    try:
        # 尝试获取 GitHub 项目的基本 URL
        base = app.config.github_project_url
        # 如果未设置基本 URL，则抛出错误
        if not base:
            raise AttributeError
        # 如果基本 URL 不以 '/' 结尾，则补充 '/'
        if not base.endswith('/'):
            base += '/'
    except AttributeError as err:
        # 捕获未设置基本 URL 的异常，并抛出详细错误信息
        raise ValueError(
            f'github_project_url configuration value is not set '
            f'({err})') from err

    # 构建 GitHub 提交的完整链接地址
    ref = base + text
    # 创建一个文档节点，表示 GitHub 提交的链接（仅显示前6个字符）
    node = nodes.reference(rawtext, text[:6], refuri=ref, **options)
    # 返回一个包含文档节点的列表和空列表（系统消息）
    return [node], []


def setup(app):
    """
    Install the plugin.

    :param app: Sphinx application context.
    """
    # 添加 GitHub 相关角色和配置
    app.add_role('ghissue', ghissue_role)
    app.add_role('ghpull', ghissue_role)  # 使用相同的角色处理 GitHub 问题和拉取请求
    app.add_role('ghuser', ghuser_role)   # 添加 GitHub 用户角色
    app.add_role('ghcommit', ghcommit_role)  # 添加 GitHub 提交角色
    # 添加配置选项 github_project_url，用于存储 GitHub 项目的基本 URL
    app.add_config_value('github_project_url', None, 'env')

    # 返回插件的元数据，表明其在读写时都是线程安全的
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
```