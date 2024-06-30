# `D:\src\scipysrc\scipy\doc\source\doi_role.py`

```
"""
    doilinks
    ~~~~~~~~
    Extension to add links to DOIs. With this extension you can use e.g.
    :doi:`10.1016/S0022-2836(05)80360-2` in your documents. This will
    create a link to a DOI resolver
    (``https://doi.org/10.1016/S0022-2836(05)80360-2``).
    The link caption will be the raw DOI.
    You can also give an explicit caption, e.g.
    :doi:`Basic local alignment search tool <10.1016/S0022-2836(05)80360-2>`.

    :copyright: Copyright 2015  Jon Lund Steffensen. Based on extlinks by
        the Sphinx team.
    :license: BSD.
"""

# 从 docutils 模块导入 nodes 和 utils
from docutils import nodes, utils
# 从 sphinx.util.nodes 模块导入 split_explicit_title 函数
from sphinx.util.nodes import split_explicit_title

# 定义 DOI 角色处理函数
def doi_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    # 反转义文本内容
    text = utils.unescape(text)
    # 解析是否有显示标题，标题内容，以及 DOI 的部分
    has_explicit_title, title, part = split_explicit_title(text)
    # 构建完整的 DOI 链接 URL
    full_url = 'https://doi.org/' + part
    # 如果没有显示标题，则默认标题为 'DOI:' + DOI 的部分
    if not has_explicit_title:
        title = 'DOI:' + part
    # 创建一个指向 DOI 的节点对象
    pnode = nodes.reference(title, title, internal=False, refuri=full_url)
    return [pnode], []

# 定义 arXiv 角色处理函数
def arxiv_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    # 反转义文本内容
    text = utils.unescape(text)
    # 解析是否有显示标题，标题内容，以及 arXiv 的部分
    has_explicit_title, title, part = split_explicit_title(text)
    # 构建完整的 arXiv 链接 URL
    full_url = 'https://arxiv.org/abs/' + part
    # 如果没有显示标题，则默认标题为 'arXiv:' + arXiv 的部分
    if not has_explicit_title:
        title = 'arXiv:' + part
    # 创建一个指向 arXiv 的节点对象
    pnode = nodes.reference(title, title, internal=False, refuri=full_url)
    return [pnode], []

# 定义设置链接角色的函数
def setup_link_role(app):
    # 添加 'doi' 角色，并指定使用 doi_role 处理函数来处理它
    app.add_role('doi', doi_role, override=True)
    # 添加 'DOI' 角色，同样使用 doi_role 处理函数来处理它
    app.add_role('DOI', doi_role, override=True)
    # 添加 'arXiv' 角色，使用 arxiv_role 处理函数来处理它
    app.add_role('arXiv', arxiv_role, override=True)
    # 添加 'arxiv' 角色，同样使用 arxiv_role 处理函数来处理它
    app.add_role('arxiv', arxiv_role, override=True)

# 定义设置函数，连接到 'builder-inited' 事件，调用 setup_link_role 函数
def setup(app):
    app.connect('builder-inited', setup_link_role)
    return {'version': '0.1', 'parallel_read_safe': True}
```