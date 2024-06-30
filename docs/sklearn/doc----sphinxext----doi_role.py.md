# `D:\src\scipysrc\scikit-learn\doc\sphinxext\doi_role.py`

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

# 导入所需模块和函数
from docutils import nodes, utils
from sphinx.util.nodes import split_explicit_title


# 定义处理引用角色的函数
def reference_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    # 解析并取消转义角色文本
    text = utils.unescape(text)
    # 判断是否存在显式标题，分割出标题和部分
    has_explicit_title, title, part = split_explicit_title(text)
    
    # 如果引用类型是 arXiv 或 arxiv
    if typ in ["arXiv", "arxiv"]:
        # 构建 arXiv 的完整 URL
        full_url = "https://arxiv.org/abs/" + part
        # 如果没有显式标题，则使用默认标题格式
        if not has_explicit_title:
            title = "arXiv:" + part
        # 创建一个链接节点对象，指向 arXiv 的 URL
        pnode = nodes.reference(title, title, internal=False, refuri=full_url)
        return [pnode], []  # 返回创建的节点对象列表和空内容列表
    
    # 如果引用类型是 doi 或 DOI
    if typ in ["doi", "DOI"]:
        # 构建 DOI 的完整 URL
        full_url = "https://doi.org/" + part
        # 如果没有显式标题，则使用默认标题格式
        if not has_explicit_title:
            title = "DOI:" + part
        # 创建一个链接节点对象，指向 DOI 的 URL
        pnode = nodes.reference(title, title, internal=False, refuri=full_url)
        return [pnode], []  # 返回创建的节点对象列表和空内容列表


# 定义设置链接角色的函数
def setup_link_role(app):
    # 注册 arXiv 和 doi 等角色，使用上面定义的 reference_role 处理函数
    app.add_role("arxiv", reference_role, override=True)
    app.add_role("arXiv", reference_role, override=True)
    app.add_role("doi", reference_role, override=True)
    app.add_role("DOI", reference_role, override=True)


# 定义设置函数，连接到 Sphinx 构建初始化事件
def setup(app):
    # 在 Sphinx 的 builder-inited 事件中调用 setup_link_role 函数
    app.connect("builder-inited", setup_link_role)
    # 返回字典，包含插件版本和并行读取安全性标志
    return {"version": "0.1", "parallel_read_safe": True}
```