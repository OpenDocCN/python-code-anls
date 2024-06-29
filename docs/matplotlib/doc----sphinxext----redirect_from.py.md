# `D:\src\scipysrc\matplotlib\doc\sphinxext\redirect_from.py`

```py
"""
Redirecting old docs to new location
====================================

If an rst file is moved or its content subsumed in a different file, it
is desirable to redirect the old file to the new or existing file. This
extension enables this with a simple html refresh.

For example suppose ``doc/topic/old-page.rst`` is removed and its content
included in ``doc/topic/new-page.rst``.  We use the ``redirect-from``
directive in ``doc/topic/new-page.rst``::

    .. redirect-from:: /topic/old-page

This creates in the build directory a file ``build/html/topic/old-page.html``
that contains a relative refresh::

    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="0; url=new-page.html">
      </head>
    </html>

If you need to redirect across subdirectory trees, that works as well.  For
instance if ``doc/topic/subdir1/old-page.rst`` is now found at
``doc/topic/subdir2/new-page.rst`` then ``new-page.rst`` just lists the
full path::

    .. redirect-from:: /topic/subdir1/old-page.rst

"""

from pathlib import Path
from sphinx.util.docutils import SphinxDirective  # 导入SphinxDirective类，用于自定义Sphinx指令
from sphinx.domains import Domain  # 导入Domain类，用于创建自定义域
from sphinx.util import logging  # 导入logging模块，用于日志记录

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url={v}">
  </head>
</html>
"""


def setup(app):
    app.add_directive("redirect-from", RedirectFrom)  # 向Sphinx应用添加名为"redirect-from"的自定义指令
    app.add_domain(RedirectFromDomain)  # 向Sphinx应用添加自定义域RedirectFromDomain
    app.connect("builder-inited", _clear_redirects)  # 在文档构建初始化阶段连接_clear_redirects函数
    app.connect("build-finished", _generate_redirects)  # 在文档构建完成阶段连接_generate_redirects函数

    metadata = {'parallel_read_safe': True}  # 定义并返回元数据，表明插件支持并行读取
    return metadata


class RedirectFromDomain(Domain):
    """
    The sole purpose of this domain is a parallel_read_safe data store for the
    redirects mapping.
    """
    name = 'redirect_from'  # 设置自定义域的名称为'redirect_from'
    label = 'redirect_from'  # 设置自定义域的标签为'redirect_from'

    @property
    def redirects(self):
        """The mapping of the redirects."""
        return self.data.setdefault('redirects', {})  # 返回重定向映射，如果不存在则创建一个空字典

    def clear_doc(self, docname):
        self.redirects.pop(docname, None)  # 移除指定文档名在重定向映射中的条目，如果不存在则不操作

    def merge_domaindata(self, docnames, otherdata):
        for src, dst in otherdata['redirects'].items():
            if src not in self.redirects:
                self.redirects[src] = dst  # 将其他数据中的重定向源到目标映射合并到当前域中
            elif self.redirects[src] != dst:
                raise ValueError(
                    f"Inconsistent redirections from {src} to "
                    f"{self.redirects[src]} and {otherdata['redirects'][src]}")


class RedirectFrom(SphinxDirective):
    required_arguments = 1  # 指定自定义指令需要一个必须参数
    # 定义一个方法 `run`，该方法属于某个类的实例方法
    def run(self):
        # 解构赋值，从参数中获取被重定向的文档路径
        redirected_doc, = self.arguments
        # 从环境中获取 'redirect_from' 域对象
        domain = self.env.get_domain('redirect_from')
        # 获取当前正在处理的文档的路径
        current_doc = self.env.path2doc(self.state.document.current_source)
        # 将重定向的相对文档路径转换为绝对路径，并忽略第二个返回值
        redirected_reldoc, _ = self.env.relfn2path(redirected_doc, current_doc)
        # 检查被重定向的相对文档路径是否已经在重定向域对象的重定向字典中
        if redirected_reldoc in domain.redirects:
            # 如果已经存在，则抛出 ValueError 异常，显示已经重定向的信息
            raise ValueError(
                f"{redirected_reldoc} is already noted as redirecting to "
                f"{domain.redirects[redirected_reldoc]}")
        # 将当前文档路径作为值，被重定向的相对文档路径作为键，添加到重定向域对象的重定向字典中
        domain.redirects[redirected_reldoc] = current_doc
        # 返回空列表，表示操作成功完成
        return []
def _generate_redirects(app, exception):
    # 获取应用程序构建器对象
    builder = app.builder
    # 如果构建器的名称不是"html"或者发生异常，则直接返回
    if builder.name != "html" or exception:
        return
    # 遍历应用程序环境中重定向域的所有重定向条目
    for k, v in app.env.get_domain('redirect_from').redirects.items():
        # 构建重定向文件的路径
        p = Path(app.outdir, k + builder.out_suffix)
        # 使用 HTML 模板生成 HTML 内容
        html = HTML_TEMPLATE.format(v=builder.get_relative_uri(k, v))
        # 如果路径已存在且内容不同，记录警告信息
        if p.is_file():
            logger.warning('A redirect-from directive is trying to '
                           'create %s, but that file already exists '
                           '(perhaps you need to run "make clean")', p)
        else:
            # 否则，记录信息，准备创建新的 HTML 文件
            logger.info('making refresh html file: %s redirect to %s', k, v)
            # 创建父目录（如果不存在）并写入 HTML 内容到文件
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html, encoding='utf-8')


def _clear_redirects(app):
    # 获取重定向域对象
    domain = app.env.get_domain('redirect_from')
    # 如果存在重定向条目，则清空缓存的重定向条目
    if domain.redirects:
        logger.info('clearing cached redirects')
        domain.redirects.clear()
```