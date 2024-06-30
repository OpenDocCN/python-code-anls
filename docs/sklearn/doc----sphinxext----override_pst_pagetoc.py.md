# `D:\src\scipysrc\scikit-learn\doc\sphinxext\override_pst_pagetoc.py`

```
from functools import cache  # 导入functools模块中的cache装饰器

from sphinx.util.logging import getLogger  # 从sphinx.util.logging模块中导入getLogger函数

logger = getLogger(__name__)  # 使用__name__参数获取当前模块的日志记录器对象


def override_pst_pagetoc(app, pagename, templatename, context, doctree):
    """Overrides the `generate_toc_html` function of pydata-sphinx-theme for API."""
    # 定义一个函数override_pst_pagetoc，用于重写pydata-sphinx-theme中的generate_toc_html函数，以支持API。
    
    @cache
    # 使用functools模块中的cache装饰器，对下面的函数进行缓存处理
    def generate_api_toc_html(kind="html"):
        """生成 API 页面的目录树 HTML。

        这依赖于 pydata-sphinx-theme 添加到上下文中的 `generate_toc_html` 函数。
        我们将原始函数保存在 `pst_generate_toc_html` 中，并将 `generate_toc_html`
        覆盖为生成的 API 页面使用的此函数。

        API 页面的页面目录树如下所示：

        <ul class="visible ...">               <!-- 去掉外部标签 -->
         <li class="toc-h1 ...">               <!-- 去掉外部标签 -->
          <a class="..." href="#">{{obj}}</a>  <!-- 删除链接 -->
          <ul class="visible ...">             <!-- 设置为可见（如果存在） -->
           <li class="toc-h2 ...">             <!-- 简化 -->
            ...object
            <ul class="...">                          <!-- 设置为可见（如果存在） -->
             <li class="toc-h3 ...">...method 1</li>  <!-- 简化 -->
             <li class="toc-h3 ...">...method 2</li>  <!-- 简化 -->
             ...more methods                          <!-- 简化 -->
            </ul>
           </li>
           <li class="toc-h2 ...">...gallery examples</li>  <!-- 简化 -->
          </ul>
         </li>
        </ul>
        """
        soup = context["pst_generate_toc_html"](kind="soup")

        try:
            # 去掉最外层标签
            soup.ul.unwrap()
            soup.li.unwrap()
            soup.a.decompose()

            # 获取所有的 toc-h2 级别条目，第一个应该是函数或类，第二个（如果存在）应该是示例；
            # 对于生成的 API 页面，该级别下不应有超过两个条目
            lis = soup.ul.select("li.toc-h2")
            main_li = lis[0]
            meth_list = main_li.ul

            if meth_list is not None:
                # 这是一个类的 API 页面，我们从方法名称中移除类名以更好地适应辅助侧边栏；
                # 同时，我们始终使 toc-h3 级别条目可见，以便更轻松地浏览方法
                meth_list["class"].append("visible")
                for meth in meth_list.find_all("li", {"class": "toc-h3"}):
                    target = meth.a.code.span
                    target.string = target.string.split(".", 1)[1]

            # 这与 `generate_toc_html` 的行为对应
            return str(soup) if kind == "html" else soup

        except Exception as e:
            # 在任何失败情况下，返回原始的页面目录树
            logger.warning(
                f"Failed to generate API pagetoc for {pagename}: {e}; falling back"
            )
            return context["pst_generate_toc_html"](kind=kind)

    # 覆盖 pydata-sphinx-theme 的实现以生成 API 页面
    # 如果 pagename 字符串以 "modules/generated/" 开头，则执行以下操作
    if pagename.startswith("modules/generated/"):
        # 将 context 字典中的 "generate_toc_html" 键的值复制到 "pst_generate_toc_html" 键中
        context["pst_generate_toc_html"] = context["generate_toc_html"]
        # 将 "generate_toc_html" 键的值替换为 generate_api_toc_html 函数的引用
        context["generate_toc_html"] = generate_api_toc_html
# 定义一个函数 `setup`，接受一个参数 `app`
def setup(app):
    # 当 `html-page-context` 事件被触发时，调用 `override_pst_pagetoc` 函数。
    # 由于默认优先级为 500，为确保安全性，设置优先级为 900。
    app.connect("html-page-context", override_pst_pagetoc, priority=900)
```