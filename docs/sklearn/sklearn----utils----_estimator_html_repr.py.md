# `D:\src\scipysrc\scikit-learn\sklearn\utils\_estimator_html_repr.py`

```
# 导入html模块，用于HTML转义
import html
# 导入itertools模块，用于高效循环和迭代操作
import itertools
# 导入contextlib模块中的closing函数，用于确保资源在使用后被关闭
from contextlib import closing
# 导入inspect模块中的isclass函数，用于检查对象是否是类
from inspect import isclass
# 导入io模块中的StringIO类，用于在内存中读写str
from io import StringIO
# 导入pathlib模块中的Path类，用于处理文件路径
from pathlib import Path
# 导入string模块中的Template类，用于字符串模板替换
from string import Template

# 导入当前包的__version__和config_context模块
from .. import __version__, config_context
# 导入当前包中的fixes模块中的parse_version函数
from .fixes import parse_version

# 定义_IDCounter类，用于生成带有前缀的顺序id
class _IDCounter:
    """Generate sequential ids with a prefix."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.count = 0

    def get_id(self):
        """Return the next id with the prefix."""
        self.count += 1
        return f"{self.prefix}-{self.count}"

# 定义获取CSS样式文件内容的函数
def _get_css_style():
    return Path(__file__).with_suffix(".css").read_text(encoding="utf-8")

# 创建两个_IDCounter实例，用于生成特定前缀的顺序id
_CONTAINER_ID_COUNTER = _IDCounter("sk-container-id")
_ESTIMATOR_ID_COUNTER = _IDCounter("sk-estimator-id")
# 获取当前文件同名但扩展名为.css的文件内容作为CSS样式字符串
_CSS_STYLE = _get_css_style()

# 定义_VisualBlock类，用于HTML表示估算器
class _VisualBlock:
    """HTML Representation of Estimator

    Parameters
    ----------
    kind : {'serial', 'parallel', 'single'}
        kind of HTML block

    estimators : list of estimators or `_VisualBlock`s or a single estimator
        If kind != 'single', then `estimators` is a list of
        estimators.
        If kind == 'single', then `estimators` is a single estimator.

    names : list of str, default=None
        If kind != 'single', then `names` corresponds to estimators.
        If kind == 'single', then `names` is a single string corresponding to
        the single estimator.

    name_details : list of str, str, or None, default=None
        If kind != 'single', then `name_details` corresponds to `names`.
        If kind == 'single', then `name_details` is a single string
        corresponding to the single estimator.

    dash_wrapped : bool, default=True
        If true, wrapped HTML element will be wrapped with a dashed border.
        Only active when kind != 'single'.
    """

    def __init__(
        self, kind, estimators, *, names=None, name_details=None, dash_wrapped=True
    ):
        self.kind = kind
        self.estimators = estimators
        self.dash_wrapped = dash_wrapped

        if self.kind in ("parallel", "serial"):
            if names is None:
                names = (None,) * len(estimators)
            if name_details is None:
                name_details = (None,) * len(estimators)

        self.names = names
        self.name_details = name_details

    def _sk_visual_block_(self):
        """Return the visual block object."""
        return self

# 定义_write_label_html函数，用于生成带有标签和命名细节的HTML内容
def _write_label_html(
    out,
    name,
    name_details,
    outer_class="sk-label-container",
    inner_class="sk-label",
    checked=False,
    doc_link="",
    is_fitted_css_class="",
    is_fitted_icon="",
):
    """Write labeled html with or without a dropdown with named details.

    Parameters
    ----------
    out : file-like object
        The file to write the HTML representation to.
    name : str
        The label for the estimator. It corresponds either to the estimator class name
        for a simple estimator or in the case of a `Pipeline` and `ColumnTransformer`,
        it corresponds to the name of the step.
    """
    name_details : str
        下拉标签可展示的详细信息，如非默认参数或ColumnTransformer的列信息。
    outer_class : {"sk-label-container", "sk-item"}, default="sk-label-container"
        外部容器的CSS类。
    inner_class : {"sk-label", "sk-estimator"}, default="sk-label"
        内部容器的CSS类。
    checked : bool, default=False
        下拉标签是否展开。对于单个评估器，我们希望展开内容。
    doc_link : str, default=""
        评估器文档的链接。如果为空字符串，则在图表中不添加链接。如果评估器使用了 `_HTMLDocumentationLinkMixin`，则可以生成此链接。
    is_fitted_css_class : {"", "fitted"}
        表示评估器是否已拟合的CSS类。空字符串表示评估器未拟合，"fitted"表示评估器已拟合。
    is_fitted_icon : str, default=""
        在图表中显示已拟合信息的HTML表示。空字符串表示不显示任何信息。
    """
    # 为了确保标签居中，我们需要在标签左侧添加一些填充
    padding_label = "&nbsp;" if is_fitted_icon else ""  # 为了字符 "i" 添加填充

    out.write(
        f'<div class="{outer_class}"><div'
        f' class="{inner_class} {is_fitted_css_class} sk-toggleable">'
    )
    name = html.escape(name)  # 对名称进行HTML转义

    if name_details is not None:
        name_details = html.escape(str(name_details))  # 对名称详细信息进行HTML转义
        label_class = (
            f"sk-toggleable__label {is_fitted_css_class} sk-toggleable__label-arrow"
        )

        checked_str = "checked" if checked else ""  # 根据checked参数确定是否勾选
        est_id = _ESTIMATOR_ID_COUNTER.get_id()  # 获取评估器的唯一标识符

        if doc_link:
            doc_label = "<span>Online documentation</span>"
            if name is not None:
                doc_label = f"<span>Documentation for {name}</span>"
            doc_link = (
                f'<a class="sk-estimator-doc-link {is_fitted_css_class}"'
                f' rel="noreferrer" target="_blank" href="{doc_link}">?{doc_label}</a>'
            )
            padding_label += "&nbsp;"  # 为 "?" 字符添加额外的填充

        fmt_str = (
            '<input class="sk-toggleable__control sk-hidden--visually"'
            f' id="{est_id}" '
            f'type="checkbox" {checked_str}><label for="{est_id}" '
            f'class="{label_class} {is_fitted_css_class}">{padding_label}{name}'
            f"{doc_link}{is_fitted_icon}</label><div "
            f'class="sk-toggleable__content {is_fitted_css_class}">'
            f"<pre>{name_details}</pre></div> "
        )
        out.write(fmt_str)
    else:
        out.write(f"<label>{name}</label>")
    out.write("</div></div>")  # outer_class inner_class
# 生成关于如何显示估算器的信息。
def _get_visual_block(estimator):
    # 如果估算器具有 "_sk_visual_block_" 属性
    if hasattr(estimator, "_sk_visual_block_"):
        try:
            # 调用 "_sk_visual_block_" 方法获取可视化块信息
            return estimator._sk_visual_block_()
        except Exception:
            # 如果调用方法时出现异常，则创建一个 _VisualBlock 对象表示单个估算器
            return _VisualBlock(
                "single",
                estimator,
                names=estimator.__class__.__name__,
                name_details=str(estimator),
            )

    # 如果估算器是字符串类型，则创建一个 _VisualBlock 对象表示单个估算器
    if isinstance(estimator, str):
        return _VisualBlock(
            "single", estimator, names=estimator, name_details=estimator
        )
    # 如果估算器是 None，则创建一个 _VisualBlock 对象表示单个估算器
    elif estimator is None:
        return _VisualBlock("single", estimator, names="None", name_details="None")

    # 检查估算器是否像元估算器（包装其他估算器）
    if hasattr(estimator, "get_params") and not isclass(estimator):
        # 获取估算器的参数，筛选出具有 "get_params" 和 "fit" 方法的估算器
        estimators = [
            (key, est)
            for key, est in estimator.get_params(deep=False).items()
            if hasattr(est, "get_params") and hasattr(est, "fit") and not isclass(est)
        ]
        if estimators:
            # 如果找到符合条件的估算器，则创建一个 _VisualBlock 对象表示并行估算器
            return _VisualBlock(
                "parallel",
                [est for _, est in estimators],
                names=[f"{key}: {est.__class__.__name__}" for key, est in estimators],
                name_details=[str(est) for _, est in estimators],
            )

    # 默认情况下，创建一个 _VisualBlock 对象表示单个估算器
    return _VisualBlock(
        "single",
        estimator,
        names=estimator.__class__.__name__,
        name_details=str(estimator),
    )


# 将估算器以串行、并行或单独的形式写入 HTML 中。
# 对于多个估算器，此函数递归调用。
def _write_estimator_html(
    out,
    estimator,
    estimator_label,
    estimator_label_details,
    is_fitted_css_class,
    is_fitted_icon="",
    first_call=False,
):
    """Write estimator to html in serial, parallel, or by itself (single).

    Parameters
    ----------
    out : file-like object
        The file to write the HTML representation to.
    estimator : estimator object
        The estimator to visualize.
    estimator_label : str
        The label for the estimator. It corresponds either to the estimator class name
        for simple estimator or in the case of `Pipeline` and `ColumnTransformer`, it
        corresponds to the name of the step.
    estimator_label_details : str
        The details to show as content in the dropdown part of the toggleable label.
        It can contain information as non-default parameters or column information for
        `ColumnTransformer`.
    is_fitted_css_class : {"", "fitted"}
        The CSS class to indicate whether or not the estimator is fitted or not. The
        empty string means that the estimator is not fitted and "fitted" means that the
        estimator is fitted.
    is_fitted_icon : str, optional
        Optional icon to display for fitted estimator.
    first_call : bool, optional
        Indicates if this is the first call in recursion, default is False.
    """
    # 如果是第一次调用该函数
    if first_call:
        # 获取 estimator 的可视化块
        est_block = _get_visual_block(estimator)
    else:
        # 否则，将 is_fitted_icon 置为空字符串
        is_fitted_icon = ""
        # 使用配置上下文，仅打印变化的部分
        with config_context(print_changed_only=True):
            # 获取 estimator 的可视化块
            est_block = _get_visual_block(estimator)

    # 如果 estimator 也可能是 `_VisualBlock` 的实例
    if hasattr(estimator, "_get_doc_link"):
        # 获取文档链接
        doc_link = estimator._get_doc_link()
    else:
        # 否则，文档链接为空字符串
        doc_link = ""

    # 如果 est_block 的类型是 "serial" 或者 "parallel"
    if est_block.kind in ("serial", "parallel"):
        # 是否需要虚线包装，如果是第一次调用或者 est_block.dash_wrapped 为真，则需要
        dashed_wrapped = first_call or est_block.dash_wrapped
        # 虚线类，如果需要虚线包装，则加上 " sk-dashed-wrapped"
        dash_cls = " sk-dashed-wrapped" if dashed_wrapped else ""
        # 输出一个 div，类名为 "sk-item"，可能包含虚线类
        out.write(f'<div class="sk-item{dash_cls}">')

        # 如果有 estimator_label
        if estimator_label:
            # 写入标签的 HTML，包括文档链接和是否拟合的图标等信息
            _write_label_html(
                out,
                estimator_label,
                estimator_label_details,
                doc_link=doc_link,
                is_fitted_css_class=is_fitted_css_class,
                is_fitted_icon=is_fitted_icon,
            )

        # kind 表示 est_block 的类型，输出一个 div，类名为 "sk-{kind}"
        kind = est_block.kind
        out.write(f'<div class="sk-{kind}">')
        # 获取 est_block 中的估算器、名称及名称细节的组合
        est_infos = zip(est_block.estimators, est_block.names, est_block.name_details)

        # 遍历估算器信息
        for est, name, name_details in est_infos:
            # 如果 kind 是 "serial"
            if kind == "serial":
                # 写入估算器的 HTML
                _write_estimator_html(
                    out,
                    est,
                    name,
                    name_details,
                    is_fitted_css_class=is_fitted_css_class,
                )
            else:  # 如果 kind 是 "parallel"
                # 输出一个并行项目的 div
                out.write('<div class="sk-parallel-item">')
                # 将元素包装在一个串行的可视块中
                serial_block = _VisualBlock("serial", [est], dash_wrapped=False)
                # 写入估算器的 HTML
                _write_estimator_html(
                    out,
                    serial_block,
                    name,
                    name_details,
                    is_fitted_css_class=is_fitted_css_class,
                )
                out.write("</div>")  # 关闭 sk-parallel-item 的 div

        out.write("</div></div>")  # 关闭 sk-{kind} 和 sk-item 的 div
    elif est_block.kind == "single":
        # 写入标签的 HTML，用于单个估算器的情况
        _write_label_html(
            out,
            est_block.names,
            est_block.name_details,
            outer_class="sk-item",
            inner_class="sk-estimator",
            checked=first_call,
            doc_link=doc_link,
            is_fitted_css_class=is_fitted_css_class,
            is_fitted_icon=is_fitted_icon,
        )
def estimator_html_repr(estimator):
    """Build a HTML representation of an estimator.

    Read more in the :ref:`User Guide <visualizing_composite_estimators>`.

    Parameters
    ----------
    estimator : estimator object
        The estimator to visualize.

    Returns
    -------
    html: str
        HTML representation of estimator.

    Examples
    --------
    >>> from sklearn.utils._estimator_html_repr import estimator_html_repr
    >>> from sklearn.linear_model import LogisticRegression
    >>> estimator_html_repr(LogisticRegression())
    '<style>...</div>'
    """
    # 导入异常类和验证函数
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    # 检查估算器是否具有'fit'属性
    if not hasattr(estimator, "fit"):
        # 如果没有'fit'属性，标记为未拟合状态
        status_label = "<span>Not fitted</span>"
        is_fitted_css_class = ""
    else:
        try:
            # 尝试检查估算器是否已拟合
            check_is_fitted(estimator)
            status_label = "<span>Fitted</span>"
            is_fitted_css_class = "fitted"
        except NotFittedError:
            # 如果未拟合，则标记为未拟合状态
            status_label = "<span>Not fitted</span>"
            is_fitted_css_class = ""

    # 构建HTML表示，包括拟合状态标签和相关的CSS类
    is_fitted_icon = (
        f'<span class="sk-estimator-doc-link {is_fitted_css_class}">'
        f"i{status_label}</span>"
    )
    # 使用 closing(StringIO()) 上下文管理器创建一个字符串IO对象，命名为out
    with closing(StringIO()) as out:
        # 从全局计数器获取一个容器ID
        container_id = _CONTAINER_ID_COUNTER.get_id()
        # 使用_CSS_STYLE创建一个模板对象
        style_template = Template(_CSS_STYLE)
        # 使用模板对象替换id字段，生成具有唯一ID的样式表
        style_with_id = style_template.substitute(id=container_id)
        # 将estimator对象转换为字符串
        estimator_str = str(estimator)

        # 默认情况下显示回退消息，并且加载CSS会将div.sk-text-repr-fallback的显示设置为none以隐藏回退消息。
        #
        # 如果笔记本被信任，则加载CSS会隐藏回退消息。如果笔记本不被信任，则不加载CSS，显示回退消息。
        #
        # 对HTML repr div.sk-container应用相反的逻辑。
        # 默认情况下，div.sk-container被隐藏，加载CSS会显示它。
        fallback_msg = (
            "In a Jupyter environment, please rerun this cell to show the HTML"
            " representation or trust the notebook. <br />On GitHub, the"
            " HTML representation is unable to render, please try loading this page"
            " with nbviewer.org."
        )
        # 创建HTML模板，包括样式和两个div容器：一个用于回退消息，另一个默认隐藏
        html_template = (
            f"<style>{style_with_id}</style>"
            f'<div id="{container_id}" class="sk-top-container">'
            '<div class="sk-text-repr-fallback">'
            f"<pre>{html.escape(estimator_str)}</pre><b>{fallback_msg}</b>"
            "</div>"
            '<div class="sk-container" hidden>'
        )

        # 将HTML模板写入字符串IO对象中
        out.write(html_template)

        # 调用函数_write_estimator_html，将estimator的HTML表示写入out对象
        _write_estimator_html(
            out,
            estimator,
            estimator.__class__.__name__,
            estimator_str,
            first_call=True,
            is_fitted_css_class=is_fitted_css_class,
            is_fitted_icon=is_fitted_icon,
        )
        # 在HTML模板的闭合div标签后添加结尾标记
        out.write("</div></div>")

        # 获取out对象的字符串值，作为最终的HTML输出
        html_output = out.getvalue()
        # 返回最终的HTML输出
        return html_output
class _HTMLDocumentationLinkMixin:
    """Mixin class allowing to generate a link to the API documentation.

    This mixin relies on three attributes:
    - `_doc_link_module`: it corresponds to the root module (e.g. `sklearn`). Using this
      mixin, the default value is `sklearn`.
    - `_doc_link_template`: it corresponds to the template used to generate the
      link to the API documentation. Using this mixin, the default value is
      `"https://scikit-learn.org/{version_url}/modules/generated/
      {estimator_module}.{estimator_name}.html"`.
    - `_doc_link_url_param_generator`: it corresponds to a function that generates the
      parameters to be used in the template when the estimator module and name are not
      sufficient.

    The method :meth:`_get_doc_link` generates the link to the API documentation for a
    given estimator.

    This useful provides all the necessary states for
    :func:`sklearn.utils.estimator_html_repr` to generate a link to the API
    documentation for the estimator HTML diagram.

    Examples
    --------
    If the default values for `_doc_link_module`, `_doc_link_template` are not suitable,
    then you can override them:
    >>> from sklearn.base import BaseEstimator
    >>> estimator = BaseEstimator()
    >>> estimator._doc_link_template = "https://website.com/{single_param}.html"
    >>> def url_param_generator(estimator):
    ...     return {"single_param": estimator.__class__.__name__}
    >>> estimator._doc_link_url_param_generator = url_param_generator
    >>> estimator._get_doc_link()
    'https://website.com/BaseEstimator.html'
    """

    _doc_link_module = "sklearn"  # 默认的根模块名为'sklearn'

    _doc_link_url_param_generator = None  # 参数生成器函数，默认为None

    @property
    def _doc_link_template(self):
        # 解析当前安装的 scikit-learn 版本
        sklearn_version = parse_version(__version__)
        # 根据版本信息生成对应的 URL 部分
        if sklearn_version.dev is None:
            version_url = f"{sklearn_version.major}.{sklearn_version.minor}"
        else:
            version_url = "dev"
        # 返回默认的文档链接模板，可以被子类重载
        return getattr(
            self,
            "__doc_link_template",
            (
                f"https://scikit-learn.org/{version_url}/modules/generated/"
                "{estimator_module}.{estimator_name}.html"
            ),
        )

    @_doc_link_template.setter
    def _doc_link_template(self, value):
        setattr(self, "__doc_link_template", value)  # 设置文档链接模板的属性值
    # 生成给定估计器的 API 文档链接
    def _get_doc_link(self):
        """Generates a link to the API documentation for a given estimator.

        This method generates the link to the estimator's documentation page
        by using the template defined by the attribute `_doc_link_template`.

        Returns
        -------
        url : str
            The URL to the API documentation for this estimator. If the estimator does
            not belong to module `_doc_link_module`, the empty string (i.e. `""`) is
            returned.
        """
        # 检查当前实例所属的模块是否与指定的 `_doc_link_module` 相符
        if self.__class__.__module__.split(".")[0] != self._doc_link_module:
            return ""

        # 如果没有提供 `_doc_link_url_param_generator` 函数，直接构建链接模板
        if self._doc_link_url_param_generator is None:
            estimator_name = self.__class__.__name__
            # 构造估计器的模块名称，直到第一个私有子模块为止
            # 这种方式适用于 scikit-learn 中的公共估计器，即使它们实际上位于私有子模块中
            estimator_module = ".".join(
                itertools.takewhile(
                    lambda part: not part.startswith("_"),
                    self.__class__.__module__.split("."),
                )
            )
            return self._doc_link_template.format(
                estimator_module=estimator_module, estimator_name=estimator_name
            )
        
        # 如果提供了 `_doc_link_url_param_generator` 函数，则调用该函数生成链接参数
        return self._doc_link_template.format(
            **self._doc_link_url_param_generator(self)
        )
```