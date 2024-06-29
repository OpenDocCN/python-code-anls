# `D:\src\scipysrc\matplotlib\lib\matplotlib\sphinxext\figmpl_directive.py`

```py
"""
为Sphinx添加一个响应式的``figure-mpl``指令，类似于``figure``指令的实现。

这个实现与``.. figure::``非常相似，但它允许传递一个``srcset=``参数给图像标签，从而支持响应式分辨率的图片。

这个指令可以独立使用，但通常与:doc:`/api/sphinxext_plot_directive_api`一起使用。

注意，目录结构与``.. figure::``有所不同。查看下面的*FigureMpl*文档了解更多信息。
"""

from docutils import nodes  # 导入docutils模块中的nodes模块

from docutils.parsers.rst import directives  # 导入docutils模块中的directives模块
from docutils.parsers.rst.directives.images import Figure, Image  # 从docutils.parsers.rst.directives.images中导入Figure和Image类

import os  # 导入标准库os
from os.path import relpath  # 从os.path模块导入relpath函数
from pathlib import PurePath, Path  # 从pathlib模块导入PurePath和Path类
import shutil  # 导入标准库shutil

from sphinx.errors import ExtensionError  # 从sphinx.errors模块导入ExtensionError异常类

import matplotlib  # 导入matplotlib库

class figmplnode(nodes.General, nodes.Element):
    """
    定义一个自定义节点类figmplnode，继承自nodes.General和nodes.Element类。
    """

class FigureMpl(Figure):
    """
    实现一个指令，允许使用hidpi图像的可选指令。

    用于在conf.py中设置*plot_srcset*配置选项，并在plot_directive.py的TEMPLATE中设置。

    例如::

        .. figure-mpl:: plot_directive/some_plots-1.png
            :alt: bar
            :srcset: plot_directive/some_plots-1.png,
                     plot_directive/some_plots-1.2x.png 2.00x
            :class: plot-directive

    生成的html（在``some_plots.html``中）为::

        <img src="sphx_glr_bar_001_hidpi.png"
            srcset="_images/some_plot-1.png,
                    _images/some_plots-1.2x.png 2.00x",
            alt="bar"
            class="plot_directive" />

    注意与sphinx figure指令处理子目录的方式不同的地方::

        .. figure-mpl:: plot_directive/nestedpage/index-1.png
            :alt: bar
            :srcset: plot_directive/nestedpage/index-1.png
                     plot_directive/nestedpage/index-1.2x.png 2.00x
            :class: plot_directive

    生成的html（在``nestedpage/index.html``中）为::

        <img src="../_images/nestedpage-index-1.png"
            srcset="../_images/nestedpage-index-1.png,
                    ../_images/_images/nestedpage-index-1.2x.png 2.00x",
            alt="bar"
            class="sphx-glr-single-img" />

    其中子目录包含在图像名称中，以确保唯一性。
    """

    has_content = False  # 指定该指令没有内容部分
    required_arguments = 1  # 指定必须有一个参数
    optional_arguments = 2  # 指定可以有两个可选参数
    final_argument_whitespace = False  # 指定最后一个参数不能包含空格
    option_spec = {  # 指定可接受的选项及其类型
        'alt': directives.unchanged,  # alt选项，不改变其内容
        'height': directives.length_or_unitless,  # height选项，接受长度或无单位
        'width': directives.length_or_percentage_or_unitless,  # width选项，接受长度、百分比或无单位
        'scale': directives.nonnegative_int,  # scale选项，非负整数
        'align': Image.align,  # align选项，图像对齐方式
        'class': directives.class_option,  # class选项，类选项
        'caption': directives.unchanged,  # caption选项，不改变其内容
        'srcset': directives.unchanged,  # srcset选项，不改变其内容
    }
    # 定义一个方法 run，该方法是一个对象的成员方法
    def run(self):

        # 创建一个 figmplnode 对象，表示图像节点
        image_node = figmplnode()

        # 从参数中获取图像的文件名
        imagenm = self.arguments[0]

        # 设置图像节点的属性，如果选项中没有指定则使用默认值
        image_node['alt'] = self.options.get('alt', '')
        image_node['align'] = self.options.get('align', None)
        image_node['class'] = self.options.get('class', None)
        image_node['width'] = self.options.get('width', None)
        image_node['height'] = self.options.get('height', None)
        image_node['scale'] = self.options.get('scale', None)
        image_node['caption'] = self.options.get('caption', None)

        # 设置图像节点的 URI 属性为给定的 imagenm，表示图像的位置
        image_node['uri'] = imagenm

        # 设置图像节点的 srcset 属性，用于指定多种分辨率下的图片地址
        image_node['srcset'] = self.options.get('srcset', None)

        # 返回一个包含图像节点的列表，作为方法的结果
        return [image_node]
def _parse_srcsetNodes(st):
    """
    解析srcset字符串，将其转换为字典格式。
    """
    entries = st.split(',')  # 将srcset字符串按逗号分割成多个条目
    srcset = {}  # 初始化空字典，用于存放解析后的srcset信息
    for entry in entries:
        spl = entry.strip().split(' ')  # 去除首尾空格后，按空格分割每个条目
        if len(spl) == 1:
            srcset[0] = spl[0]  # 如果分割后长度为1，将图像密度设为0，URL为对应的图像URL
        elif len(spl) == 2:
            mult = spl[1][:-1]  # 获取图像密度，并去除末尾的'x'
            srcset[float(mult)] = spl[0]  # 将图像密度和对应的URL存入字典，密度转换为浮点数
        else:
            raise ExtensionError(f'srcset argument "{entry}" is invalid.')  # 如果条目长度不为1或2，抛出异常
    return srcset  # 返回解析后的srcset字典


def _copy_images_figmpl(self, node):
    """
    复制图片到指定目录，并返回相关信息。
    """
    # 图像源集不为空时，解析srcset字符串
    if node['srcset']:
        srcset = _parse_srcsetNodes(node['srcset'])
    else:
        srcset = None  # 否则置srcset为None

    # 获取文档源文件的路径
    docsource = PurePath(self.document['source']).parent

    # 获取构建目录的根路径
    srctop = self.builder.srcdir

    # 计算相对路径，并将路径分隔符替换为连字符
    rel = relpath(docsource, srctop).replace('.', '').replace(os.sep, '-')
    if len(rel):
        rel += '-'  # 如果rel不为空，添加连字符

    # 构建图片目录的完整路径
    imagedir = PurePath(self.builder.outdir, self.builder.imagedir)

    # 确保图片目录存在，不存在则创建
    Path(imagedir).mkdir(parents=True, exist_ok=True)

    # 如果srcset不为空，复制所有图片到图片目录中
    if srcset:
        for src in srcset.values():
            # 图片路径为相对于文档源文件的绝对路径
            abspath = PurePath(docsource, src)
            # 图片文件名为rel加上路径中的文件名
            name = rel + abspath.name
            shutil.copyfile(abspath, imagedir / name)
    else:
        # 否则，复制node['uri']指定的图片到图片目录中
        abspath = PurePath(docsource, node['uri'])
        name = rel + abspath.name
        shutil.copyfile(abspath, imagedir / name)

    # 返回图片目录路径、srcset字典和相对路径
    return imagedir, srcset, rel


def visit_figmpl_html(self, node):
    """
    访问HTML节点并处理图像相关内容。
    """
    imagedir, srcset, rel = _copy_images_figmpl(self, node)

    # 获取文档源文件的路径
    docsource = PurePath(self.document['source'])

    # 获取构建目录的根路径，并确保以斜杠结尾
    srctop = PurePath(self.builder.srcdir, '')

    # 计算相对路径
    relsource = relpath(docsource, srctop)

    # 获取构建目录的根路径，并确保以斜杠结尾
    desttop = PurePath(self.builder.outdir, '')

    # 构建目标路径
    dest = desttop / relsource

    # 计算图片相对路径
    imagerel = PurePath(relpath(imagedir, dest.parent)).as_posix()

    # 如果构建器名称为"dirhtml"，调整图片相对路径
    if self.builder.name == "dirhtml":
        imagerel = f'..{imagerel}'

    # 获取图片文件名，并构建URI
    nm = PurePath(node['uri'][1:]).name
    uri = f'{imagerel}/{rel}{nm}'

    # 初始化srcset字符串
    maxsrc = uri
    srcsetst = ''
    # 如果srcset非空，则执行以下代码块
    if srcset:
        # 初始化最大倍数为-1
        maxmult = -1
        # 遍历srcset字典中的每个倍数和对应的源路径
        for mult, src in srcset.items():
            # 从源路径中提取文件名部分
            nm = PurePath(src[1:]).name
            # 构建完整的路径，结合imagerel和rel变量
            path = f'{imagerel}/{rel}{nm}'
            # 将当前路径添加到srcsetst字符串中
            srcsetst += path
            # 根据当前倍数将相应的分隔符和倍数信息添加到srcsetst字符串中
            if mult == 0:
                srcsetst += ', '
            else:
                srcsetst += f' {mult:1.2f}x, '

            # 如果当前倍数大于最大倍数，更新最大倍数和对应的最大路径
            if mult > maxmult:
                maxmult = mult
                maxsrc = path

        # 去除末尾的逗号和空格
        srcsetst = srcsetst[:-2]

    # 获取节点中的alt属性值
    alt = node['alt']
    # 如果节点中存在class属性，则将其转换为HTML格式的class字符串
    if node['class'] is not None:
        classst = ' '.join(node['class'])
        classst = f'class="{classst}"'
    # 如果节点中不存在class属性，则classst为空字符串
    else:
        classst = ''

    # 定义需要处理的样式属性列表
    stylers = ['width', 'height', 'scale']
    stylest = ''
    # 遍历样式属性列表，如果节点中存在该属性，则将其格式化为CSS样式字符串
    for style in stylers:
        if node[style]:
            stylest += f'{style}: {node[style]};'

    # 获取节点中的对齐属性，如果不存在则默认为'center'
    figalign = node['align'] if node['align'] else 'center'
# 定义一个函数，用于处理 figure-mpl 指令生成的节点
def visit_figmpl_latex(self, node):
    # 如果节点的 srcset 属性不为空，则复制图像并获取相关信息
    if node['srcset'] is not None:
        imagedir, srcset = _copy_images_figmpl(self, node)
        maxmult = -1
        # 选择最高分辨率版本的图像用于 LaTeX
        maxmult = max(srcset, default=-1)
        # 获取图像的文件名，并将其设为节点的 uri 属性
        node['uri'] = PurePath(srcset[maxmult]).name

    # 调用 visit_figure 方法处理节点
    self.visit_figure(node)


# 定义一个函数，用于处理 figure-mpl 指令生成的节点在 HTML 中的处理方法
def depart_figmpl_html(self, node):
    # 在 HTML 中不进行任何操作，直接 pass
    pass


# 定义一个函数，用于处理 figure-mpl 指令生成的节点在 LaTeX 中的处理方法
def depart_figmpl_latex(self, node):
    # 调用 depart_figure 方法处理节点
    self.depart_figure(node)


# 定义一个函数，用于添加 figure-mpl 指令的节点到应用程序中
def figurempl_addnode(app):
    # 添加 figmplnode 节点类型到应用程序
    app.add_node(figmplnode,
                 html=(visit_figmpl_html, depart_figmpl_html),
                 latex=(visit_figmpl_latex, depart_figmpl_latex))


# 定义一个函数，用于设置应用程序，添加 figure-mpl 指令和其节点处理方法
def setup(app):
    # 添加 "figure-mpl" 指令到应用程序，使用 FigureMpl 类处理
    app.add_directive("figure-mpl", FigureMpl)
    # 添加 figure-mpl 指令的节点处理方法到应用程序
    figurempl_addnode(app)
    # 定义元数据，表明此设置是并行读写安全的，并包含 matplotlib 版本信息
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True,
                'version': matplotlib.__version__}
    # 返回设置的元数据
    return metadata
```