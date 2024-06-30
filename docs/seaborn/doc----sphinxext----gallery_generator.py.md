# `D:\src\scipysrc\seaborn\doc\sphinxext\gallery_generator.py`

```
"""
Sphinx plugin to run example scripts and create a gallery page.

Lightly modified from the mpld3 project.

"""
# 导入必要的库
import os  # 导入操作系统相关功能
import os.path as op  # 导入路径操作模块，并命名为op
import re  # 导入正则表达式模块
import glob  # 导入文件名匹配模块
import token  # 导入token模块
import tokenize  # 导入tokenize模块
import shutil  # 导入文件操作模块
import warnings  # 导入警告模块

# 设置matplotlib使用的后端为Agg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块，并忽略E402警告


# Python 3 没有execfile函数，定义一个用于执行文件的函数
def execfile(filename, globals=None, locals=None):
    with open(filename, "rb") as fp:
        exec(compile(fp.read(), filename, 'exec'), globals, locals)


RST_TEMPLATE = """

.. currentmodule:: seaborn

.. _{sphinx_tag}:

{docstring}

.. image:: {img_file}

**seaborn components used:** {components}

.. literalinclude:: {fname}
    :lines: {end_line}-

"""

INDEX_TEMPLATE = """
:html_theme.sidebar_secondary.remove:

.. raw:: html

    <style type="text/css">
    .thumb {{
        position: relative;
        float: left;
        width: 180px;
        height: 180px;
        margin: 0;
    }}

    .thumb img {{
        position: absolute;
        display: inline;
        left: 0;
        width: 170px;
        height: 170px;
        opacity:1.0;
        filter:alpha(opacity=100); /* For IE8 and earlier */
    }}

    .thumb:hover img {{
        -webkit-filter: blur(3px);
        -moz-filter: blur(3px);
        -o-filter: blur(3px);
        -ms-filter: blur(3px);
        filter: blur(3px);
        opacity:1.0;
        filter:alpha(opacity=100); /* For IE8 and earlier */
    }}

    .thumb span {{
        position: absolute;
        display: inline;
        left: 0;
        width: 170px;
        height: 170px;
        background: #000;
        color: #fff;
        visibility: hidden;
        opacity: 0;
        z-index: 100;
    }}

    .thumb p {{
        position: absolute;
        top: 45%;
        width: 170px;
        font-size: 110%;
        color: #fff;
    }}

    .thumb:hover span {{
        visibility: visible;
        opacity: .4;
    }}

    .caption {{
        position: absolute;
        width: 180px;
        top: 170px;
        text-align: center !important;
    }}
    </style>

.. _{sphinx_tag}:

Example gallery
===============

{toctree}

{contents}

.. raw:: html

    <div style="clear: both"></div>
"""


def create_thumbnail(infile, thumbfile,
                     width=275, height=275,
                     cx=0.5, cy=0.5, border=4):
    baseout, extout = op.splitext(thumbfile)

    # 读取输入文件的图像数据
    im = matplotlib.image.imread(infile)
    rows, cols = im.shape[:2]
    # 计算缩略图的起始位置和切片
    x0 = int(cx * cols - .5 * width)
    y0 = int(cy * rows - .5 * height)
    xslice = slice(x0, x0 + width)
    yslice = slice(y0, y0 + height)
    # 创建缩略图
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    # 创建matplotlib图形对象
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax = fig.add_axes([0, 0, 1, 1], aspect='auto',
                      frameon=False, xticks=[], yticks=[])
    # 检查缩略图的形状是否全部有效（非零）
    if all(thumb.shape):
        # 如果形状有效，使用imshow方法显示缩略图在当前图形上
        ax.imshow(thumb, aspect='auto', resample=True,
                  interpolation='bilinear')
    else:
        # 如果形状无效，发出警告并指出缩略图文件将为空
        warnings.warn(
            f"Bad thumbnail crop. {thumbfile} will be empty."
        )
    # 将当前图形保存为缩略图文件，使用指定的分辨率（dpi）
    fig.savefig(thumbfile, dpi=dpi)
    # 返回保存后的图形对象
    return fig
def indent(s, N=4):
    """indent a string"""
    # 将字符串中的换行符替换为指定数量空格加换行符，实现字符串缩进
    return s.replace('\n', '\n' + N * ' ')

class ExampleGenerator:
    """Tools for generating an example page from a file"""
    
    def __init__(self, filename, target_dir):
        # 初始化 ExampleGenerator 类的实例
        self.filename = filename
        self.target_dir = target_dir
        self.thumbloc = .5, .5  # 设置默认的缩略图位置为中心点（0.5, 0.5）
        self.extract_docstring()  # 调用提取文档字符串的方法
        
        with open(filename) as fid:
            self.filetext = fid.read()  # 读取文件内容并保存在实例变量中

        outfilename = op.join(target_dir, self.rstfilename)  # 组合输出文件的完整路径名

        # 只有当输出的 RST 文件不存在或者修改时间早于源文件时才执行以下操作
        file_mtime = op.getmtime(filename)
        if not op.exists(outfilename) or op.getmtime(outfilename) < file_mtime:
            self.exec_file()  # 执行文件内容
        else:
            print(f"skipping {self.filename}")  # 输出跳过信息

    @property
    def dirname(self):
        return op.split(self.filename)[0]  # 返回文件的目录路径

    @property
    def fname(self):
        return op.split(self.filename)[1]  # 返回文件的名称部分

    @property
    def modulename(self):
        return op.splitext(self.fname)[0]  # 返回文件的模块名称部分

    @property
    def pyfilename(self):
        return self.modulename + '.py'  # 返回对应的 Python 文件名

    @property
    def rstfilename(self):
        return self.modulename + ".rst"  # 返回对应的 RST 文件名

    @property
    def htmlfilename(self):
        return self.modulename + '.html'  # 返回对应的 HTML 文件名

    @property
    def pngfilename(self):
        pngfile = self.modulename + '.png'
        return "_images/" + pngfile  # 返回对应的 PNG 图片文件名，并放置在 _images 目录下

    @property
    def thumbfilename(self):
        pngfile = self.modulename + '_thumb.png'
        return pngfile  # 返回缩略图文件名，带有 _thumb 后缀

    @property
    def sphinxtag(self):
        return self.modulename  # 返回模块名称，作为 Sphinx 标签使用

    @property
    def pagetitle(self):
        return self.docstring.strip().split('\n')[0].strip()  # 返回页面标题，从文档字符串中提取第一行作为标题

    @property
    def plotfunc(self):
        match = re.search(r"sns\.(.+plot)\(", self.filetext)
        if match:
            return match.group(1)  # 从文件文本中匹配并返回第一个 sns.plot 相关函数名
        match = re.search(r"sns\.(.+map)\(", self.filetext)
        if match:
            return match.group(1)  # 从文件文本中匹配并返回第一个 sns.map 相关函数名
        match = re.search(r"sns\.(.+Grid)\(", self.filetext)
        if match:
            return match.group(1)  # 从文件文本中匹配并返回第一个 sns.Grid 相关函数名
        return ""  # 如果没有匹配到则返回空字符串

    @property
    def components(self):
        objects = re.findall(r"sns\.(\w+)\(", self.filetext)  # 从文件文本中查找所有以 sns. 开头的函数名
        refs = []
        for obj in objects:
            if obj[0].isupper():
                refs.append(f":class:`{obj}`")  # 如果函数名首字母大写，作为类名处理
            else:
                refs.append(f":func:`{obj}`")  # 否则作为函数名处理
        return ", ".join(refs)  # 返回处理后的函数名字符串，用逗号分隔
    def extract_docstring(self):
        """ Extract a module-level docstring
        """
        # 读取文件的所有行
        lines = open(self.filename).readlines()
        start_row = 0
        # 如果第一行以 '#!' 开始，将其移除，并更新起始行数
        if lines[0].startswith('#!'):
            lines.pop(0)
            start_row = 1

        docstring = ''
        first_par = ''
        line_iter = lines.__iter__()
        # 使用 tokenize 生成器来解析代码行
        tokens = tokenize.generate_tokens(lambda: next(line_iter))
        for tok_type, tok_content, _, (erow, _), _ in tokens:
            tok_type = token.tok_name[tok_type]
            # 跳过不需要处理的 token 类型
            if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
                continue
            elif tok_type == 'STRING':
                # 解析字符串型的 token，将其作为文档字符串进行评估
                docstring = eval(tok_content)
                # 如果文档字符串包含多个段落，提取第一个段落
                paragraphs = '\n'.join(line.rstrip()
                                       for line in docstring.split('\n')
                                       ).split('\n\n')
                if len(paragraphs) > 0:
                    first_par = paragraphs[0]
            break

        thumbloc = None
        # 在文档字符串中查找 '_thumb: (.数字), (.数字)' 格式的内容
        for i, line in enumerate(docstring.split("\n")):
            m = re.match(r"^_thumb: (\.\d+),\s*(\.\d+)", line)
            if m:
                # 如果找到匹配的内容，提取缩略图的位置信息
                thumbloc = float(m.group(1)), float(m.group(2))
                break
        # 如果找到缩略图位置信息，更新对象的缩略图位置，并从文档字符串中移除 '_thumb' 相关行
        if thumbloc is not None:
            self.thumbloc = thumbloc
            docstring = "\n".join([l for l in docstring.split("\n")
                                   if not l.startswith("_thumb")])

        # 更新对象的文档字符串、简短描述和结束行数
        self.docstring = docstring
        self.short_desc = first_par
        self.end_line = erow + 1 + start_row

    def exec_file(self):
        """ Execute the Python file specified by self.filename
        """
        # 打印正在运行的文件名
        print(f"running {self.filename}")

        # 关闭所有当前的 matplotlib 图形
        plt.close('all')
        my_globals = {'pl': plt,
                      'plt': plt}
        # 在指定的全局命名空间中执行文件
        execfile(self.filename, my_globals)

        # 获取当前的 matplotlib 图形对象，并进行绘制
        fig = plt.gcf()
        fig.canvas.draw()

        # 保存图形为 PNG 文件，并指定路径
        pngfile = op.join(self.target_dir, self.pngfilename)
        thumbfile = op.join("example_thumbs", self.thumbfilename)
        # 生成 HTML 代码，用于显示图像
        self.html = f"<img src=../{self.pngfilename}>"
        fig.savefig(pngfile, dpi=75, bbox_inches="tight")

        # 创建缩略图并指定位置
        cx, cy = self.thumbloc
        create_thumbnail(pngfile, thumbfile, cx=cx, cy=cy)

    def toctree_entry(self):
        """ Generate a Sphinx toctree entry for the HTML output
        """
        # 返回一个用于 Sphinx toctree 的条目
        return f"   ./{op.splitext(self.htmlfilename)[0]}\n\n"

    def contents_entry(self):
        """ Generate an HTML contents entry with thumbnail and label
        """
        # 返回一个包含缩略图和标签的 HTML 内容条目
        return (".. raw:: html\n\n"
                "    <div class='thumb align-center'>\n"
                "    <a href=./{}>\n"
                "    <img src=../_static/{}>\n"
                "    <span class='thumb-label'>\n"
                "    <p>{}</p>\n"
                "    </span>\n"
                "    </a>\n"
                "    </div>\n\n"
                "\n\n"
                "".format(self.htmlfilename,
                          self.thumbfilename,
                          self.plotfunc))
def main(app):
    # 定义静态文件目录为应用构建器的源目录下的 '_static'
    static_dir = op.join(app.builder.srcdir, '_static')
    # 定义目标目录为应用构建器的源目录下的 'examples'
    target_dir = op.join(app.builder.srcdir, 'examples')
    # 定义图片目录为应用构建器的源目录下的 'examples/_images'
    image_dir = op.join(app.builder.srcdir, 'examples/_images')
    # 定义缩略图目录为应用构建器的源目录下的 'example_thumbs'
    thumb_dir = op.join(app.builder.srcdir, "example_thumbs")
    # 定义源文件目录为应用构建器的源目录的上级目录下的 'examples'
    source_dir = op.abspath(op.join(app.builder.srcdir, '..', 'examples'))
    
    # 如果静态文件目录不存在，则创建
    if not op.exists(static_dir):
        os.makedirs(static_dir)

    # 如果目标目录不存在，则创建
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # 如果图片目录不存在，则创建
    if not op.exists(image_dir):
        os.makedirs(image_dir)

    # 如果缩略图目录不存在，则创建
    if not op.exists(thumb_dir):
        os.makedirs(thumb_dir)

    # 如果源文件目录不存在，则创建
    if not op.exists(source_dir):
        os.makedirs(source_dir)

    # 存储用于展示的横幅数据的列表
    banner_data = []

    # 创建 TOC 树的初始部分
    toctree = ("\n\n"
               ".. toctree::\n"
               "   :hidden:\n\n")
    contents = "\n\n"

    # 遍历源目录下的所有 Python 文件
    for filename in sorted(glob.glob(op.join(source_dir, "*.py"))):
        # 使用 ExampleGenerator 类处理每个文件
        ex = ExampleGenerator(filename, target_dir)

        # 添加横幅数据到列表中
        banner_data.append({"title": ex.pagetitle,
                            "url": op.join('examples', ex.htmlfilename),
                            "thumb": op.join(ex.thumbfilename)})
        
        # 复制源文件到目标目录
        shutil.copyfile(filename, op.join(target_dir, ex.pyfilename))
        
        # 根据模板生成 RST 输出内容
        output = RST_TEMPLATE.format(sphinx_tag=ex.sphinxtag,
                                     docstring=ex.docstring,
                                     end_line=ex.end_line,
                                     components=ex.components,
                                     fname=ex.pyfilename,
                                     img_file=ex.pngfilename)
        
        # 将 RST 内容写入到目标目录下的文件中
        with open(op.join(target_dir, ex.rstfilename), 'w') as f:
            f.write(output)

        # 添加到 TOC 树和内容列表中的条目
        toctree += ex.toctree_entry()
        contents += ex.contents_entry()

    # 如果横幅数据不足 10 条，则重复列表以达到最少 10 条
    if len(banner_data) < 10:
        banner_data = (4 * banner_data)[:10]

    # 写入索引文件
    index_file = op.join(target_dir, 'index.rst')
    with open(index_file, 'w') as index:
        index.write(INDEX_TEMPLATE.format(sphinx_tag="example_gallery",
                                          toctree=toctree,
                                          contents=contents))


def setup(app):
    # 将 main 函数与应用的 'builder-inited' 事件连接起来
    app.connect('builder-inited', main)
```