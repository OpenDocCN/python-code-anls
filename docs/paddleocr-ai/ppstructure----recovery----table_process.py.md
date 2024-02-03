# `.\PaddleOCR\ppstructure\recovery\table_process.py`

```py
# 版权声明
# 本代码版权归 PaddlePaddle 作者所有。保留所有权利。
# 根据 Apache 许可证 2.0 版本授权；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。

"""
# 本代码参考自：https://github.com/weizwx/html2docx/blob/master/htmldocx/h2d.py
"""

# 导入所需的库
import re
import docx
from docx import Document
from bs4 import BeautifulSoup
from html.parser import HTMLParser

# 定义函数：获取表格的行
def get_table_rows(table_soup):
    # 定义表格行的选择器
    table_row_selectors = [
        'table > tr', 'table > thead > tr', 'table > tbody > tr',
        'table > tfoot > tr'
    ]
    # 如果存在表头、主体、页脚或直接子级 tr 标签，则从中添加行维度
    return table_soup.select(', '.join(table_row_selectors), recursive=False)

# 定义函数：获取表格的列
def get_table_columns(row):
    # 获取指定行标签的所有列
    return row.find_all(['th', 'td'], recursive=False) if row else []

# 定义函数：获取表格的维度
def get_table_dimensions(table_soup):
    # 获取表格的行
    rows = get_table_rows(table_soup)
    # 如果表格为空或在 table 和 tr 标签之间有非直接子级，则假定行维度和列维度为 0

    cols = get_table_columns(rows[0]) if rows else []
    # 添加 colspan 计算列数
    col_count = 0
    for col in cols:
        colspan = col.attrs.get('colspan', 1)
        col_count += int(colspan)

    return rows, col_count

# 定义函数：获取单元格的 HTML 内容
def get_cell_html(soup):
    # 返回不包含开头和结尾 <td> 标签的 td 元素的字符串
    # 由于 find_all 只能找到元素标签，无法找到不在元素内的文本，因此不能使用它
    # 将 BeautifulSoup 对象的内容转换为字符串，并用空格连接起来
    return ' '.join([str(i) for i in soup.contents])
# 删除给定段落对象
def delete_paragraph(paragraph):
    # 获取段落对象的 XML 元素
    p = paragraph._element
    # 从父节点中移除该 XML 元素
    p.getparent().remove(p)
    # 将段落对象的 XML 元素置空
    p._p = p._element = None

# 从字符串中移除空白字符
def remove_whitespace(string, leading=False, trailing=False):
    """Remove white space from a string.
    Args:
        string(str): The string to remove white space from.
        leading(bool, optional): Remove leading new lines when True.
        trailing(bool, optional): Remove trailing new lines when False.
    Returns:
        str: The input string with new line characters removed and white space squashed.
    Examples:
        Single or multiple new line characters are replaced with space.
            >>> remove_whitespace("abc\\ndef")
            'abc def'
            >>> remove_whitespace("abc\\n\\n\\ndef")
            'abc def'
        New line characters surrounded by white space are replaced with a single space.
            >>> remove_whitespace("abc \\n \\n \\n def")
            'abc def'
            >>> remove_whitespace("abc  \\n  \\n  \\n  def")
            'abc def'
        Leading and trailing new lines are replaced with a single space.
            >>> remove_whitespace("\\nabc")
            ' abc'
            >>> remove_whitespace("  \\n  abc")
            ' abc'
            >>> remove_whitespace("abc\\n")
            'abc '
            >>> remove_whitespace("abc  \\n  ")
            'abc '
        Use ``leading=True`` to remove leading new line characters, including any surrounding
        white space:
            >>> remove_whitespace("\\nabc", leading=True)
            'abc'
            >>> remove_whitespace("  \\n  abc", leading=True)
            'abc'
        Use ``trailing=True`` to remove trailing new line characters, including any surrounding
        white space:
            >>> remove_whitespace("abc  \\n  ", trailing=True)
            'abc'
    """
    # 移除任何前导换行符以及周围的空白字符
    # 如果参数 leading 为真，则去除字符串开头的空行和空格
    if leading:
        string = re.sub(r'^\s*\n+\s*', '', string)

    # 如果参数 trailing 为真，则去除字符串末尾的空行和空格
    # Remove any trailing new line characters along with any surrounding white space
    if trailing:
        string = re.sub(r'\s*\n+\s*$', '', string)

    # 替换字符串中的换行符，并吸收周围的空格
    string = re.sub(r'\s*\n\s*', ' ', string)
    # TODO 需要一种方法来去除例如文本 <span>   </span>  文本 中的额外空格
    # 替换多余的空格为一个空格
    return re.sub(r'\s+', ' ', string)
# 定义字体样式映射关系，将 HTML 标签对应的样式映射到 Word 文档中的样式
font_styles = {
    'b': 'bold',  # 加粗
    'strong': 'bold',  # 加粗
    'em': 'italic',  # 斜体
    'i': 'italic',  # 斜体
    'u': 'underline',  # 下划线
    's': 'strike',  # 删除线
    'sup': 'superscript',  # 上标
    'sub': 'subscript',  # 下标
    'th': 'bold',  # 表头
}

# 定义字体名称映射关系，将 HTML 标签对应的字体名称映射到 Word 文档中的字体名称
font_names = {
    'code': 'Courier',  # 代码字体
    'pre': 'Courier',  # 预格式化字体
}

# 定义 HtmlToDocx 类，继承自 HTMLParser 类
class HtmlToDocx(HTMLParser):
    # 初始化方法
    def __init__(self):
        super().__init__()
        # 初始化选项字典，包括修复 HTML、包含图片、包含表格、包含样式
        self.options = {
            'fix-html': True,
            'images': True,
            'tables': True,
            'styles': True,
        }
        # 表格行选择器列表
        self.table_row_selectors = [
            'table > tr', 'table > thead > tr', 'table > tbody > tr',
            'table > tfoot > tr'
        ]
        self.table_style = None  # 表格样式
        self.paragraph_style = None  # 段落样式

    # 设置初始属性方法
    def set_initial_attrs(self, document=None):
        # 初始化标签字典，包括 span 和 list
        self.tags = {
            'span': [],
            'list': [],
        }
        if document:
            self.doc = document
        else:
            self.doc = Document()  # 创建 Word 文档对象
        self.bs = self.options[
            'fix-html']  # 是否使用 BeautifulSoup 进行清理
        self.document = self.doc
        self.include_tables = True  # 是否包含表格
        self.include_images = self.options['images']  # 是否包含图片
        self.include_styles = self.options['styles']  # 是否包含样式
        self.paragraph = None  # 当前段落
        self.skip = False  # 是否跳过
        self.skip_tag = None  # 要跳过的标签
        self.instances_to_skip = 0  # 要跳过的实例数

    # 从另一个实例中复制设置方法
    def copy_settings_from(self, other):
        """Copy settings from another instance of HtmlToDocx"""
        self.table_style = other.table_style  # 复制表格样式
        self.paragraph_style = other.paragraph_style  # 复制段落样式
    # 忽略嵌套表格，返回仅包含最高级别表格的数组
    def ignore_nested_tables(self, tables_soup):
        """
        Returns array containing only the highest level tables
        Operates on the assumption that bs4 returns child elements immediately after
        the parent element in `find_all`. If this changes in the future, this method will need to be updated
        :return:
        """
        new_tables = []
        nest = 0
        遍历传入的表格元素
        for table in tables_soup:
            如果嵌套层级不为0，则继续减少嵌套层级并跳过当前表格
            if nest:
                nest -= 1
                continue
            将当前表格添加到新表格数组中
            new_tables.append(table)
            计算当前表格内的子表格数量，作为下一个需要跳过的嵌套层级
            nest = len(table.find_all('table'))
        返回最高级别的表格数组
        return new_tables

    # 获取表格数据
    def get_tables(self):
        如果对象中没有'soup'属性
        if not hasattr(self, 'soup'):
            将'include_tables'属性设置为False并返回
            self.include_tables = False
            return
            # find other way to do it, or require this dependency?
        使用'ignore_nested_tables'方法获取最高级别的表格
        self.tables = self.ignore_nested_tables(self.soup.find_all('table'))
        初始化表格编号
        self.table_no = 0

    # 运行处理过程
    def run_process(self, html):
        如果存在BeautifulSoup库
        if self.bs and BeautifulSoup:
            使用BeautifulSoup解析HTML内容
            self.soup = BeautifulSoup(html, 'html.parser')
            将HTML内容转换为字符串
            html = str(self.soup)
        如果需要包含表格
        if self.include_tables:
            获取表格数据
            self.get_tables()
        将HTML内容传递给处理方法
        self.feed(html)

    # 将HTML内容添加到单元格
    def add_html_to_cell(self, html, cell):
        如果第二个参数不是docx.table._Cell类型
        if not isinstance(cell, docx.table._Cell):
            抛出数值错误异常
            raise ValueError('Second argument needs to be a %s' %
                             docx.table._Cell)
        获取单元格中的第一个段落
        unwanted_paragraph = cell.paragraphs[0]
        如果不需要的段落内容为空
        if unwanted_paragraph.text == "":
            删除不需要的段落
            delete_paragraph(unwanted_paragraph)
        设置单元格的初始属性
        self.set_initial_attrs(cell)
        运行处理过程
        self.run_process(html)
        # 单元格必须以段落结尾，否则会收到有关损坏文件的消息
        # https://stackoverflow.com/a/29287121
        如果文档中没有段落
        if not self.doc.paragraphs:
            添加一个空段落到文档中
            self.doc.add_paragraph('')
    # 定义一个方法，用于应用段落样式
    def apply_paragraph_style(self, style=None):
        # 尝试执行以下代码块，捕获可能出现的异常
        try:
            # 如果传入了样式参数，则将段落样式设置为传入的样式
            if style:
                self.paragraph.style = style
            # 如果没有传入样式参数，但存在实例变量中的段落样式，则将段落样式设置为实例变量中的段落样式
            elif self.paragraph_style:
                self.paragraph.style = self.paragraph_style
        # 捕获 KeyError 异常，并将其转换为 ValueError 异常抛出
        except KeyError as e:
            raise ValueError(
                f"Unable to apply style {self.paragraph_style}.") from e
    def handle_table(self, html, doc):
        """
        To handle nested tables, we will parse tables manually as follows:
        Get table soup
        Create docx table
        Iterate over soup and fill docx table with new instances of this parser
        Tell HTMLParser to ignore any tags until the corresponding closing table tag
        """
        # 使用 BeautifulSoup 解析 HTML 表格
        table_soup = BeautifulSoup(html, 'html.parser')
        # 获取表格的行数和列数
        rows, cols_len = get_table_dimensions(table_soup)
        # 在文档中添加一个表格，行数为表格的行数，列数为表格的列数
        table = doc.add_table(len(rows), cols_len)
        # 设置表格的样式为 'Table Grid'

        table.style = doc.styles['Table Grid']

        cell_row = 0
        # 遍历表格的每一行
        for index, row in enumerate(rows):
            # 获取当前行的列
            cols = get_table_columns(row)
            cell_col = 0
            # 遍历当前行的每一列
            for col in cols:
                if cell_col >= cols_len:
                    break

                # 获取当前单元格的列合并数和行合并数
                colspan = int(col.attrs.get('colspan', 1))
                rowspan = int(col.attrs.get('rowspan', 1))
                # 获取当前单元格的 HTML 内容
                cell_html = get_cell_html(col)
                # 如果当前单元格是表头单元格，则加粗显示
                if col.name == 'th':
                    cell_html = "<b>%s</b>" % cell_html

                # 获取当前单元格对象
                docx_cell = table.cell(cell_row, cell_col)
                if (cell_col + colspan -1) >= cols_len:
                    colspan -= 1

                # 获取需要合并的单元格对象
                cell_to_merge = table.cell(cell_row + rowspan - 1,
                                           cell_col + colspan - 1)
                # 如果当前单元格和需要合并的单元格不相同，则合并它们
                if docx_cell != cell_to_merge:
                    docx_cell.merge(cell_to_merge)

                # 创建一个新的 HtmlToDocx 实例，将当前设置复制到子实例中
                child_parser = HtmlToDocx()
                child_parser.copy_settings_from(self)
                # 将 HTML 内容添加到当前单元格中
                child_parser.add_html_to_cell(cell_html or ' ', docx_cell)

                cell_col += colspan
            cell_row += 1
    # 处理传入的数据
    def handle_data(self, data):
        # 如果标记为跳过，则直接返回
        if self.skip:
            return

        # 只有在不在 <pre> 块中时才移除空白字符
        if 'pre' not in self.tags:
            # 移除所有实例中的前导和尾随空格
            data = remove_whitespace(data, True, True)

        # 如果段落不存在，则创建一个新段落并应用段落样式
        if not self.paragraph:
            self.paragraph = self.doc.add_paragraph()
            self.apply_paragraph_style()

        # 在有效的 HTML 文档中只能有一个嵌套链接
        # 不能在 A 标签中有交互内容，包括链接
        link = self.tags.get('a')
        if link:
            self.handle_link(link['href'], data)
        else:
            # 如果存在链接，则不直接将数据放入运行中
            self.run = self.paragraph.add_run(data)
            spans = self.tags['span']
            for span in spans:
                if 'style' in span:
                    style = self.parse_dict_string(span['style'])
                    self.add_styles_to_run(style)

            # 添加字体样式和名称
            for tag in self.tags:
                if tag in font_styles:
                    font_style = font_styles[tag]
                    setattr(self.run.font, font_style, True)

                if tag in font_names:
                    font_name = font_names[tag]
                    self.run.font.name = font_name
```