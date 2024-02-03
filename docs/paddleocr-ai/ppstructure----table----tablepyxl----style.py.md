# `.\PaddleOCR\ppstructure\table\tablepyxl\style.py`

```py
# 这里处理将 CSS 样式转换为 openpyxl 样式，并在 DOM 中从父级传递到子级。

from openpyxl.cell import cell
from openpyxl.styles import Font, Alignment, PatternFill, NamedStyle, Border, Side, Color
from openpyxl.styles.fills import FILL_SOLID
from openpyxl.styles.numbers import FORMAT_CURRENCY_USD_SIMPLE, FORMAT_PERCENTAGE
from openpyxl.styles.colors import BLACK

FORMAT_DATE_MMDDYYYY = 'mm/dd/yyyy'


def colormap(color):
    """
    便于查找已知颜色
    """
    cmap = {'black': BLACK}
    return cmap.get(color, color)


def style_string_to_dict(style):
    """
    将 CSS 样式字符串转换为 Python 字典
    """
    def clean_split(string, delim):
        return (s.strip() for s in string.split(delim))
    styles = [clean_split(s, ":") for s in style.split(";") if ":" in s]
    return dict(styles)


def get_side(style, name):
    return {'border_style': style.get('border-{}-style'.format(name)),
            'color': colormap(style.get('border-{}-color'.format(name)))}

known_styles = {}


def style_dict_to_named_style(style_dict, number_format=None):
    """
    将 CSS 样式（存储在 Python 字典中）转换为 openpyxl NamedStyle
    """

    style_and_format_string = str({
        'style_dict': style_dict,
        'parent': style_dict.parent,
        'number_format': number_format,
    })
    # 如果样式和格式字符串不在已知样式中
    if style_and_format_string not in known_styles:
        # 创建字体对象
        font = Font(bold=style_dict.get('font-weight') == 'bold',
                    color=style_dict.get_color('color', None),
                    size=style_dict.get('font-size'))

        # 创建对齐对象
        alignment = Alignment(horizontal=style_dict.get('text-align', 'general'),
                              vertical=style_dict.get('vertical-align'),
                              wrap_text=style_dict.get('white-space', 'nowrap') == 'normal')

        # 创建填充对象
        bg_color = style_dict.get_color('background-color')
        fg_color = style_dict.get_color('foreground-color', Color())
        fill_type = style_dict.get('fill-type')
        if bg_color and bg_color != 'transparent':
            fill = PatternFill(fill_type=fill_type or FILL_SOLID,
                               start_color=bg_color,
                               end_color=fg_color)
        else:
            fill = PatternFill()

        # 创建边框对象
        border = Border(left=Side(**get_side(style_dict, 'left')),
                        right=Side(**get_side(style_dict, 'right')),
                        top=Side(**get_side(style_dict, 'top')),
                        bottom=Side(**get_side(style_dict, 'bottom')),
                        diagonal=Side(**get_side(style_dict, 'diagonal')),
                        diagonal_direction=None,
                        outline=Side(**get_side(style_dict, 'outline')),
                        vertical=None,
                        horizontal=None)

        # 生成样式名称
        name = 'Style {}'.format(len(known_styles) + 1)

        # 创建命名样式对象
        pyxl_style = NamedStyle(name=name, font=font, fill=fill, alignment=alignment, border=border,
                                number_format=number_format)

        # 将新样式添加到已知样式中
        known_styles[style_and_format_string] = pyxl_style

    # 返回样式对象
    return known_styles[style_and_format_string]
# 定义一个继承自 dict 的类 StyleDict，用于查找父字典中的项
class StyleDict(dict):
    """
    It's like a dictionary, but it looks for items in the parent dictionary
    """
    # 初始化方法，接受任意参数和关键字参数，将 parent 参数从 kwargs 中弹出
    def __init__(self, *args, **kwargs):
        # 从 kwargs 中获取 parent 参数，如果没有则为 None
        self.parent = kwargs.pop('parent', None)
        # 调用父类 dict 的初始化方法
        super(StyleDict, self).__init__(*args, **kwargs)

    # 重写 __getitem__ 方法，查找元素时先在当前字典中查找，如果没有则在父字典中查找
    def __getitem__(self, item):
        if item in self:
            return super(StyleDict, self).__getitem__(item)
        elif self.parent:
            return self.parent[item]
        else:
            raise KeyError('{} not found'.format(item))

    # 重写 __hash__ 方法，返回当前字典的哈希值
    def __hash__(self):
        return hash(tuple([(k, self.get(k)) for k in self._keys()]))

    # 定义一个方法 _keys，通过 yield 关键字返回当前字典的键，避免创建不必要的数据结构
    # 适用于 python2 和 python3，因为 python3 中 .keys() 方法返回的是 dictionary_view，python2 返回的是 list
    def _keys(self):
        yielded = set()
        for k in self.keys():
            yielded.add(k)
            yield k
        if self.parent:
            for k in self.parent._keys():
                if k not in yielded:
                    yielded.add(k)
                    yield k

    # 定义一个方法 get，获取指定键的值，如果不存在则返回默认值
    def get(self, k, d=None):
        try:
            return self[k]
        except KeyError:
            return d

    # 定义一个方法 get_color，获取指定键的颜色值，如果存在颜色值以 '#' 开头，则去掉 '#'
    # 如果颜色值长度为 3，则扩展为 6 位，以符合 openpyxl 的要求
    def get_color(self, k, d=None):
        """
        Strip leading # off colors if necessary
        """
        color = self.get(k, d)
        if hasattr(color, 'startswith') and color.startswith('#'):
            color = color[1:]
            if len(color) == 3:  # Premailers reduces colors like #00ff00 to #0f0, openpyxl doesn't like that
                color = ''.join(2 * c for c in color)
        return color


# 定义一个基类 Element，用于表示一个带有级联样式的 html 元素
# 创建元素时会同时创建一个父元素，以便存储的 StyleDict 可以指向父元素的 StyleDict
class Element(object):
    """
    Our base class for representing an html element along with a cascading style.
    The element is created along with a parent so that the StyleDict that we store
    can point to the parent's StyleDict.
    """
    # 初始化方法，接受一个元素和一个可选的父样式对象作为参数
    def __init__(self, element, parent=None):
        # 将传入的元素赋值给对象的 element 属性
        self.element = element
        # 初始化 number_format 属性为 None
        self.number_format = None
        # 如果有父样式对象，则获取其样式字典，否则为 None
        parent_style = parent.style_dict if parent else None
        # 将元素的 style 属性转换为字典，然后使用 StyleDict 类创建样式字典对象，并赋值给对象的 style_dict 属性
        self.style_dict = StyleDict(style_string_to_dict(element.get('style', '')), parent=parent_style)
        # 初始化样式缓存为 None
        self._style_cache = None

    # 将元素的 CSS 样式转换为 openpyxl 的 NamedStyle 对象
    def style(self):
        """
        Turn the css styles for this element into an openpyxl NamedStyle.
        """
        # 如果样式缓存为空，则将样式字典转换为 NamedStyle 对象，并缓存起来
        if not self._style_cache:
            self._style_cache = style_dict_to_named_style(self.style_dict, number_format=self.number_format)
        # 返回样式缓存
        return self._style_cache

    # 从元素的样式字典中提取指定维度的值，并将其作为浮点数返回
    def get_dimension(self, dimension_key):
        """
        Extracts the dimension from the style dict of the Element and returns it as a float.
        """
        # 从样式字典中获取指定维度的值
        dimension = self.style_dict.get(dimension_key)
        # 如果存在维度值
        if dimension:
            # 如果维度值的最后两个字符是 ['px', 'em', 'pt', 'in', 'cm'] 中的一个，则去掉单位
            if dimension[-2:] in ['px', 'em', 'pt', 'in', 'cm']:
                dimension = dimension[:-2]
            # 将维度值转换为浮点数
            dimension = float(dimension)
        # 返回维度值
        return dimension
class Table(Element):
    """
    The concrete implementations of Elements are semantically named for the types of elements we are interested in.
    This defines a very concrete tree structure for html tables that we expect to deal with. I prefer this compared to
    allowing Element to have an arbitrary number of children and dealing with an abstract element tree.
    """
    # 定义 Table 类，继承自 Element 类，用于表示 HTML 表格
    def __init__(self, table):
        """
        takes an html table object (from lxml)
        """
        # 初始化方法，接受一个 lxml 中的 html 表格对象
        super(Table, self).__init__(table)
        # 查找表格中的 thead 元素
        table_head = table.find('thead')
        # 如果找到 thead 元素，则创建 TableHead 对象，否则为 None
        self.head = TableHead(table_head, parent=self) if table_head is not None else None
        # 查找表格中的 tbody 元素
        table_body = table.find('tbody')
        # 创建 TableBody 对象，如果找到 tbody 元素，则使用 tbody，否则使用整个表格
        self.body = TableBody(table_body if table_body is not None else table, parent=self)


class TableHead(Element):
    """
    This class maps to the `<th>` element of the html table.
    """
    # 定义 TableHead 类，继承自 Element 类，用于表示 HTML 表格中的 th 元素
    def __init__(self, head, parent=None):
        # 初始化方法，接受一个 th 元素和可选的父元素
        super(TableHead, self).__init__(head, parent=parent)
        # 遍历 th 元素中的 tr 元素，创建 TableRow 对象
        self.rows = [TableRow(tr, parent=self) for tr in head.findall('tr')]


class TableBody(Element):
    """
    This class maps to the `<tbody>` element of the html table.
    """
    # 定义 TableBody 类，继承自 Element 类，用于表示 HTML 表格中的 tbody 元素
    def __init__(self, body, parent=None):
        # 初始化方法，接受一个 tbody 元素和可选的父元素
        super(TableBody, self).__init__(body, parent=parent)
        # 遍历 tbody 元素中的 tr 元素，创建 TableRow 对象
        self.rows = [TableRow(tr, parent=self) for tr in body.findall('tr')]


class TableRow(Element):
    """
    This class maps to the `<tr>` element of the html table.
    """
    # 定义 TableRow 类，继承自 Element 类，用于表示 HTML 表格中的 tr 元素
    def __init__(self, tr, parent=None):
        # 初始化方法，接受一个 tr 元素和可选的父元素
        super(TableRow, self).__init__(tr, parent=parent)
        # 遍历 tr 元素中的 th 和 td 元素，创建 TableCell 对象
        self.cells = [TableCell(cell, parent=self) for cell in tr.findall('th') + tr.findall('td')]


def element_to_string(el):
    # 将元素转换为字符串并去除首尾空格
    return _element_to_string(el).strip()


def _element_to_string(el):
    string = ''

    for x in el.iterchildren():
        string += '\n' + _element_to_string(x)

    text = el.text.strip() if el.text else ''
    tail = el.tail.strip() if el.tail else ''

    return text + string + '\n' + tail
class TableCell(Element):
    """
    This class maps to the `<td>` element of the html table.
    """
    CELL_TYPES = {'TYPE_STRING', 'TYPE_FORMULA', 'TYPE_NUMERIC', 'TYPE_BOOL', 'TYPE_CURRENCY', 'TYPE_PERCENTAGE',
                  'TYPE_NULL', 'TYPE_INLINE', 'TYPE_ERROR', 'TYPE_FORMULA_CACHE_STRING', 'TYPE_INTEGER'}

    def __init__(self, cell, parent=None):
        # 调用父类的构造函数，初始化 TableCell 对象
        super(TableCell, self).__init__(cell, parent=parent)
        # 将单元格的值转换为字符串
        self.value = element_to_string(cell)
        # 获取单元格的数字格式
        self.number_format = self.get_number_format()

    def data_type(self):
        # 获取单元格的数据类型
        cell_types = self.CELL_TYPES & set(self.element.get('class', '').split())
        if cell_types:
            if 'TYPE_FORMULA' in cell_types:
                # 确保 TYPE_FORMULA 优先于集合中的其他类
                cell_type = 'TYPE_FORMULA'
            elif cell_types & {'TYPE_CURRENCY', 'TYPE_INTEGER', 'TYPE_PERCENTAGE'}:
                cell_type = 'TYPE_NUMERIC'
            else:
                cell_type = cell_types.pop()
        else:
            cell_type = 'TYPE_STRING'
        return getattr(cell, cell_type)

    def get_number_format(self):
        # 获取单元格的数字格式
        if 'TYPE_CURRENCY' in self.element.get('class', '').split():
            return FORMAT_CURRENCY_USD_SIMPLE
        if 'TYPE_INTEGER' in self.element.get('class', '').split():
            return '#,##0'
        if 'TYPE_PERCENTAGE' in self.element.get('class', '').split():
            return FORMAT_PERCENTAGE
        if 'TYPE_DATE' in self.element.get('class', '').split():
            return FORMAT_DATE_MMDDYYYY
        if self.data_type() == cell.TYPE_NUMERIC:
            try:
                int(self.value)
            except ValueError:
                return '#,##0.##'
            else:
                return '#,##0'

    def format(self, cell):
        # 格式化单元格
        cell.style = self.style()
        data_type = self.data_type()
        if data_type:
            cell.data_type = data_type
```