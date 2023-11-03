# PythonMarkdown源码解析 10

# `/markdown/markdown/extensions/tables.py`

这段代码是一个Python脚本，它定义了一个名为“Tables Extension for Python-Markdown”的扩展。这个扩展提供了对表格的解析支持，以便在Python-Markdown中更方便地生成和渲染表格。

具体来说，这段代码实现了以下功能：

1. 解析Markdown表格，使其可以被Python代码顺利读取和解析。
2. 通过扩展定义，可以生成带有数据的表格，也可以在已有Markdown表格的基础上继续生成新的表格。
3. 支持表格的嵌套，即在多层表格中使用相对引用或绝对引用生成新的表格。
4. 通过设置`css`参数，可以定义表格的样式，如字体、颜色、边框等。
5. 可以设置表格的行/列数量以及标题，这将在渲染时使用。

代码的贡献者是Waylan Limberg，最初于2009年开发。


```py
# Tables Extension for Python-Markdown
# ====================================

# Added parsing of tables to Python-Markdown.

# See https://Python-Markdown.github.io/extensions/tables
# for documentation.

# Original code Copyright 2009 [Waylan Limberg](http://achinghead.com)

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
```

这段代码是一个Python源代码，定义了一个名为`AddParsingOfTablesToPythonMarkdown`的扩展。这个扩展将解析表格内容并添加到Python-Markdown中。

具体来说，这段代码实现了以下功能：

1. 从Python-Markdown中提取表格内容。
2. 遍历表格内容中的每个元素（如`<tr>`、`<td>`等）。
3. 对每个元素，解析其内容并将其添加到Python-Markdown的表格内容中。

由于这段代码使用了`@Extension`注解，因此它被视为Python-Markdown的扩展。这意味着，这段代码可以在Python-Markdown的配置文件中声明，并根据需要动态加载和卸载。

例如，在Python-Markdown的配置文件中，可以像下面这样声明扩展：
python
# config.py
import config

config.register_extension('add_parsing_of_tables', 'add_parsing_of_tables')

然后，在Python-Markdown的渲染进程配置文件中，可以定义扩展的名称：
python
# render.py
from render.py import render_block
from pyconfig import Config

class AddParsingOfTables(Config):
   def configure(self):
       self.block_processors.add_rules('header', self.add_header_blocks)
       self.block_processors.add_rules('table', self.add_table_blocks)

在`add_header_blocks`和`add_table_blocks`函数中，分别实现了对表格头和表格内容的解析和添加。


```py
Added parsing of tables to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/tables)
for details.
"""

from __future__ import annotations

from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .. import blockparser

```

This looks like a function that reads in a table of tics and a table of pipes, and returns a list of elements that correspond to each region in the table pipes. It does this by first storing the location of each pipe in a list of tics. Then, for each pipe, it looks through the tics to see if it regions, and if it's in a region, it doesn't want it, so it throws it out. If it's not in a region, it's a table pipe and it's included in the list of good pipes. Finally, it splits the table pipes elements by the table delimiters and returns the elements.



```py
PIPE_NONE = 0
PIPE_LEFT = 1
PIPE_RIGHT = 2


class TableProcessor(BlockProcessor):
    """ Process Tables. """

    RE_CODE_PIPES = re.compile(r'(?:(\\\\)|(\\`+)|(`+)|(\\\|)|(\|))')
    RE_END_BORDER = re.compile(r'(?<!\\)(?:\\\\)*\|$')

    def __init__(self, parser: blockparser.BlockParser, config: dict[str, Any]):
        self.border: bool | int = False
        self.separator: Sequence[str] = ''
        self.config = config

        super().__init__(parser)

    def test(self, parent: etree.Element, block: str) -> bool:
        """
        Ensure first two rows (column header and separator row) are valid table rows.

        Keep border check and separator row do avoid repeating the work.
        """
        is_table = False
        rows = [row.strip(' ') for row in block.split('\n')]
        if len(rows) > 1:
            header0 = rows[0]
            self.border = PIPE_NONE
            if header0.startswith('|'):
                self.border |= PIPE_LEFT
            if self.RE_END_BORDER.search(header0) is not None:
                self.border |= PIPE_RIGHT
            row = self._split_row(header0)
            row0_len = len(row)
            is_table = row0_len > 1

            # Each row in a single column table needs at least one pipe.
            if not is_table and row0_len == 1 and self.border:
                for index in range(1, len(rows)):
                    is_table = rows[index].startswith('|')
                    if not is_table:
                        is_table = self.RE_END_BORDER.search(rows[index]) is not None
                    if not is_table:
                        break

            if is_table:
                row = self._split_row(rows[1])
                is_table = (len(row) == row0_len) and set(''.join(row)) <= set('|:- ')
                if is_table:
                    self.separator = row

        return is_table

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        """ Parse a table block and build table. """
        block = blocks.pop(0).split('\n')
        header = block[0].strip(' ')
        rows = [] if len(block) < 3 else block[2:]

        # Get alignment of columns
        align: list[str | None] = []
        for c in self.separator:
            c = c.strip(' ')
            if c.startswith(':') and c.endswith(':'):
                align.append('center')
            elif c.startswith(':'):
                align.append('left')
            elif c.endswith(':'):
                align.append('right')
            else:
                align.append(None)

        # Build table
        table = etree.SubElement(parent, 'table')
        thead = etree.SubElement(table, 'thead')
        self._build_row(header, thead, align)
        tbody = etree.SubElement(table, 'tbody')
        if len(rows) == 0:
            # Handle empty table
            self._build_empty_row(tbody, align)
        else:
            for row in rows:
                self._build_row(row.strip(' '), tbody, align)

    def _build_empty_row(self, parent: etree.Element, align: Sequence[str | None]) -> None:
        """Build an empty row."""
        tr = etree.SubElement(parent, 'tr')
        count = len(align)
        while count:
            etree.SubElement(tr, 'td')
            count -= 1

    def _build_row(self, row: str, parent: etree.Element, align: Sequence[str | None]) -> None:
        """ Given a row of text, build table cells. """
        tr = etree.SubElement(parent, 'tr')
        tag = 'td'
        if parent.tag == 'thead':
            tag = 'th'
        cells = self._split_row(row)
        # We use align here rather than cells to ensure every row
        # contains the same number of columns.
        for i, a in enumerate(align):
            c = etree.SubElement(tr, tag)
            try:
                c.text = cells[i].strip(' ')
            except IndexError:  # pragma: no cover
                c.text = ""
            if a:
                if self.config['use_align_attribute']:
                    c.set('align', a)
                else:
                    c.set('style', f'text-align: {a};')

    def _split_row(self, row: str) -> list[str]:
        """ split a row of text into list of cells. """
        if self.border:
            if row.startswith('|'):
                row = row[1:]
            row = self.RE_END_BORDER.sub('', row)
        return self._split(row)

    def _split(self, row: str) -> list[str]:
        """ split a row of text with some code into a list of cells. """
        elements = []
        pipes = []
        tics = []
        tic_points = []
        tic_region = []
        good_pipes = []

        # Parse row
        # Throw out \\, and \|
        for m in self.RE_CODE_PIPES.finditer(row):
            # Store ` data (len, start_pos, end_pos)
            if m.group(2):
                # \`+
                # Store length of each tic group: subtract \
                tics.append(len(m.group(2)) - 1)
                # Store start of group, end of group, and escape length
                tic_points.append((m.start(2), m.end(2) - 1, 1))
            elif m.group(3):
                # `+
                # Store length of each tic group
                tics.append(len(m.group(3)))
                # Store start of group, end of group, and escape length
                tic_points.append((m.start(3), m.end(3) - 1, 0))
            # Store pipe location
            elif m.group(5):
                pipes.append(m.start(5))

        # Pair up tics according to size if possible
        # Subtract the escape length *only* from the opening.
        # Walk through tic list and see if tic has a close.
        # Store the tic region (start of region, end of region).
        pos = 0
        tic_len = len(tics)
        while pos < tic_len:
            try:
                tic_size = tics[pos] - tic_points[pos][2]
                if tic_size == 0:
                    raise ValueError
                index = tics[pos + 1:].index(tic_size) + 1
                tic_region.append((tic_points[pos][0], tic_points[pos + index][1]))
                pos += index + 1
            except ValueError:
                pos += 1

        # Resolve pipes.  Check if they are within a tic pair region.
        # Walk through pipes comparing them to each region.
        #     - If pipe position is less that a region, it isn't in a region
        #     - If it is within a region, we don't want it, so throw it out
        #     - If we didn't throw it out, it must be a table pipe
        for pipe in pipes:
            throw_out = False
            for region in tic_region:
                if pipe < region[0]:
                    # Pipe is not in a region
                    break
                elif region[0] <= pipe <= region[1]:
                    # Pipe is within a code region.  Throw it out.
                    throw_out = True
                    break
            if not throw_out:
                good_pipes.append(pipe)

        # Split row according to table delimiters.
        pos = 0
        for pipe in good_pipes:
            elements.append(row[pos:pipe])
            pos = pipe + 1
        elements.append(row[pos:])
        return elements


```

这段代码定义了一个名为 "TableExtension" 的类，继承自 "Extension" 类。这个类的实现中包含了一个名为 "**init**" 的方法，该方法接受一个或多个参数并将其存储在内部状态中。

在 "**init**" 方法中，首先调用父类的 "**init**" 方法，然后设置本类的配置选项。在这个配置选项中，使用 "use\_align\_attribute" 参数设置是否使用垂直居中样式，如果设置为 "True"，则将 "|" 符号替换为垂直居中样式。

接着，定义了一个名为 "extendMarkdown" 的方法，该方法接受一个 "md" 对象并将其传递给 "BlockParser"。在 "extendMarkdown" 方法中，首先检查 md 对象中是否存在 "使用 align 属性" 的选项，如果不存在，则将 "|" 符号添加到 "md.ESCAPED_CHARS" 列表中。然后，创建一个名为 "processor" 的 "TableProcessor" 实例，使用 md.parser 和 self.config 设置来获取正确的配置选项。最后，使用 md.parser 的 blockprocessors 注册表长程序器，使得 "TableProcessor" 可以被 "md.parser" 对象的块处理程序调用。


```py
class TableExtension(Extension):
    """ Add tables to Markdown. """

    def __init__(self, **kwargs):
        self.config = {
            'use_align_attribute': [False, 'True to use align attribute instead of style.'],
        }
        """ Default configuration options. """

        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """ Add an instance of `TableProcessor` to `BlockParser`. """
        if '|' not in md.ESCAPED_CHARS:
            md.ESCAPED_CHARS.append('|')
        processor = TableProcessor(md.parser, self.getConfigs())
        md.parser.blockprocessors.register(processor, 'table', 75)


```

这段代码定义了一个名为`makeExtension`的函数，它接受一个或多个参数`**kwargs`，并将其传递给名为`TableExtension`的类。

具体来说，这段代码的作用是创建一个名为`TableExtension`的类，该类具有一个名为`**kwargs`的参数接收函数。这个接收函数可以接受任何数量的参数，并将它们存储在一个字典中。然后，这个接收函数返回一个`TableExtension`类的实例，将这个实例的`**kwargs`参数设置为接收到的参数对象。

`**kwargs`在这里的作用是接收函数的参数，并将其存储到一个`**kwargs`字典中。这个字典可以被用作参数传递给`TableExtension`类的构造函数，以初始化该类的实例。

由于`makeExtension`函数创建了一个新的类实例，并且返回了该实例，因此我们可以将其视为一个装饰器。当我们在需要使用这个装饰器时，只需在需要的地方导入它，而无需创建一个新的类实例。


```py
def makeExtension(**kwargs):  # pragma: no cover
    return TableExtension(**kwargs)

```

# `/markdown/markdown/extensions/toc.py`

这段代码是一个 Python-Markdown 的自定义表格 of contents 扩展，旨在添加表格 of contents 支持。该功能使得用户可以轻松地创建和渲染表格 of contents，类似于在 HTML 中使用 `<table>` 标签来显示表格数据。

具体来说，这段代码实现了以下操作：

1. 引入必要的 Python-Markdown 和扩展的类和函数。
2. 创建一个名为 `toc` 的自定义模板类，用于生成表格 of contents。
3. 在模板类中定义了一个名为 `add_toc` 的方法，该方法接受一个列表作为参数，其中包含要包含在表格 of contents 中的项目。
4. 在 `add_toc` 方法中，使用 `print` 函数将自定义的表格 of contents 信息添加到控制台。
5. 在 `print` 函数中，使用 `<table>` 标签来显示自定义表格 of contents。
6. 在 `<table>` 标签中，使用 `<tr>` 标签来定义表格行，使用 `<th>` 标签来定义表格单元格中的标题，使用 `<td>` 标签来定义表格单元格中的内容。
7. 在代码文件中包含一个 `#` 注释，用于描述该自定义表格 of contents 扩展的来源和版权信息。

总之，这段代码是一个用于在 Python-Markdown 中添加表格 of contents 支持的功能，使得用户可以轻松地创建和渲染表格 of contents。


```py
# Table of Contents Extension for Python-Markdown
# ===============================================

# See https://Python-Markdown.github.io/extensions/toc
# for documentation.

# Original code Copyright 2008 [Jack Miller](https://codezen.org/)

# All changes Copyright 2008-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
Add table of contents support to Python-Markdown.

```

这段代码是一个Python文档的扩展，定义了一个名为“extensions”的类，包含一个名为“Treeprocessor”的接口和一个名为“UnescapeTreeprocessor”的子接口。这个类提供了一个遍历树形数据的功能，并对数据进行了一些处理，最后将结果返回。

详细说明：

1. 首先定义了一个名为“extensions”的类，这是一个扩展，继承自Python的“typing”模块中的“MutableSet”类型，表示将内部数据饱和技术。

2. 在类中引入了三个外部库的引用：Python的“documentation”库、Python的“treeprocessors”库、Python的“util”库，分别用于获取文档信息、处理树形数据和处理字符串。

3. 在类中定义了一个名为“Treeprocessor”的接口，这个接口提供了一个遍历树形数据的方法“extension_api”，但没有实现这个接口，因为在后面还需要对其进行自定义处理。

4. 在类中定义了一个名为“UnescapeTreeprocessor”的子接口，这个接口实现了“Treeprocessor”接口，并对其进行了一些自定义处理，最后返回处理后的结果。

5. 在“extensions”类中定义了一个名为“code_escape”的函数，这个函数的作用是使用Python的“code_escape”库将编码的字符串解码成原始字符串，并返回原始字符串。

6. 在“extensions”类中定义了一个名为“parseBoolValue”的函数，这个函数的作用是使用Python的“parse_bool”库将字符串解析成布尔值，并返回布尔值。

7. 在“extensions”类中定义了一个名为“AMP_SUBSTITUTE”的函数，这个函数的作用是使用给定的模板字符串替换字符串中的所有“AMP_SUBSTITUTE”标签，并返回替换后的字符串。

8. 在“extensions”类中定义了一个名为“html_placeholder_re”的函数，这个函数的作用是使用Python的“html.parser”库将一个字符串解析为HTML，并返回解析后的字符串。

9. 在“extensions”类中定义了一个名为“AtomicString”的类，这个类提供了一个原子操作，可以保证多个操作同时进行的字符串是互不干扰的，并可以保证所有的操作都返回同一个内部数据类型，同时使用了Python的“typing”库中的“Literal”类型。

10. 在“extensions”类中定义了一个名为“Treeprocessor”的接口，这个接口提供了一个遍历树形数据的方法“extension_api”，这个方法使用了一个名为“UnescapeTreeprocessor”的子接口，并在内部数据中使用了“AtomicString”的类。

11. 在“extensions”类中定义了一个名为“UnescapeTreeprocessor”的接口，这个接口实现了“Treeprocessor”接口，并在内部数据中使用了“code_escape”函数对字符串进行编码和解码。

12. 在“extensions”类中定义了一个名为“code_escape”的函数，这个函数的作用是使用Python的“code_escape”库将编码的字符串解码成原始字符串，并返回原始字符串。

13. 在“extensions”类中定义了一个名为“parseBoolValue”的函数，这个函数的作用是使用Python的“parse_bool”库将字符串解析成布尔值，并返回布尔值。


```py
See the [documentation](https://Python-Markdown.github.io/extensions/toc)
for details.
"""

from __future__ import annotations

from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import code_escape, parseBoolValue, AMP_SUBSTITUTE, HTML_PLACEHOLDER_RE, AtomicString
from ..treeprocessors import UnescapeTreeprocessor
import re
import html
import unicodedata
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any, Iterator, MutableSet

```

这段代码定义了一个名为 `slugify` 的函数，用于将给定的字符串（通过 `value` 参数）进行 URL 友好化。函数可以分为以下几个部分：

1. 从 `markdown` 模块中导入了一个名为 `Markdown` 的函数，这可能是用于将 Markdown 对象转换为普通 Python 函数的模块。

2. 定义了一个名为 `TYPE_CHECKING` 的条件判断，这意味着这个函数不会在编译时进行类型检查。这个条件判断在函数内部使用，用于在运行时确保函数可以正确地使用 `from markdown import Markdown`。

3. 定义了 `slugify` 函数的实现，这个函数的主要作用是将给定的字符串进行 URL 友好化。函数的实现包括以下几个步骤：

a. 通过 `unicodedata.normalize('NFKD', value)` 将字符串转换为扩展的标记符号（例如，`é丝绸之路` 转换为 `滇巫本初毛造icこ厭我心里应该 appreciation classicalunity低俗恶心`）。

b. 通过 `unicodedata.normalize('NFKD', value)` 将字符串转换为 ASCII 编码，通过使用 `ignore` 参数来忽略掉那些属于 "挤压ISO-2002 ASCII替换范围" 的字符（例如，`é丝绸之路` 中的 `é` 字符）。

c. 通过 `re.sub(r'[^\w\s-]', '', value)` 删除字符串中的所有非字母数字字符（例如，空格、下划线、破折号等），并将结果设置为新的字符串。

d. 通过 `re.sub(r'[{}\s]+'.format(separator), separator, value)` 在字符串中查找所有属于 `{}` 模式的子串（例如，`{}妊娠全过程 Suction在整个手术过程中她只打算经历两次的分娩和手术过程` 中的 `{}` 模式），并使用指定的分隔符将它们与周围的字符连接起来。这样，将不会对已经使用分隔符连接起来的子串进行操作，从而保证函数的兼容性。

e. 通过 `strip()` 和 `lower()` 函数来删除字符串中的所有空格和将字符串转换为小写。

f. 通过 `re.sub()` 函数将所有的异质子串（例如，`é丝绸之路` 中的 `é` 字符）转换为相同的字符（例如，`zlutiny`）。

4. 定义了一个名为 `slugify_unicode` 的函数，这个函数与 `slugify` 函数的实现几乎完全相同，但是使用 `unicode=True` 选项时，会忽略掉所有非 ASCII 字符。

总的来说，这些函数将帮助开发者将 Markdown 对象转换为兼容的、统一的接口，以进行更丰富的 URL 友好化处理。


```py
if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """ Slugify a string, to make it URL friendly. """
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[{}\s]+'.format(separator), separator, value)


def slugify_unicode(value: str, separator: str) -> str:
    """ Slugify a string, to make it URL friendly while preserving Unicode characters. """
    return slugify(value, separator, unicode=True)


```

这段代码是一个Python函数，名为`unique`。它接受两个参数：`id`是输入的字符串，`ids`是一个可变集合，包含输入的字符串。

函数的作用是确保`id`这个字符串只属于输入的`ids`集合中的一个元素，如果已经存在，则输出一个新的字符串，否则将`_1`、`_2`等数字添加到`ids`中，并将`id`添加到`ids`中。

具体实现可以分为以下几个步骤：

1. 使用Python标准库中的`re`模块，定义一个正则表达式`IDCOUNT_RE`，其含义是：匹配ID号，形式为`^(.*)_(0-9)+$`，其中`^`匹配字符串的开始位置，`(.*)`匹配字符串中的任意字符，`_`匹配一个下划线，`(0-9)+`匹配1到9任意个数字。

2. 定义一个函数`unique(id: str, ids: MutableSet[str]) -> str`，其中`id`是输入的字符串参数，`ids`是一个可变集合参数。

3. 在函数体中，使用Python标准库中的`id`变量，将其与`ids`中的元素进行比较。如果是`re.match`的结果，则说明`id`匹配了这个正则表达式，可以提取到一个包含`id`和数字的元组`m`中；否则，直接将`id`和数字1组成一个新的字符串，并将`_1`、`_2`等数字添加到`ids`中。

4. 最后，返回`id`这个新的字符串。

由于正则表达式中没有重复匹配的情况，因此该函数可以确保每个`id`都只属于`ids`中的一个元素，如果已经存在的话，则输出一个新的字符串。


```py
IDCOUNT_RE = re.compile(r'^(.*)_([0-9]+)$')


def unique(id: str, ids: MutableSet[str]) -> str:
    """ Ensure id is unique in set of ids. Append '_1', '_2'... if not """
    while id in ids or not id:
        m = IDCOUNT_RE.match(id)
        if m:
            id = '%s_%d' % (m.group(1), int(m.group(2))+1)
        else:
            id = '%s_%d' % (id, 1)
    ids.add(id)
    return id


```

这段代码的作用是提取一个元素（el）的标题（title）名称，然后将其返回。如果元素无法获取标题，则返回一个空字符串。

具体来说，代码首先定义了一个名为`get_name`的函数，它接受一个`etree.Element`对象（即一个XML元素对象）作为参数。这个函数通过遍历`el`的文本内容，提取出所有`<title>`标签，并将它们转换为`<html>`标签和`<title>`标签之间的内容。如果提取出的内容是一个`<title>`标签，那么它将直接返回这个标题名称。否则，它将返回一个空字符串。

接下来，定义了一个名为`stashedHTML2text`的函数，它接受一个`str`类型的参数（即一段HTML文本），一个`Markdown`类型的参数（即一个Markdown解析器对象），和一个`strip_entities`参数（即一个布尔值，表示是否去除文本中的实体）。这个函数的作用是将从`md`解析器中提取的HTML文本，返回原始的、经过转换的文本，并且可以将其与`strip_entities`参数一起设置，以去除文本中的实体。

`stashedHTML2text`函数的核心部分是一个名为`_html_sub`的函数，它接受一个`re.Match`对象作为参数，并返回一个经过转换的HTML文本。这个函数的作用是在保留原始HTML文本的同时，将其中的标签、属性等经过解析和转义，使得它能够被`stashedHTML2text`函数接受。


```py
def get_name(el: etree.Element) -> str:
    """Get title name."""

    text = []
    for c in el.itertext():
        if isinstance(c, AtomicString):
            text.append(html.unescape(c))
        else:
            text.append(c)
    return ''.join(text).strip()


def stashedHTML2text(text: str, md: Markdown, strip_entities: bool = True) -> str:
    """ Extract raw HTML from stash, reduce to plain text and swap with placeholder. """
    def _html_sub(m: re.Match[str]) -> str:
        """ Substitute raw html with plain text. """
        try:
            raw = md.htmlStash.rawHtmlBlocks[int(m.group(1))]
        except (IndexError, TypeError):  # pragma: no cover
            return m.group(0)
        # Strip out tags and/or entities - leaving text
        res = re.sub(r'(<[^>]+>)', '', raw)
        if strip_entities:
            res = re.sub(r'(&[\#a-zA-Z0-9]+;)', '', res)
        return res

    return HTML_PLACEHOLDER_RE.sub(_html_sub, text)


```

This function takes a list of entries and returns a new list with the same entries, but organized differently. The entries are processed in depth first, meaning that the deepest level is processed first.

The function has two main functionalities:

1. If the list is empty, it initializes everything by processing the first entry and returns a new list with the same structure.
2. If the list is not empty, it iterates through the entries and builds a new list with the same structure.

The new list is constructed by populating it with the entries that have not yet been processed. The processed entries are stored in the new list, and the parents of the entries are stored in a list.

The new list is constructed recursively, with the last level being constructed last. This is done to ensure that the list is constructed correctly. The level of each entry is determined by the value of the `level` attribute of the entry. If the `level` value is less than the `level` of the last processed entry, the last level is reduced.

If the `level` value is greater than the `level` of the last processed entry, the last entry is added as a child to the parent, which is stored in the new list. The level of the last entry is set to the same as the level of the last processed entry.

If the `level` value is the same, the current entry is added as a child to the parent, which is stored in the new list. The new list is then populated with the children of the current parent, if they have not yet been processed.


```py
def unescape(text: str) -> str:
    """ Unescape escaped text. """
    c = UnescapeTreeprocessor()
    return c.unescape(text)


def nest_toc_tokens(toc_list):
    """Given an unsorted list with errors and skips, return a nested one.

        [{'level': 1}, {'level': 2}]
        =>
        [{'level': 1, 'children': [{'level': 2, 'children': []}]}]

    A wrong list is also converted:

        [{'level': 2}, {'level': 1}]
        =>
        [{'level': 2, 'children': []}, {'level': 1, 'children': []}]
    """

    ordered_list = []
    if len(toc_list):
        # Initialize everything by processing the first entry
        last = toc_list.pop(0)
        last['children'] = []
        levels = [last['level']]
        ordered_list.append(last)
        parents = []

        # Walk the rest nesting the entries properly
        while toc_list:
            t = toc_list.pop(0)
            current_level = t['level']
            t['children'] = []

            # Reduce depth if current level < last item's level
            if current_level < levels[-1]:
                # Pop last level since we know we are less than it
                levels.pop()

                # Pop parents and levels we are less than or equal to
                to_pop = 0
                for p in reversed(parents):
                    if current_level <= p['level']:
                        to_pop += 1
                    else:  # pragma: no cover
                        break
                if to_pop:
                    levels = levels[:-to_pop]
                    parents = parents[:-to_pop]

                # Note current level as last
                levels.append(current_level)

            # Level is the same, so append to
            # the current parent (if available)
            if current_level == levels[-1]:
                (parents[-1]['children'] if parents
                 else ordered_list).append(t)

            # Current level is > last item's level,
            # So make last item a parent and append current as child
            else:
                last['children'].append(t)
                parents.append(last)
                levels.append(current_level)
            last = t

    return ordered_list


```

This is a JavaScript class that appears to be for creating tables of contents (TOC) for Markdown documents. It uses the Python `docx` library for reading the Markdown and generates the TOC using JavaScript.

Here is a high-level overview of the class:

* The `match` method reads the TOC from the Markdown element and adds it to the `toc_tokens` list.
* The `set_level` method sets the text level of the TOC to the level specified by the `level` attribute of the TOC token.
* The `get_name` method retrieves the name of the TOC.
* The `append_tokens` method appends the TOC tokens (TOC elements with `level` attribute) to the `toc_tokens` list.
* The `build_toc_div` method creates a div element with the TOC tokens and adds it to the document.
* The `attach_toc_to_div` method attaches the TOC to the div element and serializes the TOC tokens.
* The `build_toc_markup` method builds the markup for the TOC using the TOC tokens.
* The `serialize_toc` method serializes the TOC tokens.
* The `add_ marks` method adds the specified marks (e.g., `<script>` tags) to the TOC.
* The `add_permalink` method adds a link to the TOC.
* The `use_anchors` property determines whether to use the `<a>` element for links to the TOC.
* The `add_script` method adds a script to the TOC.


```py
class TocTreeprocessor(Treeprocessor):
    """ Step through document and build TOC. """

    def __init__(self, md: Markdown, config: dict[str, Any]):
        super().__init__(md)

        self.marker: str = config["marker"]
        self.title: str = config["title"]
        self.base_level = int(config["baselevel"]) - 1
        self.slugify = config["slugify"]
        self.sep = config["separator"]
        self.toc_class = config["toc_class"]
        self.title_class: str = config["title_class"]
        self.use_anchors: bool = parseBoolValue(config["anchorlink"])
        self.anchorlink_class: str = config["anchorlink_class"]
        self.use_permalinks = parseBoolValue(config["permalink"], False)
        if self.use_permalinks is None:
            self.use_permalinks = config["permalink"]
        self.permalink_class: str = config["permalink_class"]
        self.permalink_title: str = config["permalink_title"]
        self.permalink_leading: bool | None = parseBoolValue(config["permalink_leading"], False)
        self.header_rgx = re.compile("[Hh][123456]")
        if isinstance(config["toc_depth"], str) and '-' in config["toc_depth"]:
            self.toc_top, self.toc_bottom = [int(x) for x in config["toc_depth"].split('-')]
        else:
            self.toc_top = 1
            self.toc_bottom = int(config["toc_depth"])

    def iterparent(self, node: etree.Element) -> Iterator[tuple[etree.Element, etree.Element]]:
        """ Iterator wrapper to get allowed parent and child all at once. """

        # We do not allow the marker inside a header as that
        # would causes an endless loop of placing a new TOC
        # inside previously generated TOC.
        for child in node:
            if not self.header_rgx.match(child.tag) and child.tag not in ['pre', 'code']:
                yield node, child
                yield from self.iterparent(child)

    def replace_marker(self, root: etree.Element, elem: etree.Element) -> None:
        """ Replace marker with elem. """
        for (p, c) in self.iterparent(root):
            text = ''.join(c.itertext()).strip()
            if not text:
                continue

            # To keep the output from screwing up the
            # validation by putting a `<div>` inside of a `<p>`
            # we actually replace the `<p>` in its entirety.

            # The `<p>` element may contain more than a single text content
            # (`nl2br` can introduce a `<br>`). In this situation, `c.text` returns
            # the very first content, ignore children contents or tail content.
            # `len(c) == 0` is here to ensure there is only text in the `<p>`.
            if c.text and c.text.strip() == self.marker and len(c) == 0:
                for i in range(len(p)):
                    if p[i] == c:
                        p[i] = elem
                        break

    def set_level(self, elem: etree.Element) -> None:
        """ Adjust header level according to base level. """
        level = int(elem.tag[-1]) + self.base_level
        if level > 6:
            level = 6
        elem.tag = 'h%d' % level

    def add_anchor(self, c: etree.Element, elem_id: str) -> None:
        anchor = etree.Element("a")
        anchor.text = c.text
        anchor.attrib["href"] = "#" + elem_id
        anchor.attrib["class"] = self.anchorlink_class
        c.text = ""
        for elem in c:
            anchor.append(elem)
        while len(c):
            c.remove(c[0])
        c.append(anchor)

    def add_permalink(self, c: etree.Element, elem_id: str) -> None:
        permalink = etree.Element("a")
        permalink.text = ("%spara;" % AMP_SUBSTITUTE
                          if self.use_permalinks is True
                          else self.use_permalinks)
        permalink.attrib["href"] = "#" + elem_id
        permalink.attrib["class"] = self.permalink_class
        if self.permalink_title:
            permalink.attrib["title"] = self.permalink_title
        if self.permalink_leading:
            permalink.tail = c.text
            c.text = ""
            c.insert(0, permalink)
        else:
            c.append(permalink)

    def build_toc_div(self, toc_list: list) -> etree.Element:
        """ Return a string div given a toc list. """
        div = etree.Element("div")
        div.attrib["class"] = self.toc_class

        # Add title to the div
        if self.title:
            header = etree.SubElement(div, "span")
            if self.title_class:
                header.attrib["class"] = self.title_class
            header.text = self.title

        def build_etree_ul(toc_list: list, parent: etree.Element) -> etree.Element:
            ul = etree.SubElement(parent, "ul")
            for item in toc_list:
                # List item link, to be inserted into the toc div
                li = etree.SubElement(ul, "li")
                link = etree.SubElement(li, "a")
                link.text = item.get('name', '')
                link.attrib["href"] = '#' + item.get('id', '')
                if item['children']:
                    build_etree_ul(item['children'], li)
            return ul

        build_etree_ul(toc_list, div)

        if 'prettify' in self.md.treeprocessors:
            self.md.treeprocessors['prettify'].run(div)

        return div

    def run(self, doc: etree.Element) -> None:
        # Get a list of id attributes
        used_ids = set()
        for el in doc.iter():
            if "id" in el.attrib:
                used_ids.add(el.attrib["id"])

        toc_tokens = []
        for el in doc.iter():
            if isinstance(el.tag, str) and self.header_rgx.match(el.tag):
                self.set_level(el)
                text = get_name(el)

                # Do not override pre-existing ids
                if "id" not in el.attrib:
                    innertext = unescape(stashedHTML2text(text, self.md))
                    el.attrib["id"] = unique(self.slugify(innertext, self.sep), used_ids)

                if int(el.tag[-1]) >= self.toc_top and int(el.tag[-1]) <= self.toc_bottom:
                    toc_tokens.append({
                        'level': int(el.tag[-1]),
                        'id': el.attrib["id"],
                        'name': unescape(stashedHTML2text(
                            code_escape(el.attrib.get('data-toc-label', text)),
                            self.md, strip_entities=False
                        ))
                    })

                # Remove the data-toc-label attribute as it is no longer needed
                if 'data-toc-label' in el.attrib:
                    del el.attrib['data-toc-label']

                if self.use_anchors:
                    self.add_anchor(el, el.attrib["id"])
                if self.use_permalinks not in [False, None]:
                    self.add_permalink(el, el.attrib["id"])

        toc_tokens = nest_toc_tokens(toc_tokens)
        div = self.build_toc_div(toc_tokens)
        if self.marker:
            self.replace_marker(doc, div)

        # serialize and attach to markdown instance.
        toc = self.md.serializer(div)
        for pp in self.md.postprocessors:
            toc = pp.run(toc)
        self.md.toc_tokens = toc_tokens
        self.md.toc = toc


```

This appears to be a Python class for configuring a Markdown toc (Table of Contents) and is meant to be used with the Sphinx-style peppers to generate TOC marks. Here are the main configuration options of the class:

* `permalink_class`: A class to decide which class to use for the link text in a Sphinx-style permeal link.
* `permalink_title`: The title of the link in a Sphinx-style permeal link.
* `permalink_leading`: A boolean indicating whether the link should open at the start of the header or the end.
* `baselevel`: The base level for the headers, where 1 is the default level.
* `slugify`: A function for generating anchors based on header text.
* `separator`: A word separator for the slugify function.
* `toc_depth`: The minimum depth of the TOC tree to be included in the generated TOC. A single integer (b) defines the bottom section level (<h1>..<hb>) only. A string consisting of two digits separated by a hyphen defines the top (<tt>..<tt>) and the bottom (..<hb>) section levels.

To use this class, you would first need to install the `sphinx` package, which is not included in the default Python installation. Then, you could use the `Sphinx` class to create a `Sphinx` object and pass it to the `extendMarkdown` method of the `TocConfig` class to enable the TOC extension in your Markdown.


```py
class TocExtension(Extension):

    TreeProcessorClass = TocTreeprocessor

    def __init__(self, **kwargs):
        self.config = {
            'marker': [
                '[TOC]',
                'Text to find and replace with Table of Contents. Set to an empty string to disable. '
                'Default: `[TOC]`.'
            ],
            'title': [
                '', 'Title to insert into TOC `<div>`. Default: an empty string.'
            ],
            'title_class': [
                'toctitle', 'CSS class used for the title. Default: `toctitle`.'
            ],
            'toc_class': [
                'toc', 'CSS class(es) used for the link. Default: `toclink`.'
            ],
            'anchorlink': [
                False, 'True if header should be a self link. Default: `False`.'
            ],
            'anchorlink_class': [
                'toclink', 'CSS class(es) used for the link. Defaults: `toclink`.'
            ],
            'permalink': [
                0, 'True or link text if a Sphinx-style permalink should be added. Default: `False`.'
            ],
            'permalink_class': [
                'headerlink', 'CSS class(es) used for the link. Default: `headerlink`.'
            ],
            'permalink_title': [
                'Permanent link', 'Title attribute of the permalink. Default: `Permanent link`.'
            ],
            'permalink_leading': [
                False,
                'True if permalinks should be placed at start of the header, rather than end. Default: False.'
            ],
            'baselevel': ['1', 'Base level for headers. Default: `1`.'],
            'slugify': [
                slugify, 'Function to generate anchors based on header text. Default: `slugify`.'
            ],
            'separator': ['-', 'Word separator. Default: `-`.'],
            'toc_depth': [
                6,
                'Define the range of section levels to include in the Table of Contents. A single integer '
                '(b) defines the bottom section level (<h1>..<hb>) only. A string consisting of two digits '
                'separated by a hyphen in between (`2-5`) defines the top (t) and the bottom (b) (<ht>..<hb>). '
                'Default: `6` (bottom).'
            ],
        }
        """ Default configuration options. """

        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """ Add TOC tree processor to Markdown. """
        md.registerExtension(self)
        self.md = md
        self.reset()
        tocext = self.TreeProcessorClass(md, self.getConfigs())
        md.treeprocessors.register(tocext, 'toc', 5)

    def reset(self) -> None:
        self.md.toc = ''
        self.md.toc_tokens = []


```

这段代码定义了一个名为 `makeExtension` 的函数，它接收一个或多个参数 `**kwargs`，并将其传递给 `TocExtension` 函数，最终返回一个扩展名为 `**kwargs` 的对象。

具体来说，这段代码使用的是 Python 的 `**` 修饰符，这使得函数可以接受任意数量的参数。在函数内部，接收的参数通过 `**kwargs` 的形式传递给了 `TocExtension` 函数，这种传递方式类似于参数列表，但是可以同时传递多个参数。

函数的实现还使用了一个 pragma 指导原则，即 `# pragma: no cover`，这告诉我们在该函数内部不会对 `**kwargs` 中的参数进行访问，因此不会输出任何错误信息。

最后，函数返回了一个扩展名为 `**kwargs` 的对象，这个对象将传递给 `TocExtension` 函数，用于扩展 `TocExtension` 的功能。


```py
def makeExtension(**kwargs):  # pragma: no cover
    return TocExtension(**kwargs)

```

# `/markdown/markdown/extensions/wikilinks.py`

这是一段Python代码，定义了一个名为`WikiLinksExtension`的扩展，可以用来将WikiLinks格式的链接转换为相对路径。

具体来说，这段代码实现了一个`WikiLinks`类，用于处理WikiLinks链接。在这个类中，通过使用Python的`markdown`包，解析WikiLinks链接，并将其转换为内部链接。

此外，定义了一个`config`属性，用于存储配置信息，例如WikiLinks服务器的地址等。

最后，定义了一个`__init__`方法，用于初始化WikiLinks扩展，并输出一条文档信息。

值得注意的是，这段代码使用了Python的第三方库`markdown`和`http`。`markdown`库可以解析Markdown文本，将其转换为HTML；`http`库用于发送HTTP请求。


```py
# WikiLinks Extension for Python-Markdown
# ======================================

# Converts [[WikiLinks]] to relative links.

# See https://Python-Markdown.github.io/extensions/wikilinks
# for documentation.

# Original code Copyright [Waylan Limberg](http://achinghead.com/).

# All changes Copyright The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
```

这段代码是一个Python extension，名为"Linkify"，作用是将一个包含链接的XML文档转换为具有相对链接的XML文档。其实现基于Python的[[WikiLinks]]库，通过使用inks.py库进行转换。具体来说，这段代码做了以下几件事情：

1. 导入需要用到的库：WikiLinks库、extension.py库、inlinepatterns.py库、xml.etree.ElementTree库、re库以及typing库。

2. 定义了输入参数和输出参数，输入参数是一个包含至少一个有序无限元组的字符串列表，输出参数是一个包含至少一个有序无限元组的字符串列表。

3. 导入并实例化extension.py库中的Linkify类，然后调用其main函数进行转换。

4. 通过调用 etree.parse.sort_values 和 etree.model.function.create_Element 将XML文档中的元数据解析为树状结构，并创建一个空的Linkify对象。

5. 遍历输入文档中的每一个元组，提取出其中包含链接的元组，并使用Linkify对象将链接转换为具有相对链接的元组。

6. 将转换后的元组添加到输出结果列表中，最终得到一个完整的文档。

7. 由于Linkify库中没有定义导出，因此需要手动导入导出的部分。


```py
Converts `[[WikiLinks]]` to relative links.

See the [documentation](https://Python-Markdown.github.io/extensions/wikilinks)
for details.
"""

from __future__ import annotations

from . import Extension
from ..inlinepatterns import InlineProcessor
import xml.etree.ElementTree as etree
import re
from typing import Any


```

这段代码定义了一个名为 build_url 的函数，它接受三个参数 label、base 和 end，并返回一个 URL。函数使用正则表达式从 label 中提取出标签名、在标签名前添加下划线、去除标签名中的空格，并在 end 前添加下划线，最后将所有部分拼接起来。

下一个定义了一个名为 WikiLinkExtension 的类，这个类继承自 Extensions 类，提供了添加 Markdown inline 处理器的功能。这个类的构造函数接受一个字典，其中包含两个键：'base\_url' 和 'end\_url'，它们分别指代要包含在 URL 中的前缀和后缀。这个类的实例化过程包括在 Self 初始化过程中加载配置项，然后在扩展 Markdown 的过程中注册一个名为 WIKILINK\_RE 的正则表达式，以及将注册好的正则表达式实例化并注册到 md.inlinePatterns 列表中。


```py
def build_url(label: str, base: str, end: str) -> str:
    """ Build a URL from the label, a base, and an end. """
    clean_label = re.sub(r'([ ]+_)|(_[ ]+)|([ ]+)', '_', label)
    return '{}{}{}'.format(base, clean_label, end)


class WikiLinkExtension(Extension):
    """ Add inline processor to Markdown. """

    def __init__(self, **kwargs):
        self.config = {
            'base_url': ['/', 'String to append to beginning or URL.'],
            'end_url': ['/', 'String to append to end of URL.'],
            'html_class': ['wikilink', 'CSS hook. Leave blank for none.'],
            'build_url': [build_url, 'Callable formats URL from label.'],
        }
        """ Default configuration options. """
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        self.md = md

        # append to end of inline patterns
        WIKILINK_RE = r'\[\[([\w0-9_ -]+)\]\]'
        wikilinkPattern = WikiLinksInlineProcessor(WIKILINK_RE, self.getConfigs())
        wikilinkPattern.md = md
        md.inlinePatterns.register(wikilinkPattern, 'wikilink', 75)


```

这段代码是一个名为 `WikiLinksInlineProcessor` 的类，它是 `InlineProcessor` 的子类。这个类的作用是构建链接，它从 `wikilink` 构造链接。

在 `__init__` 方法中，首先调用父类的 `__init__` 方法，然后初始化自己的 `__config` 参数对。

在 `handleMatch` 方法中，如果匹配到了正则表达式，就代表找到了一个匹配项。在这个方法中，首先根据 `__config` 参数中的规则，解析出元数据（如 base_url、end_url 和 html_class）。如果没有解析出元数据，就直接返回根元素。然后，创建一个链接元素 `a`，设置其文本为解析出的元数据中的第一个，设置其 href 为 `build_url` 方法的结果，如果解析出了 html_class，就设置其 class 属性。最后，将 `a` 元素添加到结果元素中，并根据需要调整匹配结果。

在 `_getMeta` 方法中，根据 `__config` 参数中的规则，返回元数据中的 base_url、end_url 和 html_class，如果没有这些配置项，就返回元数据中的第一个。


```py
class WikiLinksInlineProcessor(InlineProcessor):
    """ Build link from `wikilink`. """

    def __init__(self, pattern: str, config: dict[str, Any]):
        super().__init__(pattern)
        self.config = config

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | str, int, int]:
        if m.group(1).strip():
            base_url, end_url, html_class = self._getMeta()
            label = m.group(1).strip()
            url = self.config['build_url'](label, base_url, end_url)
            a = etree.Element('a')
            a.text = label
            a.set('href', url)
            if html_class:
                a.set('class', html_class)
        else:
            a = ''
        return a, m.start(0), m.end(0)

    def _getMeta(self) -> tuple[str, str, str]:
        """ Return meta data or `config` data. """
        base_url = self.config['base_url']
        end_url = self.config['end_url']
        html_class = self.config['html_class']
        if hasattr(self.md, 'Meta'):
            if 'wiki_base_url' in self.md.Meta:
                base_url = self.md.Meta['wiki_base_url'][0]
            if 'wiki_end_url' in self.md.Meta:
                end_url = self.md.Meta['wiki_end_url'][0]
            if 'wiki_html_class' in self.md.Meta:
                html_class = self.md.Meta['wiki_html_class'][0]
        return base_url, end_url, html_class


```

这段代码定义了一个名为 `makeExtension` 的函数，它接受一个或多个参数 `**kwargs`，并将它们传递给 `WikiLinkExtension` 类。

具体来说，这个函数的作用是创建一个类 `WikiLinkExtension`，如果这个类已经定义好的话，则直接返回；否则，根据传递的参数创建一个新的类实例，并将其返回。

在这个函数中，`**kwargs` 表示一个或多个参数，可以是任何类型（包括命名参数和可变参数）。这个函数将这个或多个参数传递给 `WikiLinkExtension` 类，根据这些参数来创建一个新的类实例。


```py
def makeExtension(**kwargs):  # pragma: no cover
    return WikiLinkExtension(**kwargs)

```

# `/markdown/markdown/extensions/__init__.py`

这是一个 Python 实现的 John Gruber 的 Markdown 代码。它定义了一个 Python 类，该类实现了 Markdown 的语法，可以轻松地将 Markdown 语法转换为 Python 代码。这个项目最初由 Manfred Stienstra 开始，后被 Yuri Takhteyev、Waylan Limberg 和 Dmitry Shachnev 维护和开发。

该代码定义了一个 `Markdown` 类，这个类包含了一些方法来处理 Markdown 语法，例如：

- `parse()`：解析一段 Markdown 文本并返回一个 Python 对象，这个对象包含了 Markdown 语法元素，例如标题、段落、列表等。
- `code()`：将一个 Markdown 段落转换为 Python 代码。
- `render()`：将一个 Python 列表或元组渲染为 Markdown 格式。

这个项目的文档在 [https://python-markdown.github.io/](https://python-markdown.github.io/)，GitHub 地址为 [https://github.com/Python-Markdown/markdown/](https://github.com/Python-Markdown/markdown/)，PyPI 地址为 [https://pypi.org/project/Markdown/](https://pypi.org/project/Markdown/)。


```py
# Python Markdown

# A Python implementation of John Gruber's Markdown.

# Documentation: https://python-markdown.github.io/
# GitHub: https://github.com/Python-Markdown/markdown/
# PyPI: https://pypi.org/project/Markdown/

# Started by Manfred Stienstra (http://www.dwerg.net/).
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
# Currently maintained by Waylan Limberg (https://github.com/waylan),
# Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

# Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
# Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
```

这段代码定义了一个`Extension`类，用于每个Markdown扩展。每个扩展类必须继承自`Extension`类并重写`extendMarkdown`方法。这个类提供了一个统一的接口，让用户可以管理扩展的配置选项，并将它们附加到`Markdown`实例上。

具体来说，这个类接受一个`Extension`实例，允许用户设置扩展的选项。然后，这个类定义了一个`__init__`方法，用于初始化扩展类实例。在`__init__`方法中，用户可以设置扩展的选项，并将其附加到`Markdown`实例上。

这个类的目的是让扩展的开发者可以更轻松地管理扩展的配置选项，并将其附加到`Markdown`实例上。这对于使用Markdown的开发者来说非常有用，因为他们可以使用`Markdown`提供的API来创建和配置扩展。


```py
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
Markdown accepts an [`Extension`][markdown.extensions.Extension] instance for each extension. Therefore, each extension
must to define a class that extends [`Extension`][markdown.extensions.Extension] and over-rides the
[`extendMarkdown`][markdown.extensions.Extension.extendMarkdown] method. Within this class one can manage configuration
options for their extension and attach the various processors and patterns which make up an extension to the
[`Markdown`][markdown.Markdown] instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping
```

It looks like `MarkdownExtension` is a base class for extensions that can be added to a `Markdown` instance. It has a `setConfig` method for setting configuration options, and a `setConfigs` method for setting multiple configuration options in a single call. The `extendMarkdown` method is overridden in the `MarkdownExtension` class, but it is recommended that any subclasses do not define this method as it is already implemented in the base class.


```py
from ..util import parseBoolValue

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


class Extension:
    """ Base class for extensions to subclass. """

    config: Mapping[str, list] = {}
    """
    Default configuration for an extension.

    This attribute is to be defined in a subclass and must be of the following format:

    ``` python
    config = {
        'key': ['value', 'description']
    }
    ```py

    Note that [`setConfig`][markdown.extensions.Extension.setConfig] will raise a [`KeyError`][]
    if a default is not set for each option.
    """

    def __init__(self, **kwargs):
        """ Initiate Extension and set up configs. """
        self.setConfigs(kwargs)

    def getConfig(self, key: str, default: Any = '') -> Any:
        """
        Return a single configuration option value.

        Arguments:
            key: The configuration option name.
            default: Default value to return if key is not set.

        Returns:
            Value of stored configuration option.
        """
        if key in self.config:
            return self.config[key][0]
        else:
            return default

    def getConfigs(self) -> dict[str, Any]:
        """
        Return all configuration options.

        Returns:
            All configuration options.
        """
        return {key: self.getConfig(key) for key in self.config.keys()}

    def getConfigInfo(self) -> list[tuple[str, str]]:
        """
        Return descriptions of all configuration options.

        Returns:
            All descriptions of configuration options.
        """
        return [(key, self.config[key][1]) for key in self.config.keys()]

    def setConfig(self, key: str, value: Any) -> None:
        """
        Set a configuration option.

        If the corresponding default value set in [`config`][markdown.extensions.Extension.config]
        is a `bool` value or `None`, then `value` is passed through
        [`parseBoolValue`][markdown.util.parseBoolValue] before being stored.

        Arguments:
            key: Name of configuration option to set.
            value: Value to assign to option.

        Raises:
            KeyError: If `key` is not known.
        """
        if isinstance(self.config[key][0], bool):
            value = parseBoolValue(value)
        if self.config[key][0] is None:
            value = parseBoolValue(value, preserve_none=True)
        self.config[key][0] = value

    def setConfigs(self, items: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> None:
        """
        Loop through a collection of configuration options, passing each to
        [`setConfig`][markdown.extensions.Extension.setConfig].

        Arguments:
            items: Collection of configuration options.

        Raises:
            KeyError: for any unknown key.
        """
        if hasattr(items, 'items'):
            # it's a dict
            items = items.items()
        for key, value in items:
            self.setConfig(key, value)

    def extendMarkdown(self, md: Markdown) -> None:
        """
        Add the various processors and patterns to the Markdown Instance.

        This method must be overridden by every extension.

        Arguments:
            md: The Markdown instance.

        """
        raise NotImplementedError(
            'Extension "%s.%s" must define an "extendMarkdown"'
            'method.' % (self.__class__.__module__, self.__class__.__name__)
        )

```

# `/markdown/scripts/gen_ref_nav.py`

这段代码的主要作用是生成代码文档。它包括以下几个步骤：

1. 导入需要用到的库：textwrap、yaml、pathlib、mkdocs_gen_files。
2. 导入 mkdocs_gen_files 的 Nav 类。
3. 创建一个空 Nav 对象。
4. 设置每个模块的选项，包括 "markdown" 选项。
5. 设置代码文档的基础路径。
6. 生成文档。

具体来说，这段代码会在当前目录下创建一个名为 "../content度量衡表.yml" 的文档。这个文档包括了一些头部信息（如文档的名称、描述、作者等），以及一些定义好的函数、类和模块。通过这些定义，mkdocs_gen_files 库就能够根据需要在文档中插入这些内容，从而生成一个完整的文档。


```py
"""Generate the code reference pages and navigation."""

import textwrap
import yaml
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

per_module_options = {
    "markdown": {"summary": {"attributes": True, "functions": True, "classes": True}}
}

base_path = Path(__file__).resolve().parent.parent

```

这段代码的作用是定义了一个名为 "modules" 的列表，其中包含了一系列Python模块的路径。这些模块都是 "markdown" 目录下的子目录中命名的。

具体来说，这些模块包含了：

- `base_path.joinpath("markdown", "__init__.py")`:Python标准库中的 `__init__.py` 文件，用于定义模块的导入路径。
- `base_path.joinpath("markdown", "preprocessors.py")`:Python标准库中的 `preprocessors.py` 文件，可能是一个自定义的预处理器。
- `base_path.joinpath("markdown", "blockparser.py")`:Python标准库中的 `blockparser.py` 文件，可能是一个自定义的解析器。
- `base_path.joinpath("markdown", "blockprocessors.py")`:Python标准库中的 `blockprocessors.py` 文件，可能是一个自定义的处理器。
- `base_path.joinpath("markdown", "treeprocessors.py")`:Python标准库中的 `treeprocessors.py` 文件，可能是一个自定义的遍历器。
- `base_path.joinpath("markdown", "inlinepatterns.py")`:Python标准库中的 `inlinepatterns.py` 文件，可能是一个自定义的正则表达式。
- `base_path.joinpath("markdown", "postprocessors.py")`:Python标准库中的 `postprocessors.py` 文件，可能是一个自定义的处理器。
- `base_path.joinpath("markdown", "serializers.py")`:Python标准库中的 `serializers.py` 文件，可能是一个自定义的序列化器。
- `base_path.joinpath("markdown", "util.py")`:Python标准库中的 `util.py` 文件，可能是一个自定义的工具函数或类。
- `base_path.joinpath("markdown", "htmlparser.py")`:Python标准库中的 `htmlparser.py` 文件，可能是一个自定义的解析器。
- `base_path.joinpath("markdown", "test_tools.py")`:Python标准库中的 `test_tools.py` 文件，可能是一个自定义的测试工具类。

这些模块可能是在 "markdown" 目录下使用的，因此可能包含了与 Markdown 格式相关的处理和转换。


```py
modules = [
    base_path.joinpath("markdown", "__init__.py"),
    base_path.joinpath("markdown", "preprocessors.py"),
    base_path.joinpath("markdown", "blockparser.py"),
    base_path.joinpath("markdown", "blockprocessors.py"),
    base_path.joinpath("markdown", "treeprocessors.py"),
    base_path.joinpath("markdown", "inlinepatterns.py"),
    base_path.joinpath("markdown", "postprocessors.py"),
    base_path.joinpath("markdown", "serializers.py"),
    base_path.joinpath("markdown", "util.py"),
    base_path.joinpath("markdown", "htmlparser.py"),
    base_path.joinpath("markdown", "test_tools.py"),
    *sorted(base_path.joinpath("markdown", "extensions").rglob("*.py")),
]

```

这段代码的作用是生成一个带有导航栏的Markdown文档。它将给定的源代码文件夹(modules)中的所有文件的路径存储在一个元组(ModulePath)中，并对这些路径进行处理，以生成一个带有导航栏的Markdown文档。

具体来说，代码首先定义了一系列变量，然后使用Path.relative_to()方法将源文件路径相对于一个基础路径(未定义)调整。接下来，使用Path.with_suffix()方法为每个路径添加文件后缀。然后，定义了一个full_doc_path变量，用于存储一个指向Markdown文件的路径。

接着，代码使用tuple()函数提取出ModulePath对象的元组元素，并将这些元素传递给一个列表。然后，代码检查列表最后一个元素是否为".__init__"或以"_"开头的名称，如果是，则将列表的后半部分作为新文件名，并将文档路径更改为新文件名。如果元组元素不以这些名称之一结束，则代码将继续执行。

接下来，代码使用一个列表循环遍历ModulePath对象的每个元素。对于每个元素，代码使用Python的with语句打开一个Markdown文件，并将其写入到full_doc_path中。如果full_doc_path已经存在，则代码会使用Python的os模块获取当前工作目录并将其作为新文件夹。

最后，代码使用Python的yaml库读取per_module_options字典中的所有选项，并将其中的键值对存储在一个新的字典中。然后，代码使用textwrap库中的indent函数对每个选项的字符串进行格式化，并将它们写入到Markdown文件中。如果选项字典中没有指定的选项，则代码将在文件中写入一个空字符串。


```py
for src_path in modules:
    path = src_path.relative_to(base_path)
    module_path = path.with_suffix("")
    doc_path = path.with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    nav_parts = [f"<code>{part}</code>" for part in parts]
    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")
        if ident in per_module_options:
            yaml_options = yaml.dump({"options": per_module_options[ident]})
            fd.write(f"\n{textwrap.indent(yaml_options, prefix='    ')}")
        elif ident.startswith("markdown.extensions."):
            yaml_options = yaml.dump({"options": {"inherited_members": False}})
            fd.write(f"\n{textwrap.indent(yaml_options, prefix='    ')}")

    mkdocs_gen_files.set_edit_path(full_doc_path, ".." / path)

```

这段代码使用了Python的with语句和mkdocs_gen_files库来创建一个名为"reference/SUMMARY.md"的文件并输出其中包含的nav对象。

首先，代码使用with语句创建了一个文件对象nav_file，然后使用nav.build_literate_nav()方法创建一个nav对象并将其写入nav_file中。其中，markdown文件中的\n换行符被替换为制表符，从而可以方便地在多个nav对象之间添加换行符。

最后，通过with语句的break语句可以提前终止with语句中的内容，如果没有break语句，则会一直运行with语句中的代码，直到文件被正确关闭。


```py
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

```

# `/markdown/scripts/griffe_extensions.py`

这段代码是一个Griffe扩展，主要作用是在类的定义中自动生成docs.txt格式的文档。具体来说，它定义了一个名为`Docstring`的类，以及一个名为`Extension`的类，它们可以用来生成类的文档。

这里使用了Python的类型提示(type hints)，用来提示函数参数和返回值的类型。同时，使用了`ast`模块来解析Python的语法定义，以便更好地理解文档的含义。

`Docstring`类有两个方法，`DocstringSectionAdmonition`和`DocstringSectionText`，可以用来定义和展示文档的特定部分。其中，`DocstringSectionAdmonition`用来在类定义中显示文档的概述，`DocstringSectionText`则用来在类定义中显示具体的文档内容。

`Extension`类有两个方法，`generate_documentation`和`generate_source_documentation`，可以用来生成指定类别的文档。其中，`generate_documentation`用来生成指定类别的文档，包括类定义以及其中的子类、函数、类成员等信息；而`generate_source_documentation`则用来生成指定类别的源代码文档。

最后，如果需要的话，可以通过调用`Docstring`类中的方法来设置或清除`Docstring`的作用，例如：


from grifference.extensions import Extension

class MyClass(Extension):
   @my_class_doc
   def my_method(self):
       ...



def some_function(self):
   ...



```py
"""Griffe extensions."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
import textwrap

from griffe import Docstring, Extension
from griffe.docstrings.dataclasses import DocstringSectionAdmonition, DocstringSectionText

if TYPE_CHECKING:
    from griffe import Class, Function, ObjectNode


```

这段代码定义了一个名为 `_deprecated` 的 deprecated decorator，它会将一个对象（如函数或类）的调用者（也就是使用该对象的函数或类）作为参数传递给内部函数。如果这个对象是一个函数，函数内部的 `callable_path` 将指向一个参数，该参数将引发一个警告。

如果一个使用该对象的函数或类在一个被 deprecated 的类中，将会引发一个错误消息。这段代码将会返回一个警告字符串，除非字符串是 `None`。否则，它会引发警告，该警告信息将作为 `ast. literal_eval` 中的参数传递给 `str` 函数，以返回一个字符串对象。

该代码还定义了一个名为 `DeprecatedExtension` 的类，该类继承自 `Extension` 类。该 `DeprecatedExtension` 类实现了两个方法：`_insert_message` 和 `on_class_instance` 和 `on_function_instance`。

`_insert_message` 方法接收两个参数：`obj` 是目标函数或类的对象，`message` 是要返回的警告消息。它将在函数或类被导入时，将警告消息添加到对象的文档字符串中。

`on_class_instance` 方法接收两个参数：`node` 是目标函数或类的 `ast.AST` 或 `ast.ObjectNode` 对象，`cls` 是类的实例。它将在类被导入时，将警告消息添加到类的文档字符串中。

`on_function_instance` 方法接收两个参数：`node` 是目标函数或类的 `ast.AST` 或 `ast.ObjectNode` 对象，`func` 是函数的实例。它将在函数被导入时，将警告消息添加到函数的文档字符串中。


```py
def _deprecated(obj: Class | Function) -> str | None:
    for decorator in obj.decorators:
        if decorator.callable_path == "markdown.util.deprecated":
            return ast.literal_eval(str(decorator.value.arguments[0]))
    return None


class DeprecatedExtension(Extension):
    """Griffe extension for `@markdown.util.deprecated` decorator support."""

    def _insert_message(self, obj: Function | Class, message: str) -> None:
        if not obj.docstring:
            obj.docstring = Docstring("", parent=obj)
        sections = obj.docstring.parsed
        sections.insert(0, DocstringSectionAdmonition(kind="warning", text=message, title="Deprecated"))

    def on_class_instance(self, node: ast.AST | ObjectNode, cls: Class) -> None:  # noqa: ARG002
        """Add section to docstrings of deprecated classes."""
        if message := _deprecated(cls):
            self._insert_message(cls, message)
            cls.labels.add("deprecated")

    def on_function_instance(self, node: ast.AST | ObjectNode, func: Function) -> None:  # noqa: ARG002
        """Add section to docstrings of deprecated functions."""
        if message := _deprecated(func):
            self._insert_message(func, message)
            func.labels.add("deprecated")


```

This is a Python function that generates a table of priorities for a given class hierarchy. It takes an object containing the class hierarchy and a code object as inputs.

The function first extracts the table header and then iterates over the methods in the hierarchy. For each method, it checks if it is a method registered by the `register_code` function and if it is called with a function argument. If both conditions are met, it extracts the arguments passed to the function and constructs the class name, priority, and table body.

The table is generated by joining the prioritized class instances with their names and priorities. The priority queue is implemented using the ` collections.deque` class to store the classes.

Finally, the table is added to the function's docstring, which is generated using the ` Docstring` class.


```py
class PriorityTableExtension(Extension):
    """ Griffe extension to insert a table of processor priority in specified functions. """

    def __init__(self, paths: list[str] | None = None) -> None:
        super().__init__()
        self.paths = paths

    def linked_obj(self, value: str, path: str) -> str:
        """ Wrap object name in reference link. """
        return f'[`{value}`][{path}.{value}]'

    def on_function_instance(self, node: ast.AST | ObjectNode, func: Function) -> None:  # noqa: ARG002
        """Add table to specified function docstrings."""
        if self.paths and func.path not in self.paths:
            return  # skip objects that were not selected

        # Table header
        data = [
            'Class Instance | Name | Priority',
            '-------------- | ---- | :------:'
        ]

        # Extract table body from source code of function.
        for obj in node.body:
            # Extract the arguments passed to `util.Registry.register`.
            if isinstance(obj, ast.Expr) and isinstance(obj.value, ast.Call) and obj.value.func.attr == 'register':
                _args = obj.value.args
                cls = self.linked_obj(_args[0].func.id, func.path.rsplit('.', 1)[0])
                name = _args[1].value
                priority = str(_args[2].value)
                if func.name == ('build_inlinepatterns'):
                    # Include Pattern: first arg passed to class
                    if isinstance(_args[0].args[0], ast.Constant):
                        # Pattern is a string
                        value = f'`"{_args[0].args[0].value}"`'
                    else:
                        # Pattern is a variable
                        value = self.linked_obj(_args[0].args[0].id, func.path.rsplit('.', 1)[0])
                    cls = f'{cls}({value})'
                data.append(f'{cls} | `{name}` | `{priority}`')

        table = '\n'.join(data)
        body = (
            f"Return a [`{func.returns.canonical_name}`][{func.returns.canonical_path}] instance which contains "
            "the following collection of classes with their assigned names and priorities.\n\n"
            f"{table}"
        )

        # Add to docstring.
        if not func.docstring:
            func.docstring = Docstring("", parent=func)
        sections = func.docstring.parsed
        sections.append(DocstringSectionText(body, title="Priority Table"))

```