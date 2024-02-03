# `.\PaddleOCR\ppstructure\table\tablepyxl\tablepyxl.py`

```py
# 导入必要的模块，确保代码能够同时在 Python 2 和 Python 3 中运行
from __future__ import absolute_import

# 导入 lxml 库中的 html 模块
from lxml import html
# 导入 openpyxl 库中的 Workbook 类
from openpyxl import Workbook
# 导入 openpyxl 库中的 get_column_letter 函数
from openpyxl.utils import get_column_letter
# 导入 premailer 库中的 Premailer 类
from premailer import Premailer
# 导入 tablepyxl 库中的 Table 类
from tablepyxl.style import Table

# 定义一个函数，将字符串转换为整数
def string_to_int(s):
    # 如果字符串只包含数字，则转换为整数并返回
    if s.isdigit():
        return int(s)
    # 否则返回 0
    return 0

# 定义一个函数，从 HTML 文档中获取所有的表格元素
def get_Tables(doc):
    # 从 HTML 文档中构建一个树形结构
    tree = html.fromstring(doc)
    # 获取树中所有的注释节点，并移除这些节点
    comments = tree.xpath('//comment()')
    for comment in comments:
        comment.drop_tag()
    # 返回树中所有表格元素构成的列表
    return [Table(table) for table in tree.xpath('//table')]

# 定义一个函数，将表格的每一行写入到工作表中
def write_rows(worksheet, elem, row, column=1):
    """
    Writes every tr child element of elem to a row in the worksheet
    returns the next row after all rows are written
    """
    # 导入 openpyxl 库中的 MergedCell 类
    from openpyxl.cell.cell import MergedCell

    # 记录初始列号
    initial_column = column
    # 遍历表格中的每一行
    for table_row in elem.rows:
        # 遍历表格行中的每一个单元格
        for table_cell in table_row.cells:
            # 获取当前单元格的行和列索引，初始化为指定行和列
            cell = worksheet.cell(row=row, column=column)
            # 处理合并单元格情况，如果当前单元格是合并单元格，则更新列索引
            while isinstance(cell, MergedCell):
                column += 1
                cell = worksheet.cell(row=row, column=column)

            # 获取单元格的列合并数和行合并数
            colspan = string_to_int(table_cell.element.get("colspan", "1"))
            rowspan = string_to_int(table_cell.element.get("rowspan", "1"))
            # 如果行合并数或列合并数大于1，则合并单元格
            if rowspan > 1 or colspan > 1:
                worksheet.merge_cells(start_row=row, start_column=column,
                                      end_row=row + rowspan - 1, end_column=column + colspan - 1)

            # 将单元格的值设置为表格单元格的值
            cell.value = table_cell.value
            # 格式化单元格
            table_cell.format(cell)
            # 获取单元格的最小宽度和最大宽度
            min_width = table_cell.get_dimension('min-width')
            max_width = table_cell.get_dimension('max-width')

            # 如果列合并数为1
            if colspan == 1:
                # 初始时，当第一次通过循环迭代时，所有单元格的宽度都为None。
                # 随着我们开始填充内容，单元格的初始宽度（可以通过以下方式检索：worksheet.column_dimensions[get_column_letter(column)].width）
                # 等于同一列中前一个单元格的宽度（即A2的宽度等于A1的宽度）
                width = max(worksheet.column_dimensions[get_column_letter(column)].width or 0, len(table_cell.value) + 2)
                # 如果存在最大宽度并且当前宽度大于最大宽度，则使用最大宽度
                if max_width and width > max_width:
                    width = max_width
                # 如果存在最小宽度并且当前宽度小于最小宽度，则使用最小宽度
                elif min_width and width < min_width:
                    width = min_width
                # 设置列的宽度
                worksheet.column_dimensions[get_column_letter(column)].width = width
            # 更新列索引
            column += colspan
        # 更新行索引
        row += 1
        # 重置列索引为初始列索引
        column = initial_column
    # 返回最终的行索引
    return row
# 将表格写入工作簿的新工作表中，工作表的标题将与表格属性名称相同
def table_to_sheet(table, wb):
    # 创建一个新的工作表，标题为表格元素的名称
    ws = wb.create_sheet(title=table.element.get('name'))
    # 在指定位置插入表格
    insert_table(table, ws, 1, 1)


# 将 HTML 文档转换为工作簿，为文档中的每个表格创建一个工作表，返回工作簿
def document_to_workbook(doc, wb=None, base_url=None):
    # 如果没有提供工作簿，则创建一个新的工作簿并移除默认的工作表
    if not wb:
        wb = Workbook()
        wb.remove(wb.active)

    # 将内联样式的文档转换为带有基础 URL 的 Premailer 对象，并保留类名
    inline_styles_doc = Premailer(doc, base_url=base_url, remove_classes=False).transform()
    # 获取文档中的所有表格
    tables = get_Tables(inline_styles_doc)

    # 遍历每个表格，并将其写入工作簿中的工作表
    for table in tables:
        table_to_sheet(table, wb)

    # 返回工作簿
    return wb


# 将 HTML 文档转换为 Excel 文件，为文档中的每个表格创建一个工作表，并将工作簿写入名为 filename 的文件中
def document_to_xl(doc, filename, base_url=None):
    # 将文档转换为工作簿
    wb = document_to_workbook(doc, base_url=base_url)
    # 将工作簿保存到指定的文件中
    wb.save(filename)


# 在指定的工作表、列和行位置插入表格
def insert_table(table, worksheet, column, row):
    # 如果表格有表头，则将表头写入工作表
    if table.head:
        row = write_rows(worksheet, table.head, row, column)
    # 如果表格有主体，则将主体写入工作表
    if table.body:
        row = write_rows(worksheet, table.body, row, column)


# 在 openpyxl 的 Cell 对象位置插入表格
def insert_table_at_cell(table, cell):
    # 获取 Cell 对象所在的工作表、列和行位置
    ws = cell.parent
    column, row = cell.column, cell.row
    # 在指定位置插入表格
    insert_table(table, ws, column, row)
```