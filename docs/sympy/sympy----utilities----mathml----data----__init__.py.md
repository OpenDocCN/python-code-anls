# `D:\src\scipysrc\sympy\sympy\utilities\mathml\data\__init__.py`

```
# 导入操作 Excel 文件的库 openpyxl
import openpyxl

# 打开名为 'data.xlsx' 的 Excel 文件
wb = openpyxl.load_workbook('data.xlsx')

# 选择第一个工作表
sheet = wb.active

# 获取 A 列的所有单元格对象
column_a = sheet['A']

# 遍历 A 列的所有单元格对象，并打印它们的值
for cell in column_a:
    print(cell.value)
```