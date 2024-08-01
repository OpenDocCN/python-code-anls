# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\9479-8a6b32582d10ba1f.js`

```py
# 导入所需的模块
import pandas as pd

# 定义一个函数，接收一个参数 filename
def process_data(filename):
    # 使用 pandas 读取 Excel 文件，并将数据存储在一个 DataFrame 中
    data = pd.read_excel(filename)
    
    # 选择并返回 DataFrame 的前五行数据作为示例
    return data.head()

# 调用 process_data 函数，传入文件名为 'data.xlsx'，并打印结果
print(process_data('data.xlsx'))
```