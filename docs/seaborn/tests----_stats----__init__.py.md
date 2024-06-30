# `D:\src\scipysrc\seaborn\tests\_stats\__init__.py`

```
# 导入pandas库，通常用于数据处理和分析
import pandas as pd

# 读取CSV文件，并将其解析为DataFrame对象，赋值给变量df
df = pd.read_csv('data.csv')

# 打印DataFrame的前5行数据，用于快速预览数据结构和内容
print(df.head())

# 计算DataFrame中列名为'column1'的平均值，存储结果到变量mean_value中
mean_value = df['column1'].mean()

# 将计算得到的平均值打印输出，展示分析结果
print('Mean of column1:', mean_value)
```