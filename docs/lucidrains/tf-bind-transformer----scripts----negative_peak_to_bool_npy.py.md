# `.\lucidrains\tf-bind-transformer\scripts\negative_peak_to_bool_npy.py`

```py
#/usr/bin/python

# 导入必要的库
import polars as pl
import numpy as np
from pathlib import Path
import sys

# 从命令行参数中获取负峰文件路径和行数
NEGATIVE_PEAK_PATH = sys.argv[1]
NUMROWS = int(sys.argv[2])
ID_COLUMN = 'column_6'

# 读取以制表符分隔的无标题负峰文件
df = pl.read_csv(NEGATIVE_PEAK_PATH, sep = '\t', has_headers = False)

# 获取指定列的数据并转换为 NumPy 数组
np_array = df.get_column(ID_COLUMN).to_numpy()

# 创建一个布尔数组，用于标记需要保存的行
to_save = np.full((NUMROWS,), False)
to_save[np_array - 1] = True

# 获取文件路径的 stem 部分，并创建保存布尔数组的文件名
p = Path(NEGATIVE_PEAK_PATH)
filename = f'{p.stem}.bool'

# 将布尔数组保存为 NumPy 文件
np.save(filename, to_save)

# 打印保存文件的信息
print(f'{filename} saved')
```