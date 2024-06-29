# `.\numpy\doc\source\reference\random\performance.py`

```py
from timeit import repeat  # 导入计时器函数
import pandas as pd  # 导入 pandas 库

import numpy as np  # 导入 numpy 库
from numpy.random import MT19937, PCG64, PCG64DXSM, Philox, SFC64  # 从 numpy.random 导入多个随机数生成器类

PRNGS = [MT19937, PCG64, PCG64DXSM, Philox, SFC64]  # 定义随机数生成器类列表

funcs = {}  # 初始化空字典 funcs 用于存储函数调用字符串
integers = 'integers(0, 2**{bits},size=1000000, dtype="uint{bits}")'  # 定义整数生成函数模板
funcs['32-bit Unsigned Ints'] = integers.format(bits=32)  # 添加生成 32 位无符号整数的函数调用字符串
funcs['64-bit Unsigned Ints'] = integers.format(bits=64)  # 添加生成 64 位无符号整数的函数调用字符串
funcs['Uniforms'] = 'random(size=1000000)'  # 添加生成均匀分布随机数的函数调用字符串
funcs['Normals'] = 'standard_normal(size=1000000)'  # 添加生成标准正态分布随机数的函数调用字符串
funcs['Exponentials'] = 'standard_exponential(size=1000000)'  # 添加生成指数分布随机数的函数调用字符串
funcs['Gammas'] = 'standard_gamma(3.0,size=1000000)'  # 添加生成 Gamma 分布随机数的函数调用字符串
funcs['Binomials'] = 'binomial(9, .1, size=1000000)'  # 添加生成二项分布随机数的函数调用字符串
funcs['Laplaces'] = 'laplace(size=1000000)'  # 添加生成拉普拉斯分布随机数的函数调用字符串
funcs['Poissons'] = 'poisson(3.0, size=1000000)'  # 添加生成泊松分布随机数的函数调用字符串

setup = """
from numpy.random import {prng}, Generator
rg = Generator({prng}())
"""  # 定义设置字符串模板，用于设置随机数生成器

test = "rg.{func}"  # 定义测试字符串模板，用于测试随机数生成器性能

table = {}  # 初始化空字典 table 用于存储测试结果表格
for prng in PRNGS:  # 遍历每个随机数生成器类
    print(prng)  # 打印当前随机数生成器类的名称
    col = {}  # 初始化空字典 col 用于存储每种函数调用的最小执行时间
    for key in funcs:  # 遍历每种函数调用字符串
        t = repeat(test.format(func=funcs[key]),  # 测试当前函数调用的执行时间
                   setup.format(prng=prng().__class__.__name__),  # 设置随机数生成器
                   number=1, repeat=3)  # 设置测试参数
        col[key] = 1000 * min(t)  # 将最小执行时间（毫秒）添加到 col 字典中
    col = pd.Series(col)  # 将 col 字典转换为 pandas Series 对象
    table[prng().__class__.__name__] = col  # 将当前随机数生成器类的测试结果添加到 table 字典中

npfuncs = {}  # 初始化空字典 npfuncs 用于存储 numpy 函数调用字符串
npfuncs.update(funcs)  # 将 funcs 字典中的内容复制到 npfuncs 字典中
npfuncs['32-bit Unsigned Ints'] = 'randint(2**32,dtype="uint32",size=1000000)'  # 替换生成 32 位无符号整数的函数调用字符串
npfuncs['64-bit Unsigned Ints'] = 'randint(2**64,dtype="uint64",size=1000000)'  # 替换生成 64 位无符号整数的函数调用字符串
setup = """
from numpy.random import RandomState
rg = RandomState()
"""  # 更新设置字符串模板，用于设置随机数生成器

col = {}  # 初始化空字典 col 用于存储每种函数调用的最小执行时间
for key in npfuncs:  # 遍历每种函数调用字符串
    t = repeat(test.format(func=npfuncs[key]),  # 测试当前函数调用的执行时间
               setup.format(prng=prng().__class__.__name__),  # 设置随机数生成器
               number=1, repeat=3)  # 设置测试参数
    col[key] = 1000 * min(t)  # 将最小执行时间（毫秒）添加到 col 字典中
table['RandomState'] = pd.Series(col)  # 将 RandomState 类的测试结果添加到 table 字典中

columns = ['MT19937', 'PCG64', 'PCG64DXSM', 'Philox', 'SFC64', 'RandomState']  # 定义表格的列名列表
table = pd.DataFrame(table)  # 将 table 字典转换为 pandas DataFrame 对象
order = np.log(table).mean().sort_values().index  # 对表格进行排序，按照均值对数值排序
table = table.T  # 转置表格
table = table.reindex(columns)  # 根据指定列名重新索引表格
table = table.T  # 再次转置表格
table = table.reindex([k for k in funcs], axis=0)  # 根据函数调用字符串的键重新索引表格
print(table.to_csv(float_format='%0.1f'))  # 将表格输出为 CSV 格式，保留一位小数

rel = table.loc[:, ['RandomState']].values @ np.ones(  # 计算相对性能比率
    (1, table.shape[1])) / table  # 计算比率
rel.pop('RandomState')  # 删除 RandomState 列
rel = rel.T  # 转置比率表格
rel['Overall'] = np.exp(np.log(rel).mean(1))  # 计算平均对数值，再取指数，作为 Overall 列
rel *= 100  # 将比率转换为百分比
rel = np.round(rel)  # 四舍五入到整数
rel = rel.T  # 再次转置比率表格
print(rel.to_csv(float_format='%0d'))  # 将比率表格输出为 CSV 格式，保留整数

# Cross-platform table
rows = ['32-bit Unsigned Ints','64-bit Unsigned Ints','Uniforms','Normals','Exponentials']  # 定义跨平台表格的行名列表
xplat = rel.reindex(rows, axis=0)  # 根据指定行名重新索引比率表格
xplat = 100 * (xplat / xplat.MT19937.values[:,None])  # 计算相对于 MT19937 的百分比
overall = np.exp(np.log(xplat).mean(0))  # 计算平均对数值，再取指数，作为 Overall 行
xplat = xplat.T.copy()  # 转置并复制比率表格
xplat['Overall']=overall  # 添加 Overall 行
print(xplat.T.round(1))  # 将跨平台表格输出为 CSV 格式，保留一位小数
```