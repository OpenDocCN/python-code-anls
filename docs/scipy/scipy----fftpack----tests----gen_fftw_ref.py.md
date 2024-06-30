# `D:\src\scipysrc\scipy\scipy\fftpack\tests\gen_fftw_ref.py`

```
# 导入所需模块
from subprocess import Popen, PIPE, STDOUT
# 导入 NumPy 库并重命名为 np
import numpy as np

# 定义一个包含多个大小的列表
SZ = [2, 3, 4, 8, 12, 15, 16, 17, 32, 64, 128, 256, 512, 1024]

# 根据给定的数据类型生成数据
def gen_data(dt):
    # 创建一个空字典用于存储数据数组
    arrays = {}

    # 根据不同的数据类型选择不同的程序路径
    if dt == np.float128:
        pg = './fftw_longdouble'
    elif dt == np.float64:
        pg = './fftw_double'
    elif dt == np.float32:
        pg = './fftw_single'
    else:
        # 如果数据类型不在已知范围内，抛出错误
        raise ValueError("unknown: %s" % dt)
    
    # 使用 FFTW 生成测试数据作为参考
    for type in [1, 2, 3, 4, 5, 6, 7, 8]:
        # 初始化字典嵌套结构
        arrays[type] = {}
        # 遍历给定的大小列表
        for sz in SZ:
            # 启动一个子进程来执行外部程序，捕获输出
            a = Popen([pg, str(type), str(sz)], stdout=PIPE, stderr=STDOUT)
            # 读取并解码子进程的输出，整理成一个列表
            st = [i.decode('ascii').strip() for i in a.stdout.readlines()]
            # 将输出转换为 NumPy 数组并存储到字典中
            arrays[type][sz] = np.fromstring(",".join(st), sep=',', dtype=dt)

    # 返回生成的数据字典
    return arrays

# 生成单精度浮点数据
data = gen_data(np.float32)
# 设定保存文件的文件名
filename = 'fftw_single_ref'
# 将数据保存到 npz 格式文件中
d = {'sizes': SZ}
for type in [1, 2, 3, 4]:
    for sz in SZ:
        d['dct_%d_%d' % (type, sz)] = data[type][sz]

d['sizes'] = SZ
for type in [5, 6, 7, 8]:
    for sz in SZ:
        d['dst_%d_%d' % (type-4, sz)] = data[type][sz]
np.savez(filename, **d)

# 生成双精度浮点数据
data = gen_data(np.float64)
filename = 'fftw_double_ref'
# 将数据保存到 npz 格式文件中
d = {'sizes': SZ}
for type in [1, 2, 3, 4]:
    for sz in SZ:
        d['dct_%d_%d' % (type, sz)] = data[type][sz]

d['sizes'] = SZ
for type in [5, 6, 7, 8]:
    for sz in SZ:
        d['dst_%d_%d' % (type-4, sz)] = data[type][sz]
np.savez(filename, **d)

# 生成长双精度浮点数据
data = gen_data(np.float128)
filename = 'fftw_longdouble_ref'
# 将数据保存到 npz 格式文件中
d = {'sizes': SZ}
for type in [1, 2, 3, 4]:
    for sz in SZ:
        d['dct_%d_%d' % (type, sz)] = data[type][sz]

d['sizes'] = SZ
for type in [5, 6, 7, 8]:
    for sz in SZ:
        d['dst_%d_%d' % (type-4, sz)] = data[type][sz]
np.savez(filename, **d)
```