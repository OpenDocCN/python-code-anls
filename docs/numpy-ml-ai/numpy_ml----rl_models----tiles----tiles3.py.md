# `numpy-ml\numpy_ml\rl_models\tiles\tiles3.py`

```
"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed.
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)),
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile
coordinates are to be returned without being converted to indices).
"""

# 导入 math 模块中的 floor 函数和 itertools 模块中的 zip_longest 函数
from math import floor
from itertools import zip_longest

# 将 basehash 函数赋值给 basehash 变量
basehash = hash

# 定义 IHT 类
class IHT:
    "Structure to handle collisions"

    # 初始化方法，接受一个参数 sizeval
    def __init__(self, sizeval):
        # 设置实例变量 size 为传入的 sizeval
        self.size = sizeval
        # 初始化 overfullCount 为 0
        self.overfullCount = 0
        # 初始化 dictionary 为空字典
        self.dictionary = {}

    # 定义 __str__ 方法，用于对象打印时返回字符串
    def __str__(self):
        # 返回包含对象信息的字符串
        return (
            "Collision table:"
            + " size:"
            + str(self.size)
            + " overfullCount:"
            + str(self.overfullCount)
            + " dictionary:"
            + str(len(self.dictionary))
            + " items"
        )
    # 返回字典中键值对的数量
    def count(self):
        return len(self.dictionary)

    # 检查字典是否已满
    def fullp(self):
        return len(self.dictionary) >= self.size

    # 获取对象在字典中的索引，如果对象不存在且只读模式，则返回None
    def getindex(self, obj, readonly=False):
        # 获取字典引用
        d = self.dictionary
        # 如果对象在字典中存在，则返回其索引
        if obj in d:
            return d[obj]
        # 如果对象不存在且为只读模式，则返回None
        elif readonly:
            return None
        # 获取字典大小和当前键值对数量
        size = self.size
        count = self.count()
        # 如果键值对数量大于等于字典大小
        if count >= size:
            # 如果超出计数为0，则打印信息
            if self.overfullCount == 0:
                print("IHT full, starting to allow collisions")
            # 增加超出计数
            self.overfullCount += 1
            # 返回对象的哈希值对字典大小取模作为索引
            return basehash(obj) % self.size
        else:
            # 将对象添加到字典中，并返回其索引
            d[obj] = count
            return count
# 根据输入的坐标、哈希表或大小、只读标志，返回哈希索引
def hashcoords(coordinates, m, readonly=False):
    # 如果哈希表类型为IHT，则调用getindex方法获取索引
    if type(m) == IHT:
        return m.getindex(tuple(coordinates), readonly)
    # 如果哈希表类型为整数，则对坐标进行哈希运算并取模
    if type(m) == int:
        return basehash(tuple(coordinates)) % m
    # 如果哈希表为None，则直接返回坐标
    if m == None:
        return coordinates


# 返回num-tilings个瓦片索引，对应于浮点数和整数
def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    # 将浮点数乘以numtilings并向下取整
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // numtilings)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles


# 返回num-tilings个瓦片索引，对应于浮点数和整数，其中一些浮点数进行了包装
def tileswrap(ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints,
    wrapping some floats"""
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b % numtilings) // numtilings
            coords.append(c % width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles
```