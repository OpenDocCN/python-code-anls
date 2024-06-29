# `D:\src\scipysrc\pandas\asv_bench\benchmarks\multiindex_object.py`

```
import string  # 导入字符串模块，用于生成字母序列

import numpy as np  # 导入NumPy库，用于数值计算

from pandas import (  # 从Pandas库中导入以下对象：
    NA,  # 缺失值标识符
    DataFrame,  # 数据框对象，用于操作二维数据
    Index,  # 索引对象，用于标记轴
    MultiIndex,  # 多级索引对象，用于多维数据标记
    RangeIndex,  # 范围索引对象，表示连续整数索引
    Series,  # 系列对象，用于操作一维数据
    array,  # 数组对象，用于创建NumPy数组
    date_range,  # 日期范围生成器
)


class GetLoc:
    def setup(self):
        self.mi_large = MultiIndex.from_product(
            [np.arange(1000), np.arange(20), list(string.ascii_letters)],
            names=["one", "two", "three"],
        )  # 创建一个包含大量数据的多级索引对象，包括三个级别：one, two, three

        self.mi_med = MultiIndex.from_product(
            [np.arange(1000), np.arange(10), list("A")],
            names=["one", "two", "three"]
        )  # 创建一个中等大小的多级索引对象，包括三个级别：one, two, three

        self.mi_small = MultiIndex.from_product(
            [np.arange(100), list("A"), list("A")],
            names=["one", "two", "three"]
        )  # 创建一个较小的多级索引对象，包括三个级别：one, two, three

    def time_large_get_loc(self):
        self.mi_large.get_loc((999, 19, "Z"))  # 获取大型多级索引对象中特定标签元组的位置

    def time_large_get_loc_warm(self):
        for _ in range(1000):
            self.mi_large.get_loc((999, 19, "Z"))  # 重复多次获取大型多级索引对象中特定标签元组的位置，用于预热

    def time_med_get_loc(self):
        self.mi_med.get_loc((999, 9, "A"))  # 获取中等大小多级索引对象中特定标签元组的位置

    def time_med_get_loc_warm(self):
        for _ in range(1000):
            self.mi_med.get_loc((999, 9, "A"))  # 重复多次获取中等大小多级索引对象中特定标签元组的位置，用于预热

    def time_string_get_loc(self):
        self.mi_small.get_loc((99, "A", "A"))  # 获取小型多级索引对象中特定标签元组的位置

    def time_small_get_loc_warm(self):
        for _ in range(1000):
            self.mi_small.get_loc((99, "A", "A"))  # 重复多次获取小型多级索引对象中特定标签元组的位置，用于预热


class GetLocs:
    def setup(self):
        self.mi_large = MultiIndex.from_product(
            [np.arange(1000), np.arange(20), list(string.ascii_letters)],
            names=["one", "two", "three"],
        )  # 创建一个包含大量数据的多级索引对象，包括三个级别：one, two, three

        self.mi_med = MultiIndex.from_product(
            [np.arange(1000), np.arange(10), list("A")],
            names=["one", "two", "three"]
        )  # 创建一个中等大小的多级索引对象，包括三个级别：one, two, three

        self.mi_small = MultiIndex.from_product(
            [np.arange(100), list("A"), list("A")],
            names=["one", "two", "three"]
        )  # 创建一个较小的多级索引对象，包括三个级别：one, two, three

    def time_large_get_locs(self):
        self.mi_large.get_locs([999, 19, "Z"])  # 获取大型多级索引对象中特定标签列表对应的位置列表

    def time_med_get_locs(self):
        self.mi_med.get_locs([999, 9, "A"])  # 获取中等大小多级索引对象中特定标签列表对应的位置列表

    def time_small_get_locs(self):
        self.mi_small.get_locs([99, "A", "A"])  # 获取小型多级索引对象中特定标签列表对应的位置列表


class Duplicates:
    def setup(self):
        size = 65536
        arrays = [np.random.randint(0, 8192, size), np.random.randint(0, 1024, size)]
        mask = np.random.rand(size) < 0.1
        self.mi_unused_levels = MultiIndex.from_arrays(arrays)
        self.mi_unused_levels = self.mi_unused_levels[mask]  # 创建一个多级索引对象，然后根据掩码筛选出使用的级别

    def time_remove_unused_levels(self):
        self.mi_unused_levels.remove_unused_levels()  # 移除多级索引对象中未使用的级别


class Integer:
    # 定义一个方法 `setup`，用于设置测试环境
    def setup(self):
        # 创建一个多级索引对象 `mi_int`，包含两个维度各 1000 个元素的笛卡尔积
        self.mi_int = MultiIndex.from_product(
            [np.arange(1000), np.arange(1000)], names=["one", "two"]
        )
        # 创建一个包含元组的 NumPy 数组 `obj_index`，元组中包含 object 类型的数据
        self.obj_index = np.array(
            [
                (0, 10),
                (0, 11),
                (0, 12),
                (0, 13),
                (0, 14),
                (0, 15),
                (0, 16),
                (0, 17),
                (0, 18),
                (0, 19),
            ],
            dtype=object,
        )
        # 使用元组列表创建另一个多级索引对象 `other_mi_many_mismatches`
        self.other_mi_many_mismatches = MultiIndex.from_tuples(
            [
                (-7, 41),
                (-2, 3),
                (-0.7, 5),
                (0, 0),
                (0, 1.5),
                (0, 340),
                (0, 1001),
                (1, -4),
                (1, 20),
                (1, 1040),
                (432, -5),
                (432, 17),
                (439, 165.5),
                (998, -4),
                (998, 24065),
                (999, 865.2),
                (999, 1000),
                (1045, -843),
            ]
        )

    # 定义一个方法 `time_get_indexer`，测试获取 `mi_int` 对象中 `obj_index` 的索引
    def time_get_indexer(self):
        self.mi_int.get_indexer(self.obj_index)

    # 定义一个方法 `time_get_indexer_and_backfill`，测试获取并回填 `mi_int` 对象中 `other_mi_many_mismatches` 的索引
    def time_get_indexer_and_backfill(self):
        self.mi_int.get_indexer(self.other_mi_many_mismatches, method="backfill")

    # 定义一个方法 `time_get_indexer_and_pad`，测试获取并填充 `mi_int` 对象中 `other_mi_many_mismatches` 的索引
    def time_get_indexer_and_pad(self):
        self.mi_int.get_indexer(self.other_mi_many_mismatches, method="pad")

    # 定义一个方法 `time_is_monotonic`，测试检查 `mi_int` 对象是否单调递增
    def time_is_monotonic(self):
        self.mi_int.is_monotonic_increasing
class Duplicated:
    # 初始化方法，设置数组长度和重复次数
    def setup(self):
        n, k = 200, 5000
        # 定义多层索引的级别
        levels = [
            np.arange(n),  # 第一级：长度为 n 的序列
            Index([f"i-{i}" for i in range(n)], dtype=object).values,  # 第二级：以字符串命名的索引
            1000 + np.arange(n),  # 第三级：从1000开始的长度为 n 的序列
        ]
        # 随机生成每个级别的代码
        codes = [np.random.choice(n, (k * n)) for lev in levels]
        # 创建多层索引对象
        self.mi = MultiIndex(levels=levels, codes=codes)

    # 计算重复条目的时间性能
    def time_duplicated(self):
        self.mi.duplicated()


class Sortlevel:
    # 初始化方法，设置数组长度和值范围
    def setup(self):
        n = 1182720
        low, high = -4096, 4096
        # 创建不同长度的重复数组
        arrs = [
            np.repeat(np.random.randint(low, high, (n // k)), k)
            for k in [11, 7, 5, 3, 1]
        ]
        # 通过数组创建多层索引，并随机排列
        self.mi_int = MultiIndex.from_arrays(arrs)[np.random.permutation(n)]

        # 创建另一个多层索引，按指定顺序排列
        a = np.repeat(np.arange(100), 1000)
        b = np.tile(np.arange(1000), 100)
        self.mi = MultiIndex.from_arrays([a, b])
        self.mi = self.mi.take(np.random.permutation(np.arange(100000)))

    # 对整数类型的索引按层级排序的时间性能
    def time_sortlevel_int64(self):
        self.mi_int.sortlevel()

    # 对第一级别索引排序的时间性能
    def time_sortlevel_zero(self):
        self.mi.sortlevel(0)

    # 对第二级别索引排序的时间性能
    def time_sortlevel_one(self):
        self.mi.sortlevel(1)


class SortValues:
    params = ["int64", "Int64"]
    param_names = ["dtype"]

    # 初始化方法，根据数据类型创建多层索引
    def setup(self, dtype):
        a = array(np.tile(np.arange(100), 1000), dtype=dtype)
        b = array(np.tile(np.arange(1000), 100), dtype=dtype)
        self.mi = MultiIndex.from_arrays([a, b])

    # 按值排序的时间性能
    def time_sort_values(self, dtype):
        self.mi.sort_values()


class Values:
    # 初始化方法，创建带有日期和整数级别的多层索引
    def setup_cache(self):
        level1 = range(1000)
        level2 = date_range(start="1/1/2012", periods=100)
        mi = MultiIndex.from_product([level1, level2])
        return mi

    # 复制并获取索引值的时间性能（全部复制）
    def time_datetime_level_values_copy(self, mi):
        mi.copy().values

    # 获取索引值的时间性能（前10个条目）
    def time_datetime_level_values_sliced(self, mi):
        mi[:10].values


class CategoricalLevel:
    # 初始化方法，创建包含大量数据的数据框，并将其中某些列转换为分类类型
    def setup(self):
        self.df = DataFrame(
            {
                "a": np.arange(1_000_000, dtype=np.int32),
                "b": np.arange(1_000_000, dtype=np.int64),
                "c": np.arange(1_000_000, dtype=float),
            }
        ).astype({"a": "category", "b": "category"})

    # 将某些列设置为索引并计算时间性能
    def time_categorical_level(self):
        self.df.set_index(["a", "b"])


class Equals:
    # 初始化方法，创建两个多层索引对象，一个是深拷贝，另一个是非对象类型的索引
    def setup(self):
        self.mi = MultiIndex.from_product(
            [
                date_range("2000-01-01", periods=1000),
                RangeIndex(1000),
            ]
        )
        self.mi_deepcopy = self.mi.copy(deep=True)
        self.idx_non_object = RangeIndex(1)

    # 比较两个多层索引对象（深拷贝）的时间性能
    def time_equals_deepcopy(self):
        self.mi.equals(self.mi_deepcopy)

    # 比较多层索引对象和非对象索引的时间性能
    def time_equals_non_object_index(self):
        self.mi.equals(self.idx_non_object)


class SetOperations:
    params = [
        ("monotonic", "non_monotonic"),
        ("datetime", "int", "string", "ea_int"),
        ("intersection", "union", "symmetric_difference"),
        (False, None),
    ]
    param_names = ["index_structure", "dtype", "method", "sort"]
    # 定义一个设置方法，用于初始化数据结构和参数
    def setup(self, index_structure, dtype, method, sort):
        # 定义一个常量 N，表示数据量级
        N = 10**5
        # 创建一个包含 1000 个整数的序列，作为第一级索引
        level1 = range(1000)

        # 创建一个日期范围，从"1/1/2000"开始，包含 N//1000 个时间点，作为第二级索引
        level2 = date_range(start="1/1/2000", periods=N // 1000)
        # 创建一个多级索引对象 dates_left，包含所有可能的 (level1, level2) 组合
        dates_left = MultiIndex.from_product([level1, level2])

        # 创建一个包含 0 到 N//1000-1 的整数序列，作为第二级索引
        level2 = range(N // 1000)
        # 创建一个多级索引对象 int_left，包含所有可能的 (level1, level2) 组合
        int_left = MultiIndex.from_product([level1, level2])

        # 创建一个包含以字符串形式命名的对象的索引，作为第二级索引
        level2 = Index([f"i-{i}" for i in range(N // 1000)], dtype=object).values
        # 创建一个多级索引对象 str_left，包含所有可能的 (level1, level2) 组合
        str_left = MultiIndex.from_product([level1, level2])

        # 创建一个包含整数序列的 Series 对象作为第二级索引
        level2 = range(N // 1000)
        # 创建一个多级索引对象 ea_int_left，包含所有可能的 (level1, level2) 组合
        ea_int_left = MultiIndex.from_product([level1, Series(level2, dtype="Int64")])

        # 将创建的各种多级索引对象存储在 data 字典中，用于后续操作
        data = {
            "datetime": dates_left,
            "int": int_left,
            "string": str_left,
            "ea_int": ea_int_left,
        }

        # 如果索引结构选择为非单调递增，反转每个键对应的索引值顺序
        if index_structure == "non_monotonic":
            data = {k: mi[::-1] for k, mi in data.items()}

        # 将每个键对应的索引值转换为字典形式，包括左右两侧的索引
        data = {k: {"left": mi, "right": mi[:-1]} for k, mi in data.items()}
        # 将指定类型 dtype 对应的左右索引存储在类实例的属性中
        self.left = data[dtype]["left"]
        self.right = data[dtype]["right"]

    # 定义一个时间操作方法，执行左侧索引对象上的指定方法，传入右侧索引对象作为参数
    def time_operation(self, index_structure, dtype, method, sort):
        getattr(self.left, method)(self.right, sort=sort)
# 定义一个名为 Difference 的类
class Difference:
    # 定义一个类级别的参数列表，包含四个元组，每个元组有四个字符串
    params = [
        ("datetime", "int", "string", "ea_int"),
    ]
    # 定义一个类级别的参数名列表，只包含一个字符串 "dtype"
    param_names = ["dtype"]

    # 定义一个初始化方法 setup，接收一个参数 dtype
    def setup(self, dtype):
        # 设定变量 N 为 20000
        N = 10**4 * 2
        # 创建一个范围为 0 到 999 的整数序列，赋值给 level1
        level1 = range(1000)

        # 创建一个日期范围，从 "1/1/2000" 开始，周期为 N//1000，赋值给 level2
        level2 = date_range(start="1/1/2000", periods=N // 1000)
        # 使用 MultiIndex.from_product 方法，将 level1 和 level2 组合成多级索引，赋值给 dates_left

        dates_left = MultiIndex.from_product([level1, level2])

        # 创建一个范围为 0 到 N//1000-1 的整数序列，赋值给 level2
        level2 = range(N // 1000)
        # 使用 MultiIndex.from_product 方法，将 level1 和 level2 组合成多级索引，赋值给 int_left
        int_left = MultiIndex.from_product([level1, level2])

        # 创建一个整数序列，范围为 0 到 N//1000-1，使用 "Int64" 数据类型，赋值给 level2
        level2 = Series(range(N // 1000), dtype="Int64")
        # 将第一个元素设为 NA（缺失值）
        level2[0] = NA
        # 使用 MultiIndex.from_product 方法，将 level1 和 level2 组合成多级索引，赋值给 ea_int_left
        ea_int_left = MultiIndex.from_product([level1, level2])

        # 创建一个对象数组，包含 N//1000 个字符串，格式为 "i-0" 到 "i-(N//1000-1)"
        level2 = Index([f"i-{i}" for i in range(N // 1000)], dtype=object).values
        # 使用 MultiIndex.from_product 方法，将 level1 和 level2 组合成多级索引，赋值给 str_left
        str_left = MultiIndex.from_product([level1, level2])

        # 创建一个字典，包含四个键值对，每个值都是一个字典，包含 "left" 和 "right" 两个键，值是对应的 MultiIndex 对象
        data = {
            "datetime": dates_left,
            "int": int_left,
            "ea_int": ea_int_left,
            "string": str_left,
        }
        # 将每个值的 "left" 和 "right" 对象截取前五个元素，更新原数据字典
        data = {k: {"left": mi, "right": mi[:5]} for k, mi in data.items()}
        # 根据输入的 dtype 选择更新后的数据字典中对应的 "left" 多级索引，赋值给 self.left
        self.left = data[dtype]["left"]
        # 根据输入的 dtype 选择更新后的数据字典中对应的 "right" 多级索引（截取前五个元素），赋值给 self.right
        self.right = data[dtype]["right"]

    # 定义一个方法 time_difference，接收一个参数 dtype
    def time_difference(self, dtype):
        # 计算 self.left 与 self.right 之间的差异
        self.left.difference(self.right)


# 定义一个名为 Unique 的类
class Unique:
    # 定义一个类级别的参数列表，包含两个元组，每个元组包含两个值
    params = [
        (("Int64", NA), ("int64", 0)),
    ]
    # 定义一个类级别的参数名列表，只包含一个字符串 "dtype_val"
    param_names = ["dtype_val"]

    # 定义一个初始化方法 setup，接收一个参数 dtype_val
    def setup(self, dtype_val):
        # 创建一个 Series，包含 [1, 2, dtype_val[1], dtype_val[1]] + range(1_000_000) 的序列，使用 dtype_val[0] 数据类型，赋值给 level
        level = Series(
            [1, 2, dtype_val[1], dtype_val[1]] + list(range(1_000_000)),
            dtype=dtype_val[0],
        )
        # 使用 MultiIndex.from_arrays 方法，将 level 两次作为数组元素，创建多级索引，赋值给 self.midx
        self.midx = MultiIndex.from_arrays([level, level])

        # 创建一个 Series，包含 [1, 2, dtype_val[1], dtype_val[1]] + range(500_000) * 2 的序列，使用 dtype_val[0] 数据类型，赋值给 level_dups
        level_dups = Series(
            [1, 2, dtype_val[1], dtype_val[1]] + list(range(500_000)) * 2,
            dtype=dtype_val[0],
        )
        # 使用 MultiIndex.from_arrays 方法，将 level_dups 两次作为数组元素，创建多级索引，赋值给 self.midx_dups
        self.midx_dups = MultiIndex.from_arrays([level_dups, level_dups])

    # 定义一个方法 time_unique，接收一个参数 dtype_val
    def time_unique(self, dtype_val):
        # 对 self.midx 执行唯一化操作
        self.midx.unique()

    # 定义一个方法 time_unique_dups，接收一个参数 dtype_val
    def time_unique_dups(self, dtype_val):
        # 对 self.midx_dups 执行唯一化操作
        self.midx_dups.unique()


# 定义一个名为 Isin 的类
class Isin:
    # 定义一个类级别的参数列表，包含三个字符串
    params = [
        ("string", "int", "datetime"),
    ]
    # 定义一个类级别的参数名列表，只包含一个字符串 "dtype"
    param_names = ["dtype"]

    # 定义一个初始化方法 setup，接收一个参数 dtype
    def setup(self, dtype):
        # 设定变量 N 为 100000
        N = 10**5
        # 创建一个范围为 0 到 999 的整数序列，赋值给 level1
        level1 = range(1000)

        # 创建一个日期范围，从 "1/1/2000" 开始，周期为 N//1000，赋值给 level2
        level2 = date_range(start="1/1/2000", periods=N // 1000)
        # 使用 MultiIndex.from_product 方法，将 level1 和 level2 组合成多级索引，赋值给 dates_midx
        dates_midx = MultiIndex.from_product([level1, level2])

        # 创建一个范围为 0 到 N//1000-1 的整数序列，赋值给 level2
        level2 = range(N // 1000)
        # 使用 MultiIndex.from_product 方法，将 level1 和 level2 组合成多级索引，赋值给 int_midx
        int_midx = MultiIndex.from_product([level1, level2])

        # 创建一个对象数组，包含 N//1000 个字符串，格式为 "i-0" 到 "i-(N//1000-1)"
        level2 = Index([f"i-{i}" for i in range(N // 1000)], dtype=object).values
        # 使用 MultiIndex.from_product 方法，将 level1 和 level2 组合成多级索引，赋值给 str_midx
        str_midx = MultiIndex.from_product([level1, level2])

        # 创建一个字典，包含三个键值对，每个值都是一个 MultiIndex 对象
        data = {
            "datetime": dates_midx,
            "int": int_midx,
            "string": str_midx,
        }
        # 根据输入的 dtype 选择更新原数据字典中对应的 MultiIndex 对象，赋值给 self.midx
        self.midx = data[dtype]
        # 分别从 self.midx 中截取前 100 个元素和后面的元素，赋值给 self.values_small 和 self.values_large
        self.values_small = self.m
    # 在设置方法中初始化一个常量 N，并赋值为 10 的 5 次方
    def setup(self):
        N = 10**5
        # 创建一个范围为 0 到 999 的整数序列，作为第一级索引
        level1 = range(1_000)

        # 生成一个日期范围，从 "1/1/2000" 开始，长度为 N//1000
        level2 = date_range(start="1/1/2000", periods=N // 1000)
        # 使用 level1 和 level2 创建一个多重索引对象 self.midx
        self.midx = MultiIndex.from_product([level1, level2])

        # 修改 level1 的范围为 1000 到 1999，作为第一级索引的备选值
        level1 = range(1_000, 2_000)
        # 使用修改后的 level1 和之前的 level2 创建另一个多重索引对象 self.midx_values
        self.midx_values = MultiIndex.from_product([level1, level2])

        # 将 level2 的起始日期改为 "1/1/2010"，保持长度为 N//1000
        level2 = date_range(start="1/1/2010", periods=N // 1000)
        # 使用新的 level1 和更新后的 level2 创建第三个多重索引对象 self.midx_values_different
        self.midx_values_different = MultiIndex.from_product([level1, level2])
        
        # 创建一个布尔类型的 NumPy 数组 self.mask，长度为 N//2，交替为 True 和 False
        self.mask = np.array([True, False] * (N // 2))

    # 在 time_putmask 方法中，根据 self.mask 对 self.midx 进行部分更新
    def time_putmask(self):
        self.midx.putmask(self.mask, self.midx_values)

    # 在 time_putmask_all_different 方法中，使用不同的 self.midx_values_different 更新 self.midx
    def time_putmask_all_different(self):
        self.midx.putmask(self.mask, self.midx_values_different)
# 定义一个类 Append，用于处理数据追加操作
class Append:
    # 类变量 params 存储支持的数据类型列表
    params = ["datetime64[ns]", "int64", "string"]
    # 类变量 param_names 存储参数名称列表
    param_names = ["dtype"]

    # 实例方法 setup 用于初始化数据结构
    def setup(self, dtype):
        # 设定常量 N1 和 N2
        N1 = 1000
        N2 = 500
        # 创建左侧和右侧 MultiIndex 的一级标签
        left_level1 = range(N1)
        right_level1 = range(N1, N1 + N1)

        # 根据参数 dtype 不同，选择不同的 level2 数据类型
        if dtype == "datetime64[ns]":
            level2 = date_range(start="2000-01-01", periods=N2)
        elif dtype == "int64":
            level2 = range(N2)
        elif dtype == "string":
            level2 = Index([f"i-{i}" for i in range(N2)], dtype=object)
        else:
            # 如果参数不在预定义的类型中，则抛出未实现错误
            raise NotImplementedError

        # 使用 MultiIndex.from_product 创建左侧和右侧的 MultiIndex 对象
        self.left = MultiIndex.from_product([left_level1, level2])
        self.right = MultiIndex.from_product([right_level1, level2])

    # 实例方法 time_append 用于执行追加操作
    def time_append(self, dtype):
        # 将右侧 MultiIndex 追加到左侧 MultiIndex
        self.left.append(self.right)


# 导入 .pandas_vb_common 模块中的 setup 函数，并跳过 isort 校验
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```