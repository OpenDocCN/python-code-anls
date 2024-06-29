# `.\numpy\numpy\_core\tests\test_umath_accuracy.py`

```
# 导入 NumPy 库，别名为 np
import numpy as np
# 导入操作系统相关的功能模块 os
import os
# 从 os 模块中导入 path 函数
from os import path
# 导入系统相关的模块 sys
import sys
# 导入 pytest 库，用于编写测试
import pytest
# 从 ctypes 模块中导入若干数据类型和函数
from ctypes import c_longlong, c_double, c_float, c_int, cast, pointer, POINTER
# 从 numpy.testing 模块中导入 assert_array_max_ulp 函数
from numpy.testing import assert_array_max_ulp
# 从 numpy.testing._private.utils 模块中导入 _glibc_older_than 函数
from numpy.testing._private.utils import _glibc_older_than
# 从 numpy._core._multiarray_umath 模块中导入 __cpu_features__ 函数

# 获取所有一元通用函数的列表，存储在 UNARY_UFUNCS 中
UNARY_UFUNCS = [obj for obj in np._core.umath.__dict__.values() if
        isinstance(obj, np.ufunc)]
# 从 UNARY_UFUNCS 中筛选出输入输出类型为对象的一元通用函数，存储在 UNARY_OBJECT_UFUNCS 中
UNARY_OBJECT_UFUNCS = [uf for uf in UNARY_UFUNCS if "O->O" in uf.types]

# 从 UNARY_OBJECT_UFUNCS 中移除不支持浮点数的函数
UNARY_OBJECT_UFUNCS.remove(getattr(np, 'invert'))
UNARY_OBJECT_UFUNCS.remove(getattr(np, 'bitwise_count'))

# 检查当前系统是否支持 AVX 指令集，存储在 IS_AVX 中
IS_AVX = __cpu_features__.get('AVX512F', False) or \
        (__cpu_features__.get('FMA3', False) and __cpu_features__.get('AVX2', False))

# 检查当前系统是否支持 AVX512FP16 指令集，存储在 IS_AVX512FP16 中
IS_AVX512FP16 = __cpu_features__.get('AVX512FP16', False)

# 只在 Linux 系统且支持 AVX 指令集，同时避免旧版 glibc（参考 numpy/numpy#20448）时才运行测试
runtest = (sys.platform.startswith('linux')
           and IS_AVX and not _glibc_older_than("2.17"))
# 如果不满足运行条件，则通过 pytest.mark.skipif 标记跳过测试
platform_skip = pytest.mark.skipif(not runtest,
                                   reason="avoid testing inconsistent platform "
                                   "library implementations")

# 将字符串转换为十六进制表示的浮点数的函数定义，来源于 Stack Overflow 上的解决方案
def convert(s, datatype="np.float32"):
    i = int(s, 16)                   # 将字符串 s 转换为 Python 的整数
    if (datatype == "np.float64"):
        cp = pointer(c_longlong(i))           # 将整数转换为 c 长整型
        fp = cast(cp, POINTER(c_double))  # 将整数指针转换为双精度浮点数指针
    else:
        cp = pointer(c_int(i))           # 将整数转换为 c 整型
        fp = cast(cp, POINTER(c_float))  # 将整数指针转换为单精度浮点数指针

    return fp.contents.value         # 返回浮点数指针所指向的值

# 将 convert 函数向量化，命名为 str_to_float，以便能够处理 NumPy 数组
str_to_float = np.vectorize(convert)

# 定义一个测试类 TestAccuracy，使用 platform_skip 标记以决定是否跳过测试
class TestAccuracy:
    @platform_skip
    # 定义一个测试函数，用于验证超越函数的正确性
    def test_validate_transcendentals(self):
        # 忽略 NumPy 的所有错误，以便在处理数据时不受错误的影响
        with np.errstate(all='ignore'):
            # 确定数据目录的路径，使用当前文件的路径和 'data' 目录名拼接而成
            data_dir = path.join(path.dirname(__file__), 'data')
            # 获取数据目录下所有文件的列表
            files = os.listdir(data_dir)
            # 过滤出以 '.csv' 结尾的文件
            files = list(filter(lambda f: f.endswith('.csv'), files))
            # 遍历每个符合条件的文件
            for filename in files:
                # 构建文件的完整路径
                filepath = path.join(data_dir, filename)
                # 打开文件，生成一个生成器，用于跳过以 '$' 或 '#' 开头的行
                with open(filepath) as fid:
                    file_without_comments = (r for r in fid if not r[0] in ('$', '#'))
                    # 从文件中读取数据并解析为 NumPy 数组
                    data = np.genfromtxt(file_without_comments,
                                         dtype=('|S39','|S39','|S39',int),
                                         names=('type','input','output','ulperr'),
                                         delimiter=',',
                                         skip_header=1)
                    # 从文件名中提取出相应的 NumPy 函数名
                    npname = path.splitext(filename)[0].split('-')[3]
                    npfunc = getattr(np, npname)
                    # 对数据按照 'type' 列中的唯一值进行分组处理
                    for datatype in np.unique(data['type']):
                        data_subset = data[data['type'] == datatype]
                        # 将输入和输出数据转换为相应的数据类型，并进行数值计算
                        inval  = np.array(str_to_float(data_subset['input'].astype(str), data_subset['type'].astype(str)), dtype=eval(datatype))
                        outval = np.array(str_to_float(data_subset['output'].astype(str), data_subset['type'].astype(str)), dtype=eval(datatype))
                        # 随机排列输入数据，以便进行最大误差比较
                        perm = np.random.permutation(len(inval))
                        inval = inval[perm]
                        outval = outval[perm]
                        # 获取当前数据集的最大 ULP 误差
                        maxulperr = data_subset['ulperr'].max()
                        # 断言调用 NumPy 函数后的输出与预期输出之间的最大 ULP 误差不超过指定值
                        assert_array_max_ulp(npfunc(inval), outval, maxulperr)

    # 使用 pytest 的装饰器标记，用于跳过 AVX512FP16 指令集支持的测试
    @pytest.mark.skipif(IS_AVX512FP16,
                        reason = "SVML FP16 have slightly higher ULP errors")
    # 参数化测试函数，验证浮点数转换及其超越函数在 FP16 下的正确性
    @pytest.mark.parametrize("ufunc", UNARY_OBJECT_UFUNCS)
    def test_validate_fp16_transcendentals(self, ufunc):
        # 忽略 NumPy 的所有错误
        with np.errstate(all='ignore'):
            # 创建一个 int16 类型的数组，转换为 float16 和 float32 类型的数组
            arr = np.arange(65536, dtype=np.int16)
            datafp16 = np.frombuffer(arr.tobytes(), dtype=np.float16)
            datafp32 = datafp16.astype(np.float32)
            # 断言在 FP16 和 FP32 下调用超越函数后的输出最大 ULP 误差不超过 1
            assert_array_max_ulp(ufunc(datafp16), ufunc(datafp32),
                                 maxulp=1, dtype=np.float16)

    # 使用 pytest 的装饰器标记，用于跳过非 AVX512FP16 指令集支持的测试
    @pytest.mark.skipif(not IS_AVX512FP16,
                        reason="lower ULP only apply for SVML FP16")
    # 定义一个测试函数，用于验证半精度浮点数的数值计算精度
    def test_validate_svml_fp16(self):
        # 定义最大误差字典，每个函数名对应其允许的最大误差
        max_ulp_err = {
                "arccos": 2.54,
                "arccosh": 2.09,
                "arcsin": 3.06,
                "arcsinh": 1.51,
                "arctan": 2.61,
                "arctanh": 1.88,
                "cbrt": 1.57,
                "cos": 1.43,
                "cosh": 1.33,
                "exp2": 1.33,
                "exp": 1.27,
                "expm1": 0.53,
                "log": 1.80,
                "log10": 1.27,
                "log1p": 1.88,
                "log2": 1.80,
                "sin": 1.88,
                "sinh": 2.05,
                "tan": 2.26,
                "tanh": 3.00,
                }
        
        # 忽略所有的 NumPy 运行时错误
        with np.errstate(all='ignore'):
            # 创建一个包含 65536 个元素的 int16 类型数组
            arr = np.arange(65536, dtype=np.int16)
            # 将 int16 数组转换为 float16 类型数组
            datafp16 = np.frombuffer(arr.tobytes(), dtype=np.float16)
            # 将 float16 类型数组转换为 float32 类型数组
            datafp32 = datafp16.astype(np.float32)
            
            # 遍历每个函数名及其对应的最大误差
            for func in max_ulp_err:
                # 根据函数名获取 NumPy 中的对应函数
                ufunc = getattr(np, func)
                # 对最大误差向上取整，作为断言函数 assert_array_max_ulp 的参数
                ulp = np.ceil(max_ulp_err[func])
                # 断言两个函数应用到 datafp16 和 datafp32 上的结果的最大 ULP 差值
                assert_array_max_ulp(ufunc(datafp16), ufunc(datafp32),
                        maxulp=ulp, dtype=np.float16)
```