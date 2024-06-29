# `.\numpy\numpy\f2py\tests\test_array_from_pyobj.py`

```
import os
import sys
import copy
import platform
import pytest
from pathlib import Path

import numpy as np

from numpy.testing import assert_, assert_equal
from numpy._core._type_aliases import c_names_dict as _c_names_dict
from . import util

wrap = None

# 扩展核心类型信息，添加 CHARACTER 类型以测试 dtype('c')
# c_names_dict 是一个字典，将字符类型映射到对应的 NumPy 数据类型
c_names_dict = dict(
    CHARACTER=np.dtype("c"),
    **_c_names_dict
)

# 获取测试目录路径
def get_testdir():
    # 使用当前文件的绝对路径获取其父目录，然后添加 "src/array_from_pyobj" 作为测试根目录的子目录
    testroot = Path(__file__).resolve().parent / "src"
    return testroot / "array_from_pyobj"

# 在模块设置阶段构建必需的测试扩展模块
def setup_module():
    """
    构建必需的测试扩展模块

    """
    global wrap

    if wrap is None:
        # 定义源文件路径列表，包括 "src/array_from_pyobj/wrapmodule.c"
        src = [
            get_testdir() / "wrapmodule.c",
        ]
        # 使用 util.build_meson 函数构建 Meson 构建系统的扩展模块
        wrap = util.build_meson(src, module_name="test_array_from_pyobj_ext")

# 根据数组的 flags 属性获取其信息
def flags_info(arr):
    # 调用 wrap.array_attrs(arr) 获取数组的属性信息中的第 6 项，即 flags
    flags = wrap.array_attrs(arr)[6]
    return flags2names(flags)

# 将 flags 转换为对应的标志名称列表
def flags2names(flags):
    info = []
    for flagname in [
            "CONTIGUOUS",
            "FORTRAN",
            "OWNDATA",
            "ENSURECOPY",
            "ENSUREARRAY",
            "ALIGNED",
            "NOTSWAPPED",
            "WRITEABLE",
            "WRITEBACKIFCOPY",
            "UPDATEIFCOPY",
            "BEHAVED",
            "BEHAVED_RO",
            "CARRAY",
            "FARRAY",
    ]:
        # 检查 flags 中是否存在指定的标志位，若存在则将标志名称加入 info 列表
        if abs(flags) & getattr(wrap, flagname, 0):
            info.append(flagname)
    return info

# 定义 Intent 类，用于管理参数的意图
class Intent:
    def __init__(self, intent_list=[]):
        # 将传入的 intent_list 复制到对象的 intent_list 属性中
        self.intent_list = intent_list[:]
        # 初始化 flags 为 0
        flags = 0
        # 遍历 intent_list 中的每一项
        for i in intent_list:
            # 根据每一项的名称构建相应的标志位并加入 flags
            if i == "optional":
                flags |= wrap.F2PY_OPTIONAL
            else:
                flags |= getattr(wrap, "F2PY_INTENT_" + i.upper())
        # 将构建好的 flags 赋值给对象的 flags 属性
        self.flags = flags

    # 允许通过属性访问 intent_list 的内容
    def __getattr__(self, name):
        name = name.lower()
        if name == "in_":
            name = "in"
        return self.__class__(self.intent_list + [name])

    # 返回对象的字符串表示，格式为 intent(列表内容)
    def __str__(self):
        return "intent(%s)" % (",".join(self.intent_list))

    # 返回对象的表示形式
    def __repr__(self):
        return "Intent(%r)" % (self.intent_list)

    # 检查当前意图是否包含指定的参数意图
    def is_intent(self, *names):
        for name in names:
            if name not in self.intent_list:
                return False
        return True

    # 检查当前意图是否严格包含指定的参数意图
    def is_intent_exact(self, *names):
        return len(self.intent_list) == len(names) and self.is_intent(*names)

# 定义预定义的数据类型名称列表
_type_names = [
    "BOOL",
    "BYTE",
    "UBYTE",
    "SHORT",
    "USHORT",
    "INT",
    "UINT",
    "LONG",
    "ULONG",
    "LONGLONG",
    "ULONGLONG",
    "FLOAT",
    "DOUBLE",
    "CFLOAT",
    "STRING1",
    "STRING5",
    "CHARACTER",
]

# 定义数据类型强制转换字典，将数据类型映射到允许的强制转换列表
_cast_dict = {
    "BOOL": ["BOOL"],
    "BYTE": _cast_dict["BOOL"] + ["BYTE"],
    "UBYTE": _cast_dict["BOOL"] + ["UBYTE"],
    "BYTE": ["BYTE"],
    "UBYTE": ["UBYTE"],
    "SHORT": _cast_dict["BYTE"] + ["UBYTE", "SHORT"],
    "USHORT": _cast_dict["UBYTE"] + ["BYTE", "USHORT"],
    "INT": _cast_dict["SHORT"] + ["USHORT", "INT"],
}
# 将 UINT 类型转换为 USHORT 类型，添加 SHORT 和 UINT 到 UINT 类型的转换字典中
_cast_dict["UINT"] = _cast_dict["USHORT"] + ["SHORT", "UINT"]

# 将 LONG 类型转换为 INT 类型，添加 LONG 到 LONG 类型的转换字典中
_cast_dict["LONG"] = _cast_dict["INT"] + ["LONG"]

# 将 ULONG 类型转换为 UINT 类型，添加 ULONG 到 ULONG 类型的转换字典中
_cast_dict["ULONG"] = _cast_dict["UINT"] + ["ULONG"]

# 将 LONGLONG 类型转换为 LONG 类型，添加 LONGLONG 到 LONGLONG 类型的转换字典中
_cast_dict["LONGLONG"] = _cast_dict["LONG"] + ["LONGLONG"]

# 将 ULONGLONG 类型转换为 ULONG 类型，添加 ULONGLONG 到 ULONGLONG 类型的转换字典中
_cast_dict["ULONGLONG"] = _cast_dict["ULONG"] + ["ULONGLONG"]

# 将 FLOAT 类型转换为 SHORT 类型，添加 USHORT、FLOAT 到 FLOAT 类型的转换字典中
_cast_dict["FLOAT"] = _cast_dict["SHORT"] + ["USHORT", "FLOAT"]

# 将 DOUBLE 类型转换为 INT 类型，添加 UINT、FLOAT、DOUBLE 到 DOUBLE 类型的转换字典中
_cast_dict["DOUBLE"] = _cast_dict["INT"] + ["UINT", "FLOAT", "DOUBLE"]

# 将 CFLOAT 类型转换为 FLOAT 类型，添加 CFLOAT 到 CFLOAT 类型的转换字典中
_cast_dict["CFLOAT"] = _cast_dict["FLOAT"] + ["CFLOAT"]

# 添加 STRING1 到 STRING1 类型的转换字典中
_cast_dict['STRING1'] = ['STRING1']

# 添加 STRING5 到 STRING5 类型的转换字典中
_cast_dict['STRING5'] = ['STRING5']

# 添加 CHARACTER 到 CHARACTER 类型的转换字典中
_cast_dict['CHARACTER'] = ['CHARACTER']

# 检查条件：32位系统的 malloc 通常不提供长双精度类型所需的16字节对齐，
# 这意味着无法满足输入意图，导致多个测试失败，因为对齐标志可能随机为真或假。
# 当 numpy 获得对齐分配器时，可以重新启用这些测试。
#
# 此外，在 macOS ARM64 上，LONGDOUBLE 是 DOUBLE 的别名。
if ((np.intp().dtype.itemsize != 4 or np.clongdouble().dtype.alignment <= 8)
        and sys.platform != "win32"
        and (platform.system(), platform.processor()) != ("Darwin", "arm")):
    # 扩展 _type_names 列表，添加 LONGDOUBLE、CDOUBLE、CLONGDOUBLE 类型
    _type_names.extend(["LONGDOUBLE", "CDOUBLE", "CLONGDOUBLE"])
    
    # 将 LONGDOUBLE 类型转换为 LONG 类型，并添加 ULONG、FLOAT、DOUBLE、LONGDOUBLE 到 LONGDOUBLE 类型的转换字典中
    _cast_dict["LONGDOUBLE"] = _cast_dict["LONG"] + [
        "ULONG",
        "FLOAT",
        "DOUBLE",
        "LONGDOUBLE",
    ]
    
    # 将 CLONGDOUBLE 类型转换为 LONGDOUBLE 类型，并添加 CFLOAT、CDOUBLE、CLONGDOUBLE 到 CLONGDOUBLE 类型的转换字典中
    _cast_dict["CLONGDOUBLE"] = _cast_dict["LONGDOUBLE"] + [
        "CFLOAT",
        "CDOUBLE",
        "CLONGDOUBLE",
    ]
    
    # 将 CDOUBLE 类型转换为 DOUBLE 类型，并添加 CFLOAT、CDOUBLE 到 CDOUBLE 类型的转换字典中
    _cast_dict["CDOUBLE"] = _cast_dict["DOUBLE"] + ["CFLOAT", "CDOUBLE"]


class Type:
    _type_cache = {}

    def __new__(cls, name):
        if isinstance(name, np.dtype):
            dtype0 = name
            name = None
            for n, i in c_names_dict.items():
                if not isinstance(i, type) and dtype0.type is i.type:
                    name = n
                    break
        obj = cls._type_cache.get(name.upper(), None)
        if obj is not None:
            return obj
        obj = object.__new__(cls)
        obj._init(name)
        cls._type_cache[name.upper()] = obj
        return obj

    def _init(self, name):
        self.NAME = name.upper()

        if self.NAME == 'CHARACTER':
            info = c_names_dict[self.NAME]
            self.type_num = getattr(wrap, 'NPY_STRING')
            self.elsize = 1
            self.dtype = np.dtype('c')
        elif self.NAME.startswith('STRING'):
            info = c_names_dict[self.NAME[:6]]
            self.type_num = getattr(wrap, 'NPY_STRING')
            self.elsize = int(self.NAME[6:] or 0)
            self.dtype = np.dtype(f'S{self.elsize}')
        else:
            info = c_names_dict[self.NAME]
            self.type_num = getattr(wrap, 'NPY_' + self.NAME)
            self.elsize = info.itemsize
            self.dtype = np.dtype(info.type)

        assert self.type_num == info.num
        self.type = info.type
        self.dtypechar = info.char
    # 定义对象的字符串表示形式，返回格式化后的字符串，包括类型名称和相关属性
    def __repr__(self):
        return (f"Type({self.NAME})|type_num={self.type_num},"
                f" dtype={self.dtype},"
                f" type={self.type}, elsize={self.elsize},"
                f" dtypechar={self.dtypechar}")

    # 将当前类型转换为其它类型的对象列表，使用_cast_dict中当前类型的映射
    def cast_types(self):
        return [self.__class__(_m) for _m in _cast_dict[self.NAME]]

    # 返回所有已定义类型的对象列表，每个对象表示一个类型
    def all_types(self):
        return [self.__class__(_m) for _m in _type_names]

    # 返回比当前类型对齐位数小的所有类型的对象列表
    def smaller_types(self):
        bits = c_names_dict[self.NAME].alignment  # 获取当前类型的对齐位数
        types = []
        for name in _type_names:
            if c_names_dict[name].alignment < bits:  # 检查每个类型的对齐位数是否小于当前类型
                types.append(Type(name))  # 如果是，则添加对应类型的对象到列表
        return types

    # 返回与当前类型对齐位数相同的所有类型的对象列表
    def equal_types(self):
        bits = c_names_dict[self.NAME].alignment  # 获取当前类型的对齐位数
        types = []
        for name in _type_names:
            if name == self.NAME:
                continue
            if c_names_dict[name].alignment == bits:  # 检查每个类型的对齐位数是否与当前类型相同
                types.append(Type(name))  # 如果是，则添加对应类型的对象到列表
        return types

    # 返回比当前类型对齐位数大的所有类型的对象列表
    def larger_types(self):
        bits = c_names_dict[self.NAME].alignment  # 获取当前类型的对齐位数
        types = []
        for name in _type_names:
            if c_names_dict[name].alignment > bits:  # 检查每个类型的对齐位数是否大于当前类型
                types.append(Type(name))  # 如果是，则添加对应类型的对象到列表
        return types
class Array:
    # 返回数组的字符串表示形式，包括类型、维度、意图和对象信息
    def __repr__(self):
        return (f'Array({self.type}, {self.dims}, {self.intent},'
                f' {self.obj})|arr={self.arr}')

    # 比较两个数组是否相等
    def arr_equal(self, arr1, arr2):
        # 检查数组形状是否相同
        if arr1.shape != arr2.shape:
            return False
        # 检查数组元素是否完全相同
        return (arr1 == arr2).all()

    # 返回数组的字符串表示形式
    def __str__(self):
        return str(self.arr)

    # 检查是否创建的数组与输入数组共享内存
    def has_shared_memory(self):
        # 检查对象是否与数组相同
        if self.obj is self.arr:
            return True
        # 如果对象不是 numpy 数组，返回 False
        if not isinstance(self.obj, np.ndarray):
            return False
        # 使用 wrap 模块获取对象的属性
        obj_attr = wrap.array_attrs(self.obj)
        # 比较对象属性的第一个元素与数组属性的第一个元素是否相同
        return obj_attr[0] == self.arr_attr[0]


class TestIntent:
    # 测试 intent 对象的输入输出字符串表示
    def test_in_out(self):
        assert str(intent.in_.out) == "intent(in,out)"
        # 检查是否 c 是 intent
        assert intent.in_.c.is_intent("c")
        # 检查是否 c 是确切的 intent
        assert not intent.in_.c.is_intent_exact("c")
        # 检查是否 c 是确切的 intent
        assert intent.in_.c.is_intent_exact("c", "in")
        # 检查是否 c 是确切的 intent
        assert intent.in_.c.is_intent_exact("in", "c")
        # 检查是否 c 是 intent
        assert not intent.in_.is_intent("c")


class TestSharedMemory:
    # 设置测试类型的 fixture
    @pytest.fixture(autouse=True, scope="class", params=_type_names)
    def setup_type(self, request):
        request.cls.type = Type(request.param)
        # 创建 Array 对象的 lambda 函数
        request.cls.array = lambda self, dims, intent, obj: Array(
            Type(request.param), dims, intent, obj)

    # 返回 num2seq 属性
    @property
    def num2seq(self):
        # 如果类型以 'STRING' 开头，返回字符串序列
        if self.type.NAME.startswith('STRING'):
            elsize = self.type.elsize
            return ['1' * elsize, '2' * elsize]
        # 否则返回整数序列
        return [1, 2]

    # 返回 num23seq 属性
    @property
    def num23seq(self):
        # 如果类型以 'STRING' 开头，返回二维字符串序列
        if self.type.NAME.startswith('STRING'):
            elsize = self.type.elsize
            return [['1' * elsize, '2' * elsize, '3' * elsize],
                    ['4' * elsize, '5' * elsize, '6' * elsize]]
        # 否则返回二维整数序列
        return [[1, 2, 3], [4, 5, 6]]

    # 测试从两个元素序列创建输入数组
    def test_in_from_2seq(self):
        # 使用 Array 创建数组对象 a
        a = self.array([2], intent.in_, self.num2seq)
        # 断言 a 没有共享内存
        assert not a.has_shared_memory()

    # 测试从两个转换类型创建输入数组
    def test_in_from_2casttype(self):
        # 遍历类型的转换类型
        for t in self.type.cast_types():
            # 使用 numpy 创建数组对象 obj
            obj = np.array(self.num2seq, dtype=t.dtype)
            # 使用 Array 创建数组对象 a
            a = self.array([len(self.num2seq)], intent.in_, obj)
            # 如果类型的字节大小与转换类型的字节大小相同，断言 a 共享内存
            if t.elsize == self.type.elsize:
                assert a.has_shared_memory(), repr((self.type.dtype, t.dtype))
            else:
                assert not a.has_shared_memory()

    # 测试是否可以传递 intent(in) 数组而不进行复制
    @pytest.mark.parametrize("write", ["w", "ro"])
    @pytest.mark.parametrize("order", ["C", "F"])
    @pytest.mark.parametrize("inp", ["2seq", "23seq"])
    def test_in_nocopy(self, write, order, inp):
        """Test if intent(in) array can be passed without copies"""
        # 获取相应的序列属性
        seq = getattr(self, "num" + inp)
        # 使用 numpy 创建数组对象 obj
        obj = np.array(seq, dtype=self.type.dtype, order=order)
        # 设置数组的写入标志
        obj.setflags(write=(write == 'w'))
        # 使用 Array 创建数组对象 a
        a = self.array(obj.shape,
                       ((order == 'C' and intent.in_.c) or intent.in_), obj)
        # 断言 a 共享内存
        assert a.has_shared_memory()
    # 定义测试函数，用于测试特定的输入输出意图
    def test_inout_2seq(self):
        # 使用self.num2seq生成NumPy数组，并指定数据类型
        obj = np.array(self.num2seq, dtype=self.type.dtype)
        # 创建一个具有意图输入输出的数组对象，并传入长度信息和数据对象
        a = self.array([len(self.num2seq)], intent.inout, obj)
        # 断言数组具有共享内存特性
        assert a.has_shared_memory()

        try:
            # 尝试创建一个意图为输入输出的数组对象，但传入了序列对象self.num2seq，预期会抛出TypeError异常
            a = self.array([2], intent.in_.inout, self.num2seq)
        except TypeError as msg:
            # 如果异常消息不是以指定的字符串开头，则抛出异常
            if not str(msg).startswith(
                    "failed to initialize intent(inout|inplace|cache) array"):
                raise
        else:
            # 如果没有抛出TypeError异常，则抛出SystemError异常
            raise SystemError("intent(inout) should have failed on sequence")

    # 定义测试函数，测试Fortran顺序的输入输出意图
    def test_f_inout_23seq(self):
        # 使用self.num23seq生成NumPy数组，指定数据类型和Fortran顺序
        obj = np.array(self.num23seq, dtype=self.type.dtype, order="F")
        # 创建一个具有意图输入输出的数组对象，并传入形状信息和数据对象
        shape = (len(self.num23seq), len(self.num23seq[0]))
        a = self.array(shape, intent.in_.inout, obj)
        # 断言数组具有共享内存特性
        assert a.has_shared_memory()

        # 使用self.num23seq生成NumPy数组，指定数据类型和C顺序
        obj = np.array(self.num23seq, dtype=self.type.dtype, order="C")
        # 尝试创建一个意图为输入输出的数组对象，但传入了错误的数组对象，预期会抛出ValueError异常
        try:
            a = self.array(shape, intent.in_.inout, obj)
        except ValueError as msg:
            # 如果异常消息不是以指定的字符串开头，则抛出异常
            if not str(msg).startswith(
                    "failed to initialize intent(inout) array"):
                raise
        else:
            # 如果没有抛出ValueError异常，则抛出SystemError异常
            raise SystemError(
                "intent(inout) should have failed on improper array")

    # 定义测试函数，测试复杂顺序的输入输出意图
    def test_c_inout_23seq(self):
        # 使用self.num23seq生成NumPy数组，并指定数据类型
        obj = np.array(self.num23seq, dtype=self.type.dtype)
        # 创建一个具有意图输入输出的数组对象，并传入形状信息和数据对象
        shape = (len(self.num23seq), len(self.num23seq[0]))
        a = self.array(shape, intent.in_.c.inout, obj)
        # 断言数组具有共享内存特性
        assert a.has_shared_memory()

    # 定义测试函数，测试从不同类型转换而来的输入复制意图
    def test_in_copy_from_2casttype(self):
        # 遍历类型转换后的数据类型列表
        for t in self.type.cast_types():
            # 使用self.num2seq生成NumPy数组，并指定数据类型为当前类型t的dtype
            obj = np.array(self.num2seq, dtype=t.dtype)
            # 创建一个具有意图输入复制的数组对象，并传入长度信息和数据对象
            a = self.array([len(self.num2seq)], intent.in_.copy, obj)
            # 断言数组不具有共享内存特性
            assert not a.has_shared_memory()

    # 定义测试函数，测试从序列对象转换而来的输入意图
    def test_c_in_from_23seq(self):
        # 创建一个具有意图输入的数组对象，并传入形状信息和数据对象self.num23seq
        a = self.array(
            [len(self.num23seq), len(self.num23seq[0])], intent.in_,
            self.num23seq)
        # 断言数组不具有共享内存特性
        assert not a.has_shared_memory()

    # 定义测试函数，测试从不同类型转换而来的输入意图
    def test_in_from_23casttype(self):
        # 遍历类型转换后的数据类型列表
        for t in self.type.cast_types():
            # 使用self.num23seq生成NumPy数组，并指定数据类型为当前类型t的dtype
            obj = np.array(self.num23seq, dtype=t.dtype)
            # 创建一个具有意图输入的数组对象，并传入形状信息和数据对象
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_, obj)
            # 断言数组不具有共享内存特性
            assert not a.has_shared_memory()

    # 定义测试函数，测试从不同类型转换而来的Fortran顺序输入意图
    def test_f_in_from_23casttype(self):
        # 遍历类型转换后的数据类型列表
        for t in self.type.cast_types():
            # 使用self.num23seq生成NumPy数组，并指定数据类型为当前类型t的dtype和Fortran顺序
            obj = np.array(self.num23seq, dtype=t.dtype, order="F")
            # 创建一个具有意图输入的数组对象，并传入形状信息和数据对象
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_, obj)
            # 如果当前类型t的元素大小等于self.type的元素大小，则断言数组具有共享内存特性，否则断言数组不具有共享内存特性
            if t.elsize == self.type.elsize:
                assert a.has_shared_memory()
            else:
                assert not a.has_shared_memory()
    # 对于类型系统中的每种类型进行测试
    def test_c_in_from_23casttype(self):
        for t in self.type.cast_types():
            # 使用给定的类型创建一个 numpy 数组对象
            obj = np.array(self.num23seq, dtype=t.dtype)
            # 创建一个数组对象，指定其意图为输入 (intent.in_.c)，传入 numpy 数组对象
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_.c, obj)
            # 如果类型的字节大小与数组对象的字节大小相同，则断言数组具有共享内存
            if t.elsize == self.type.elsize:
                assert a.has_shared_memory()
            else:
                assert not a.has_shared_memory()

    # 对于类型系统中的每种类型进行测试，强制使用 Fortran 的顺序
    def test_f_copy_in_from_23casttype(self):
        for t in self.type.cast_types():
            # 使用给定的类型创建一个 numpy 数组对象，强制使用 Fortran 的顺序
            obj = np.array(self.num23seq, dtype=t.dtype, order="F")
            # 创建一个数组对象，指定其意图为输入并且要求复制 (intent.in_.copy)，传入 numpy 数组对象
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_.copy,
                obj)
            # 断言数组不具有共享内存
            assert not a.has_shared_memory()

    # 对于类型系统中的每种类型进行测试，强制使用 C 的顺序并要求复制
    def test_c_copy_in_from_23casttype(self):
        for t in self.type.cast_types():
            # 使用给定的类型创建一个 numpy 数组对象
            obj = np.array(self.num23seq, dtype=t.dtype)
            # 创建一个数组对象，指定其意图为输入并且要求复制 (intent.in_.c.copy)，传入 numpy 数组对象
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_.c.copy,
                obj)
            # 断言数组不具有共享内存
            assert not a.has_shared_memory()

    # 对于类型系统中的每种类型进行测试，检查缓存意图 (intent.in_.cache)
    def test_in_cache_from_2casttype(self):
        for t in self.type.all_types():
            # 如果类型的字节大小与系统的字节大小不同，则跳过该类型的测试
            if t.elsize != self.type.elsize:
                continue
            # 使用给定的类型创建一个 numpy 数组对象
            obj = np.array(self.num2seq, dtype=t.dtype)
            shape = (len(self.num2seq), )
            # 创建一个数组对象，指定其意图为输入并且要求缓存 (intent.in_.c.cache)，传入 numpy 数组对象
            a = self.array(shape, intent.in_.c.cache, obj)
            # 断言数组具有共享内存
            assert a.has_shared_memory()

            # 创建一个数组对象，指定其意图为输入并且要求缓存 (intent.in_.cache)，传入 numpy 数组对象
            a = self.array(shape, intent.in_.cache, obj)
            # 断言数组具有共享内存
            assert a.has_shared_memory()

            # 使用给定的类型创建一个 numpy 数组对象，强制使用 Fortran 的顺序
            obj = np.array(self.num2seq, dtype=t.dtype, order="F")
            # 创建一个数组对象，指定其意图为输入并且要求缓存 (intent.in_.c.cache)，传入 numpy 数组对象
            a = self.array(shape, intent.in_.c.cache, obj)
            # 断言数组具有共享内存
            assert a.has_shared_memory()

            # 创建一个数组对象，指定其意图为输入并且要求缓存 (intent.in_.cache)，传入 numpy 数组对象
            a = self.array(shape, intent.in_.cache, obj)
            # 断言数组具有共享内存，并输出该类型的数据类型信息
            assert a.has_shared_memory(), repr(t.dtype)

            # 尝试使用反转后的数组对象创建一个数组对象，指定其意图为输入并且要求缓存 (intent.in_.cache)
            try:
                a = self.array(shape, intent.in_.cache, obj[::-1])
            # 如果抛出 ValueError 异常且消息不以指定的前缀开头，则重新抛出异常
            except ValueError as msg:
                if not str(msg).startswith(
                        "failed to initialize intent(cache) array"):
                    raise
            # 否则，抛出 SystemError 异常，表明意图 (cache) 应该在多段数组上失败
            else:
                raise SystemError(
                    "intent(cache) should have failed on multisegmented array")
    # 对 self.type 中的每种类型进行迭代，self.type 是一个类型对象的集合
    def test_in_cache_from_2casttype_failure(self):
        for t in self.type.all_types():
            # 如果类型 t 的名称为 'STRING'，跳过当前循环，不执行后续代码
            if t.NAME == 'STRING':
                # string elsize is 0, so skipping the test
                continue
            # 如果 t 的元素大小大于等于 self.type 的元素大小，跳过当前循环，不执行后续代码
            if t.elsize >= self.type.elsize:
                continue
            # 判断 t.dtype 是否为整数类型
            is_int = np.issubdtype(t.dtype, np.integer)
            # 如果是整数类型，并且将 self.num2seq[0] 转换为整数后大于 t.dtype 的最大值，跳过当前循环，不执行后续代码
            if is_int and int(self.num2seq[0]) > np.iinfo(t.dtype).max:
                # skip test if num2seq would trigger an overflow error
                continue
            # 将 self.num2seq 转换为类型为 t.dtype 的 NumPy 数组
            obj = np.array(self.num2seq, dtype=t.dtype)
            # 创建形状为 (len(self.num2seq), ) 的数组
            shape = (len(self.num2seq), )
            try:
                # 调用 self.array 方法，传入参数 shape, intent.in_.cache, obj，期望操作成功
                self.array(shape, intent.in_.cache, obj)  # Should succeed
            except ValueError as msg:
                # 如果捕获到 ValueError 异常，并且异常消息不以 "failed to initialize intent(cache) array" 开头，抛出异常
                if not str(msg).startswith(
                        "failed to initialize intent(cache) array"):
                    raise
            else:
                # 如果没有捕获到异常，抛出 SystemError 异常，说明 intent(cache) 应该在更小的数组上失败
                raise SystemError(
                    "intent(cache) should have failed on smaller array")

    # 测试 intent.cache.hide 模式
    def test_cache_hidden(self):
        # 测试形状为 (2, ) 的数组
        shape = (2, )
        # 调用 self.array 方法，使用 intent.cache.hide 模式创建数组 a，不传入对象
        a = self.array(shape, intent.cache.hide, None)
        # 断言数组 a 的形状为 shape
        assert a.arr.shape == shape

        # 测试形状为 (2, 3) 的数组
        shape = (2, 3)
        # 调用 self.array 方法，使用 intent.cache.hide 模式创建数组 a，不传入对象
        a = self.array(shape, intent.cache.hide, None)
        # 断言数组 a 的形状为 shape
        assert a.arr.shape == shape

        # 测试形状为 (-1, 3) 的数组
        shape = (-1, 3)
        try:
            # 尝试调用 self.array 方法，使用 intent.cache.hide 模式创建数组 a，不传入对象
            a = self.array(shape, intent.cache.hide, None)
        except ValueError as msg:
            # 如果捕获到 ValueError 异常，并且异常消息不以 "failed to create intent(cache|hide)|optional array" 开头，抛出异常
            if not str(msg).startswith(
                    "failed to create intent(cache|hide)|optional array"):
                raise
        else:
            # 如果没有捕获到异常，抛出 SystemError 异常，说明 intent(cache) 应该在未定义维度上失败
            raise SystemError(
                "intent(cache) should have failed on undefined dimensions")

    # 测试 intent.hide 模式
    def test_hidden(self):
        # 测试形状为 (2, ) 的数组
        shape = (2, )
        # 调用 self.array 方法，使用 intent.hide 模式创建数组 a，不传入对象
        a = self.array(shape, intent.hide, None)
        # 断言数组 a 的形状为 shape
        assert a.arr.shape == shape
        # 断言数组 a 的内容与形状为 shape、数据类型为 self.type.dtype 的零数组相等
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))

        # 测试形状为 (2, 3) 的数组
        shape = (2, 3)
        # 调用 self.array 方法，使用 intent.hide 模式创建数组 a，不传入对象
        a = self.array(shape, intent.hide, None)
        # 断言数组 a 的形状为 shape
        assert a.arr.shape == shape
        # 断言数组 a 的内容与形状为 shape、数据类型为 self.type.dtype 的零数组相等
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        # 断言数组 a 是按 Fortran 顺序存储而不是连续存储
        assert a.arr.flags["FORTRAN"] and not a.arr.flags["CONTIGUOUS"]

        # 测试形状为 (2, 3) 的数组
        shape = (2, 3)
        # 调用 self.array 方法，使用 intent.c.hide 模式创建数组 a，不传入对象
        a = self.array(shape, intent.c.hide, None)
        # 断言数组 a 的形状为 shape
        assert a.arr.shape == shape
        # 断言数组 a 的内容与形状为 shape、数据类型为 self.type.dtype 的零数组相等
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        # 断言数组 a 不是按 Fortran 顺序存储而是连续存储
        assert not a.arr.flags["FORTRAN"] and a.arr.flags["CONTIGUOUS"]

        # 测试形状为 (-1, 3) 的数组
        shape = (-1, 3)
        try:
            # 尝试调用 self.array 方法，使用 intent.hide 模式创建数组 a，不传入对象
            a = self.array(shape, intent.hide, None)
        except ValueError as msg:
            # 如果捕获到 ValueError 异常，并且异常消息不以 "failed to create intent(cache|hide)|optional array" 开头，抛出异常
            if not str(msg).startswith(
                    "failed to create intent(cache|hide)|optional array"):
                raise
        else:
            # 如果没有捕获到异常，抛出 SystemError 异常，说明 intent(hide) 应该在未定义维度上失败
            raise SystemError(
                "intent(hide) should have failed on undefined dimensions")
    # 测试函数，验证处理可选参数为 None 的情况
    def test_optional_none(self):
        shape = (2, )
        # 调用 array 方法创建一个数组对象，形状为 (2,)，可选参数为 None
        a = self.array(shape, intent.optional, None)
        # 断言数组对象的形状与预期相同
        assert a.arr.shape == shape
        # 断言数组对象与用零填充后的数组在值上相等
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))

        shape = (2, 3)
        # 再次调用 array 方法创建一个数组对象，形状为 (2,3)，可选参数为 None
        a = self.array(shape, intent.optional, None)
        # 断言数组对象的形状与预期相同
        assert a.arr.shape == shape
        # 断言数组对象与用零填充后的数组在值上相等
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        # 断言数组对象为 Fortran 风格，且不是连续的
        assert a.arr.flags["FORTRAN"] and not a.arr.flags["CONTIGUOUS"]

        shape = (2, 3)
        # 第三次调用 array 方法创建一个数组对象，形状为 (2,3)，可选参数为 None
        a = self.array(shape, intent.c.optional, None)
        # 断言数组对象的形状与预期相同
        assert a.arr.shape == shape
        # 断言数组对象与用零填充后的数组在值上相等
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        # 断言数组对象不是 Fortran 风格，且是连续的
        assert not a.arr.flags["FORTRAN"] and a.arr.flags["CONTIGUOUS"]

    # 测试函数，验证处理从可迭代对象创建数组的情况
    def test_optional_from_2seq(self):
        obj = self.num2seq
        shape = (len(obj), )
        # 调用 array 方法创建一个数组对象，形状为 (len(obj),)，可选参数为 obj
        a = self.array(shape, intent.optional, obj)
        # 断言数组对象的形状与预期相同
        assert a.arr.shape == shape
        # 断言数组对象没有共享内存
        assert not a.has_shared_memory()

    # 测试函数，验证处理从二维可迭代对象创建数组的情况
    def test_optional_from_23seq(self):
        obj = self.num23seq
        shape = (len(obj), len(obj[0]))
        # 调用 array 方法创建一个数组对象，形状为 (len(obj), len(obj[0]))，可选参数为 obj
        a = self.array(shape, intent.optional, obj)
        # 断言数组对象的形状与预期相同
        assert a.arr.shape == shape
        # 断言数组对象没有共享内存
        assert not a.has_shared_memory()

        # 再次调用 array 方法创建一个数组对象，形状为 (len(obj), len(obj[0]))，可选参数为 obj
        a = self.array(shape, intent.optional.c, obj)
        # 断言数组对象的形状与预期相同
        assert a.arr.shape == shape
        # 断言数组对象没有共享内存
        assert not a.has_shared_memory()

    # 测试函数，验证处理 inplace 操作的情况
    def test_inplace(self):
        obj = np.array(self.num23seq, dtype=self.type.dtype)
        # 断言对象不是 Fortran 风格且是连续的
        assert not obj.flags["FORTRAN"] and obj.flags["CONTIGUOUS"]
        shape = obj.shape
        # 调用 array 方法创建一个数组对象，形状与 obj 相同，操作类型为 inplace，对象为 obj
        a = self.array(shape, intent.inplace, obj)
        # 断言数组对象的某个元素与 obj 相同
        assert obj[1][2] == a.arr[1][2], repr((obj, a.arr))
        # 修改数组对象的某个元素
        a.arr[1][2] = 54
        # 断言 obj 和数组对象的相同元素值为 54
        assert obj[1][2] == a.arr[1][2] == np.array(54, dtype=self.type.dtype)
        # 断言数组对象与 obj 是同一个对象
        assert a.arr is obj
        # 断言 obj 是 Fortran 风格的（因为 inplace 操作）
        assert obj.flags["FORTRAN"]  # obj attributes are changed inplace!
        # 断言 obj 不是连续的
        assert not obj.flags["CONTIGUOUS"]

    # 测试函数，验证从类型转换后进行 inplace 操作的情况
    def test_inplace_from_casttype(self):
        for t in self.type.cast_types():
            if t is self.type:
                continue
            obj = np.array(self.num23seq, dtype=t.dtype)
            # 断言对象的数据类型为 t
            assert obj.dtype.type == t.type
            # 断言对象的数据类型不是 self.type.type
            assert obj.dtype.type is not self.type.type
            # 断言对象不是 Fortran 风格且是连续的
            assert not obj.flags["FORTRAN"] and obj.flags["CONTIGUOUS"]
            shape = obj.shape
            # 调用 array 方法创建一个数组对象，形状与 obj 相同，操作类型为 inplace，对象为 obj
            a = self.array(shape, intent.inplace, obj)
            # 断言数组对象的某个元素与 obj 相同
            assert obj[1][2] == a.arr[1][2], repr((obj, a.arr))
            # 修改数组对象的某个元素
            a.arr[1][2] = 54
            # 断言 obj 和数组对象的相同元素值为 54
            assert obj[1][2] == a.arr[1][2] == np.array(54, dtype=self.type.dtype)
            # 断言数组对象与 obj 是同一个对象
            assert a.arr is obj
            # 断言 obj 是 Fortran 风格的（因为 inplace 操作）
            assert obj.flags["FORTRAN"]  # obj attributes changed inplace!
            # 断言 obj 不是连续的
            assert not obj.flags["CONTIGUOUS"]
            # 断言 obj 的数据类型是 self.type.type（因为 inplace 操作）
            assert obj.dtype.type is self.type.type  # obj changed inplace!
```