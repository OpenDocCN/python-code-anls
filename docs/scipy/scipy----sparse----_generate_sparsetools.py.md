# `D:\src\scipysrc\scipy\scipy\sparse\_generate_sparsetools.py`

```
#!/usr/bin/env python3
"""
python generate_sparsetools.py

Generate manual wrappers for C++ sparsetools code.

Type codes used:

    'i':  integer scalar
    'I':  integer array
    'T':  data array
    'B':  boolean array
    'V':  std::vector<integer>*
    'W':  std::vector<data>*
    '*':  indicates that the next argument is an output argument
    'v':  void
    'l':  64-bit integer scalar

See sparsetools.cxx for more details.

"""
import argparse  # 导入处理命令行参数的库
import os  # 导入与操作系统相关的功能
from stat import ST_MTIME  # 从stat模块中导入获取文件修改时间的常量


#
# List of all routines and their argument types.
#
# The first code indicates the return value, the rest the arguments.
#

# bsr.h
BSR_ROUTINES = """
bsr_diagonal        v iiiiiIIT*T
bsr_tocsr           v iiiiIIT*I*I*T
bsr_scale_rows      v iiiiII*TT
bsr_scale_columns   v iiiiII*TT
bsr_sort_indices    v iiii*I*I*T
bsr_transpose       v iiiiIIT*I*I*T
bsr_matmat          v iiiiiiIITIIT*I*I*T
bsr_matvec          v iiiiIITT*T
bsr_matvecs         v iiiiiIITT*T
bsr_elmul_bsr       v iiiiIITIIT*I*I*T
bsr_eldiv_bsr       v iiiiIITIIT*I*I*T
bsr_plus_bsr        v iiiiIITIIT*I*I*T
bsr_minus_bsr       v iiiiIITIIT*I*I*T
bsr_maximum_bsr     v iiiiIITIIT*I*I*T
bsr_minimum_bsr     v iiiiIITIIT*I*I*T
bsr_ne_bsr          v iiiiIITIIT*I*I*B
bsr_lt_bsr          v iiiiIITIIT*I*I*B
bsr_gt_bsr          v iiiiIITIIT*I*I*B
bsr_le_bsr          v iiiiIITIIT*I*I*B
bsr_ge_bsr          v iiiiIITIIT*I*I*B
"""

# csc.h
CSC_ROUTINES = """
csc_diagonal        v iiiIIT*T
csc_tocsr           v iiIIT*I*I*T
csc_matmat_maxnnz   l iiIIII
csc_matmat          v iiIITIIT*I*I*T
csc_matvec          v iiIITT*T
csc_matvecs         v iiiIITT*T
csc_elmul_csc       v iiIITIIT*I*I*T
csc_eldiv_csc       v iiIITIIT*I*I*T
csc_plus_csc        v iiIITIIT*I*I*T
csc_minus_csc       v iiIITIIT*I*I*T
csc_maximum_csc     v iiIITIIT*I*I*T
csc_minimum_csc     v iiIITIIT*I*I*T
csc_ne_csc          v iiIITIIT*I*I*B
csc_lt_csc          v iiIITIIT*I*I*B
csc_gt_csc          v iiIITIIT*I*I*B
csc_le_csc          v iiIITIIT*I*I*B
csc_ge_csc          v iiIITIIT*I*I*B
"""

# csr.h
CSR_ROUTINES = """
csr_matmat_maxnnz   l iiIIII
csr_matmat          v iiIITIIT*I*I*T
csr_diagonal        v iiiIIT*T
csr_tocsc           v iiIIT*I*I*T
csr_tobsr           v iiiiIIT*I*I*T
csr_todense         v iiIIT*T
csr_matvec          v iiIITT*T
csr_matvecs         v iiiIITT*T
csr_elmul_csr       v iiIITIIT*I*I*T
csr_eldiv_csr       v iiIITIIT*I*I*T
csr_plus_csr        v iiIITIIT*I*I*T
csr_minus_csr       v iiIITIIT*I*I*T
csr_maximum_csr     v iiIITIIT*I*I*T
csr_minimum_csr     v iiIITIIT*I*I*T
csr_ne_csr          v iiIITIIT*I*I*B
csr_lt_csr          v iiIITIIT*I*I*B
csr_gt_csr          v iiIITIIT*I*I*B
csr_le_csr          v iiIITIIT*I*I*B
csr_ge_csr          v iiIITIIT*I*I*B
csr_scale_rows      v iiII*TT
csr_scale_columns   v iiII*TT
csr_sort_indices    v iI*I*T
csr_eliminate_zeros v ii*I*I*T
csr_sum_duplicates  v ii*I*I*T
get_csr_submatrix   v iiIITiiii*V*V*W
csr_row_index       v iIIIT*I*T
csr_row_slice       v iiiIIT*I*T
"""
# BSR_ROUTINES字符串：包含了BSR格式相关的函数定义和原型
BSR_ROUTINES = """
bsr_tocsr           v iiiIIT*I*I*T
bsr_todense         v iilIIT*Ti
"""

# CSR_ROUTINES字符串：包含了CSR格式相关的函数定义和原型
CSR_ROUTINES = """
csr_column_index1   v iIiiII*I*I
csr_column_index2   v IIiIT*I*T
csr_sample_values   v iiIITiII*T
csr_count_blocks    i iiiiII
csr_sample_offsets  i iiIIiII*I
csr_hstack          v iiIIIT*I*I*T
expandptr           v iI*I
test_throw_error    i
csr_has_sorted_indices    i iII
csr_has_canonical_format  i iII
"""

# CSC_ROUTINES字符串：包含了CSC格式相关的函数定义和原型
CSC_ROUTINES = """
csc_column_index1   v iIiiII*I*I
csc_column_index2   v IIiIT*I*T
csc_sample_values   v iiIITiII*T
csc_count_blocks    i iiiiII
csc_sample_offsets  i iiIIiII*I
csc_vstack          v iiIIIT*I*I*T
expandoncscptr      v iI*I
test_throw_error    i
csc_has_sorted_indices    i iII
csc_has_canonical_format  i iII
"""

# OTHER_ROUTINES字符串：包含了COO、DIA和CSGraph中其他格式的函数定义和原型
OTHER_ROUTINES = """
coo_tocsr           v iiiIIT*I*I*T
coo_todense         v iilIIT*Ti
coo_matvec          v lIITT*T
dia_matvec          v iiiiITT*T
cs_graph_components i iII*I
"""

# COMPILATION_UNITS列表：存储了编译单元的名称及其对应的函数字符串
COMPILATION_UNITS = [
    ('bsr', BSR_ROUTINES),
    ('csr', CSR_ROUTINES),
    ('csc', CSC_ROUTINES),
    ('other', OTHER_ROUTINES),
]

# I_TYPES列表：存储了支持的索引类型及其对应的C++类型
I_TYPES = [
    ('NPY_INT32', 'npy_int32'),
    ('NPY_INT64', 'npy_int64'),
]

# T_TYPES列表：存储了支持的数据类型及其对应的C++类型
T_TYPES = [
    ('NPY_BOOL', 'npy_bool_wrapper'),
    ('NPY_BYTE', 'npy_byte'),
    ('NPY_UBYTE', 'npy_ubyte'),
    ('NPY_SHORT', 'npy_short'),
    ('NPY_USHORT', 'npy_ushort'),
    ('NPY_INT', 'npy_int'),
    ('NPY_UINT', 'npy_uint'),
    ('NPY_LONG', 'npy_long'),
    ('NPY_ULONG', 'npy_ulong'),
    ('NPY_LONGLONG', 'npy_longlong'),
    ('NPY_ULONGLONG', 'npy_ulonglong'),
    ('NPY_FLOAT', 'npy_float'),
    ('NPY_DOUBLE', 'npy_double'),
    ('NPY_LONGDOUBLE', 'npy_longdouble'),
    ('NPY_CFLOAT', 'npy_cfloat_wrapper'),
    ('NPY_CDOUBLE', 'npy_cdouble_wrapper'),
    ('NPY_CLONGDOUBLE', 'npy_clongdouble_wrapper'),
]

# THUNK_TEMPLATE字符串：用于生成thunk函数的模板
THUNK_TEMPLATE = """
static PY_LONG_LONG %(name)s_thunk(int I_typenum, int T_typenum, void **a)
{
    %(thunk_content)s
}
"""

# METHOD_TEMPLATE字符串：用于生成方法函数的模板
METHOD_TEMPLATE = """
PyObject *
%(name)s_method(PyObject *self, PyObject *args)
{
    return call_thunk('%(ret_spec)s', "%(arg_spec)s", %(name)s_thunk, args);
}
"""

# GET_THUNK_CASE_TEMPLATE字符串：用于生成获取thunk case的函数模板
GET_THUNK_CASE_TEMPLATE = """
static int get_thunk_case(int I_typenum, int T_typenum)
{
    %(content)s;
    return -1;
}
"""

# newer函数：用于判断源文件是否比目标文件更新
def newer(source, target):
    """
    Return true if 'source' exists and is more recently modified than
    'target', or if 'source' exists and 'target' doesn't.  Return false if
    both exist and 'target' is the same age or younger than 'source'.
    """
    if not os.path.exists(source):
        raise ValueError("file '%s' does not exist" % os.path.abspath(source))
    if not os.path.exists(target):
        return 1

    mtime1 = os.stat(source)[ST_MTIME]
    mtime2 = os.stat(target)[ST_MTIME]

    return mtime1 > mtime2

# get_thunk_type_set函数：生成数据类型的笛卡尔积，并包含一个获取器例程
def get_thunk_type_set():
    """
    Get a list containing cartesian product of data types, plus a getter routine.

    Returns
    -------
    i_types : list [(j, I_typenum, None, I_type, None), ...]
         Pairing of index type numbers and the corresponding C++ types,
         and an unique index `j`. This is for routines that are parameterized
         only by I but not by T.
    """
    # 初始化空列表，用于存储参数化了T和I的例程类型信息的元组列表
    it_types : list [(j, I_typenum, T_typenum, I_type, T_type), ...]
         Same as `i_types`, but for routines parameterized both by T and I.
         
    # 初始化空列表，用于存储参数化了I的例程类型信息的元组列表
    i_types = []

    # 初始化计数器j，用于唯一索引
    j = 0

    # 初始化getter_code字符串，表示C++函数的代码模板
    getter_code = "    if (0) {}"

    # 遍历I_TYPES中的每个元素，生成对应的C++代码片段
    for I_typenum, I_type in I_TYPES:
        # 构建I_typenum的匹配部分代码
        piece = """
        else if (I_typenum == %(I_typenum)s) {
            if (T_typenum == -1) { return %(j)s; }"""
        getter_code += piece % dict(I_typenum=I_typenum, j=j)

        # 将当前I_typenum的信息添加到i_types列表中
        i_types.append((j, I_typenum, None, I_type, None))
        j += 1

        # 遍历T_TYPES中的每个元素，生成对应的C++代码片段
        for T_typenum, T_type in T_TYPES:
            # 构建T_typenum的匹配部分代码
            piece = """
            else if (T_typenum == %(T_typenum)s) { return %(j)s; }"""
            getter_code += piece % dict(T_typenum=T_typenum, j=j)

            # 将当前(I_typenum, T_typenum)的信息添加到it_types列表中
            it_types.append((j, I_typenum, T_typenum, I_type, T_type))
            j += 1

        # 完成当前I_typenum的代码块
        getter_code += """
        }"""

    # 返回结果列表i_types, it_types和GET_THUNK_CASE_TEMPLATE格式化的getter_code
    return i_types, it_types, GET_THUNK_CASE_TEMPLATE % dict(content=getter_code)
# 为给定的例程生成 thunk 和方法代码

def parse_routine(name, args, types):
    """
    生成指定例程的 thunk 和方法代码。

    Parameters
    ----------
    name : str
        C++例程的名称
    args : str
        参数列表规范（格式如上所述）
    types : list
        要实例化的类型列表，由 `get_thunk_type_set` 返回

    """

    # 从参数规范中获取返回类型和参数类型列表
    ret_spec = args[0]
    arg_spec = args[1:]

    def get_arglist(I_type, T_type):
        """
        生成调用C++函数的参数列表
        """
        args = []
        next_is_writeable = False
        j = 0
        for t in arg_spec:
            const = '' if next_is_writeable else 'const '
            next_is_writeable = False
            if t == '*':
                next_is_writeable = True
                continue
            elif t == 'i':
                args.append("*(%s*)a[%d]" % (const + I_type, j))
            elif t == 'I':
                args.append("(%s*)a[%d]" % (const + I_type, j))
            elif t == 'T':
                args.append("(%s*)a[%d]" % (const + T_type, j))
            elif t == 'B':
                args.append("(npy_bool_wrapper*)a[%d]" % (j,))
            elif t == 'V':
                if const:
                    raise ValueError("'V' argument must be an output arg")
                args.append("(std::vector<%s>*)a[%d]" % (I_type, j,))
            elif t == 'W':
                if const:
                    raise ValueError("'W' argument must be an output arg")
                args.append("(std::vector<%s>*)a[%d]" % (T_type, j,))
            elif t == 'l':
                args.append("*(%snpy_int64*)a[%d]" % (const, j))
            else:
                raise ValueError(f"Invalid spec character {t!r}")
            j += 1
        return ", ".join(args)

    # 生成 thunk 代码：一个包含不同类型组合的巨大 switch 语句
    thunk_content = """int j = get_thunk_case(I_typenum, T_typenum);
    switch (j) {"""
    for j, I_typenum, T_typenum, I_type, T_type in types:
        arglist = get_arglist(I_type, T_type)

        piece = """
        case %(j)s:"""
        if ret_spec == 'v':
            piece += """
            (void)%(name)s(%(arglist)s);
            return 0;"""
        else:
            piece += """
            return %(name)s(%(arglist)s);"""
        thunk_content += piece % dict(j=j, I_type=I_type, T_type=T_type,
                                      I_typenum=I_typenum, T_typenum=T_typenum,
                                      arglist=arglist, name=name)

    # 添加默认情况，如果类型编号无效则抛出运行时错误
    thunk_content += """
    default:
        throw std::runtime_error("internal error: invalid argument typenums");
    }"""

    # 使用 THUNK_TEMPLATE 构建 thunk 代码
    thunk_code = THUNK_TEMPLATE % dict(name=name,
                                       thunk_content=thunk_content)

    # 使用 METHOD_TEMPLATE 构建方法代码
    method_code = METHOD_TEMPLATE % dict(name=name,
                                         ret_spec=ret_spec,
                                         arg_spec=arg_spec)
    # 返回两个变量 thunk_code 和 method_code
    return thunk_code, method_code
def main():
    # 解析命令行参数
    p = argparse.ArgumentParser(usage=(__doc__ or '').strip())

    # 添加命令行选项：--no-force，表示不强制生成，将其设置为False并指定默认值为True
    p.add_argument("--no-force", action="store_false",
                   dest="force", default=True)
    
    # 添加命令行选项：-o 或 --outdir，指定输出目录的相对路径
    p.add_argument("-o", "--outdir", type=str,
                   help="Relative path to the output directory")
    
    # 解析命令行参数
    options = p.parse_args()

    # 存储已解析的函数名列表
    names = []

    # 获取 thunk 类型集合和相应的 getter 代码
    i_types, it_types, getter_code = get_thunk_type_set()

    # 为每个编译单元生成 *_impl.h 文件
    for unit_name, routines in COMPILATION_UNITS:
        thunks = []
        methods = []

        # 为所有 routine 生成 thunk 和 method
        for line in routines.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                # 解析行中的函数名和参数
                name, args = line.split(None, 1)
            except ValueError as e:
                raise ValueError(f"Malformed line: {line!r}") from e

            # 删除参数中的空白字符并检查 thunk 类型
            args = "".join(args.split())
            if 't' in args or 'T' in args:
                thunk, method = parse_routine(name, args, it_types)
            else:
                thunk, method = parse_routine(name, args, i_types)

            # 检查函数名是否重复
            if name in names:
                raise ValueError(f"Duplicate routine {name!r}")

            # 将函数名添加到列表中
            names.append(name)
            thunks.append(thunk)
            methods.append(method)

        # 生成输出文件
        if options.outdir:
            # 如果指定了输出目录，则设置 outdir 为当前工作目录下的 options.outdir
            outdir = os.path.join(os.getcwd(), options.outdir)

        # 设置目标文件路径
        dst = os.path.join(outdir,
                           unit_name + '_impl.h')

        # 检查源文件是否比目标文件更新，或者是否强制生成
        if newer(__file__, dst) or options.force:
            if not options.outdir:
                # 如果未指定输出目录，且使用 Meson，则静默生成，未来可以添加 --verbose 选项
                print(f"[generate_sparsetools] generating {dst!r}")
            with open(dst, 'w') as f:
                # 写入自动生成的说明
                write_autogen_blurb(f)
                # 写入 getter_code
                f.write(getter_code)
                # 分别写入所有 thunks 和 methods
                for thunk in thunks:
                    f.write(thunk)
                for method in methods:
                    f.write(method)
        else:
            if not options.outdir:
                # 如果未指定输出目录，且使用 Meson，则静默不生成
                print(f"[generate_sparsetools] {dst!r} already up-to-date")

    # 生成方法结构的代码
    method_defs = ""
    for name in names:
        # 生成每个方法的声明
        method_defs += (f"PyObject *{name}"
                        f"_method(PyObject *, PyObject *);\n")

    method_struct = """\nstatic struct PyMethodDef sparsetools_methods[] = {"""
    for name in names:
        # 生成方法结构体的初始化
        method_struct += ("""
            {{"{name}", (PyCFunction){name}_method, METH_VARARGS, NULL}},"""
            .format(**dict(name=name))
        )
    method_struct += """
        {NULL, NULL, 0, NULL}
    };"""

    # 生成 sparsetools_impl.h 文件路径
    dst = os.path.join(outdir, 'sparsetools_impl.h')
    # 检查当前文件是否比目标文件新，或者强制更新选项被设置
    if newer(__file__, dst) or options.force:
        # 如果不指定输出目录，则打印生成消息，针对生成稀疏工具
        if not options.outdir:
            print(f"[generate_sparsetools] generating {dst!r}")
        # 使用写入模式打开目标文件，准备写入内容
        with open(dst, 'w') as f:
            # 写入自动生成的说明文本
            write_autogen_blurb(f)
            # 写入方法定义
            f.write(method_defs)
            # 写入方法结构
            f.write(method_struct)
    else:
        # 如果不指定输出目录，则打印已经是最新消息，针对生成稀疏工具
        if not options.outdir:
            print(f"[generate_sparsetools] {dst!r} already up-to-date")
# 定义一个函数，用于向给定的流对象写入自动生成的注释块
def write_autogen_blurb(stream):
    # 向流对象写入自动生成的注释块，包括生成工具的信息和禁止手动编辑或版本控制系统中检入的警告
    stream.write("""\
/* This file is autogenerated by generate_sparsetools.py
 * Do not edit manually or check into VCS.
 */
""")


# 主程序入口，检查是否在主模块中执行
if __name__ == "__main__":
    # 调用主函数 main()，这里假设主函数是在其他地方定义的
    main()
```