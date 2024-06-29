# `.\numpy\numpy\f2py\capi_maps.py`

```py
"""
Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
# 导入当前模块的版本信息
from . import __version__
# 获取当前版本号
f2py_version = __version__.version

# 导入标准库模块
import copy
import re
import os
# 导入当前包的特定模块
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map

# auxfuncs.py 中的环境函数对于某些 eval 调用是必需的。
# 由于这些函数无法通过静态代码分析确定，因此在 f2py 进行重大重构之前，
# 使用 import * 是最安全的做法。
from .auxfuncs import *

# 定义 __all__ 列表，指定模块中公开的符号
__all__ = [
    'getctype', 'getstrlength', 'getarrdims', 'getpydocsign',
    'getarrdocsign', 'getinit', 'sign2map', 'routsign2map', 'modsign2map',
    'cb_sign2map', 'cb_routsign2map', 'common_sign2map', 'process_f2cmap_dict'
]

# 初始化几个空列表和字典
depargs = []
lcb_map = {}
lcb2_map = {}

# 强制类型转换映射表：主要是因为 Python 或 Numeric C/API 不支持相应的 C 类型而导致的强制转换。
c2py_map = {'double': 'float',
            'float': 'float',                          # 强制类型转换
            'long_double': 'float',                    # 强制类型转换
            'char': 'int',                             # 强制类型转换
            'signed_char': 'int',                      # 强制类型转换
            'unsigned_char': 'int',                    # 强制类型转换
            'short': 'int',                            # 强制类型转换
            'unsigned_short': 'int',                   # 强制类型转换
            'int': 'int',                              # 强制类型转换
            'long': 'int',
            'long_long': 'long',
            'unsigned': 'int',                         # 强制类型转换
            'complex_float': 'complex',                # 强制类型转换
            'complex_double': 'complex',
            'complex_long_double': 'complex',          # 强制类型转换
            'string': 'string',
            'character': 'bytes',
            }

# C 类型到 C API 映射表
c2capi_map = {'double': 'NPY_DOUBLE',
                'float': 'NPY_FLOAT',
                'long_double': 'NPY_LONGDOUBLE',
                'char': 'NPY_BYTE',
                'unsigned_char': 'NPY_UBYTE',
                'signed_char': 'NPY_BYTE',
                'short': 'NPY_SHORT',
                'unsigned_short': 'NPY_USHORT',
                'int': 'NPY_INT',
                'unsigned': 'NPY_UINT',
                'long': 'NPY_LONG',
                'unsigned_long': 'NPY_ULONG',
                'long_long': 'NPY_LONGLONG',
                'unsigned_long_long': 'NPY_ULONGLONG',
                'complex_float': 'NPY_CFLOAT',
                'complex_double': 'NPY_CDOUBLE',
                'complex_long_double': 'NPY_CDOUBLE',
                'string': 'NPY_STRING',
                'character': 'NPY_STRING'}
# 映射C语言类型到Python格式化字符的字典
c2pycode_map = {'double': 'd',
                'float': 'f',
                'long_double': 'g',
                'char': 'b',
                'unsigned_char': 'B',
                'signed_char': 'b',
                'short': 'h',
                'unsigned_short': 'H',
                'int': 'i',
                'unsigned': 'I',
                'long': 'l',
                'unsigned_long': 'L',
                'long_long': 'q',
                'unsigned_long_long': 'Q',
                'complex_float': 'F',
                'complex_double': 'D',
                'complex_long_double': 'G',
                'string': 'S',
                'character': 'c'}

# 根据Python C API中的要求，将C类型映射到构建值的格式化字符
c2buildvalue_map = {'double': 'd',
                    'float': 'f',
                    'char': 'b',
                    'signed_char': 'b',
                    'short': 'h',
                    'int': 'i',
                    'long': 'l',
                    'long_long': 'L',
                    'complex_float': 'N',
                    'complex_double': 'N',
                    'complex_long_double': 'N',
                    'string': 'y',
                    'character': 'c'}

# 包含了Fortran到C类型映射的全局字典
f2cmap_all = {'real': {'': 'float', '4': 'float', '8': 'double',
                       '12': 'long_double', '16': 'long_double'},
              'integer': {'': 'int', '1': 'signed_char', '2': 'short',
                          '4': 'int', '8': 'long_long',
                          '-1': 'unsigned_char', '-2': 'unsigned_short',
                          '-4': 'unsigned', '-8': 'unsigned_long_long'},
              'complex': {'': 'complex_float', '8': 'complex_float',
                          '16': 'complex_double', '24': 'complex_long_double',
                          '32': 'complex_long_double'},
              'complexkind': {'': 'complex_float', '4': 'complex_float',
                              '8': 'complex_double', '12': 'complex_long_double',
                              '16': 'complex_long_double'},
              'logical': {'': 'int', '1': 'char', '2': 'short', '4': 'int',
                          '8': 'long_long'},
              'double complex': {'': 'complex_double'},
              'double precision': {'': 'double'},
              'byte': {'': 'char'},
              }

# 将ISO C处理后的映射添加到c2pycode_map和c2py_map中
c2pycode_map.update(isoc_c2pycode_map)
c2py_map.update(iso_c2py_map)
# 处理Fortran到C类型映射，返回更新后的f2cmap_all和空列表
f2cmap_all, _ = process_f2cmap_dict(f2cmap_all, iso_c_binding_map, c2py_map)
# ISO_C处理结束

# 深拷贝f2cmap_all到f2cmap_default
f2cmap_default = copy.deepcopy(f2cmap_all)

# 初始化f2cmap_mapped为空列表
f2cmap_mapped = []

# 加载f2cmap_file中的映射数据到f2cmap_all和f2cmap_mapped
def load_f2cmap_file(f2cmap_file):
    global f2cmap_all, f2cmap_mapped

    # 深拷贝f2cmap_default到f2cmap_all
    f2cmap_all = copy.deepcopy(f2cmap_default)

    # 如果f2cmap_file为None，则使用默认文件'.f2py_f2cmap'
    if f2cmap_file is None:
        # 默认文件路径
        f2cmap_file = '.f2py_f2cmap'
        # 如果默认文件不存在，则返回
        if not os.path.isfile(f2cmap_file):
            return

    # 用户自定义的f2cmap_all的添加部分
    # f2cmap_file必须包含一个仅为字典的字典
    # 结尾处缺少完整的注释
    # 尝试读取 f2cmap_file 文件，并将其内容转换为小写后使用 eval 函数进行求值
    try:
        # 输出消息，指示正在从 f2cmap_file 中读取 f2cmap
        outmess('Reading f2cmap from {!r} ...\n'.format(f2cmap_file))
        # 打开文件 f2cmap_file 并将其内容转换为小写后求值
        with open(f2cmap_file) as f:
            d = eval(f.read().lower(), {}, {})
        # 对 f2cmap 进行处理，应用到全局和映射的 f2cmap_all 中，并更新 c2py_map
        f2cmap_all, f2cmap_mapped = process_f2cmap_dict(f2cmap_all, d, c2py_map, True)
        # 输出成功应用用户定义的 f2cmap 更改的消息
        outmess('Successfully applied user defined f2cmap changes\n')
    # 捕获任何异常并记录错误消息，指示无法应用用户定义的 f2cmap 更改
    except Exception as msg:
        errmess('Failed to apply user defined f2cmap changes: %s. Skipping.\n' % (msg))
# 定义一个映射表，将 C 类型映射为格式化字符串
cformat_map = {'double': '%g',
               'float': '%g',
               'long_double': '%Lg',
               'char': '%d',
               'signed_char': '%d',
               'unsigned_char': '%hhu',
               'short': '%hd',
               'unsigned_short': '%hu',
               'int': '%d',
               'unsigned': '%u',
               'long': '%ld',
               'unsigned_long': '%lu',
               'long_long': '%ld',
               'complex_float': '(%g,%g)',
               'complex_double': '(%g,%g)',
               'complex_long_double': '(%Lg,%Lg)',
               'string': '\\"%s\\"',
               'character': "'%c'",
               }

# 辅助函数

def getctype(var):
    """
    根据变量类型信息确定其对应的 C 类型

    参数:
    var -- 包含类型信息的变量

    返回:
    ctype -- 变量的 C 类型
    """
    ctype = 'void'
    if isfunction(var):  # 如果是函数类型
        if 'result' in var:
            a = var['result']
        else:
            a = var['name']
        if a in var['vars']:
            return getctype(var['vars'][a])
        else:
            errmess('getctype: function %s has no return value?!\n' % a)
    elif issubroutine(var):  # 如果是子程序类型
        return ctype
    elif ischaracter_or_characterarray(var):  # 如果是字符或字符数组类型
        return 'character'
    elif isstring_or_stringarray(var):  # 如果是字符串或字符串数组类型
        return 'string'
    elif 'typespec' in var and var['typespec'].lower() in f2cmap_all:
        # 根据类型规范从映射表中获取 C 类型
        typespec = var['typespec'].lower()
        f2cmap = f2cmap_all[typespec]
        ctype = f2cmap['']  # 默认类型
        if 'kindselector' in var:
            if '*' in var['kindselector']:
                try:
                    ctype = f2cmap[var['kindselector']['*']]
                except KeyError:
                    errmess('getctype: "%s %s %s" not supported.\n' %
                            (var['typespec'], '*', var['kindselector']['*']))
            elif 'kind' in var['kindselector']:
                if typespec + 'kind' in f2cmap_all:
                    f2cmap = f2cmap_all[typespec + 'kind']
                try:
                    ctype = f2cmap[var['kindselector']['kind']]
                except KeyError:
                    if typespec in f2cmap_all:
                        f2cmap = f2cmap_all[typespec]
                    try:
                        ctype = f2cmap[str(var['kindselector']['kind'])]
                    except KeyError:
                        errmess('getctype: "%s(kind=%s)" is mapped to C "%s" (to override define dict(%s = dict(%s="<C typespec>")) in %s/.f2py_f2cmap file).\n'
                                % (typespec, var['kindselector']['kind'], ctype,
                                   typespec, var['kindselector']['kind'], os.getcwd()))
    else:
        if not isexternal(var):
            errmess('getctype: No C-type found in "%s", assuming void.\n' % var)
    return ctype


def f2cexpr(expr):
    """
    将 Fortran 表达式重写为 f2py 支持的 C 表达式

    由于 f2py 缺乏合适的表达式解析器，该函数使用一种启发式方法，
    假设 Fortran
    # 替换表达式中的 Fortran `len` 函数为 `f2py_slen`，用于支持 Fortran 到 C/C++ 的映射
    """
    arithmetic expressions are valid C arithmetic expressions when
    mapping Fortran function calls to the corresponding C function/CPP
    macros calls.
    
    """
    expr = re.sub(r'\blen\b', 'f2py_slen', expr)
    # 返回替换后的表达式
    return expr
# 如果变量是字符串类型的函数
def getstrlength(var):
    if isstringfunction(var):  # 检查变量是否是字符串类型的函数
        if 'result' in var:  # 如果变量包含'result'字段
            a = var['result']  # 将'result'字段赋给变量a
        else:
            a = var['name']  # 否则将'name'字段赋给变量a
        if a in var['vars']:  # 如果a存在于变量的'vars'字段中
            return getstrlength(var['vars'][a])  # 递归调用getstrlength函数，传入a对应的变量
        else:
            errmess('getstrlength: function %s has no return value?!\n' % a)  # 报告错误，函数没有返回值
    if not isstring(var):  # 如果变量不是字符串类型
        errmess(  # 报告错误，期望一个字符串的签名，但实际得到了其他类型
            'getstrlength: expected a signature of a string but got: %s\n' % (repr(var)))
    len = '1'  # 将长度设置为默认值'1'
    if 'charselector' in var:  # 如果变量中包含'charselector'字段
        a = var['charselector']  # 将'charselector'字段赋给变量a
        if '*' in a:  # 如果a中包含'*'
            len = a['*']  # 将'*'对应的值赋给len
        elif 'len' in a:  # 否则如果a中包含'len'
            len = f2cexpr(a['len'])  # 将'len'对应的值转换成C表达式并赋给len
    # 如果len符合预期的正则表达式条件，设置长度为'-1'
    if re.match(r'\(\s*(\*|:)\s*\)', len) or re.match(r'(\*|:)', len):
        if isintent_hide(var):  # 如果意图是隐藏
            errmess('getstrlength:intent(hide): expected a string with defined length but got: %s\n' % (
                repr(var)))  # 报告错误，期望一个具有定义长度的字符串，但实际得到其他类型
        len = '-1'  # 设置长度为'-1'
    return len  # 返回计算得到的长度


# 获取数组维度信息
def getarrdims(a, var, verbose=0):
    ret = {}  # 创建一个空字典作为返回结果
    if isstring(var) and not isarray(var):  # 如果变量是字符串并且不是数组
        ret['size'] = getstrlength(var)  # 获取字符串的长度信息
        ret['rank'] = '0'  # 数组的秩为0，表示不是数组
        ret['dims'] = ''  # 维度为空字符串，表示不是数组
    elif isscalar(var):  # 如果变量是标量（非数组，非字符串）
        ret['size'] = '1'  # 大小为1，表示单个元素
        ret['rank'] = '0'  # 数组的秩为0，表示不是数组
        ret['dims'] = ''  # 维度为空字符串，表示不是数组
    # 如果变量是数组，执行以下操作
    elif isarray(var):
        # 复制数组的维度信息
        dim = copy.copy(var['dimension'])
        # 将维度转换为字符串，并用 '*' 连接起来，赋给返回字典中的 'size' 键
        ret['size'] = '*'.join(dim)
        try:
            # 尝试使用 eval 函数计算 'size' 字符串表达式的值，并将结果转换为字符串
            ret['size'] = repr(eval(ret['size']))
        except Exception:
            # 如果计算失败，则忽略异常
            pass
        # 将维度列表转换为以逗号分隔的字符串，赋给返回字典中的 'dims' 键
        ret['dims'] = ','.join(dim)
        # 计算维度列表的长度，并将结果转换为字符串，赋给返回字典中的 'rank' 键
        ret['rank'] = repr(len(dim))
        # 根据维度列表生成一个以 -1 填充的列表，并将结果转换为字符串后去掉首尾的方括号，赋给返回字典中的 'rank*[-1]' 键
        ret['rank*[-1]'] = repr(len(dim) * [-1])[1:-1]
        
        # 遍历维度列表，解决依赖关系
        for i in range(len(dim)):  # solve dim for dependencies
            v = []
            # 如果当前维度在依赖参数列表中，则将其添加到 v 中
            if dim[i] in depargs:
                v = [dim[i]]
            else:
                # 否则，遍历依赖参数列表，查找与当前维度匹配的参数，并将其添加到 v 中
                for va in depargs:
                    if re.match(r'.*?\b%s\b.*' % va, dim[i]):
                        v.append(va)
            # 遍历 v 中的每个参数
            for va in v:
                # 如果当前参数在依赖参数列表中的位置在 a 的后面，则将当前维度设置为 '*'
                if depargs.index(va) > depargs.index(a):
                    dim[i] = '*'
                    break
        
        # 初始化 'setdims' 和计数器 i
        ret['setdims'], i = '', -1
        # 遍历维度列表 dim
        for d in dim:
            i = i + 1
            # 如果当前维度不在 ['*', ':', '(*)', '(:)'] 中
            if d not in ['*', ':', '(*)', '(:)']:
                # 将格式化后的字符串添加到 'setdims' 中，用于后续变量名的维度设置
                ret['setdims'] = '%s#varname#_Dims[%d]=%s,' % (
                    ret['setdims'], i, d)
        # 如果 'setdims' 不为空，则去掉末尾的逗号
        if ret['setdims']:
            ret['setdims'] = ret['setdims'][:-1]
        
        # 初始化 'cbsetdims' 和计数器 i
        ret['cbsetdims'], i = '', -1
        # 遍历数组的维度信息列表
        for d in var['dimension']:
            i = i + 1
            # 如果当前维度不在 ['*', ':', '(*)', '(:)'] 中
            if d not in ['*', ':', '(*)', '(:)']:
                # 将格式化后的字符串添加到 'cbsetdims' 中，用于回调函数中变量名的维度设置
                ret['cbsetdims'] = '%s#varname#_Dims[%d]=%s,' % (
                    ret['cbsetdims'], i, d)
            # 如果当前变量是输入参数，并且维度为 '*'
            elif isintent_in(var):
                # 输出警告信息，表示假定为定形数组，并用 0 替换当前维度
                outmess('getarrdims:warning: assumed shape array, using 0 instead of %r\n'
                        % (d))
                # 将格式化后的字符串添加到 'cbsetdims' 中，用于回调函数中变量名的维度设置
                ret['cbsetdims'] = '%s#varname#_Dims[%d]=%s,' % (
                    ret['cbsetdims'], i, 0)
            # 如果开启了详细输出
            elif verbose:
                # 输出错误信息，表示在回调函数中，数组参数 %s 必须有有界维度，但当前为 %s
                errmess(
                    'getarrdims: If in call-back function: array argument %s must have bounded dimensions: got %s\n' % (repr(a), repr(d)))
        # 如果 'cbsetdims' 不为空，则去掉末尾的逗号
        if ret['cbsetdims']:
            ret['cbsetdims'] = ret['cbsetdims'][:-1]
# 返回函数结果
    return ret


def getpydocsign(a, var):
    global lcb_map
    # 如果变量是函数
    if isfunction(var):
        # 如果函数有'result'属性，将其赋值给af，否则将函数名赋值给af
        if 'result' in var:
            af = var['result']
        else:
            af = var['name']
        # 如果函数名在var['vars']中
        if af in var['vars']:
            # 递归调用getpydocsign函数，处理函数返回值的情况
            return getpydocsign(af, var['vars'][af])
        else:
            # 报错，函数%s没有返回值?!
            errmess('getctype: function %s has no return value?!\\n' % af)
        return '', ''
    
    # 默认签名和输出签名初始值为a
    sig, sigout = a, a
    opt = ''
    # 如果变量是输入类型
    if isintent_in(var):
        opt = 'input'
    # 如果变量是输入输出类型
    elif isintent_inout(var):
        opt = 'in/output'
    
    out_a = a
    # 如果变量是输出类型
    if isintent_out(var):
        # 遍历var['intent']，查找以'out='开头的键
        for k in var['intent']:
            if k[:4] == 'out=':
                out_a = k[4:]  # 获取输出变量名
                break
    
    init = ''
    ctype = getctype(var)
    # 如果变量有初始值
    if hasinitvalue(var):
        # 调用getinit函数获取初始值和显示的初始值
        init, showinit = getinit(a, var)
        init = ', optional\\n    Default: %s' % showinit
    
    # 如果变量是标量类型
    if isscalar(var):
        if isintent_inout(var):
            # 设置标量的签名
            sig = '%s : %s rank-0 array(%s,\'%s\')%s' % (a, opt, c2py_map[ctype],
                                                         c2pycode_map[ctype], init)
        else:
            # 设置标量的签名
            sig = '%s : %s %s%s' % (a, opt, c2py_map[ctype], init)
        # 设置标量的输出签名
        sigout = '%s : %s' % (out_a, c2py_map[ctype])
    
    # 如果变量是字符串类型
    elif isstring(var):
        if isintent_inout(var):
            # 设置字符串的签名
            sig = '%s : %s rank-0 array(string(len=%s),\'c\')%s' % (
                a, opt, getstrlength(var), init)
        else:
            # 设置字符串的签名
            sig = '%s : %s string(len=%s)%s' % (
                a, opt, getstrlength(var), init)
        # 设置字符串的输出签名
        sigout = '%s : string(len=%s)' % (out_a, getstrlength(var))
    
    # 如果变量是数组类型
    elif isarray(var):
        dim = var['dimension']
        rank = repr(len(dim))
        # 设置数组的签名
        sig = '%s : %s rank-%s array(\'%s\') with bounds (%s)%s' % (a, opt, rank,
                                                                    c2pycode_map[
                                                                        ctype],
                                                                    ','.join(dim), init)
        # 如果输入变量名和输出变量名相同
        if a == out_a:
            # 设置数组的输出签名
            sigout = '%s : rank-%s array(\'%s\') with bounds (%s)'\
                % (a, rank, c2pycode_map[ctype], ','.join(dim))
        else:
            # 设置数组的输出签名，包括存储信息
            sigout = '%s : rank-%s array(\'%s\') with bounds (%s) and %s storage'\
                % (out_a, rank, c2pycode_map[ctype], ','.join(dim), a)
    
    # 如果变量是外部变量类型
    elif isexternal(var):
        ua = ''
        # 如果变量在lcb_map中，并且映射存在于lcb2_map中，并且lcb2_map[lcb_map[a]]中有'argname'键
        if a in lcb_map and lcb_map[a] in lcb2_map and 'argname' in lcb2_map[lcb_map[a]]:
            ua = lcb2_map[lcb_map[a]]['argname']
            # 如果ua不等于a，设置ua为'=> %s'
            if not ua == a:
                ua = ' => %s' % ua
            else:
                ua = ''
        # 设置外部变量的签名
        sig = '%s : call-back function%s' % (a, ua)
        sigout = sig
    
    else:
        # 报错，无法解析%s的文档签名
        errmess(
            'getpydocsign: Could not resolve docsignature for "%s".\\n' % a)
    
    # 返回签名和输出签名
    return sig, sigout


def getarrdocsign(a, var):
    ctype = getctype(var)
    # 如果变量 var 是字符串并且不是数组
    if isstring(var) and (not isarray(var)):
        # 根据变量 var 的长度获取字符串的长度信息，构造数组签名字符串
        sig = '%s : rank-0 array(string(len=%s),\'c\')' % (a,
                                                           getstrlength(var))
    
    # 如果变量 var 是标量
    elif isscalar(var):
        # 根据变量的类型获取对应的 Python 类型和代码映射，构造数组签名字符串
        sig = '%s : rank-0 array(%s,\'%s\')' % (a, c2py_map[ctype],
                                                c2pycode_map[ctype],)
    
    # 如果变量 var 是数组
    elif isarray(var):
        # 获取变量 var 的维度信息
        dim = var['dimension']
        # 获取数组的维度数并转换为字符串形式
        rank = repr(len(dim))
        # 根据变量的类型获取对应的代码映射，构造数组签名字符串
        sig = '%s : rank-%s array(\'%s\') with bounds (%s)' % (a, rank,
                                                               c2pycode_map[
                                                                   ctype],
                                                               ','.join(dim))
    
    # 返回构造好的数组签名字符串
    return sig
# 定义函数 getinit，接收两个参数 a 和 var
def getinit(a, var):
    # 如果 var 是字符串类型
    if isstring(var):
        # 初始化 init 和 showinit 分别为空字符串和单引号空字符串
        init, showinit = '""', "''"
    else:
        # 否则初始化 init 和 showinit 为空字符串
        init, showinit = '', ''
    
    # 如果 var 具有初始值
    if hasinitvalue(var):
        # 将 init 设置为 var['=']，showinit 设置为 init
        init = var['=']
        showinit = init
        
        # 如果 var 是复数或复数数组
        if iscomplex(var) or iscomplexarray(var):
            # 初始化一个空字典 ret
            ret = {}

            try:
                # 尝试获取 var["="] 的值并处理
                v = var["="]
                # 如果值包含逗号
                if ',' in v:
                    # 将去除首尾括号后的字符串通过 '@,@' 分隔，并存入 ret 字典
                    ret['init.r'], ret['init.i'] = markoutercomma(
                        v[1:-1]).split('@,@')
                else:
                    # 否则将值解析为表达式，并分别获取实部和虚部
                    v = eval(v, {}, {})
                    ret['init.r'], ret['init.i'] = str(v.real), str(v.imag)
            except Exception:
                # 如果出错则抛出 ValueError 异常
                raise ValueError(
                    'getinit: expected complex number `(r,i)\' but got `%s\' as initial value of %r.' % (init, a))
            
            # 如果 var 是数组类型，则重新设置 init 为复数格式字符串
            if isarray(var):
                init = '(capi_c.r=%s,capi_c.i=%s,capi_c)' % (
                    ret['init.r'], ret['init.i'])
        
        # 如果 var 是字符串类型
        elif isstring(var):
            # 如果 init 是空的，将其设置为双引号包裹的 var["="] 值的处理字符串，同时设置 showinit 为单引号包裹
            if not init:
                init, showinit = '""', "''"
            if init[0] == "'":
                init = '"%s"' % (init[1:-1].replace('"', '\\"'))
            if init[0] == '"':
                showinit = "'%s'" % (init[1:-1])
    
    # 返回 init 和 showinit
    return init, showinit
    # 如果变量是外部回调函数
    if isexternal(var):
        # 将回调函数的名称映射到 ret 字典中的 'cbnamekey'
        ret['cbnamekey'] = a
        # 如果回调函数在 lcb_map 中存在
        if a in lcb_map:
            # 设置 ret 字典中的回调函数名称为 lcb_map 中对应的值
            ret['cbname'] = lcb_map[a]
            # 设置最大参数数目为 lcb_map 对应值在 lcb2_map 中的最大参数数目
            ret['maxnofargs'] = lcb2_map[lcb_map[a]]['maxnofargs']
            # 设置可选参数数目为 lcb_map 对应值在 lcb2_map 中的可选参数数目
            ret['nofoptargs'] = lcb2_map[lcb_map[a]]['nofoptargs']
            # 设置回调函数的文档字符串为 lcb_map 对应值在 lcb2_map 中的文档字符串
            ret['cbdocstr'] = lcb2_map[lcb_map[a]]['docstr']
            # 设置回调函数的 LaTeX 文档字符串为 lcb_map 对应值在 lcb2_map 中的 LaTeX 文档字符串
            ret['cblatexdocstr'] = lcb2_map[lcb_map[a]]['latexdocstr']
        else:
            # 如果回调函数在 lcb_map 中不存在，设置 ret 字典中的回调函数名称为 a
            ret['cbname'] = a
            # 输出错误信息，说明外部回调函数 a 不在 lcb_map 中
            errmess('sign2map: Confused: external %s is not in lcb_map%s.\n' % (
                a, list(lcb_map.keys())))
    
    # 如果变量是字符串类型
    if isstring(var):
        # 设置 ret 字典中的 'length' 键为字符串 var 的长度
        ret['length'] = getstrlength(var)
    
    # 如果变量是数组类型
    if isarray(var):
        # 将 ret 字典更新为包含数组维度信息的新字典
        ret = dictappend(ret, getarrdims(a, var))
        # 复制数组变量的维度到 dim 变量中
        dim = copy.copy(var['dimension'])
    
    # 如果 ret 字典中的 'ctype' 键存在于 c2capi_map 中
    if ret['ctype'] in c2capi_map:
        # 设置 ret 字典中的 'atype' 键为 c2capi_map 中 'ctype' 键对应的值
        ret['atype'] = c2capi_map[ret['ctype']]
        # 设置 ret 字典中的 'elsize' 键为变量的元素大小
        ret['elsize'] = get_elsize(var)
    
    # 调试信息
    if debugcapi(var):
        # 定义调试信息列表 il
        il = [isintent_in, 'input', isintent_out, 'output',
              isintent_inout, 'inoutput', isrequired, 'required',
              isoptional, 'optional', isintent_hide, 'hidden',
              iscomplex, 'complex scalar',
              l_and(isscalar, l_not(iscomplex)), 'scalar',
              isstring, 'string', isarray, 'array',
              iscomplexarray, 'complex array', isstringarray, 'string array',
              iscomplexfunction, 'complex function',
              l_and(isfunction, l_not(iscomplexfunction)), 'function',
              isexternal, 'callback',
              isintent_callback, 'callback',
              isintent_aux, 'auxiliary',
              ]
        # 初始化结果列表 rl
        rl = []
        # 遍历 il 列表，每两个元素一组判断是否符合条件，并添加到 rl 中
        for i in range(0, len(il), 2):
            if il[i](var):
                rl.append(il[i + 1])
        # 如果变量是字符串类型，添加字符串长度信息到 rl
        if isstring(var):
            rl.append('slen(%s)=%s' % (a, ret['length']))
        # 如果变量是数组类型，添加数组维度信息到 rl
        if isarray(var):
            ddim = ','.join(
                map(lambda x, y: '%s|%s' % (x, y), var['dimension'], dim))
            rl.append('dims(%s)' % ddim)
        # 如果变量是外部回调函数，设置调试信息
        if isexternal(var):
            ret['vardebuginfo'] = 'debug-capi:%s=>%s:%s' % (
                a, ret['cbname'], ','.join(rl))
        else:
            # 设置其他类型变量的调试信息
            ret['vardebuginfo'] = 'debug-capi:%s %s=%s:%s' % (
                ret['ctype'], a, ret['showinit'], ','.join(rl))
        # 如果变量是标量，根据 'ctype' 设置调试显示值
        if isscalar(var):
            if ret['ctype'] in cformat_map:
                ret['vardebugshowvalue'] = 'debug-capi:%s=%s' % (
                    a, cformat_map[ret['ctype']])
        # 如果变量是字符串，设置调试显示值为字符串长度和值的格式化字符串
        if isstring(var):
            ret['vardebugshowvalue'] = 'debug-capi:slen(%s)=%s %s="%s"' % (
                a, ret['length'], a, a)
        # 如果变量是外部回调函数，设置调试显示值
        if isexternal(var):
            ret['vardebugshowvalue'] = 'debug-capi:%s=%p' % (a)
    
    # 如果 ret 字典中的 'ctype' 键存在于 cformat_map 中
    if ret['ctype'] in cformat_map:
        # 设置 ret 字典中的 'varshowvalue' 键为格式化后的值的显示名称
        ret['varshowvalue'] = '#name#:%s=%s' % (a, cformat_map[ret['ctype']])
        # 设置 ret 字典中的 'showvalueformat' 键为格式化后的值的显示格式
        ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
    
    # 如果变量是字符串，设置 ret 字典中的 'varshowvalue' 键为字符串长度和字符串值的格式化字符串
    if isstring(var):
        ret['varshowvalue'] = '#name#:slen(%s)=%d %s="%s"' % (a, ret['length'], a, a)
    
    # 获取变量的 Python 文档签名和输出签名，存储在 ret 字典中的 'pydocsign' 和 'pydocsignout' 键中
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    # 如果变量 var 中包含名为 'note' 的键
    if hasnote(var):
        # 将 'note' 键对应的值赋给返回字典 ret 的 'note' 键
        ret['note'] = var['note']
    # 返回处理后的字典 ret
    return ret
# 将函数 routsign2map 定义为将给定的 rout 字典转换为一个映射字典的函数
def routsign2map(rout):
    """
    name,NAME,begintitle,endtitle
    rname,ctype,rformat
    routdebugshowvalue
    """
    # 声明 lcb_map 为全局变量
    global lcb_map
    # 从 rout 字典中获取 'name' 键对应的值，并赋给 name 变量
    name = rout['name']
    # 使用 getfortranname 函数获取 rout 对应的 Fortran 名称，并赋给 fname 变量
    fname = getfortranname(rout)
    # 创建一个新的字典 ret，包含一些与 name 和 fname 相关的派生值
    ret = {'name': name,
           'texname': name.replace('_', '\\_'),  # 生成 LaTeX 格式的 name
           'name_lower': name.lower(),            # name 的小写形式
           'NAME': name.upper(),                  # name 的大写形式
           'begintitle': gentitle(name),          # 生成 name 的标题形式
           'endtitle': gentitle('end of %s' % name),  # 生成 name 结尾的标题形式
           'fortranname': fname,                  # Fortran 名称
           'FORTRANNAME': fname.upper(),          # Fortran 名称的大写形式
           'callstatement': getcallstatement(rout) or '',  # 获取调用语句，如果不存在则为空字符串
           'usercode': getusercode(rout) or '',  # 获取用户代码，如果不存在则为空字符串
           'usercode1': getusercode1(rout) or '',  # 获取用户代码1，如果不存在则为空字符串
           }
    # 根据 fname 是否包含下划线来确定 F_FUNC 的值
    if '_' in fname:
        ret['F_FUNC'] = 'F_FUNC_US'
    else:
        ret['F_FUNC'] = 'F_FUNC'
    # 根据 name 是否包含下划线来确定 F_WRAPPEDFUNC 的值
    if '_' in name:
        ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC_US'
    else:
        ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC'
    # 初始化空字典 lcb_map
    lcb_map = {}
    # 如果 rout 中包含 'use' 键
    if 'use' in rout:
        # 遍历 rout['use'] 中的每个键
        for u in rout['use'].keys():
            # 如果 u 在 cb_rules.cb_map 中
            if u in cb_rules.cb_map:
                # 遍历 cb_rules.cb_map[u] 中的每个元素 un
                for un in cb_rules.cb_map[u]:
                    ln = un[0]
                    # 如果 rout['use'][u] 中包含 'map' 键
                    if 'map' in rout['use'][u]:
                        # 遍历 rout['use'][u]['map'] 中的每个键 k
                        for k in rout['use'][u]['map'].keys():
                            # 如果 rout['use'][u]['map'][k] 等于 un[0]
                            if rout['use'][u]['map'][k] == un[0]:
                                ln = k  # 将 k 赋给 ln
                                break  # 结束内部循环
                    lcb_map[ln] = un[1]  # 将 un[1] 添加到 lcb_map 中
    # 如果 rout 中没有 'use' 键，但有 'externals' 键且 rout['externals'] 不为空
    elif 'externals' in rout and rout['externals']:
        # 输出错误信息，指出函数 routsign2map 中出现了错误情况
        errmess('routsign2map: Confused: function %s has externals %s but no "use" statement.\n' % (
            ret['name'], repr(rout['externals'])))
    # 获取调用协议参数，使用 rout 和 lcb_map，如果不存在则为空字符串
    ret['callprotoargument'] = getcallprotoargument(rout, lcb_map) or ''
    # 如果 rout 是一个函数对象
    if isfunction(rout):
        # 如果 rout 包含 'result' 键
        if 'result' in rout:
            # 将 rout['result'] 赋给变量 a
            a = rout['result']
        else:
            # 否则将 rout['name'] 赋给变量 a
            a = rout['name']
        
        # 将变量 a 赋给返回字典 ret 的 'rname' 键
        ret['rname'] = a
        
        # 调用 getpydocsign 函数获取签名信息，将结果分别赋给 ret['pydocsign'] 和 ret['pydocsignout']
        ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, rout)
        
        # 调用 getctype 函数获取 rout['vars'][a] 的类型，将结果赋给 ret['ctype']
        ret['ctype'] = getctype(rout['vars'][a])
        
        # 如果 rout 包含结果的注释信息
        if hasresultnote(rout):
            # 将 rout['vars'][a]['note'] 赋给 ret['resultnote']
            ret['resultnote'] = rout['vars'][a]['note']
            # 将 rout['vars'][a]['note'] 设置为 ['See elsewhere.']，替换原有注释信息
            rout['vars'][a]['note'] = ['See elsewhere.']
        
        # 如果 ret['ctype'] 在 c2buildvalue_map 中有对应的值
        if ret['ctype'] in c2buildvalue_map:
            # 将 c2buildvalue_map[ret['ctype']] 的值赋给 ret['rformat']
            ret['rformat'] = c2buildvalue_map[ret['ctype']]
        else:
            # 否则将 'O' 赋给 ret['rformat']，并记录错误消息
            ret['rformat'] = 'O'
            errmess('routsign2map: no c2buildvalue key for type %s\n' % (repr(ret['ctype'])))
        
        # 如果启用了 debugcapi(rout) 调试模式
        if debugcapi(rout):
            # 如果 ret['ctype'] 在 cformat_map 中有对应的格式化方式
            if ret['ctype'] in cformat_map:
                # 设置 ret['routdebugshowvalue'] 为 'debug-capi:%s=%s' 的格式化字符串
                ret['routdebugshowvalue'] = 'debug-capi:%s=%s' % (a, cformat_map[ret['ctype']])
            
            # 如果 rout 是字符串函数
            if isstringfunction(rout):
                # 设置 ret['routdebugshowvalue'] 为 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"' 的格式化字符串
                ret['routdebugshowvalue'] = 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"' % (a, a)
        
        # 如果 rout 是字符串函数
        if isstringfunction(rout):
            # 获取 rout['vars'][a] 的字符串长度，将结果赋给 ret['rlength']
            ret['rlength'] = getstrlength(rout['vars'][a])
            
            # 如果字符串长度为 '-1'，记录错误消息并设置 ret['rlength'] 为 '10'
            if ret['rlength'] == '-1':
                errmess('routsign2map: expected explicit specification of the length of the string returned by the fortran function %s; taking 10.\n' % (repr(rout['name'])))
                ret['rlength'] = '10'
    
    # 如果 rout 中有注释信息
    if hasnote(rout):
        # 将 rout['note'] 赋给 ret['note']
        ret['note'] = rout['note']
        # 将 rout['note'] 设置为 ['See elsewhere.']，替换原有注释信息
        rout['note'] = ['See elsewhere.']
    
    # 返回最终的结果字典 ret
    return ret
def modsign2map(m):
    """
    将模块的签名映射到特定格式的字典
    """
    if ismodule(m):
        # 如果 m 是模块，则设置特定的键值对
        ret = {'f90modulename': m['name'],
               'F90MODULENAME': m['name'].upper(),
               'texf90modulename': m['name'].replace('_', '\\_')}
    else:
        # 如果 m 不是模块，则设置另一组特定的键值对
        ret = {'modulename': m['name'],
               'MODULENAME': m['name'].upper(),
               'texmodulename': m['name'].replace('_', '\\_')}
    # 获取模块的文档字符串列表，如果不存在则返回空列表
    ret['restdoc'] = getrestdoc(m) or []
    # 如果模块有注释信息，则添加到结果字典中
    if hasnote(m):
        ret['note'] = m['note']
    # 获取模块的用户代码字符串，如果不存在则为空字符串
    ret['usercode'] = getusercode(m) or ''
    # 获取模块的另一部分用户代码字符串，如果不存在则为空字符串
    ret['usercode1'] = getusercode1(m) or ''
    # 如果模块有函数体，则获取第一个函数体的用户代码字符串，否则为空字符串
    if m['body']:
        ret['interface_usercode'] = getusercode(m['body'][0]) or ''
    else:
        ret['interface_usercode'] = ''
    # 获取模块的 Python 方法定义字符串，如果不存在则为空字符串
    ret['pymethoddef'] = getpymethoddef(m) or ''
    # 如果模块中包含 'coutput' 键，则添加到结果字典中
    if 'coutput' in m:
        ret['coutput'] = m['coutput']
    # 如果模块中包含 'f2py_wrapper_output' 键，则添加到结果字典中
    if 'f2py_wrapper_output' in m:
        ret['f2py_wrapper_output'] = m['f2py_wrapper_output']
    # 返回整理后的结果字典
    return ret


def cb_sign2map(a, var, index=None):
    """
    将回调函数参数的签名映射到特定格式的字典
    """
    ret = {'varname': a}
    ret['varname_i'] = ret['varname']
    # 获取变量的 C 类型字符串
    ret['ctype'] = getctype(var)
    # 如果 C 类型在 c2capi_map 中存在映射，则设置相关字段
    if ret['ctype'] in c2capi_map:
        ret['atype'] = c2capi_map[ret['ctype']]
        ret['elsize'] = get_elsize(var)
    # 如果 C 类型在 cformat_map 中存在格式化字符串，则设置显示值格式字段
    if ret['ctype'] in cformat_map:
        ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
    # 如果变量是数组，则追加数组维度信息到结果字典中
    if isarray(var):
        ret = dictappend(ret, getarrdims(a, var))
    # 获取变量的 Python 文档签名和输出签名
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    # 如果变量有注释信息，则添加到结果字典中，并将原变量的注释设置为 ['See elsewhere.']
    if hasnote(var):
        ret['note'] = var['note']
        var['note'] = ['See elsewhere.']
    # 返回整理后的结果字典
    return ret


def cb_routsign2map(rout, um):
    """
    将回调函数的签名映射到特定格式的字典
    """
    ret = {'name': 'cb_%s_in_%s' % (rout['name'], um),
           'returncptr': ''}
    # 如果回调函数是意图回调函数，则设置相关字段
    if isintent_callback(rout):
        if '_' in rout['name']:
            F_FUNC = 'F_FUNC_US'
        else:
            F_FUNC = 'F_FUNC'
        # 设置回调函数名称和 static 属性
        ret['callbackname'] = '%s(%s,%s)' \
                              % (F_FUNC,
                                 rout['name'].lower(),
                                 rout['name'].upper(),
                                 )
        ret['static'] = 'extern'
    else:
        # 否则，直接设置回调函数名称和 static 属性
        ret['callbackname'] = ret['name']
        ret['static'] = 'static'
    # 设置回调函数的参数名称、标题等信息
    ret['argname'] = rout['name']
    ret['begintitle'] = gentitle(ret['name'])
    ret['endtitle'] = gentitle('end of %s' % ret['name'])
    # 获取回调函数的 C 类型字符串和返回类型
    ret['ctype'] = getctype(rout)
    ret['rctype'] = 'void'
    # 如果返回类型是字符串，则设置返回类型为 void
    if ret['ctype'] == 'string':
        ret['rctype'] = 'void'
    else:
        ret['rctype'] = ret['ctype']
    # 如果返回类型不是 void，则根据是否为复杂函数设置返回值指针的相关代码
    if ret['rctype'] != 'void':
        if iscomplexfunction(rout):
            ret['returncptr'] = """
#ifdef F2PY_CB_RETURNCOMPLEX
return_value=
#endif
"""
        else:
            ret['returncptr'] = 'return_value='
    # 如果回调函数的 C 类型在 cformat_map 中存在格式化字符串，则设置显示值格式字段
    if ret['ctype'] in cformat_map:
        ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
    # 如果回调函数是字符串函数，则获取其字符串长度信息
    if isstringfunction(rout):
        ret['strlength'] = getstrlength(rout)
    # 返回整理后的结果字典
    return ret
    # 如果 rout 是一个函数
    if isfunction(rout):
        # 如果 rout 字典中包含 'result' 键
        if 'result' in rout:
            # 将 a 设为 rout['result'] 对应的值
            a = rout['result']
        else:
            # 否则将 a 设为 rout['name'] 对应的值
            a = rout['name']
        
        # 如果 rout['vars'][a] 中有注释
        if hasnote(rout['vars'][a]):
            # 将 ret 字典中的 'note' 键设为 rout['vars'][a] 中 'note' 键对应的值
            ret['note'] = rout['vars'][a]['note']
            # 将 rout['vars'][a] 中 'note' 键对应的值设为 ['See elsewhere.']
            rout['vars'][a]['note'] = ['See elsewhere.']
        
        # 将 ret 字典中的 'rname' 键设为 a 的值
        ret['rname'] = a
        
        # 调用 getpydocsign 函数，将其返回值分别设为 ret 字典中的 'pydocsign' 和 'pydocsignout' 键的值
        ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, rout)
        
        # 如果 rout 是复杂函数
        if iscomplexfunction(rout):
            # 将 ret 字典中的 'rctype' 键设为空字符串
            ret['rctype'] = ""
#ifdef F2PY_CB_RETURNCOMPLEX
#ctype#
#else
void
#endif
"""
    else:
        # 如果路由（rout）具有注释，则将其保存到返回字典中，并将路由的注释替换为默认的“See elsewhere.”
        if hasnote(rout):
            ret['note'] = rout['note']
            rout['note'] = ['See elsewhere.']
    # 初始化参数计数器
    nofargs = 0
    # 初始化可选参数计数器
    nofoptargs = 0
    # 如果路由（rout）包含 'args' 和 'vars'
    if 'args' in rout and 'vars' in rout:
        # 遍历路由（rout）的参数列表
        for a in rout['args']:
            # 获取变量的详细信息
            var = rout['vars'][a]
            # 如果变量具有输入或输入输出意图
            if l_or(isintent_in, isintent_inout)(var):
                # 增加参数计数器
                nofargs = nofargs + 1
                # 如果变量是可选的
                if isoptional(var):
                    # 增加可选参数计数器
                    nofoptargs = nofoptargs + 1
    # 将最大参数数量存入返回字典
    ret['maxnofargs'] = repr(nofargs)
    # 将可选参数数量存入返回字典
    ret['nofoptargs'] = repr(nofoptargs)
    # 如果路由（rout）具有注释且是函数，并且包含 'result'
    if hasnote(rout) and isfunction(rout) and 'result' in rout:
        # 将路由的注释存入返回字典，并将路由的注释替换为默认的“See elsewhere.”
        ret['routnote'] = rout['note']
        rout['note'] = ['See elsewhere.']
    # 返回最终的返回字典
    return ret


# 定义一个函数 common_sign2map，已经过时
def common_sign2map(a, var):
    # 初始化返回字典，包含变量名和其 C 类型
    ret = {'varname': a, 'ctype': getctype(var)}
    # 如果变量是字符串数组
    if isstringarray(var):
        ret['ctype'] = 'char'
    # 如果变量的 C 类型在 c2capi_map 中
    if ret['ctype'] in c2capi_map:
        # 将其对应的 API 类型存入返回字典，并获取元素大小
        ret['atype'] = c2capi_map[ret['ctype']]
        ret['elsize'] = get_elsize(var)
    # 如果变量的 C 类型在 cformat_map 中
    if ret['ctype'] in cformat_map:
        # 存入显示值的格式字符串到返回字典
        ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
    # 如果变量是数组
    if isarray(var):
        # 将数组维度信息附加到返回字典中
        ret = dictappend(ret, getarrdims(a, var))
    # 如果变量是字符串
    elif isstring(var):
        # 获取字符串的长度和维度信息，并存入返回字典
        ret['size'] = getstrlength(var)
        ret['rank'] = '1'
    # 获取变量的 Python 文档字符串签名和输出签名，并存入返回字典
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    # 如果变量具有注释
    if hasnote(var):
        # 将变量的注释存入返回字典，并将变量的注释替换为默认的“See elsewhere.”
        ret['note'] = var['note']
        var['note'] = ['See elsewhere.']
    # 获取数组文档字符串签名并存入返回字典
    # 对于字符串，这里返回的是 0 维，但实际上是 1 维
    ret['arrdocstr'] = getarrdocsign(a, var)
    # 返回最终的返回字典
    return ret
```