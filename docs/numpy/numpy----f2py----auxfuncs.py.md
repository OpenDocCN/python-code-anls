# `.\numpy\numpy\f2py\auxfuncs.py`

```py
# 引入 pprint 模块，用于美化打印输出
import pprint
# 引入 sys 模块，用于访问系统相关功能
import sys
# 引入 re 模块，用于正则表达式操作
import re
# 引入 types 模块，用于类型操作
import types
# 从 functools 模块中引入 reduce 函数，用于序列化操作
from functools import reduce
# 从 copy 模块中引入 deepcopy 函数，用于深拷贝对象
from copy import deepcopy

# 从当前包中导入 __version__ 和 cfuncs 模块
from . import __version__
from . import cfuncs

# 定义 __all__ 列表，包含模块中需要导出的公共符号
__all__ = [
    'applyrules', 'debugcapi', 'dictappend', 'errmess', 'gentitle',
    'getargs2', 'getcallprotoargument', 'getcallstatement',
    'getfortranname', 'getpymethoddef', 'getrestdoc', 'getusercode',
    'getusercode1', 'getdimension', 'hasbody', 'hascallstatement', 'hascommon',
    'hasexternals', 'hasinitvalue', 'hasnote', 'hasresultnote',
    'isallocatable', 'isarray', 'isarrayofstrings',
    'ischaracter', 'ischaracterarray', 'ischaracter_or_characterarray',
    'iscomplex',
    'iscomplexarray', 'iscomplexfunction', 'iscomplexfunction_warn',
    'isdouble', 'isdummyroutine', 'isexternal', 'isfunction',
    'isfunction_wrap', 'isint1', 'isint1array', 'isinteger', 'isintent_aux',
    'isintent_c', 'isintent_callback', 'isintent_copy', 'isintent_dict',
    'isintent_hide', 'isintent_in', 'isintent_inout', 'isintent_inplace',
    'isintent_nothide', 'isintent_out', 'isintent_overwrite', 'islogical',
    'islogicalfunction', 'islong_complex', 'islong_double',
    'islong_doublefunction', 'islong_long', 'islong_longfunction',
    'ismodule', 'ismoduleroutine', 'isoptional', 'isprivate', 'isrequired',
    'isroutine', 'isscalar', 'issigned_long_longarray', 'isstring',
    'isstringarray', 'isstring_or_stringarray', 'isstringfunction',
    'issubroutine', 'get_f2py_modulename',
    'issubroutine_wrap', 'isthreadsafe', 'isunsigned', 'isunsigned_char',
    'isunsigned_chararray', 'isunsigned_long_long',
    'isunsigned_long_longarray', 'isunsigned_short',
    'isunsigned_shortarray', 'l_and', 'l_not', 'l_or', 'outmess',
    'replace', 'show', 'stripcomma', 'throw_error', 'isattr_value',
    'getuseblocks', 'process_f2cmap_dict'
]

# 获取当前模块的版本号并赋值给 f2py_version
f2py_version = __version__.version

# 将 sys.stderr.write 赋值给 errmess，用于输出错误信息
errmess = sys.stderr.write

# 将 pprint.pprint 赋值给 show，用于美化打印输出
show = pprint.pprint

# 初始化空字典 options，用于存储选项设置
options = {}

# 初始化空列表 debugoptions，用于存储调试选项
debugoptions = []

# 设置 wrapfuncs 为 1，表示启用函数包装
wrapfuncs = 1


# 定义函数 outmess，根据选项输出消息到标准输出
def outmess(t):
    if options.get('verbose', 1):
        sys.stdout.write(t)


# 定义函数 debugcapi，检查变量是否在调试选项中
def debugcapi(var):
    return 'capi' in debugoptions


# 定义函数 _ischaracter，检查变量是否为字符类型且非外部变量
def _ischaracter(var):
    return 'typespec' in var and var['typespec'] == 'character' and \
           not isexternal(var)


# 定义函数 _isstring，检查变量是否为字符串类型且非外部变量
def _isstring(var):
    return 'typespec' in var and var['typespec'] == 'character' and \
           not isexternal(var)


# 定义函数 ischaracter_or_characterarray，检查变量是否为字符或字符数组，且未指定 charselector
def ischaracter_or_characterarray(var):
    return _ischaracter(var) and 'charselector' not in var


# 定义函数 ischaracter，检查变量是否为字符且非数组
def ischaracter(var):
    return ischaracter_or_characterarray(var) and not isarray(var)


# 定义函数 ischaracterarray，检查变量是否为字符数组
def ischaracterarray(var):
    return ischaracter_or_characterarray(var) and isarray(var)


# 定义函数 isstring_or_stringarray，检查变量是否为字符串或字符串数组
    # 检查变量是否为字符类型，并且在变量中包含 'charselector' 字符串
    return _ischaracter(var) and 'charselector' in var
# 判断变量是否为字符串类型（单个字符串）
def isstring(var):
    # 调用 isstring_or_stringarray 函数判断是否为字符串或字符串数组，同时不是数组
    return isstring_or_stringarray(var) and not isarray(var)


# 判断变量是否为字符串数组类型
def isstringarray(var):
    # 调用 isstring_or_stringarray 函数判断是否为字符串或字符串数组，并且是数组
    return isstring_or_stringarray(var) and isarray(var)


# 判断变量是否为字符串数组，并且数组的最后一个维度为(*)，已经过时（obsolete）
def isarrayofstrings(var):  # obsolete?
    # 暂时略过 '*'，以便将 `character*(*) a(m)` 和 `character a(m,*)` 区分对待。
    # 幸运的是 `character**` 是非法的。
    return isstringarray(var) and var['dimension'][-1] == '(*)'


# 判断变量是否为数组类型
def isarray(var):
    # 判断变量中是否有 'dimension' 键，并且不是外部变量
    return 'dimension' in var and not isexternal(var)


# 判断变量是否为标量（非数组、非字符串）
def isscalar(var):
    # 判断变量不是数组、不是字符串、不是外部变量
    return not (isarray(var) or isstring(var) or isexternal(var))


# 判断变量是否为复数类型（实数和双精度复数）
def iscomplex(var):
    # 判断变量是标量，并且其 'typespec' 键对应的值为 'complex' 或 'double complex'
    return isscalar(var) and \
           var.get('typespec') in ['complex', 'double complex']


# 判断变量是否为逻辑类型
def islogical(var):
    # 判断变量是标量，并且其 'typespec' 键对应的值为 'logical'
    return isscalar(var) and var.get('typespec') == 'logical'


# 判断变量是否为整数类型
def isinteger(var):
    # 判断变量是标量，并且其 'typespec' 键对应的值为 'integer'
    return isscalar(var) and var.get('typespec') == 'integer'


# 判断变量是否为实数类型
def isreal(var):
    # 判断变量是标量，并且其 'typespec' 键对应的值为 'real'
    return isscalar(var) and var.get('typespec') == 'real'


# 获取变量的 'kind' 属性值
def get_kind(var):
    try:
        return var['kindselector']['*']
    except KeyError:
        try:
            return var['kindselector']['kind']
        except KeyError:
            pass


# 判断变量是否为 'integer' 类型且 'kind' 为 '1' 的整数（单字节整数）
def isint1(var):
    return var.get('typespec') == 'integer' \
        and get_kind(var) == '1' and not isarray(var)


# 判断变量是否为 'long long' 类型的整数
def islong_long(var):
    # 如果不是标量，则返回 0
    if not isscalar(var):
        return 0
    # 如果 'typespec' 不是 'integer' 或 'logical'，则返回 0
    if var.get('typespec') not in ['integer', 'logical']:
        return 0
    # 判断 'kind' 是否为 '8'
    return get_kind(var) == '8'


# 判断变量是否为 'unsigned char' 类型的整数
def isunsigned_char(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-1'


# 判断变量是否为 'unsigned short' 类型的整数
def isunsigned_short(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-2'


# 判断变量是否为 'unsigned' 类型的整数
def isunsigned(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-4'


# 判断变量是否为 'unsigned long long' 类型的整数
def isunsigned_long_long(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-8'


# 判断变量是否为 'double' 类型的实数
def isdouble(var):
    if not isscalar(var):
        return 0
    if not var.get('typespec') == 'real':
        return 0
    return get_kind(var) == '8'


# 判断变量是否为 'long double' 类型的实数
def islong_double(var):
    if not isscalar(var):
        return 0
    if not var.get('typespec') == 'real':
        return 0
    return get_kind(var) == '16'


# 判断变量是否为 'long complex' 类型的复数
def islong_complex(var):
    # 如果不是复数类型，则返回 0
    if not iscomplex(var):
        return 0
    # 判断 'kind' 是否为 '32'
    return get_kind(var) == '32'


# 判断变量是否为复数数组类型
def iscomplexarray(var):
    # 判断变量是否为数组，并且 'typespec' 是 'complex' 或 'double complex'
    return isarray(var) and \
           var.get('typespec') in ['complex', 'double complex']


# 判断变量是否为 'integer' 类型且 'kind' 为 '1' 的整数数组（单字节整数数组）
def isint1array(var):
    return isarray(var) and var.get('typespec') == 'integer' \
        and get_kind(var) == '1'


# 判断变量是否为 'unsigned char' 类型的整数数组
def isunsigned_chararray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-1'


# 判断变量是否为 'unsigned short' 类型的整数数组
def isunsigned_shortarray(var):
    if not isarray(var):
        return 0
    if var.get('typespec') not in ['integer', 'logical']:
        return 0
    return get_kind(var) == '-2'
    # 检查变量是否为数组并且其类型规范为整数或逻辑值
    return isarray(var) and var.get('typespec') in ['integer', 'logical'] \
        # 获取变量的 KIND 属性，判断其是否为 Fortran 的默认整数类型（'-2' 表示默认整数）
        and get_kind(var) == '-2'
# 检查变量是否为无符号整型或逻辑型数组
def isunsignedarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-4'


# 检查变量是否为无符号长长整型或逻辑型数组
def isunsigned_long_longarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-8'


# 检查变量是否为有符号字符型数组
def issigned_chararray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '1'


# 检查变量是否为有符号短整型数组
def issigned_shortarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '2'


# 检查变量是否为有符号整型或逻辑型数组
def issigned_array(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '4'


# 检查变量是否为有符号长长整型或逻辑型数组
def issigned_long_longarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '8'


# 检查变量是否具有 allocatable 属性
def isallocatable(var):
    return 'attrspec' in var and 'allocatable' in var['attrspec']


# 检查变量是否为可变类型，即不是字符串且具有维度属性
def ismutable(var):
    return not ('dimension' not in var or isstring(var))


# 检查过程是否为模块内程序
def ismoduleroutine(rout):
    return 'modulename' in rout


# 检查过程是否为模块
def ismodule(rout):
    return 'block' in rout and 'module' == rout['block']


# 检查过程是否为函数
def isfunction(rout):
    return 'block' in rout and 'function' == rout['block']


# 检查过程是否为带有特定条件的函数
def isfunction_wrap(rout):
    if isintent_c(rout):
        return 0
    return wrapfuncs and isfunction(rout) and (not isexternal(rout))


# 检查过程是否为子程序
def issubroutine(rout):
    return 'block' in rout and 'subroutine' == rout['block']


# 检查过程是否为带有特定条件的子程序
def issubroutine_wrap(rout):
    if isintent_c(rout):
        return 0
    return issubroutine(rout) and hasassumedshape(rout)


# 检查变量是否具有值属性
def isattr_value(var):
    return 'value' in var.get('attrspec', [])


# 检查过程是否具有隐式形状参数
def hasassumedshape(rout):
    if rout.get('hasassumedshape'):
        return True
    for a in rout['args']:
        for d in rout['vars'].get(a, {}).get('dimension', []):
            if d == ':':
                rout['hasassumedshape'] = True
                return True
    return False


# 检查过程是否需要使用 F90 包装器
def requiresf90wrapper(rout):
    return ismoduleroutine(rout) or hasassumedshape(rout)


# 检查过程是否为函数或子程序
def isroutine(rout):
    return isfunction(rout) or issubroutine(rout)


# 检查过程是否为逻辑型函数
def islogicalfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islogical(rout['vars'][a])
    return 0


# 检查过程是否为长长整型函数
def islong_longfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islong_long(rout['vars'][a])
    return 0


# 检查过程是否为长双精度函数
def islong_doublefunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islong_double(rout['vars'][a])
    return 0


# 检查过程是否为复数函数
def iscomplexfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    # 如果条件不成立，则将变量 a 赋值为 rout 字典中键为 'name' 的值
    else:
        a = rout['name']
    # 如果变量 a 存在于 rout 字典中的 'vars' 键中，则返回 rout 字典中 'vars' 键对应值的 iscomplex 函数结果
    if a in rout['vars']:
        return iscomplex(rout['vars'][a])
    # 否则返回 0
    return 0
# 判断给定的函数是否复杂并发出警告
def iscomplexfunction_warn(rout):
    # 如果函数返回复数值，则发出警告信息
    if iscomplexfunction(rout):
        outmess("""\
    **************************************************************
        Warning: code with a function returning complex value
        may not work correctly with your Fortran compiler.
        When using GNU gcc/g77 compilers, codes should work
        correctly for callbacks with:
        f2py -c -DF2PY_CB_RETURNCOMPLEX
    **************************************************************\n""")
        return 1
    # 如果函数不复杂，返回 0
    return 0


# 判断给定的函数是否返回字符串
def isstringfunction(rout):
    # 如果不是函数，返回 0
    if not isfunction(rout):
        return 0
    # 获取函数结果变量名称
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    # 如果结果变量存在于函数变量中，则判断其是否为字符串类型
    if a in rout['vars']:
        return isstring(rout['vars'][a])
    # 否则返回 0
    return 0


# 判断给定的函数是否具有外部变量
def hasexternals(rout):
    # 返回是否存在 'externals' 键并且其值不为空
    return 'externals' in rout and rout['externals']


# 判断给定的函数是否支持线程安全
def isthreadsafe(rout):
    # 返回是否存在 'f2pyenhancements' 键并且包含 'threadsafe' 属性
    return 'f2pyenhancements' in rout and \
           'threadsafe' in rout['f2pyenhancements']


# 判断给定的函数是否具有变量
def hasvariables(rout):
    # 返回是否存在 'vars' 键并且其值不为空
    return 'vars' in rout and rout['vars']


# 判断给定的变量是否为可选参数
def isoptional(var):
    # 判断变量是否包含 'optional' 属性且不包含 'required' 属性，并且满足非隐藏意图
    return ('attrspec' in var and 'optional' in var['attrspec'] and
            'required' not in var['attrspec']) and isintent_nothide(var)


# 判断给定的变量是否为外部变量
def isexternal(var):
    # 判断变量是否包含 'external' 属性
    return 'attrspec' in var and 'external' in var['attrspec']


# 获取给定变量的维度信息
def getdimension(var):
    dimpattern = r"\((.*?)\)"
    # 如果变量的属性中包含 'dimension'，则返回其维度信息
    if 'attrspec' in var.keys():
        if any('dimension' in s for s in var['attrspec']):
            return [re.findall(dimpattern, v) for v in var['attrspec']][0]


# 判断给定变量是否为必需参数
def isrequired(var):
    # 判断变量既不是可选参数也不是隐藏意图
    return not isoptional(var) and isintent_nothide(var)


# 判断给定变量是否具有 'intent=in' 属性
def isintent_in(var):
    # 如果变量没有 'intent' 属性，则默认为 'intent=in'
    if 'intent' not in var:
        return 1
    # 根据 'intent' 属性值判断是否为 'in'
    if 'hide' in var['intent']:
        return 0
    if 'inplace' in var['intent']:
        return 0
    if 'in' in var['intent']:
        return 1
    if 'out' in var['intent']:
        return 0
    if 'inout' in var['intent']:
        return 0
    if 'outin' in var['intent']:
        return 0
    return 1


# 判断给定变量是否具有 'intent=inout' 属性
def isintent_inout(var):
    # 判断变量是否具有 'intent=inout' 或者 'intent=outin'，并且不具有 'in' 属性，也不具有 'hide' 和 'inplace' 属性
    return ('intent' in var and ('inout' in var['intent'] or
            'outin' in var['intent']) and 'in' not in var['intent'] and
            'hide' not in var['intent'] and 'inplace' not in var['intent'])


# 判断给定变量是否具有 'intent=out' 属性
def isintent_out(var):
    # 返回变量是否具有 'out' 属性
    return 'out' in var.get('intent', [])


# 判断给定变量是否具有 'intent=hide' 属性
def isintent_hide(var):
    # 判断变量是否具有 'hide' 属性或者 'out' 属性且没有 'in' 属性，并且不满足 isintent_inout 或 isintent_inplace 条件
    return ('intent' in var and ('hide' in var['intent'] or
            ('out' in var['intent'] and 'in' not in var['intent'] and
                (not l_or(isintent_inout, isintent_inplace)(var)))))


# 判断给定变量是否不具有 'intent=hide' 属性
def isintent_nothide(var):
    # 返回变量是否不具有 'intent=hide' 属性
    return not isintent_hide(var)


# 判断给定变量是否具有 'intent=c' 属性
def isintent_c(var):
    # 返回变量是否具有 'c' 属性
    return 'c' in var.get('intent', [])


# 判断给定变量是否具有 'intent=cache' 属性
def isintent_cache(var):
    # 返回变量是否具有 'cache' 属性
    return 'cache' in var.get('intent', [])


# 判断给定变量是否具有 'intent=copy' 属性
def isintent_copy(var):
    # 返回变量是否具有 'copy' 属性
    return 'copy' in var.get('intent', [])


# 判断给定变量是否具有 'intent=overwrite' 属性
def isintent_overwrite(var):
    # 返回变量是否具有 'overwrite' 属性
    return 'overwrite' in var.get('intent', [])


# 判断给定变量是否具有 'intent=callback' 属性
def isintent_callback(var):
    # 返回变量是否具有 'callback' 属性
    return 'callback' in var.get('intent', [])


# 判断给定变量是否具有 'intent=inplace' 属性
def isintent_inplace(var):
    # 返回变量是否具有 'inplace' 属性
    return 'inplace' in var.get('intent', [])
# 检查变量的 'intent' 属性是否包含 'aux'
def isintent_aux(var):
    return 'aux' in var.get('intent', [])


# 检查变量的 'intent' 属性是否包含 'aligned4'
def isintent_aligned4(var):
    return 'aligned4' in var.get('intent', [])


# 检查变量的 'intent' 属性是否包含 'aligned8'
def isintent_aligned8(var):
    return 'aligned8' in var.get('intent', [])


# 检查变量的 'intent' 属性是否包含 'aligned16'
def isintent_aligned16(var):
    return 'aligned16' in var.get('intent', [])


# 定义一个字典 isintent_dict，将不同的检查函数映射到对应的字符串标识
isintent_dict = {
    isintent_in: 'INTENT_IN',
    isintent_inout: 'INTENT_INOUT',
    isintent_out: 'INTENT_OUT',
    isintent_hide: 'INTENT_HIDE',
    isintent_cache: 'INTENT_CACHE',
    isintent_c: 'INTENT_C',
    isoptional: 'OPTIONAL',
    isintent_inplace: 'INTENT_INPLACE',
    isintent_aligned4: 'INTENT_ALIGNED4',
    isintent_aligned8: 'INTENT_ALIGNED8',
    isintent_aligned16: 'INTENT_ALIGNED16',
}


# 检查变量是否包含 'attrspec' 属性，并且其中包含 'private' 字段
def isprivate(var):
    return 'attrspec' in var and 'private' in var['attrspec']


# 检查变量是否具有初始值
def hasinitvalue(var):
    return '=' in var


# 检查变量的初始值是否是字符串类型
def hasinitvalueasstring(var):
    if not hasinitvalue(var):
        return 0
    return var['='][0] in ['"', "'"]


# 检查变量是否包含 'note' 属性
def hasnote(var):
    return 'note' in var


# 检查函数是否具有 'result' 属性的注释
def hasresultnote(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return hasnote(rout['vars'][a])
    return 0


# 检查函数是否包含 'common' 属性
def hascommon(rout):
    return 'common' in rout


# 检查函数或块是否包含 'common' 属性
def containscommon(rout):
    if hascommon(rout):
        return 1
    if hasbody(rout):
        for b in rout['body']:
            if containscommon(b):
                return 1
    return 0


# 检查块是否包含模块
def containsmodule(block):
    if ismodule(block):
        return 1
    if not hasbody(block):
        return 0
    for b in block['body']:
        if containsmodule(b):
            return 1
    return 0


# 检查函数是否具有 'body' 属性
def hasbody(rout):
    return 'body' in rout


# 检查函数是否包含调用语句
def hascallstatement(rout):
    return getcallstatement(rout) is not None


# 始终返回真
def istrue(var):
    return 1


# 始终返回假
def isfalse(var):
    return 0


# 定义一个自定义异常类 F2PYError
class F2PYError(Exception):
    pass


# 定义一个可抛出异常的类 throw_error
class throw_error:

    # 初始化方法，接受一个消息参数 mess
    def __init__(self, mess):
        self.mess = mess

    # 调用实例时触发异常，并包含相关的变量信息和消息
    def __call__(self, var):
        mess = '\n\n  var = %s\n  Message: %s\n' % (var, self.mess)
        raise F2PYError(mess)


# 创建一个逻辑与函数，接受多个函数作为参数，返回一个函数，对这些函数应用逻辑与操作
def l_and(*f):
    l1, l2 = 'lambda v', []
    for i in range(len(f)):
        l1 = '%s,f%d=f[%d]' % (l1, i, i)
        l2.append('f%d(v)' % (i))
    return eval('%s:%s' % (l1, ' and '.join(l2)))


# 创建一个逻辑或函数，接受多个函数作为参数，返回一个函数，对这些函数应用逻辑或操作
def l_or(*f):
    l1, l2 = 'lambda v', []
    for i in range(len(f)):
        l1 = '%s,f%d=f[%d]' % (l1, i, i)
        l2.append('f%d(v)' % (i))
    return eval('%s:%s' % (l1, ' or '.join(l2)))


# 创建一个逻辑非函数，接受一个函数作为参数，返回对该函数应用逻辑非操作后的结果
def l_not(f):
    return eval('lambda v,f=f:not f(v)')


# 检查函数是否是虚拟的（dummy routine），即是否具有 'f2pyenhancements' 属性且其中的 'fortranname' 为空字符串
def isdummyroutine(rout):
    try:
        return rout['f2pyenhancements']['fortranname'] == ''
    except KeyError:
        return 0


# 获取函数的 Fortran 名称
def getfortranname(rout):
    # 这里需要继续添加代码
    # 尝试从rout字典中获取'f2pyenhancements'键下的'fortranname'值
    try:
        # 从'rout'字典的'f2pyenhancements'键中获取'fortranname'值
        name = rout['f2pyenhancements']['fortranname']
        # 如果'fortranname'值为空字符串，则抛出KeyError异常
        if name == '':
            raise KeyError
        # 如果'fortranname'值为假值（如None），则记录错误消息并抛出KeyError异常
        if not name:
            errmess('Failed to use fortranname from %s\n' %
                    (rout['f2pyenhancements']))
            raise KeyError
    # 捕获KeyError异常，处理方式是从'rout'字典中获取'name'键的值赋给'name'
    except KeyError:
        name = rout['name']
    # 返回变量'name'的值作为函数的结果
    return name
# 根据给定的块名从 rout['f2pyenhancements'] 字典中获取多行块内容
def getmultilineblock(rout, blockname, comment=1, counter=0):
    try:
        r = rout['f2pyenhancements'].get(blockname)
    except KeyError:
        return

    # 如果未找到指定的块名，返回空
    if not r:
        return

    # 如果 counter 大于 0 且 r 是字符串类型，则直接返回
    if counter > 0 and isinstance(r, str):
        return

    # 如果 r 是列表类型，根据 counter 获取对应元素
    if isinstance(r, list):
        if counter >= len(r):
            return
        r = r[counter]

    # 如果 r 是以三引号开头的字符串，处理多行块的注释
    if r[:3] == "'''":
        # 如果需要添加注释，添加起始注释信息
        if comment:
            r = '\t/* start ' + blockname + \
                ' multiline (' + repr(counter) + ') */\n' + r[3:]
        else:
            r = r[3:]
        
        # 如果字符串以三引号结尾，添加结束注释信息
        if r[-3:] == "'''":
            if comment:
                r = r[:-3] + '\n\t/* end multiline (' + repr(counter) + ')*/'
            else:
                r = r[:-3]
        else:
            # 如果多行块不以三引号结尾，报错
            errmess("%s multiline block should end with `'''`: %s\n"
                    % (blockname, repr(r)))
    return r


# 获取 rout 中 'callstatement' 的多行块内容
def getcallstatement(rout):
    return getmultilineblock(rout, 'callstatement')


# 获取 rout 中 'callprotoargument' 的多行块内容，不添加注释
def getcallprotoargument(rout, cb_map={}):
    r = getmultilineblock(rout, 'callprotoargument', comment=0)
    if r:
        return r
    
    # 如果 'callstatement' 已定义但未找到 'callprotoargument'，输出警告信息
    if hascallstatement(rout):
        outmess(
            'warning: callstatement is defined without callprotoargument\n')
        return
    
    # 根据变量的类型和属性生成参数列表
    from .capi_maps import getctype
    arg_types, arg_types2 = [], []
    
    # 如果 rout 符合特定条件，则添加 'char*' 和 'size_t' 到参数类型列表
    if l_and(isstringfunction, l_not(isfunction_wrap))(rout):
        arg_types.extend(['char*', 'size_t'])
    
    # 遍历 rout['args'] 中的变量，根据变量属性生成参数类型列表
    for n in rout['args']:
        var = rout['vars'][n]
        if isintent_callback(var):
            continue
        if n in cb_map:
            ctype = cb_map[n] + '_typedef'
        else:
            ctype = getctype(var)
            if l_and(isintent_c, l_or(isscalar, iscomplex))(var):
                pass
            elif isstring(var):
                pass
            else:
                if not isattr_value(var):
                    ctype = ctype + '*'
            if ((isstring(var)
                 or isarrayofstrings(var)  # obsolete?
                 or isstringarray(var))):
                arg_types2.append('size_t')
        arg_types.append(ctype)

    # 将参数类型列表连接成字符串，作为函数原型的参数部分
    proto_args = ','.join(arg_types + arg_types2)
    
    # 如果参数为空，则默认为 'void'
    if not proto_args:
        proto_args = 'void'
    return proto_args


# 获取 rout 中 'usercode' 的多行块内容
def getusercode(rout):
    return getmultilineblock(rout, 'usercode')


# 获取 rout 中 'usercode' 的第二个多行块内容
def getusercode1(rout):
    return getmultilineblock(rout, 'usercode', counter=1)


# 获取 rout 中 'pymethoddef' 的多行块内容
def getpymethoddef(rout):
    return getmultilineblock(rout, 'pymethoddef')


# 获取 rout 的参数列表和排序后的参数列表
def getargs(rout):
    sortargs, args = [], []
    
    # 如果 rout 中存在 'args'，将其作为参数列表
    if 'args' in rout:
        args = rout['args']
        
        # 如果 rout 中存在 'sortvars'，根据 'sortvars' 对 'args' 进行排序
        if 'sortvars' in rout:
            for a in rout['sortvars']:
                if a in args:
                    sortargs.append(a)
            for a in args:
                if a not in sortargs:
                    sortargs.append(a)
        else:
            # 如果不存在 'sortvars'，直接使用 'args'
            sortargs = rout['args']
    
    # 返回参数列表和排序后的参数列表
    return args, sortargs


# 获取 rout 中的参数列表
def getargs2(rout):
    sortargs, args = [], rout.get('args', [])
    # 从给定的字典 rout['vars'] 中筛选出符合 isintent_aux 函数条件且不在 args 列表中的变量名，形成辅助变量列表
    auxvars = [a for a in rout['vars'].keys() if isintent_aux(rout['vars'][a])
               and a not in args]
    
    # 将辅助变量列表和原始参数列表 args 合并，形成新的参数列表 args
    args = auxvars + args
    
    # 检查 rout 字典中是否有 'sortvars' 键
    if 'sortvars' in rout:
        # 如果有，遍历 rout['sortvars'] 列表中的每个元素
        for a in rout['sortvars']:
            # 如果元素 a 在参数列表 args 中，则将其添加到排序参数列表 sortargs 中
            if a in args:
                sortargs.append(a)
        # 再次遍历参数列表 args
        for a in args:
            # 如果元素 a 不在排序参数列表 sortargs 中，则将其添加到 sortargs 中
            if a not in sortargs:
                sortargs.append(a)
    else:
        # 如果 rout 字典中没有 'sortvars' 键，则将辅助变量列表和 rout['args'] 合并，形成排序参数列表 sortargs
        sortargs = auxvars + rout['args']
    
    # 返回最终的参数列表 args 和排序后的参数列表 sortargs
    return args, sortargs
# 根据传入的字典 rout，获取对应的文档内容，若未找到 'f2pymultilines' 则返回 None
def getrestdoc(rout):
    if 'f2pymultilines' not in rout:
        return None
    k = None
    # 如果 block 属性为 'python module'，则将 block 和 name 组成元组赋值给 k
    if rout['block'] == 'python module':
        k = rout['block'], rout['name']
    # 返回 f2pymultilines 字典中对应 k 的值，找不到则返回 None
    return rout['f2pymultilines'].get(k, None)


# 根据给定的 name，生成一个标题字符串，总长度为 80，形如 /* ************ name ************ */
def gentitle(name):
    # 计算左右两侧的星号数量
    ln = (80 - len(name) - 6) // 2
    # 返回生成的标题字符串
    return '/*%s %s %s*/' % (ln * '*', name, ln * '*')


# 将嵌套的列表扁平化为一维列表
def flatlist(lst):
    # 若 lst 是列表，则递归调用 flatlist 并使用 reduce 函数拼接列表
    if isinstance(lst, list):
        return reduce(lambda x, y, f=flatlist: x + f(y), lst, [])
    # 若 lst 不是列表，则将其封装为列表返回
    return [lst]


# 去除字符串末尾的逗号
def stripcomma(s):
    # 如果 s 存在且末尾是逗号，则返回去除逗号后的字符串，否则返回原字符串
    if s and s[-1] == ',':
        return s[:-1]
    return s


# 根据字典 d 中的键值对，替换字符串 str 中的对应标记
def replace(str, d, defaultsep=''):
    # 如果 d 是列表，则对列表中的每个元素递归调用 replace，并返回结果列表
    if isinstance(d, list):
        return [replace(str, _m, defaultsep) for _m in d]
    # 如果 str 是列表，则对列表中的每个元素递归调用 replace，并返回结果列表
    if isinstance(str, list):
        return [replace(_m, d, defaultsep) for _m in str]
    # 对字典 d 中的每对键值进行替换操作
    for k in 2 * list(d.keys()):
        # 若键为 'separatorsfor' 则跳过
        if k == 'separatorsfor':
            continue
        # 根据键 'separatorsfor' 取得对应的分隔符，否则使用默认分隔符
        if 'separatorsfor' in d and k in d['separatorsfor']:
            sep = d['separatorsfor'][k]
        else:
            sep = defaultsep
        # 若值为列表，则用分隔符连接后替换对应标记；否则直接替换对应标记
        if isinstance(d[k], list):
            str = str.replace('#%s#' % (k), sep.join(flatlist(d[k])))
        else:
            str = str.replace('#%s#' % (k), d[k])
    # 返回替换后的字符串
    return str


# 将字典 ar 合并到字典 rd 中
def dictappend(rd, ar):
    # 若 ar 是列表，则逐个将列表元素添加到 rd 中并返回
    if isinstance(ar, list):
        for a in ar:
            rd = dictappend(rd, a)
        return rd
    # 遍历字典 ar 的键值对，将其合并到 rd 中
    for k in ar.keys():
        # 如果键以 '_' 开头，则跳过
        if k[0] == '_':
            continue
        # 如果键已存在于 rd 中，则根据类型合并值
        if k in rd:
            if isinstance(rd[k], str):
                rd[k] = [rd[k]]
            if isinstance(rd[k], list):
                if isinstance(ar[k], list):
                    rd[k] = rd[k] + ar[k]
                else:
                    rd[k].append(ar[k])
            elif isinstance(rd[k], dict):
                if isinstance(ar[k], dict):
                    if k == 'separatorsfor':
                        for k1 in ar[k].keys():
                            if k1 not in rd[k]:
                                rd[k][k1] = ar[k][k1]
                    else:
                        rd[k] = dictappend(rd[k], ar[k])
        else:
            rd[k] = ar[k]
    # 返回合并后的字典 rd
    return rd


# 根据给定的 rules 对象，应用规则到字典 d 中
def applyrules(rules, d, var={}):
    ret = {}
    # 若 rules 是列表，则对列表中的每个规则逐个应用，并将结果合并到 ret 中
    if isinstance(rules, list):
        for r in rules:
            rr = applyrules(r, d, var)
            ret = dictappend(ret, rr)
            # 如果应用后结果中包含 '_break'，则终止应用更多规则
            if '_break' in rr:
                break
        return ret
    # 如果规则中包含 '_check' 并且 '_check' 函数返回 False，则返回空字典 ret
    if '_check' in rules and (not rules['_check'](var)):
        return ret
    # 如果规则中包含 'need'，则根据 'need' 规则应用到字典 d 中
    if 'need' in rules:
        res = applyrules({'needs': rules['need']}, d, var)
        if 'needs' in res:
            cfuncs.append_needs(res['needs'])  # 假设 cfuncs 是全局对象，用于处理 'needs' 功能
    # 返回应用规则后的结果字典 ret
    return ret
    # 遍历规则字典的所有键
    for k in rules.keys():
        # 如果键是'separatorsfor'，直接将其值赋给返回字典，并继续下一个键的处理
        if k == 'separatorsfor':
            ret[k] = rules[k]
            continue
        
        # 如果值是字符串，对其应用替换函数replace，并将结果赋给返回字典的当前键
        if isinstance(rules[k], str):
            ret[k] = replace(rules[k], d)
        
        # 如果值是列表，初始化一个空列表，然后对列表中的每个元素应用applyrules函数，并将结果存入返回字典的当前键
        elif isinstance(rules[k], list):
            ret[k] = []
            for i in rules[k]:
                ar = applyrules({k: i}, d, var)
                if k in ar:
                    ret[k].append(ar[k])
        
        # 如果键以'_'开头，则跳过当前循环，处理下一个键
        elif k[0] == '_':
            continue
        
        # 如果值是字典，初始化一个空列表，然后遍历字典的所有键，并根据类型应用相应的规则
        elif isinstance(rules[k], dict):
            ret[k] = []
            for k1 in rules[k].keys():
                # 如果键是函数类型，并且函数返回True，则处理其值
                if isinstance(k1, types.FunctionType) and k1(var):
                    # 如果值是列表，对列表中的每个元素进行处理，如果元素是字典，使用applyrules函数处理
                    if isinstance(rules[k][k1], list):
                        for i in rules[k][k1]:
                            if isinstance(i, dict):
                                res = applyrules({'supertext': i}, d, var)
                                if 'supertext' in res:
                                    i = res['supertext']
                                else:
                                    i = ''
                            ret[k].append(replace(i, d))
                    else:
                        # 如果值是字典，使用applyrules函数处理
                        i = rules[k][k1]
                        if isinstance(i, dict):
                            res = applyrules({'supertext': i}, d)
                            if 'supertext' in res:
                                i = res['supertext']
                            else:
                                i = ''
                        ret[k].append(replace(i, d))
        
        # 如果以上情况均不符合，则调用错误信息函数errmess，并忽略当前规则
        else:
            errmess('applyrules: ignoring rule %s.\n' % repr(rules[k]))
        
        # 如果返回字典的当前键对应的值是列表
        if isinstance(ret[k], list):
            # 如果列表长度为1，将其转换为单个元素
            if len(ret[k]) == 1:
                ret[k] = ret[k][0]
            # 如果列表为空，删除当前键
            if ret[k] == []:
                del ret[k]
    
    # 返回处理后的结果字典
    return ret
# 匹配以空格开头、python module开头的正则表达式，忽略大小写，用于匹配模块名
_f2py_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]+)',
                                     re.I).match
# 匹配以空格开头、python module开头且包含__user__的正则表达式，忽略大小写，用于排除特定的模块名
_f2py_user_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]*?'
                                          r'__user__[\w_]*)', re.I).match

def get_f2py_modulename(source):
    """
    从给定的源文件中获取F2PY模块的名称。

    Parameters
    ----------
    source : str
        源文件的路径

    Returns
    -------
    str or None
        匹配到的F2PY模块的名称，如果未找到则返回None。
    """
    name = None
    with open(source) as f:
        for line in f:
            m = _f2py_module_name_match(line)
            if m:
                if _f2py_user_module_name_match(line): # 跳过包含`__user__`的模块名
                    continue
                name = m.group('name')
                break
    return name

def getuseblocks(pymod):
    """
    从Python模块的AST表示中提取所有的`use`块中的模块名。

    Parameters
    ----------
    pymod : dict
        Python模块的AST表示，包含`body`字段用于表示模块的主体。

    Returns
    -------
    list
        所有有效的模块名列表，排除包含`__`的特殊模块名。
    """
    all_uses = []
    for inner in pymod['body']:
        for modblock in inner['body']:
            if modblock.get('use'):
                all_uses.extend([x for x in modblock.get("use").keys() if "__" not in x])
    return all_uses

def process_f2cmap_dict(f2cmap_all, new_map, c2py_map, verbose = False):
    """
    更新Fortran到C类型映射字典，并返回成功映射的C类型列表。

    此函数将新的映射字典集成到现有的Fortran到C类型映射字典中。确保所有键都是小写，并针对给定的C到Python映射字典验证新条目。重定义和无效条目将报告警告。

    Parameters
    ----------
    f2cmap_all : dict
        将要更新的现有Fortran到C类型映射字典。它应该是一个字典，其中主键表示Fortran类型，嵌套字典映射Fortran类型说明符到相应的C类型。

    new_map : dict
        包含要添加到`f2cmap_all`中的新类型映射的字典。结构类似于`f2cmap_all`，主键表示Fortran类型，值是类型说明符及其C类型等价物的字典。

    c2py_map : dict
        用于验证`new_map`中的C类型的字典。它将C类型映射到相应的Python类型，用于确保`new_map`中指定的C类型是有效的。

    verbose : boolean
        一个标志，用于提供关于映射类型的信息。

    Returns
    -------
    tuple of (dict, list)
        更新后的Fortran到C类型映射字典和成功映射的C类型列表。
    """
    f2cmap_mapped = []

    new_map_lower = {}
    for k, d1 in new_map.items():
        d1_lower = {k1.lower(): v1 for k1, v1 in d1.items()}
        new_map_lower[k.lower()] = d1_lower
    # 遍历 new_map_lower 字典的键值对
    for k, d1 in new_map_lower.items():
        # 如果 k 不在 f2cmap_all 字典中，则初始化为一个空字典
        if k not in f2cmap_all:
            f2cmap_all[k] = {}

        # 遍历 d1 字典的键值对
        for k1, v1 in d1.items():
            # 如果 v1 在 c2py_map 字典中
            if v1 in c2py_map:
                # 如果 k1 已经存在于 f2cmap_all[k] 中，则输出警告信息
                if k1 in f2cmap_all[k]:
                    outmess(
                        "\tWarning: redefinition of {'%s':{'%s':'%s'->'%s'}}\n"
                        % (k, k1, f2cmap_all[k][k1], v1)
                    )
                # 将 f2cmap_all[k][k1] 设为 v1
                f2cmap_all[k][k1] = v1
                # 如果 verbose 为 True，则输出映射信息
                if verbose:
                    outmess('\tMapping "%s(kind=%s)" to "%s"\n' % (k, k1, v1))
                # 将 v1 添加到 f2cmap_mapped 列表中
                f2cmap_mapped.append(v1)
            else:
                # 如果 verbose 为 True，则输出错误信息，指出 v1 必须在 c2py_map.keys() 中
                if verbose:
                    errmess(
                        "\tIgnoring map {'%s':{'%s':'%s'}}: '%s' must be in %s\n"
                        % (k, k1, v1, v1, list(c2py_map.keys()))
                    )

    # 返回更新后的 f2cmap_all 和已映射的 v1 列表 f2cmap_mapped
    return f2cmap_all, f2cmap_mapped
```