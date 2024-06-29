# `.\numpy\numpy\_utils\_inspect.py`

```
"""
Subset of inspect module from upstream python

We use this instead of upstream because upstream inspect is slow to import, and
significantly contributes to numpy import times. Importing this copy has almost
no overhead.

"""
# 导入 types 模块，用于类型检查
import types

# 定义模块公开的函数和变量
__all__ = ['getargspec', 'formatargspec']

# ----------------------------------------------------------- type-checking
def ismethod(object):
    """Return true if the object is an instance method.

    Instance method objects provide these attributes:
        __doc__         documentation string
        __name__        name with which this method was defined
        im_class        class object in which this method belongs
        im_func         function object containing implementation of method
        im_self         instance to which this method is bound, or None

    """
    # 判断对象是否为方法类型（MethodType）
    return isinstance(object, types.MethodType)

def isfunction(object):
    """Return true if the object is a user-defined function.

    Function objects provide these attributes:
        __doc__         documentation string
        __name__        name with which this function was defined
        func_code       code object containing compiled function bytecode
        func_defaults   tuple of any default values for arguments
        func_doc        (same as __doc__)
        func_globals    global namespace in which this function was defined
        func_name       (same as __name__)

    """
    # 判断对象是否为函数类型（FunctionType）
    return isinstance(object, types.FunctionType)

def iscode(object):
    """Return true if the object is a code object.

    Code objects provide these attributes:
        co_argcount     number of arguments (not including * or ** args)
        co_code         string of raw compiled bytecode
        co_consts       tuple of constants used in the bytecode
        co_filename     name of file in which this code object was created
        co_firstlineno  number of first line in Python source code
        co_flags        bitmap: 1=optimized | 2=newlocals | 4=*arg | 8=**arg
        co_lnotab       encoded mapping of line numbers to bytecode indices
        co_name         name with which this code object was defined
        co_names        tuple of names of local variables
        co_nlocals      number of local variables
        co_stacksize    virtual machine stack space required
        co_varnames     tuple of names of arguments and local variables
        
    """
    # 判断对象是否为代码对象（CodeType）
    return isinstance(object, types.CodeType)

# ------------------------------------------------ argument list extraction
# These constants are from Python's compile.h.
# 定义一些与编译相关的常量，来源于 Python 的 compile.h 文件
CO_OPTIMIZED, CO_NEWLOCALS, CO_VARARGS, CO_VARKEYWORDS = 1, 2, 4, 8

def getargs(co):
    """Get information about the arguments accepted by a code object.

    Three things are returned: (args, varargs, varkw), where 'args' is
    a list of argument names (possibly containing nested lists), and
    'varargs' and 'varkw' are the names of the * and ** arguments or None.

    """
    # 获取代码对象（code object）的参数信息
    # 返回一个包含参数名列表的元组 (args, varargs, varkw)，其中：
    # - args 是参数名列表（可能包含嵌套列表）
    # - varargs 是 * 参数的名称或 None
    # - varkw 是 ** 参数的名称或 None
    # 如果给定的参数 `co` 不是代码对象，则抛出类型错误异常
    if not iscode(co):
        raise TypeError('arg is not a code object')

    # 获取代码对象中的参数数量
    nargs = co.co_argcount
    # 获取代码对象中的局部变量名列表
    names = co.co_varnames
    # 提取参数列表（前 nargs 个变量名）
    args = list(names[:nargs])

    # 以下部分的处理是为了处理匿名的（元组）参数。
    # 我们不需要支持这种情况，因此移除以避免引入 dis 模块。
    for i in range(nargs):
        # 如果参数名以空字符或点号开头，则抛出类型错误异常
        if args[i][:1] in ['', '.']:
            raise TypeError("tuple function arguments are not supported")
    
    varargs = None
    # 如果代码对象的标志位 CO_VARARGS 被设置
    if co.co_flags & CO_VARARGS:
        # 获取可变位置参数的变量名
        varargs = co.co_varnames[nargs]
        # 参数数量加一，以包括可变位置参数
        nargs = nargs + 1
    
    varkw = None
    # 如果代码对象的标志位 CO_VARKEYWORDS 被设置
    if co.co_flags & CO_VARKEYWORDS:
        # 获取可变关键字参数的变量名
        varkw = co.co_varnames[nargs]
    
    # 返回提取的参数列表、可变位置参数名和可变关键字参数名
    return args, varargs, varkw
# 获取函数参数的名称和默认值信息

def getargspec(func):
    """Get the names and default values of a function's arguments.

    A tuple of four things is returned: (args, varargs, varkw, defaults).
    'args' is a list of the argument names (it may contain nested lists).
    'varargs' and 'varkw' are the names of the * and ** arguments or None.
    'defaults' is an n-tuple of the default values of the last n arguments.

    """

    # 如果传入的函数是方法，则获取其函数对象
    if ismethod(func):
        func = func.__func__
    
    # 如果传入的不是函数对象，则抛出类型错误
    if not isfunction(func):
        raise TypeError('arg is not a Python function')
    
    # 使用函数对象的字节码获取参数的详细信息
    args, varargs, varkw = getargs(func.__code__)
    
    # 返回参数名称列表、*args 和 **kwargs 的名称、以及最后 n 个参数的默认值
    return args, varargs, varkw, func.__defaults__


# 获取特定帧中传入参数的信息

def getargvalues(frame):
    """Get information about arguments passed into a particular frame.

    A tuple of four things is returned: (args, varargs, varkw, locals).
    'args' is a list of the argument names (it may contain nested lists).
    'varargs' and 'varkw' are the names of the * and ** arguments or None.
    'locals' is the locals dictionary of the given frame.
    
    """
    # 使用帧对象的字节码获取参数的详细信息
    args, varargs, varkw = getargs(frame.f_code)
    
    # 返回参数名称列表、*args 和 **kwargs 的名称、以及当前帧的局部变量字典
    return args, varargs, varkw, frame.f_locals


# 将序列连接成字符串表示

def joinseq(seq):
    """Join sequence elements into a string, handling single element case.

    """
    # 如果序列只有一个元素，则返回带括号的字符串表示
    if len(seq) == 1:
        return '(' + seq[0] + ',)'
    else:
        # 否则，使用逗号连接序列中的元素，并添加括号
        return '(' + ', '.join(seq) + ')'


# 递归地将序列中的元素转换为字符串表示

def strseq(object, convert, join=joinseq):
    """Recursively walk a sequence, stringifying each element.

    """
    # 如果对象的类型是列表或元组，则递归地将每个元素转换为字符串
    if type(object) in [list, tuple]:
        return join([strseq(_o, convert, join) for _o in object])
    else:
        # 否则，使用给定的转换函数将对象转换为字符串
        return convert(object)


# 格式化函数参数规范

def formatargspec(args, varargs=None, varkw=None, defaults=None,
                  formatarg=str,
                  formatvarargs=lambda name: '*' + name,
                  formatvarkw=lambda name: '**' + name,
                  formatvalue=lambda value: '=' + repr(value),
                  join=joinseq):
    """Format an argument spec from the 4 values returned by getargspec.

    The first four arguments are (args, varargs, varkw, defaults).  The
    other four arguments are the corresponding optional formatting functions
    that are called to turn names and values into strings.  The ninth
    argument is an optional function to format the sequence of arguments.

    """
    # 初始化一个空列表用于存放格式化后的参数规范
    specs = []
    
    # 如果存在默认值，则计算第一个默认参数的索引
    if defaults:
        firstdefault = len(args) - len(defaults)
    
    # 遍历参数列表，逐个格式化成字符串并添加到列表中
    for i in range(len(args)):
        spec = strseq(args[i], formatarg, join)
        if defaults and i >= firstdefault:
            spec = spec + formatvalue(defaults[i - firstdefault])
        specs.append(spec)
    
    # 如果存在 *args 参数，则将其格式化并添加到列表中
    if varargs is not None:
        specs.append(formatvarargs(varargs))
    
    # 如果存在 **kwargs 参数，则将其格式化并添加到列表中
    if varkw is not None:
        specs.append(formatvarkw(varkw))
    
    # 使用指定的连接函数将所有参数规范连接成一个字符串
    return '(' + ', '.join(specs) + ')'


# 格式化函数参数值

def formatargvalues(args, varargs, varkw, locals,
                    formatarg=str,
                    formatvarargs=lambda name: '*' + name,
                    formatvarkw=lambda name: '**' + name,
                    formatvalue=lambda value: '=' + repr(value),
                    join=joinseq):
    """
    根据从 getargvalues 返回的四个值格式化参数规范。

    前四个参数是 (args, varargs, varkw, locals)。接下来四个参数是相应的可选格式化函数，
    用于将名称和值转换为字符串。第九个参数是一个可选的函数，用于格式化参数序列。

    """
    # 定义一个转换函数 convert，接受名称 name 和 locals 参数，默认使用 formatarg 和 formatvalue 函数进行格式化
    def convert(name, locals=locals,
                formatarg=formatarg, formatvalue=formatvalue):
        # 返回格式化后的参数名称和对应值
        return formatarg(name) + formatvalue(locals[name])
    
    # 通过列表推导式，对 args 中的每个参数调用 strseq 函数，并使用 convert 函数进行格式化，最后使用 join 函数连接成字符串
    specs = [strseq(arg, convert, join) for arg in args]

    # 如果存在可变位置参数 varargs，则将其格式化并添加到 specs 列表中
    if varargs:
        specs.append(formatvarargs(varargs) + formatvalue(locals[varargs]))
    
    # 如果存在可变关键字参数 varkw，则将其格式化并添加到 specs 列表中
    if varkw:
        specs.append(formatvarkw(varkw) + formatvalue(locals[varkw]))
    
    # 返回格式化后的参数列表，使用逗号和空格连接成一个字符串，并在两侧添加括号
    return '(' + ', '.join(specs) + ')'
```