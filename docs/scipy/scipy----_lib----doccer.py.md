# `D:\src\scipysrc\scipy\scipy\_lib\doccer.py`

```
# 导入 sys 模块
import sys

# 模块中公开的函数和方法
__all__ = [
    'docformat', 'inherit_docstring_from', 'indentcount_lines',
    'filldoc', 'unindent_dict', 'unindent_string', 'extend_notes_in_docstring',
    'replace_notes_in_docstring', 'doc_replace'
]


def docformat(docstring, docdict=None):
    ''' Fill a function docstring from variables in dictionary

    Adapt the indent of the inserted docs

    Parameters
    ----------
    docstring : string
        docstring from function, possibly with dict formatting strings
    docdict : dict, optional
        dictionary with keys that match the dict formatting strings
        and values that are docstring fragments to be inserted. The
        indentation of the inserted docstrings is set to match the
        minimum indentation of the ``docstring`` by adding this
        indentation to all lines of the inserted string, except the
        first.

    Returns
    -------
    outstring : string
        string with requested ``docdict`` strings inserted

    Examples
    --------
    >>> docformat(' Test string with %(value)s', {'value':'inserted value'})
    ' Test string with inserted value'
    >>> docstring = 'First line\\n    Second line\\n    %(value)s'
    >>> inserted_string = "indented\\nstring"
    >>> docdict = {'value': inserted_string}
    >>> docformat(docstring, docdict)
    'First line\\n    Second line\\n    indented\\n    string'
    '''
    # 如果没有输入文档字符串，直接返回空
    if not docstring:
        return docstring
    # 如果没有提供 docdict，则设为空字典
    if docdict is None:
        docdict = {}
    # 如果 docdict 为空，则直接返回原始文档字符串
    if not docdict:
        return docstring
    # 将文档字符串按制表符扩展并按行分割
    lines = docstring.expandtabs().splitlines()
    # 计算主文档字符串的最小缩进（第一行之后）
    if len(lines) < 2:
        icount = 0
    else:
        icount = indentcount_lines(lines[1:])
    indent = ' ' * icount
    # 对字典中的文档片段进行缩进处理
    indented = {}
    for name, dstr in docdict.items():
        lines = dstr.expandtabs().splitlines()
        try:
            newlines = [lines[0]]
            for line in lines[1:]:
                newlines.append(indent + line)
            indented[name] = '\n'.join(newlines)
        except IndexError:
            indented[name] = dstr
    # 将缩进后的文档片段插入原始文档字符串中的格式化字符串中
    return docstring % indented


def inherit_docstring_from(cls):
    """
    This decorator modifies the decorated function's docstring by
    replacing occurrences of '%(super)s' with the docstring of the
    method of the same name from the class `cls`.

    If the decorated method has no docstring, it is simply given the
    docstring of `cls`s method.

    Parameters
    ----------
    cls : Python class or instance
        A class with a method with the same name as the decorated method.
        The docstring of the method in this class replaces '%(super)s' in the
        docstring of the decorated method.

    Returns
    -------

    """
    f : function
        一个装饰器函数，用于修改其参数的 __doc__ 属性。

    Examples
    --------
    以下示例展示了使用 `Foo.func` 的文档字符串来创建 `Bar.func` 的文档字符串。

    >>> class Foo:
    ...     def func(self):
    ...         '''Do something useful.'''
    ...         return
    ...
    >>> class Bar(Foo):
    ...     @inherit_docstring_from(Foo)
    ...     def func(self):
    ...         '''%(super)s
    ...         Do it fast.
    ...         '''
    ...         return
    ...
    >>> b = Bar()
    >>> b.func.__doc__
    'Do something useful.\n        Do it fast.\n        '

    """
    # 定义一个内部函数 _doc，它接受一个函数作为参数 func
    def _doc(func):
        # 获取 cls 中 func 方法的文档字符串
        cls_docstring = getattr(cls, func.__name__).__doc__
        # 获取 func 自身的文档字符串
        func_docstring = func.__doc__
        # 如果 func 的文档字符串为 None，则将其设置为 cls 的文档字符串
        if func_docstring is None:
            func.__doc__ = cls_docstring
        # 否则，使用 cls 的文档字符串来格式化 func 的文档字符串
        else:
            new_docstring = func_docstring % dict(super=cls_docstring)
            func.__doc__ = new_docstring
        # 返回经过处理的 func 函数
        return func
    # 返回内部函数 _doc 本身
    return _doc
def extend_notes_in_docstring(cls, notes):
    """
    This decorator replaces the decorated function's docstring
    with the docstring from corresponding method in `cls`.
    It extends the 'Notes' section of that docstring to include
    the given `notes`.
    """
    def _doc(func):
        # 获取被装饰函数的名称对应的类方法的文档字符串
        cls_docstring = getattr(cls, func.__name__).__doc__
        # 如果类方法没有文档字符串，直接返回原始函数
        if cls_docstring is None:
            return func
        # 查找文档字符串中 'Notes' 部分的结束位置
        end_of_notes = cls_docstring.find('        References\n')
        if end_of_notes == -1:
            end_of_notes = cls_docstring.find('        Examples\n')
            if end_of_notes == -1:
                end_of_notes = len(cls_docstring)
        # 将给定的注释 notes 扩展到 'Notes' 部分后面，并更新函数的文档字符串
        func.__doc__ = (cls_docstring[:end_of_notes] + notes +
                        cls_docstring[end_of_notes:])
        return func
    return _doc


def replace_notes_in_docstring(cls, notes):
    """
    This decorator replaces the decorated function's docstring
    with the docstring from corresponding method in `cls`.
    It replaces the 'Notes' section of that docstring with
    the given `notes`.
    """
    def _doc(func):
        # 获取被装饰函数的名称对应的类方法的文档字符串
        cls_docstring = getattr(cls, func.__name__).__doc__
        # 'Notes' 部分的标头
        notes_header = '        Notes\n        -----\n'
        # 如果类方法没有文档字符串，直接返回原始函数
        if cls_docstring is None:
            return func
        # 查找文档字符串中 'Notes' 部分的起始位置和结束位置
        start_of_notes = cls_docstring.find(notes_header)
        end_of_notes = cls_docstring.find('        References\n')
        if end_of_notes == -1:
            end_of_notes = cls_docstring.find('        Examples\n')
            if end_of_notes == -1:
                end_of_notes = len(cls_docstring)
        # 替换 'Notes' 部分为给定的注释 notes，并更新函数的文档字符串
        func.__doc__ = (cls_docstring[:start_of_notes + len(notes_header)] +
                        notes +
                        cls_docstring[end_of_notes:])
        return func
    return _doc


def indentcount_lines(lines):
    ''' Minimum indent for all lines in line list

    >>> lines = [' one', '  two', '   three']
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(['    '])
    0
    '''
    # 初始化最小缩进为系统最大值
    indentno = sys.maxsize
    # 遍历所有行，找到最小缩进数
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    # 如果没有找到非空行，返回 0，否则返回最小缩进数
    if indentno == sys.maxsize:
        return 0
    return indentno


def filldoc(docdict, unindent_params=True):
    ''' Return docstring decorator using docdict variable dictionary

    Parameters
    ----------
    docdict : dictionary
        dictionary containing name, docstring fragment pairs
    unindent_params : {False, True}, boolean, optional
        If True, strip common indentation from all parameters in
        docdict

    Returns
    -------
    decfunc : function
        decorator that applies dictionary to input function docstring
    '''
    '''
    如果 unindent_params 不为空，则调用 unindent_dict 函数处理 docdict 字典。
    
    定义一个名为 decorate 的函数，接受一个函数 f 作为参数。
    将函数 f 的文档字符串用 docdict 字典中的内容格式化，然后将其赋值给 f.__doc__。
    返回经过修饰后的函数 f。
    
    返回 decorate 函数的引用。
    '''
# 将字典中的每个字符串内容去除最小缩进
def unindent_dict(docdict):
    ''' Unindent all strings in a docdict '''
    # 创建一个新的字典用于存放处理后的字符串
    can_dict = {}
    # 遍历原始字典中的每个键值对
    for name, dstr in docdict.items():
        # 对每个值应用 unindent_string 函数处理，并存入新字典
        can_dict[name] = unindent_string(dstr)
    # 返回处理后的字典
    return can_dict


def unindent_string(docstring):
    ''' Set docstring to minimum indent for all lines, including first

    >>> unindent_string(' two')
    'two'
    >>> unindent_string('  two\\n   three')
    'two\\n three'
    '''
    # 将文档字符串根据制表符扩展后拆分成行列表
    lines = docstring.expandtabs().splitlines()
    # 计算所有行的最小缩进量
    icount = indentcount_lines(lines)
    # 如果最小缩进为0，直接返回原始文档字符串
    if icount == 0:
        return docstring
    # 否则，移除每一行对应的最小缩进量，并重新拼接成字符串返回
    return '\n'.join([line[icount:] for line in lines])


def doc_replace(obj, oldval, newval):
    """Decorator to take the docstring from obj, with oldval replaced by newval

    Equivalent to ``func.__doc__ = obj.__doc__.replace(oldval, newval)``

    Parameters
    ----------
    obj : object
        The object to take the docstring from.
    oldval : string
        The string to replace from the original docstring.
    newval : string
        The string to replace ``oldval`` with.
    """
    # 对象的 __doc__ 属性在优化模式下可能为 None
    # 用 obj 的文档字符串替换 oldval 为 newval 后的结果
    doc = (obj.__doc__ or '').replace(oldval, newval)

    def inner(func):
        # 将修饰的函数的文档字符串设置为处理后的文档字符串
        func.__doc__ = doc
        return func

    return inner
```