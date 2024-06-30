# `D:\src\scipysrc\scipy\scipy\_lib\decorator.py`

```
# #########################     LICENSE     ############################ #

# Copyright (c) 2005-2015, Michele Simionato
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#   Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#   Redistributions in bytecode form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""
Decorator module, see https://pypi.python.org/pypi/decorator
for the documentation.
"""
# 导入需要的模块
import re          # 导入正则表达式模块
import sys         # 导入系统相关的模块
import inspect     # 导入用于获取对象信息的模块
import operator    # 导入操作符模块
import itertools   # 导入迭代工具模块
import collections # 导入集合模块

# 从 inspect 模块中导入 getfullargspec 函数
from inspect import getfullargspec

__version__ = '4.0.5'


def get_init(cls):
    return cls.__init__


# 定义一个命名元组 ArgSpec，用于替代被 Python 3.5 弃用的 getargspec 函数
ArgSpec = collections.namedtuple(
    'ArgSpec', 'args varargs varkw defaults')


def getargspec(f):
    """A replacement for inspect.getargspec"""
    # 使用 getfullargspec 获取函数的参数规范
    spec = getfullargspec(f)
    return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)


# 使用正则表达式定义一个模式对象，用于匹配函数定义
DEF = re.compile(r'\s*def\s*([_\w][_\w\d]*)\s*\(')


# 定义一个类 FunctionMaker，用于动态创建具有指定签名的函数
class FunctionMaker:
    """
    An object with the ability to create functions with a given signature.
    It has attributes name, doc, module, signature, defaults, dict, and
    methods update and make.
    """

    # 使用 itertools.count() 创建一个计数器对象，用于生成唯一的编译计数
    _compile_count = itertools.count()
    def __init__(self, func=None, name=None, signature=None,
                 defaults=None, doc=None, module=None, funcdict=None):
        # 初始化对象的短签名（shortsignature）属性，使用传入的签名信息
        self.shortsignature = signature
        
        # 如果传入了函数对象（func），则进行以下处理
        if func:
            # 设置对象的名称为函数的名称
            self.name = func.__name__
            
            # 对于 lambda 函数，使用 '_lambda_' 替代名称（一个小技巧）
            if self.name == '<lambda>':
                self.name = '_lambda_'
            
            # 设置对象的文档字符串为函数的文档字符串
            self.doc = func.__doc__
            
            # 设置对象的模块为函数所在的模块
            self.module = func.__module__
            
            # 如果 func 是一个函数（而不是类或实例方法），则继续处理
            if inspect.isfunction(func):
                # 获取函数的参数规范
                argspec = getfullargspec(func)
                
                # 设置对象的注解属性为函数的注解（如果有的话）
                self.annotations = getattr(func, '__annotations__', {})
                
                # 设置对象的各种参数信息属性，如 args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults
                for a in ('args', 'varargs', 'varkw', 'defaults', 'kwonlyargs',
                          'kwonlydefaults'):
                    setattr(self, a, getattr(argspec, a))
                
                # 设置对象的每个参数名为 arg0, arg1, ..., 对应的参数名
                for i, arg in enumerate(self.args):
                    setattr(self, 'arg%d' % i, arg)
                
                # 创建包含所有参数的列表和简短参数的列表
                allargs = list(self.args)
                allshortargs = list(self.args)
                
                # 处理可变位置参数（varargs）
                if self.varargs:
                    allargs.append('*' + self.varargs)
                    allshortargs.append('*' + self.varargs)
                # 处理仅关键字参数（kwonlyargs）
                elif self.kwonlyargs:
                    allargs.append('*')  # 单星号语法
                for a in self.kwonlyargs:
                    allargs.append('%s=None' % a)
                    allshortargs.append(f'{a}={a}')
                # 处理可变关键字参数（varkw）
                if self.varkw:
                    allargs.append('**' + self.varkw)
                    allshortargs.append('**' + self.varkw)
                
                # 构建对象的签名字符串，包括所有参数
                self.signature = ', '.join(allargs)
                self.shortsignature = ', '.join(allshortargs)
                
                # 复制函数的字典属性到对象的 dict 属性中
                self.dict = func.__dict__.copy()
        
        # 当 func=None 时，表示在装饰调用者时发生
        # 如果传入了名称（name），则设置对象的名称属性
        if name:
            self.name = name
        # 如果传入了签名（signature），则设置对象的签名属性
        if signature is not None:
            self.signature = signature
        # 如果传入了默认参数（defaults），则设置对象的 defaults 属性
        if defaults:
            self.defaults = defaults
        # 如果传入了文档字符串（doc），则设置对象的 doc 属性
        if doc:
            self.doc = doc
        # 如果传入了模块名称（module），则设置对象的 module 属性
        if module:
            self.module = module
        # 如果传入了函数字典（funcdict），则设置对象的 dict 属性
        if funcdict:
            self.dict = funcdict
        
        # 检查对象是否具有必需的属性 'name'
        assert hasattr(self, 'name')
        
        # 如果对象没有 'signature' 属性，则抛出 TypeError 异常
        if not hasattr(self, 'signature'):
            raise TypeError('You are decorating a non-function: %s' % func)
    # 更新函数对象的签名信息，将当前对象的属性应用到目标函数上
    def update(self, func, **kw):
        # 设置目标函数的名称为当前对象的名称
        func.__name__ = self.name
        # 如果当前对象有文档字符串，则将其设置为目标函数的文档字符串
        func.__doc__ = getattr(self, 'doc', None)
        # 将当前对象的字典属性更新到目标函数的__dict__中
        func.__dict__ = getattr(self, 'dict', {})
        # 将当前对象的默认参数更新到目标函数的__defaults__中
        func.__defaults__ = getattr(self, 'defaults', ())
        # 将当前对象的关键字默认参数更新到目标函数的__kwdefaults__中
        func.__kwdefaults__ = getattr(self, 'kwonlydefaults', None)
        # 将当前对象的注解信息更新到目标函数的__annotations__中
        func.__annotations__ = getattr(self, 'annotations', None)
        
        # 尝试获取调用者的堆栈帧信息，用于确定调用模块的名称
        try:
            frame = sys._getframe(3)
        except AttributeError:  # 处理IronPython等实现中的异常情况
            callermodule = '?'
        else:
            callermodule = frame.f_globals.get('__name__', '?')
        
        # 将当前对象的模块名称更新到目标函数的__module__中
        func.__module__ = getattr(self, 'module', callermodule)
        # 将额外的关键字参数更新到目标函数的__dict__中
        func.__dict__.update(kw)

    # 根据提供的模板src_templ创建一个新的函数，并更新其签名
    def make(self, src_templ, evaldict=None, addsource=False, **attrs):
        # 根据当前对象的属性扩展模板src_templ，生成完整的源代码src
        src = src_templ % vars(self)  # expand name and signature
        # 如果evaldict为None，则初始化为空字典
        evaldict = evaldict or {}
        # 使用正则表达式DEF匹配src是否为有效的函数模板
        mo = DEF.match(src)
        if mo is None:
            # 如果src不是有效的函数模板，则抛出SyntaxError异常
            raise SyntaxError('not a valid function template\n%s' % src)
        # 从模板中提取函数名称
        name = mo.group(1)
        # 根据函数的简短签名创建名称集合
        names = set([name] + [arg.strip(' *') for arg in self.shortsignature.split(',')])
        # 检查生成的函数是否会覆盖内置名称'_func_'或'_call_'
        for n in names:
            if n in ('_func_', '_call_'):
                raise NameError(f'{n} is overridden in\n{src}')
        # 为了安全起见，如果src没有以换行符结尾，则添加一个换行符
        if not src.endswith('\n'):
            src += '\n'  # this is needed in old versions of Python

        # 确保每个生成的函数在分析器（如cProfile）中有唯一的文件名
        filename = '<decorator-gen-%d>' % (next(self._compile_count),)
        try:
            # 编译源代码src，生成代码对象
            code = compile(src, filename, 'single')
            # 在evaldict中执行生成的代码
            exec(code, evaldict)
        except:  # noqa: E722
            # 如果生成代码出错，则在标准错误流中输出错误信息和源代码，并重新抛出异常
            print('Error in generated code:', file=sys.stderr)
            print(src, file=sys.stderr)
            raise
        # 从evaldict中获取生成的函数对象
        func = evaldict[name]
        # 如果addsource为True，则将生成的源代码添加到attrs字典中
        if addsource:
            attrs['__source__'] = src
        # 使用update方法更新func函数的属性
        self.update(func, **attrs)
        # 返回生成的函数对象
        return func
    # 创建函数的类方法，用于根据提供的字符串创建一个函数
    def create(cls, obj, body, evaldict, defaults=None,
               doc=None, module=None, addsource=True, **attrs):
        """
        Create a function from the strings name, signature, and body.
        evaldict is the evaluation dictionary. If addsource is true, an
        attribute __source__ is added to the result. The attributes attrs
        are added, if any.
        """
        # 如果 obj 是字符串，则解析出函数名和参数签名
        if isinstance(obj, str):  # "name(signature)"
            name, rest = obj.strip().split('(', 1)
            signature = rest[:-1]  # 去除末尾的右括号
            func = None
        else:  # 如果 obj 是一个函数对象
            name = None
            signature = None
            func = obj
        # 使用提供的参数初始化类实例 self
        self = cls(func, name, signature, defaults, doc, module)
        # 将函数体每一行都缩进四个空格，与 Python 函数定义的缩进格式一致
        ibody = '\n'.join('    ' + line for line in body.splitlines())
        # 调用实例方法 make 创建函数，返回结果
        return self.make('def %(name)s(%(signature)s):\n' + ibody,
                         evaldict, addsource, **attrs)
# 定义一个装饰器函数，用于装饰给定的函数，并指定调用者函数
def decorate(func, caller):
    """
    decorate(func, caller) decorates a function using a caller.
    """
    # 复制函数的全局命名空间
    evaldict = func.__globals__.copy()
    # 将调用者和函数本身添加到命名空间中
    evaldict['_call_'] = caller
    evaldict['_func_'] = func
    # 使用FunctionMaker.create创建一个新的装饰后的函数
    fun = FunctionMaker.create(
        func, "return _call_(_func_, %(shortsignature)s)",
        evaldict, __wrapped__=func)
    # 如果函数具有__qualname__属性，则将其复制到新创建的函数中
    if hasattr(func, '__qualname__'):
        fun.__qualname__ = func.__qualname__
    return fun


# 定义一个将调用者函数转换为装饰器的函数
def decorator(caller, _func=None):
    """decorator(caller) converts a caller function into a decorator"""
    if _func is not None:  # 如果提供了函数作为参数，返回一个装饰后的函数
        # 这是过时的行为；应该使用decorate代替
        return decorate(_func, caller)
    # 否则返回一个装饰器函数
    if inspect.isclass(caller):  # 如果调用者是一个类
        name = caller.__name__.lower()
        callerfunc = get_init(caller)
        doc = (f'decorator({caller.__name__}) converts functions/generators into '
               f'factories of {caller.__name__} objects')
    elif inspect.isfunction(caller):  # 如果调用者是一个函数
        if caller.__name__ == '<lambda>':
            name = '_lambda_'
        else:
            name = caller.__name__
        callerfunc = caller
        doc = caller.__doc__
    else:  # 假设调用者是一个带有__call__方法的对象
        name = caller.__class__.__name__.lower()
        callerfunc = caller.__call__.__func__
        doc = caller.__call__.__doc__
    # 复制调用者函数的全局命名空间
    evaldict = callerfunc.__globals__.copy()
    # 将调用者和装饰函数添加到命名空间中
    evaldict['_call_'] = caller
    evaldict['_decorate_'] = decorate
    # 使用FunctionMaker.create创建一个新的装饰器函数
    return FunctionMaker.create(
        '%s(func)' % name, 'return _decorate_(func, _call_)',
        evaldict, doc=doc, module=caller.__module__,
        __wrapped__=caller)


# ####################### contextmanager ####################### #

try:  # 尝试导入Python >= 3.2所需的模块
    from contextlib import _GeneratorContextManager
except ImportError:  # 对于Python >= 2.5
    from contextlib import GeneratorContextManager as _GeneratorContextManager


# 定义一个自定义的上下文管理器类，继承自_GeneratorContextManager
class ContextManager(_GeneratorContextManager):
    def __call__(self, func):
        """Context manager decorator"""
        # 创建一个新的函数，用于实现上下文管理器的装饰器功能
        return FunctionMaker.create(
            func, "with _self_: return _func_(%(shortsignature)s)",
            dict(_self_=self, _func_=func), __wrapped__=func)


# 检查_GeneratorContextManager.__init__函数的参数数量
init = getfullargspec(_GeneratorContextManager.__init__)
n_args = len(init.args)
# 根据参数数量的不同，定制化初始化函数的行为
if n_args == 2 and not init.varargs:  # (self, genobj) Python 2.7
    # 定义一个自定义的初始化函数，用于Python 2.7
    def __init__(self, g, *a, **k):
        return _GeneratorContextManager.__init__(self, g(*a, **k))
    ContextManager.__init__ = __init__
elif n_args == 2 and init.varargs:  # (self, gen, *a, **k) Python 3.4
    # 对于Python 3.4，无需进行特别处理
    pass
elif n_args == 4:  # (self, gen, args, kwds) Python 3.5
    # 定义一个自定义的初始化函数，用于Python 3.5
    def __init__(self, g, *a, **k):
        return _GeneratorContextManager.__init__(self, g, a, k)
    ContextManager.__init__ = __init__

# 将contextmanager函数定义为一个装饰器，使用上面定义的decorator函数
contextmanager = decorator(ContextManager)


# ############################ dispatch_on ############################ #

# 定义一个append函数，暂时没有提供具体实现
def append(a, vancestors):
    """
    # 将类 `a` 添加到虚拟祖先列表 `vancestors` 中，除非它已经包含在内。

    add = True
    # 遍历虚拟祖先列表 `vancestors`
    for j, va in enumerate(vancestors):
        # 如果 `a` 是 `va` 的子类，则不需要添加 `a` 到列表中
        if issubclass(va, a):
            add = False
            break
        # 如果 `va` 是 `a` 的子类，则用 `a` 替换 `va` 在列表中的位置，并且不添加 `a`
        if issubclass(a, va):
            vancestors[j] = a
            add = False
    # 如果 `a` 不是任何现有祖先类的子类，则将其添加到列表末尾
    if add:
        vancestors.append(a)
# 从 simplegeneric 由 P.J. Eby 和 functools.singledispatch 获得灵感
def dispatch_on(*dispatch_args):
    """
    工厂函数，将一个函数转变为一个通用函数，
    根据给定的参数进行分派。
    """
    assert dispatch_args, 'No dispatch args passed'
    dispatch_str = '(%s,)' % ', '.join(dispatch_args)

    def check(arguments, wrong=operator.ne, msg=''):
        """确保传入了预期数量的参数"""
        if wrong(len(arguments), len(dispatch_args)):
            raise TypeError('Expected %d arguments, got %d%s' %
                            (len(dispatch_args), len(arguments), msg))

    # 设置生成的通用函数装饰器的名称
    gen_func_dec.__name__ = 'dispatch_on' + dispatch_str
    return gen_func_dec
```