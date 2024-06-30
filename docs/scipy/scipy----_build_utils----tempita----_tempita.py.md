# `D:\src\scipysrc\scipy\scipy\_build_utils\tempita\_tempita.py`

```
"""
A small templating language

This implements a small templating language.  This language implements
if/elif/else, for/continue/break, expressions, and blocks of Python
code.  The syntax is::

  {{any expression (function calls etc)}}
  {{any expression | filter}}
  {{for x in y}}...{{endfor}}
  {{if x}}x{{elif y}}y{{else}}z{{endif}}
  {{py:x=1}}
  {{py:
  def foo(bar):
      return 'baz'
  }}
  {{default var = default_value}}
  {{# comment}}

You use this with the ``Template`` class or the ``sub`` shortcut.
The ``Template`` class takes the template string and the name of
the template (for errors) and a default namespace.  Then (like
``string.Template``) you can call the ``tmpl.substitute(**kw)``
method to make a substitution (or ``tmpl.substitute(a_dict)``).

``sub(content, **kw)`` substitutes the template immediately.  You
can use ``__name='tmpl.html'`` to set the name of the template.

If there are syntax errors ``TemplateError`` will be raised.
"""

import re
import sys
import os
import tokenize
from io import StringIO

from ._looper import looper

__all__ = ['TemplateError', 'Template', 'sub', 'bunch']

# 正则表达式，用于匹配模板语法中的 ' in '
in_re = re.compile(r'\s+in\s+')
# 正则表达式，用于验证变量名的合法性（以字母或下划线开头，后接字母、数字或下划线）
var_re = re.compile(r'^[a-z_][a-z0-9_]*$', re.I)
# 定义一个元组，包含 bytes 和 str 类型，用于文本类型的强制转换
basestring_ = (bytes, str)

def coerce_text(v):
    # 如果 v 不是字符串类型
    if not isinstance(v, basestring_):
        # 如果 v 有 '__str__' 方法，调用该方法转换为字符串
        if hasattr(v, '__str__'):
            return str(v)
        # 否则将 v 转换为 bytes 类型
        else:
            return bytes(v)
    # 如果 v 已经是字符串类型，则直接返回 v
    return v

# 自定义异常类，用于模板解析过程中的异常情况
class TemplateError(Exception):
    """Exception raised while parsing a template
    """

    def __init__(self, message, position, name=None):
        Exception.__init__(self, message)
        self.position = position
        self.name = name

    def __str__(self):
        msg = ' '.join(self.args)
        # 如果存在错误位置信息，则将其添加到错误消息中
        if self.position:
            msg = '%s at line %s column %s' % (
                msg, self.position[0], self.position[1])
        # 如果存在模板名称信息，则将其添加到错误消息中
        if self.name:
            msg += ' in %s' % self.name
        return msg

# 自定义异常类，用于模板中的 continue 语句
class _TemplateContinue(Exception):
    pass

# 自定义异常类，用于模板中的 break 语句
class _TemplateBreak(Exception):
    pass

# 根据文件名和源模板获取模板文件的路径
def get_file_template(name, from_template):
    path = os.path.join(os.path.dirname(from_template.name), name)
    return from_template.__class__.from_filename(
        path, namespace=from_template.namespace,
        get_template=from_template.get_template)

# 模板类，实现模板解析和替换功能
class Template:

    # 默认的命名空间，包含模板语法的起始符和结束符，以及 looper 函数
    default_namespace = {
        'start_braces': '{{',
        'end_braces': '}}',
        'looper': looper,
    }

    # 默认的编码方式为 utf8
    default_encoding = 'utf8'
    # 默认的继承方式为空
    default_inherit = None
    def __init__(self, content, name=None, namespace=None, stacklevel=None,
                 get_template=None, default_inherit=None, line_offset=0,
                 delimiters=None, delimeters=None):
        self.content = content  # 将传入的内容赋给对象的content属性

        # 设置分隔符
        if delimeters:
            import warnings
            warnings.warn(
                "'delimeters' kwarg is being deprecated in favor of correctly"
                " spelled 'delimiters'. Please adjust your code.",
                DeprecationWarning
            )
            if delimiters is None:
                delimiters = delimeters  # 如果delimeters不为空，则将其赋给delimiters
        if delimiters is None:
            # 如果delimiters为空，则使用默认的起止符号
            delimiters = (self.default_namespace['start_braces'],
                          self.default_namespace['end_braces'])
        else:
            # 如果delimiters不为空，则更新默认命名空间中的起止符号
            self.default_namespace = self.__class__.default_namespace.copy()
            self.default_namespace['start_braces'] = delimiters[0]
            self.default_namespace['end_braces'] = delimiters[1]
        self.delimiters = self.delimeters = delimiters  # 保留一个只读副本，但不建议使用

        self._unicode = isinstance(content, str)  # 判断传入的content是否为字符串类型
        if name is None and stacklevel is not None:
            try:
                caller = sys._getframe(stacklevel)
            except ValueError:
                pass
            else:
                globals = caller.f_globals
                lineno = caller.f_lineno
                if '__file__' in globals:
                    name = globals['__file__']
                    if name.endswith('.pyc') or name.endswith('.pyo'):
                        name = name[:-1]
                elif '__name__' in globals:
                    name = globals['__name__']
                else:
                    name = '<string>'
                if lineno:
                    name += ':%s' % lineno
        self.name = name  # 设置对象的name属性为传入的name值
        # 解析传入的内容，生成解析后的结果，存储在_parsed属性中
        self._parsed = parse(content, name=name, line_offset=line_offset, delimiters=self.delimiters)
        if namespace is None:
            namespace = {}  # 如果未传入namespace，则初始化为空字典
        self.namespace = namespace  # 设置对象的namespace属性为传入的namespace
        self.get_template = get_template  # 设置对象的get_template属性为传入的get_template
        if default_inherit is not None:
            self.default_inherit = default_inherit  # 如果传入了default_inherit，则设置对象的default_inherit属性为传入值

    def from_filename(cls, filename, namespace=None, encoding=None,
                      default_inherit=None, get_template=get_file_template):
        with open(filename, 'rb') as f:
            c = f.read()  # 读取文件内容到变量c中
        if encoding:
            c = c.decode(encoding)  # 如果指定了编码，则使用指定的编码解码文件内容
        return cls(content=c, name=filename, namespace=namespace,
                   default_inherit=default_inherit, get_template=get_template)

    from_filename = classmethod(from_filename)  # 将from_filename方法声明为类方法

    def __repr__(self):
        return '<%s %s name=%r>' % (
            self.__class__.__name__,  # 返回对象的类名
            hex(id(self))[2:],  # 返回对象的内存地址的十六进制表示
            self.name  # 返回对象的名称属性
        )
    # 定义一个方法用于变量替换，支持多种参数传递方式
    def substitute(self, *args, **kw):
        # 如果有位置参数
        if args:
            # 如果同时有关键字参数，抛出类型错误异常
            if kw:
                raise TypeError(
                    "You can only give positional *or* keyword arguments")
            # 如果位置参数超过一个，抛出类型错误异常
            if len(args) > 1:
                raise TypeError(
                    "You can only give one positional argument")
            # 如果第一个位置参数不具备 'items' 方法（不是字典类对象），抛出类型错误异常
            if not hasattr(args[0], 'items'):
                raise TypeError(
                    "If you pass in a single argument, you must pass in a dictionary-like object (with a .items() method); you gave %r"
                    % (args[0],))
            # 将位置参数作为关键字参数
            kw = args[0]
        
        # 将关键字参数保存到局部变量 ns 中
        ns = kw
        # 向 ns 字典中添加特殊键 '__template_name__'，其值为当前对象的名称 self.name
        ns['__template_name__'] = self.name
        
        # 如果存在命名空间 self.namespace，则将其内容更新到 ns 字典中
        if self.namespace:
            ns.update(self.namespace)
        
        # 调用 _interpret 方法进行模板解析，获取解析结果、定义和继承信息
        result, defs, inherit = self._interpret(ns)
        
        # 如果没有指定继承方式，使用默认的继承方式 self.default_inherit
        if not inherit:
            inherit = self.default_inherit
        
        # 如果需要继承，调用 _interpret_inherit 方法处理继承逻辑
        if inherit:
            result = self._interpret_inherit(result, defs, inherit, ns)
        
        # 返回最终的结果字符串
        return result

    # 私有方法：根据命名空间 ns 进行模板解析
    def _interpret(self, ns):
        __traceback_hide__ = True
        # 初始化空列表 parts 用于保存解析结果的片段
        parts = []
        # 初始化空字典 defs 用于保存定义
        defs = {}
        # 调用 _interpret_codes 方法解析 self._parsed 中的代码块
        self._interpret_codes(self._parsed, ns, out=parts, defs=defs)
        
        # 如果定义中包含 '__inherit__' 键，将其弹出并赋值给 inherit 变量；否则设置 inherit 为 None
        if '__inherit__' in defs:
            inherit = defs.pop('__inherit__')
        else:
            inherit = None
        
        # 返回解析后的结果字符串、定义字典及继承信息
        return ''.join(parts), defs, inherit

    # 私有方法：处理模板的继承逻辑
    def _interpret_inherit(self, body, defs, inherit_template, ns):
        __traceback_hide__ = True
        # 如果没有指定获取模板的方法 self.get_template，抛出模板错误异常
        if not self.get_template:
            raise TemplateError(
                'You cannot use inheritance without passing in get_template',
                position=None, name=self.name)
        
        # 使用 inherit_template 和当前对象 self 获取模板对象 templ
        templ = self.get_template(inherit_template, self)
        
        # 创建 TemplateObject 对象 self_
        self_ = TemplateObject(self.name)
        # 将 defs 字典中的内容作为属性设置到 self_ 对象中
        for name, value in defs.items():
            setattr(self_, name, value)
        # 设置 self_.body 为当前模板的主体内容 body
        self_.body = body
        
        # 复制一份命名空间 ns，并添加 'self' 键，其值为 self_ 对象
        ns = ns.copy()
        ns['self'] = self_
        
        # 调用 templ 对象的 substitute 方法，替换命名空间 ns 中的内容
        return templ.substitute(ns)

    # 私有方法：根据 codes 中的代码块解析命名空间 ns，将结果存储到 out 中，定义存储到 defs 中
    def _interpret_codes(self, codes, ns, out, defs):
        __traceback_hide__ = True
        # 遍历 codes 列表中的每一个元素 item
        for item in codes:
            # 如果 item 是字符串类型，直接将其添加到 out 中
            if isinstance(item, basestring_):  # basestring_ 应为合适的字符串类型判断条件，如 str
                out.append(item)
            else:
                # 否则调用 _interpret_code 方法解析 item，并传入命名空间 ns、输出列表 out 和定义字典 defs
                self._interpret_code(item, ns, out, defs)
    # 解析给定的模板代码片段，并执行相应的操作
    def _interpret_code(self, code, ns, out, defs):
        # 隐藏异常跟踪信息
        __traceback_hide__ = True
        # 提取代码片段的名称和位置信息
        name, pos = code[0], code[1]
        
        # 根据代码片段的名称执行相应的操作
        if name == 'py':
            # 执行 Python 代码片段
            self._exec(code[2], ns, pos)
        elif name == 'continue':
            # 抛出模板继续异常
            raise _TemplateContinue()
        elif name == 'break':
            # 抛出模板中断异常
            raise _TemplateBreak()
        elif name == 'for':
            # 解析循环变量、表达式和内容
            vars, expr, content = code[2], code[3], code[4]
            expr = self._eval(expr, ns, pos)
            # 执行 for 循环
            self._interpret_for(vars, expr, content, ns, out, defs)
        elif name == 'cond':
            # 解析条件语句部分
            parts = code[2:]
            self._interpret_if(parts, ns, out, defs)
        elif name == 'expr':
            # 解析表达式部分
            parts = code[2].split('|')
            base = self._eval(parts[0], ns, pos)
            # 对表达式应用管道中的函数
            for part in parts[1:]:
                func = self._eval(part, ns, pos)
                base = func(base)
            # 将结果添加到输出列表中
            out.append(self._repr(base, pos))
        elif name == 'default':
            # 处理默认值赋值操作
            var, expr = code[2], code[3]
            if var not in ns:
                # 如果变量不存在于命名空间中，则计算表达式并赋值
                result = self._eval(expr, ns, pos)
                ns[var] = result
        elif name == 'inherit':
            # 处理继承相关操作
            expr = code[2]
            value = self._eval(expr, ns, pos)
            # 将继承的值存储在定义字典中
            defs['__inherit__'] = value
        elif name == 'def':
            # 处理定义操作
            name = code[2]
            signature = code[3]
            parts = code[4]
            # 创建模板定义对象并存储在命名空间和定义字典中
            ns[name] = defs[name] = TemplateDef(self, name, signature, body=parts, ns=ns,
                                                pos=pos)
        elif name == 'comment':
            # 如果是注释类型的代码片段则直接返回，不做处理
            return
        else:
            # 如果代码片段名称未知则抛出断言错误
            assert 0, "Unknown code: %r" % name

    # 解析 for 循环代码片段
    def _interpret_for(self, vars, expr, content, ns, out, defs):
        __traceback_hide__ = True
        # 遍历表达式中的每个项目
        for item in expr:
            # 处理单变量和多变量的情况
            if len(vars) == 1:
                ns[vars[0]] = item
            else:
                # 检查变量数量是否匹配
                if len(vars) != len(item):
                    raise ValueError(
                        'Need %i items to unpack (got %i items)'
                        % (len(vars), len(item)))
                # 将变量名和对应的值绑定并存储在命名空间中
                for name, value in zip(vars, item):
                    ns[name] = value
            try:
                # 执行循环体内容的解析操作
                self._interpret_codes(content, ns, out, defs)
            except _TemplateContinue:
                # 如果遇到模板继续异常则继续下一次循环
                continue
            except _TemplateBreak:
                # 如果遇到模板中断异常则中断循环
                break

    # 解析条件语句代码片段
    def _interpret_if(self, parts, ns, out, defs):
        __traceback_hide__ = True
        # 处理条件语句的各个部分
        for part in parts:
            assert not isinstance(part, basestring_)
            name, pos = part[0], part[1]
            if name == 'else':
                result = True
            else:
                # 计算条件表达式的值
                result = self._eval(part[2], ns, pos)
            # 根据条件判断执行相应的代码片段
            if result:
                self._interpret_codes(part[3], ns, out, defs)
                break
    # 定义一个私有方法 `_eval`，用于评估给定的代码字符串
    def _eval(self, code, ns, pos):
        # 设置特殊的隐藏标志以隐藏此处的跟踪信息
        __traceback_hide__ = True
        try:
            try:
                # 使用内置的 `eval` 函数评估代码，在默认命名空间和给定命名空间中执行
                value = eval(code, self.default_namespace, ns)
            except SyntaxError as e:
                # 如果发生语法错误，重新引发一个带有更具体信息的 `SyntaxError`
                raise SyntaxError(
                    'invalid syntax in expression: %s' % code)
            # 返回评估后的值
            return value
        except Exception as e:
            # 处理捕获到的任何异常
            if getattr(e, 'args', None):
                arg0 = e.args[0]
            else:
                arg0 = coerce_text(e)
            # 将异常信息和发生位置添加到异常参数中，并重新引发异常
            e.args = (self._add_line_info(arg0, pos),)
            raise

    # 定义一个私有方法 `_exec`，用于执行给定的代码字符串
    def _exec(self, code, ns, pos):
        # 设置特殊的隐藏标志以隐藏此处的跟踪信息
        __traceback_hide__ = True
        try:
            # 使用内置的 `exec` 函数执行代码，在默认命名空间和给定命名空间中执行
            exec(code, self.default_namespace, ns)
        except Exception as e:
            # 处理捕获到的任何异常
            if e.args:
                e.args = (self._add_line_info(e.args[0], pos),)
            else:
                e.args = (self._add_line_info(None, pos),)
            # 重新引发异常
            raise

    # 定义一个私有方法 `_repr`，用于返回给定值的字符串表示
    def _repr(self, value, pos):
        # 设置特殊的隐藏标志以隐藏此处的跟踪信息
        __traceback_hide__ = True
        try:
            # 如果值为 None，则返回空字符串
            if value is None:
                return ''
            # 如果启用了 Unicode 模式
            if self._unicode:
                try:
                    # 尝试将值转换为字符串
                    value = str(value)
                except UnicodeDecodeError:
                    # 如果发生 Unicode 解码错误，则将值视为字节序列
                    value = bytes(value)
            else:
                # 如果不是字符串类型，则尝试强制转换为文本
                if not isinstance(value, basestring_):
                    value = coerce_text(value)
                # 如果值为字符串且有默认编码，则使用默认编码进行编码
                if (isinstance(value, str)
                        and self.default_encoding):
                    value = value.encode(self.default_encoding)
        except Exception as e:
            # 处理捕获到的任何异常
            e.args = (self._add_line_info(e.args[0], pos),)
            # 重新引发异常
            raise
        else:
            # 如果启用了 Unicode 模式且值为字节序列，则尝试解码为字符串
            if self._unicode and isinstance(value, bytes):
                if not self.default_encoding:
                    raise UnicodeDecodeError(
                        'Cannot decode bytes value %r into unicode '
                        '(no default_encoding provided)' % value)
                try:
                    value = value.decode(self.default_encoding)
                except UnicodeDecodeError as e:
                    raise UnicodeDecodeError(
                        e.encoding,
                        e.object,
                        e.start,
                        e.end,
                        e.reason + ' in string %r' % value)
            # 如果未启用 Unicode 模式且值为字符串，则尝试使用默认编码进行编码
            elif not self._unicode and isinstance(value, str):
                if not self.default_encoding:
                    raise UnicodeEncodeError(
                        'Cannot encode unicode value %r into bytes '
                        '(no default_encoding provided)' % value)
                value = value.encode(self.default_encoding)
            # 返回处理后的值
            return value

    # 定义一个私有方法 `_add_line_info`，用于在错误消息中添加行信息
    def _add_line_info(self, msg, pos):
        # 格式化错误消息，包括行号和列号，并可能包含文件名信息
        msg = "%s at line %s column %s" % (
            msg, pos[0], pos[1])
        if self.name:
            msg += " in file %s" % self.name
        # 返回带有完整信息的错误消息
        return msg
# 定义一个函数 `sub`，用于替换模板内容中的变量
def sub(content, delimiters=None, **kw):
    # 获取参数字典中的 '__name' 键对应的值
    name = kw.get('__name')
    # 从参数字典中获取 'delimeters' 键对应的值（用于兼容旧代码）
    delimeters = kw.pop('delimeters') if 'delimeters' in kw else None
    # 使用模板 `Template` 对象，将内容进行替换
    tmpl = Template(content, name=name, delimiters=delimiters, delimeters=delimeters)
    # 返回替换后的结果
    return tmpl.substitute(kw)


# 定义一个函数 `paste_script_template_renderer`，用于渲染粘贴脚本模板
def paste_script_template_renderer(content, vars, filename=None):
    # 使用模板 `Template` 对象，将内容进行替换
    tmpl = Template(content, name=filename)
    # 使用变量字典 `vars` 替换模板中的变量，并返回结果
    return tmpl.substitute(vars)


# 定义一个类 `bunch`，继承自 `dict`，实现了字典和属性的混合访问方式
class bunch(dict):

    # 初始化方法，将关键字参数作为属性添加到对象中
    def __init__(self, **kw):
        for name, value in kw.items():
            setattr(self, name, value)

    # 设置属性的方法，将属性值存储到字典中
    def __setattr__(self, name, value):
        self[name] = value

    # 获取属性的方法，优先从字典中获取，若不存在则抛出 `AttributeError` 异常
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    # 获取字典中指定键的值的方法，若键不存在则返回默认值（若存在的话）
    def __getitem__(self, key):
        if 'default' in self:
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                return dict.__getitem__(self, 'default')
        else:
            return dict.__getitem__(self, key)

    # 返回对象的字符串表示形式，包括类名和按键排序的键值对列表
    def __repr__(self):
        return '<%s %s>' % (
            self.__class__.__name__,
            ' '.join(['%s=%r' % (k, v) for k, v in sorted(self.items())]))


# 定义类 `TemplateDef`，表示模板定义对象
class TemplateDef:
    # 初始化方法，接收模板相关的参数和信息
    def __init__(self, template, func_name, func_signature,
                 body, ns, pos, bound_self=None):
        self._template = template
        self._func_name = func_name
        self._func_signature = func_signature
        self._body = body
        self._ns = ns
        self._pos = pos
        self._bound_self = bound_self

    # 返回对象的字符串表示形式，包括函数名、函数签名、模板名和位置信息
    def __repr__(self):
        return '<tempita function %s(%s) at %s:%s>' % (
            self._func_name, self._func_signature,
            self._template.name, self._pos)

    # 将对象作为字符串调用时的处理方法，返回模板解析后的结果
    def __str__(self):
        return self()

    # 对象被调用时的处理方法，解析参数并更新命名空间，执行模板解析并返回结果
    def __call__(self, *args, **kw):
        # 解析参数列表和关键字参数
        values = self._parse_signature(args, kw)
        ns = self._ns.copy()
        ns.update(values)
        # 如果有绑定的对象 `bound_self`，将其添加到命名空间中
        if self._bound_self is not None:
            ns['self'] = self._bound_self
        out = []
        subdefs = {}
        # 解析模板中的代码块，并将解析结果存储到 `out` 列表中
        self._template._interpret_codes(self._body, ns, out, subdefs)
        # 将 `out` 列表中的字符串连接成最终的结果并返回
        return ''.join(out)

    # 获取对象的属性方法，用于获取绑定的对象 `bound_self` 的新实例
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        return self.__class__(
            self._template, self._func_name, self._func_signature,
            self._body, self._ns, self._pos, bound_self=obj)
    # 解析函数签名，根据传入的位置参数和关键字参数生成一个字典 values
    def _parse_signature(self, args, kw):
        # 初始化一个空字典 values，用于存储最终解析出的参数值
        values = {}
        # 从对象的 _func_signature 中获取签名参数、可变位置参数、可变关键字参数和默认值
        sig_args, var_args, var_kw, defaults = self._func_signature
        # 初始化一个空字典 extra_kw，用于存储不在签名参数列表中的额外关键字参数
        extra_kw = {}
        
        # 遍历传入的关键字参数 kw 中的每一项
        for name, value in kw.items():
            # 如果没有可变关键字参数 var_kw，并且参数名 name 不在签名参数列表 sig_args 中，则抛出 TypeError
            if not var_kw and name not in sig_args:
                raise TypeError(
                    'Unexpected argument %s' % name)
            # 如果参数名 name 在签名参数列表 sig_args 中，则将其对应的值 value 存入 values 字典中
            if name in sig_args:
                values[sig_args] = value
            else:
                # 否则将其存入 extra_kw 字典中
                extra_kw[name] = value
        
        # 将位置参数 args 转换为列表
        args = list(args)
        # 将签名参数列表 sig_args 转换为列表
        sig_args = list(sig_args)
        
        # 处理剩余的位置参数 args
        while args:
            # 当 sig_args 列表非空且其第一个元素已在 values 字典中时，删除该元素
            while sig_args and sig_args[0] in values:
                sig_args.pop(0)
            if sig_args:
                # 如果 sig_args 列表非空，则取出其第一个参数名 name
                name = sig_args.pop(0)
                # 将其对应的值从 args 中取出，并存入 values 字典中
                values[name] = args.pop(0)
            elif var_args:
                # 如果存在可变位置参数 var_args，则将剩余的 args 转换为元组，存入 values 字典中，并跳出循环
                values[var_args] = tuple(args)
                break
            else:
                # 否则抛出 TypeError，指示存在额外的位置参数
                raise TypeError(
                    'Extra position arguments: %s'
                    % ', '.join([repr(v) for v in args]))
        
        # 处理默认值 defaults 中的参数
        for name, value_expr in defaults.items():
            # 如果该参数名 name 不在 values 字典中，则使用模板的 _eval 方法计算其默认值，并存入 values 字典中
            if name not in values:
                values[name] = self._template._eval(
                    value_expr, self._ns, self._pos)
        
        # 检查是否有未赋值的必需参数，如果有，则抛出 TypeError
        for name in sig_args:
            if name not in values:
                raise TypeError(
                    'Missing argument: %s' % name)
        
        # 如果存在可变关键字参数 var_kw，则将 extra_kw 存入 values 字典中
        if var_kw:
            values[var_kw] = extra_kw
        
        # 返回解析后的参数值字典 values
        return values
# TemplateObject 类定义
class TemplateObject:

    # 初始化方法，接受一个名称参数并将其保存在私有属性 __name 中
    def __init__(self, name):
        self.__name = name
        # 创建 TemplateObjectGetter 实例，并将自身传入
        self.get = TemplateObjectGetter(self)

    # 返回对象的字符串表示形式，包括类名和对象的名称
    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.__name)


# TemplateObjectGetter 类定义
class TemplateObjectGetter:

    # 初始化方法，接受一个模板对象作为参数，并保存在私有属性 __template_obj 中
    def __init__(self, template_obj):
        self.__template_obj = template_obj

    # 获取属性的魔法方法，委托给模板对象的对应属性，如果属性不存在则返回 Empty
    def __getattr__(self, attr):
        return getattr(self.__template_obj, attr, Empty)

    # 返回对象的字符串表示形式，包括类名和关联的模板对象的表示形式
    def __repr__(self):
        return '<%s around %r>' % (self.__class__.__name__, self.__template_obj)


# _Empty 类定义
class _Empty:

    # 可调用方法，返回自身
    def __call__(self, *args, **kw):
        return self

    # 返回空字符串
    def __str__(self):
        return ''

    # 返回 'Empty' 字符串
    def __repr__(self):
        return 'Empty'

    # 返回空 Unicode 字符串
    def __unicode__(self):
        return ''

    # 返回空迭代器
    def __iter__(self):
        return iter(())

    # 返回 False，表示该对象为假值
    def __bool__(self):
        return False

# 创建 Empty 实例，用于表示空值
Empty = _Empty()
del _Empty  # 删除 _Empty 类，避免外部访问

############################################################
## Lexing and Parsing
############################################################

# 词法分析函数，将字符串 s 分解为各种片段
def lex(s, name=None, trim_whitespace=True, line_offset=0, delimiters=None):
    """
    Lex a string into chunks:

        >>> lex('hey')
        ['hey']
        >>> lex('hey {{you}}')
        ['hey ', ('you', (1, 7))]
        >>> lex('hey {{')
        Traceback (most recent call last):
            ...
        TemplateError: No }} to finish last expression at line 1 column 7
        >>> lex('hey }}')
        Traceback (most recent call last):
            ...
        TemplateError: }} outside expression at line 1 column 7
        >>> lex('hey {{ {{')
        Traceback (most recent call last):
            ...
        TemplateError: {{ inside expression at line 1 column 10

    """
    # 如果未提供 delimiters 参数，则使用默认的起止分隔符
    if delimiters is None:
        delimiters = (Template.default_namespace['start_braces'],
                      Template.default_namespace['end_braces'])
    
    # 初始化标志和列表
    in_expr = False  # 表示当前是否在表达式内部
    chunks = []      # 存储分析后的片段
    last = 0         # 上一个分隔符的索引位置
    last_pos = (line_offset + 1, 1)  # 上一个分隔符所在的位置（行号，列号）

    # 定义用于匹配分隔符的正则表达式
    token_re = re.compile(r'%s|%s' % (re.escape(delimiters[0]),
                                      re.escape(delimiters[1])))

    # 遍历字符串 s 中所有匹配的分隔符
    for match in token_re.finditer(s):
        expr = match.group(0)  # 当前匹配到的分隔符
        pos = find_position(s, match.end(), last, last_pos)  # 计算当前分隔符的位置

        # 处理起始分隔符的情况
        if expr == delimiters[0] and in_expr:
            # 如果当前已经在表达式内部，则报错
            raise TemplateError('%s inside expression' % delimiters[0],
                                position=pos,
                                name=name)
        
        # 处理结束分隔符的情况
        elif expr == delimiters[1] and not in_expr:
            # 如果当前不在表达式内部，则报错
            raise TemplateError('%s outside expression' % delimiters[1],
                                position=pos,
                                name=name)
        
        # 处理起始分隔符的情况
        if expr == delimiters[0]:
            # 截取上一个分隔符到当前分隔符之间的部分作为一个片段，并添加到 chunks 中
            part = s[last:match.start()]
            if part:
                chunks.append(part)
            in_expr = True  # 进入表达式内部标志置为 True
        
        # 处理结束分隔符的情况
        else:
            # 截取上一个分隔符到当前分隔符之间的部分作为一个片段，并添加到 chunks 中
            chunks.append((s[last:match.start()], last_pos))
            in_expr = False  # 进入表达式内部标志置为 False
        
        # 更新 last 和 last_pos
        last = match.end()
        last_pos = pos
    # 如果正在表达式中，抛出模板错误异常，指示没有匹配的结束符号来完成最后一个表达式
    if in_expr:
        raise TemplateError('No %s to finish last expression' % delimiters[1],
                            name=name, position=last_pos)
    # 从字符串 s 中取出从索引 last 到末尾的部分，作为当前片段
    part = s[last:]
    # 如果 part 非空，则将其添加到 chunks 列表中
    if part:
        chunks.append(part)
    # 如果需要去除空白字符，则对 chunks 列表进行修整操作
    if trim_whitespace:
        chunks = trim_lex(chunks)
    # 返回处理后的 chunks 列表作为结果
    return chunks
# 匹配以指定关键词开头的行，用于识别代码中的特定语句和指令
statement_re = re.compile(r'^(?:if |elif |for |def |inherit |default |py:)')

# 单独的指令列表，不在其它代码块内的独立关键字
single_statements = ['else', 'endif', 'endfor', 'enddef', 'continue', 'break']

# 匹配行末尾的空白字符（空格、制表符等）
trail_whitespace_re = re.compile(r'\n\r?[\t ]*$')

# 匹配行开头的空白字符（空格、制表符等）
lead_whitespace_re = re.compile(r'^[\t ]*\n')


def trim_lex(tokens):
    r"""
    Takes a lexed set of tokens, and removes whitespace when there is
    a directive on a line by itself:

       >>> tokens = lex('{{if x}}\nx\n{{endif}}\ny', trim_whitespace=False)
       >>> tokens
       [('if x', (1, 3)), '\nx\n', ('endif', (3, 3)), '\ny']
       >>> trim_lex(tokens)
       [('if x', (1, 3)), 'x\n', ('endif', (3, 3)), 'y']
    """
    last_trim = None
    for i, current in enumerate(tokens):
        if isinstance(current, basestring_):
            # we don't trim this
            continue
        item = current[0]
        if not statement_re.search(item) and item not in single_statements:
            continue
        if not i:
            prev = ''
        else:
            prev = tokens[i - 1]
        if i + 1 >= len(tokens):
            next_chunk = ''
        else:
            next_chunk = tokens[i + 1]
        if (not isinstance(next_chunk, basestring_)
                or not isinstance(prev, basestring_)):
            continue
        prev_ok = not prev or trail_whitespace_re.search(prev)
        if i == 1 and not prev.strip():
            prev_ok = True
        if last_trim is not None and last_trim + 2 == i and not prev.strip():
            prev_ok = 'last'
        if (prev_ok
            and (not next_chunk or lead_whitespace_re.search(next_chunk)
                 or (i == len(tokens) - 2 and not next_chunk.strip()))):
            if prev:
                if ((i == 1 and not prev.strip())
                        or prev_ok == 'last'):
                    tokens[i - 1] = ''
                else:
                    m = trail_whitespace_re.search(prev)
                    # +1 to leave the leading \n on:
                    prev = prev[:m.start() + 1]
                    tokens[i - 1] = prev
            if next_chunk:
                last_trim = i
                if i == len(tokens) - 2 and not next_chunk.strip():
                    tokens[i + 1] = ''
                else:
                    m = lead_whitespace_re.search(next_chunk)
                    next_chunk = next_chunk[m.end():]
                    tokens[i + 1] = next_chunk
    return tokens


def find_position(string, index, last_index, last_pos):
    """Given a string and index, return (line, column)"""
    # 计算给定索引处的行号和列号
    lines = string.count('\n', last_index, index)
    if lines > 0:
        column = index - string.rfind('\n', last_index, index)
    else:
        column = last_pos[1] + (index - last_index)
    return (last_pos[0] + lines, column)


def parse(s, name=None, line_offset=0, delimiters=None):
    r"""
    # 如果未提供分隔符，则使用默认的起始和结束括号
    if delimiters is None:
        delimiters = ( Template.default_namespace['start_braces'],
                       Template.default_namespace['end_braces'] )
    # 对输入字符串进行词法分析，生成标记流
    tokens = lex(s, name=name, line_offset=line_offset, delimiters=delimiters)
    # 初始化结果列表
    result = []
    # 循环处理标记流中的每个标记，直到标记流为空
    while tokens:
        # 解析下一个表达式块，并更新标记流
        next_chunk, tokens = parse_expr(tokens, name)
        # 将解析得到的表达式块添加到结果列表中
        result.append(next_chunk)
    # 返回最终的解析结果列表
    return result
# 解析表达式的函数，根据给定的tokens解析表达式语句
def parse_expr(tokens, name, context=()):
    # 检查tokens的第一个元素是否是字符串类型
    if isinstance(tokens[0], basestring_):
        # 如果是字符串，直接返回该字符串和剩余的tokens
        return tokens[0], tokens[1:]
    # 否则，假设tokens的第一个元素是一个元组(expr, pos)
    expr, pos = tokens[0]
    # 去除表达式两端的空白字符
    expr = expr.strip()
    # 处理以'py:'开头的表达式
    if expr.startswith('py:'):
        expr = expr[3:].lstrip(' \t')
        # 处理多行的py块，确保以换行符开头
        if expr.startswith('\n') or expr.startswith('\r'):
            expr = expr.lstrip('\r\n')
            # 处理Windows风格的换行符
            if '\r' in expr:
                expr = expr.replace('\r\n', '\n')
                expr = expr.replace('\r', '')
            expr += '\n'
        else:
            # 如果不是以换行符开头，则抛出模板错误
            if '\n' in expr:
                raise TemplateError(
                    'Multi-line py blocks must start with a newline',
                    position=pos, name=name)
        # 返回包含'py'类型、位置信息、表达式内容的元组，以及剩余的tokens
        return ('py', pos, expr), tokens[1:]
    # 处理'continue'和'break'关键字
    elif expr in ('continue', 'break'):
        # 如果不在for循环中，则抛出模板错误
        if 'for' not in context:
            raise TemplateError(
                'continue outside of for loop',
                position=pos, name=name)
        # 返回包含关键字和位置信息的元组，以及剩余的tokens
        return (expr, pos), tokens[1:]
    # 处理以'if '开头的表达式
    elif expr.startswith('if '):
        # 调用parse_cond函数处理if语句
        return parse_cond(tokens, name, context)
    # 处理以'elif '或者'else'开头的表达式
    elif (expr.startswith('elif ')
          or expr == 'else'):
        # 抛出模板错误，因为elif和else必须在if块内部
        raise TemplateError(
            '%s outside of an if block' % expr.split()[0],
            position=pos, name=name)
    # 处理'if'、'elif'、'for'等关键字，但没有后续表达式的情况
    elif expr in ('if', 'elif', 'for'):
        # 抛出模板错误，因为这些关键字后面缺少表达式
        raise TemplateError(
            '%s with no expression' % expr,
            position=pos, name=name)
    # 处理'endif'、'endfor'、'enddef'等结束语句
    elif expr in ('endif', 'endfor', 'enddef'):
        # 抛出模板错误，因为不应该出现这些结束语句
        raise TemplateError(
            'Unexpected %s' % expr,
            position=pos, name=name)
    # 处理以'for '开头的表达式
    elif expr.startswith('for '):
        # 调用parse_for函数处理for循环语句
        return parse_for(tokens, name, context)
    # 处理以'default '开头的表达式
    elif expr.startswith('default '):
        # 调用parse_default函数处理默认值设置语句
        return parse_default(tokens, name, context)
    # 处理以'inherit '开头的表达式
    elif expr.startswith('inherit '):
        # 调用parse_inherit函数处理继承语句
        return parse_inherit(tokens, name, context)
    # 处理以'def '开头的表达式
    elif expr.startswith('def '):
        # 调用parse_def函数处理函数定义语句
        return parse_def(tokens, name, context)
    # 处理以'#'开头的注释行
    elif expr.startswith('#'):
        # 返回包含'comment'类型、位置信息、注释内容的元组，以及剩余的tokens
        return ('comment', pos, tokens[0][0]), tokens[1:]
    # 默认情况下，返回包含'expr'类型、位置信息、表达式内容的元组，以及剩余的tokens
    return ('expr', pos, tokens[0][0]), tokens[1:]


# 解析条件表达式的辅助函数，处理复杂的条件逻辑
def parse_cond(tokens, name, context):
    # 记录条件表达式开始的位置
    start = tokens[0][1]
    # 初始化条件片段列表
    pieces = []
    # 将当前上下文中加入'if'，表示正在处理if条件
    context = context + ('if',)
    while 1:
        # 如果tokens为空，则抛出模板错误，因为缺少{{endif}}
        if not tokens:
            raise TemplateError(
                'Missing {{endif}}',
                position=start, name=name)
        # 如果当前token是('endif', ...)
        if (isinstance(tokens[0], tuple)
                and tokens[0][0] == 'endif'):
            # 返回包含'cond'类型、开始位置和条件片段列表的元组，以及剩余的tokens
            return ('cond', start) + tuple(pieces), tokens[1:]
        # 调用parse_one_cond处理单个条件表达式，并将结果添加到pieces列表中
        next_chunk, tokens = parse_one_cond(tokens, name, context)
        pieces.append(next_chunk)


# 解析单个条件表达式的函数
def parse_one_cond(tokens, name, context):
    # 取出tokens的第一个元素作为(first, pos)，并将其从tokens中移除
    (first, pos), tokens = tokens[0], tokens[1:]
    # 初始化内容列表
    content = []
    # 如果first以':'结尾，则去除':'符号
    if first.endswith(':'):
        first = first[:-1]
    # 处理以'if '开头的条件语句
    if first.startswith('if '):
        part = ('if', pos, first[3:].lstrip(), content)
    # 处理以'elif '开头的条件语句
    elif first.startswith('elif '):
        part = ('elif', pos, first[5:].lstrip(), content)
    # 处理'else'语句
    elif first == 'else':
        part = ('else', pos, None, content)
    # 如果不是上述条件中的任何情况，则断言错误，显示错误消息和位置信息
    else:
        assert 0, "Unexpected token %r at %s" % (first, pos)
    # 进入无限循环，直到遇到特定条件退出
    while 1:
        # 如果 tokens 列表为空，抛出模板错误异常，指明缺少 {{endif}} 的位置和模板名称
        if not tokens:
            raise TemplateError(
                'No {{endif}}',
                position=pos, name=name)
        # 如果 tokens 的第一个元素是元组，并且以 'endif' 开头，或者是以 'elif ' 开头，或者是 'else'，则结束循环，返回当前部分和 tokens 列表
        if (isinstance(tokens[0], tuple)
            and (tokens[0][0] == 'endif'
                 or tokens[0][0].startswith('elif ')
                 or tokens[0][0] == 'else')):
            return part, tokens
        # 解析表达式，并返回解析结果和剩余的 tokens 列表
        next_chunk, tokens = parse_expr(tokens, name, context)
        # 将解析得到的内容添加到 content 列表中
        content.append(next_chunk)
# 解析 for 循环语句
def parse_for(tokens, name, context):
    # 取出第一个元素及其位置
    first, pos = tokens[0]
    # 移除第一个元素后的剩余 tokens
    tokens = tokens[1:]
    # 更新上下文，添加 'for' 标识
    context = ('for',) + context
    # 初始化内容列表
    content = []
    # 检查第一个元素是否以 'for ' 开头
    assert first.startswith('for '), first
    # 如果以 ':' 结尾，则移除末尾的 ':'
    if first.endswith(':'):
        first = first[:-1]
    # 剥离 'for ' 后面的部分，并去除首尾空格
    first = first[3:].strip()
    # 使用正则表达式查找 'in' 的位置
    match = in_re.search(first)
    # 如果未找到 'in'，则抛出模板错误
    if not match:
        raise TemplateError(
            'Bad for (no "in") in %r' % first,
            position=pos, name=name)
    # 提取变量部分直到 'in' 的位置之前的内容
    vars = first[:match.start()]
    # 如果变量部分包含 '('，则抛出模板错误
    if '(' in vars:
        raise TemplateError(
            'You cannot have () in the variable section of a for loop (%r)'
            % vars, position=pos, name=name)
    # 将变量部分按逗号分割并去除首尾空格，形成元组
    vars = tuple([
        v.strip() for v in first[:match.start()].split(',')
        if v.strip()])
    # 提取 'in' 后面的表达式部分
    expr = first[match.end():]
    # 循环解析内容块，直到遇到 {{endfor}}
    while 1:
        # 如果 tokens 为空，则抛出模板错误
        if not tokens:
            raise TemplateError(
                'No {{endfor}}',
                position=pos, name=name)
        # 如果下一个 token 是元组且以 'endfor' 开头，则返回 for 循环的信息和剩余 tokens
        if (isinstance(tokens[0], tuple)
                and tokens[0][0] == 'endfor'):
            return ('for', pos, vars, expr, content), tokens[1:]
        # 否则继续解析下一个表达式，并将结果添加到内容列表中
        next_chunk, tokens = parse_expr(tokens, name, context)
        content.append(next_chunk)


# 解析 default 语句
def parse_default(tokens, name, context):
    # 取出第一个元素及其位置
    first, pos = tokens[0]
    # 检查第一个元素是否以 'default ' 开头
    assert first.startswith('default ')
    # 剥离 'default ' 后面的部分
    first = first.split(None, 1)[1]
    # 按 '=' 分割表达式
    parts = first.split('=', 1)
    # 如果只有一个部分，则抛出模板错误
    if len(parts) == 1:
        raise TemplateError(
            "Expression must be {{default var=value}}; no = found in %r" % first,
            position=pos, name=name)
    # 提取变量部分，并去除首尾空格
    var = parts[0].strip()
    # 如果变量部分包含 ','，则抛出模板错误
    if ',' in var:
        raise TemplateError(
            "{{default x, y = ...}} is not supported",
            position=pos, name=name)
    # 如果变量名不符合正则表达式，则抛出模板错误
    if not var_re.search(var):
        raise TemplateError(
            "Not a valid variable name for {{default}}: %r"
            % var, position=pos, name=name)
    # 提取表达式部分，并去除首尾空格
    expr = parts[1].strip()
    # 返回 default 类型信息和剩余 tokens
    return ('default', pos, var, expr), tokens[1:]


# 解析 inherit 语句
def parse_inherit(tokens, name, context):
    # 取出第一个元素及其位置
    first, pos = tokens[0]
    # 检查第一个元素是否以 'inherit ' 开头
    assert first.startswith('inherit ')
    # 剥离 'inherit ' 后面的部分作为表达式
    expr = first.split(None, 1)[1]
    # 返回 inherit 类型信息和剩余 tokens
    return ('inherit', pos, expr), tokens[1:]


# 解析 def 语句
def parse_def(tokens, name, context):
    # 取出第一个元素及其位置
    first, start = tokens[0]
    # 移除第一个元素后的剩余 tokens
    tokens = tokens[1:]
    # 检查第一个元素是否以 'def ' 开头
    assert first.startswith('def ')
    # 剥离 'def ' 后面的部分
    first = first.split(None, 1)[1]
    # 如果以 ':' 结尾，则移除末尾的 ':'
    if first.endswith(':'):
        first = first[:-1]
    # 如果不包含 '('，则函数名为 first，签名默认为空
    if '(' not in first:
        func_name = first
        sig = ((), None, None, {})
    # 否则，提取函数名和签名
    elif not first.endswith(')'):
        raise TemplateError("Function definition doesn't end with ): %s" % first,
                            position=start, name=name)
    else:
        first = first[:-1]
        func_name, sig_text = first.split('(', 1)
        sig = parse_signature(sig_text, name, start)
    # 更新上下文，添加 'def' 标识
    context = context + ('def',)
    # 初始化内容列表
    content = []
    # 进入无限循环，用于处理模板定义中的内容
    while 1:
        # 检查 tokens 是否为空
        if not tokens:
            # 如果 tokens 为空，则抛出模板错误，指示缺少 {{enddef}} 结束定义
            raise TemplateError(
                'Missing {{enddef}}',
                position=start, name=name)
        # 检查 tokens 的第一个元素是否为元组，并且其第一个元素为 'enddef'
        if (isinstance(tokens[0], tuple)
                and tokens[0][0] == 'enddef'):
            # 如果满足条件，则表示找到了模板定义的结束标记，返回定义的元组及其余的 tokens
            return ('def', start, func_name, sig, content), tokens[1:]
        # 解析 tokens 中的表达式，获取下一个块，并更新 tokens
        next_chunk, tokens = parse_expr(tokens, name, context)
        # 将解析得到的下一个块添加到内容列表中，用于后续处理
        content.append(next_chunk)
# 定义函数 parse_signature，用于解析函数或方法的签名信息
def parse_signature(sig_text, name, pos):
    # 使用 tokenize 模块生成 tokens，从给定的字符串 sig_text 中读取
    tokens = tokenize.generate_tokens(StringIO(sig_text).readline)
    
    # 初始化空列表 sig_args 用于存储参数信息
    sig_args = []
    
    # 初始化可变位置参数和可变关键字参数为 None
    var_arg = None
    var_kw = None
    
    # 初始化空字典 defaults 用于存储参数的默认值信息
    defaults = {}

    # 定义内部函数 get_token，用于从 tokens 中获取下一个 token
    def get_token(pos=False):
        try:
            # 获取下一个 token 的详细信息
            tok_type, tok_string, (srow, scol), (erow, ecol), line = next(tokens)
        except StopIteration:
            # 如果 tokens 迭代结束，则返回特殊的 ENDMARKER token 和空字符串
            return tokenize.ENDMARKER, ''
        if pos:
            # 如果指定了 pos=True，则返回 token 类型、字符串、起始和结束位置信息
            return tok_type, tok_string, (srow, scol), (erow, ecol)
        else:
            # 否则只返回 token 类型和字符串内容
            return tok_type, tok_string
    # 进入无限循环，直到遇到结束符号
    while 1:
        var_arg_type = None
        # 调用函数获取下一个标记的类型和字符串值
        tok_type, tok_string = get_token()
        # 如果标记类型为文件结束符，则退出循环
        if tok_type == tokenize.ENDMARKER:
            break
        # 如果标记类型为操作符且值为 '*' 或 '**'
        if tok_type == tokenize.OP and (tok_string == '*' or tok_string == '**'):
            # 将变量参数类型设置为对应的操作符
            var_arg_type = tok_string
            # 继续获取下一个标记的类型和字符串值
            tok_type, tok_string = get_token()
        # 如果标记类型不是名称，则抛出模板错误，指示无效的签名
        if tok_type != tokenize.NAME:
            raise TemplateError('Invalid signature: (%s)' % sig_text,
                                position=pos, name=name)
        # 将变量名称设置为标记的字符串值
        var_name = tok_string
        # 继续获取下一个标记的类型和字符串值
        tok_type, tok_string = get_token()
        # 如果标记类型为文件结束符或者操作符且值为 ','，则处理变量参数
        if tok_type == tokenize.ENDMARKER or (tok_type == tokenize.OP and tok_string == ','):
            # 根据变量参数类型设置变量参数或者变量关键字参数
            if var_arg_type == '*':
                var_arg = var_name
            elif var_arg_type == '**':
                var_kw = var_name
            else:
                sig_args.append(var_name)
            # 如果标记类型为文件结束符，则退出循环
            if tok_type == tokenize.ENDMARKER:
                break
            # 继续下一次循环
            continue
        # 如果变量参数类型不为空，则抛出模板错误，指示无效的签名
        if var_arg_type is not None:
            raise TemplateError('Invalid signature: (%s)' % sig_text,
                                position=pos, name=name)
        # 如果标记类型为操作符且值为 '='，则处理默认参数表达式
        if tok_type == tokenize.OP and tok_string == '=':
            nest_type = None
            unnest_type = None
            nest_count = 0
            start_pos = end_pos = None
            parts = []
            # 进入循环，解析默认参数表达式
            while 1:
                # 获取下一个标记的类型、字符串值、起始和结束位置
                tok_type, tok_string, s, e = get_token(True)
                # 如果起始位置尚未设置，则设置起始位置
                if start_pos is None:
                    start_pos = s
                # 更新结束位置
                end_pos = e
                # 如果嵌套计数为零且遇到文件结束符，则抛出模板错误，指示无效的签名
                if tok_type == tokenize.ENDMARKER and nest_count:
                    raise TemplateError('Invalid signature: (%s)' % sig_text,
                                        position=pos, name=name)
                # 如果不处于嵌套结构内且遇到文件结束符或操作符且值为 ','，则隔离表达式并保存默认参数
                if (not nest_count and
                        (tok_type == tokenize.ENDMARKER or (tok_type == tokenize.OP and tok_string == ','))):
                    default_expr = isolate_expression(sig_text, start_pos, end_pos)
                    defaults[var_name] = default_expr
                    sig_args.append(var_name)
                    break
                # 将标记的类型和字符串值添加到部分列表中
                parts.append((tok_type, tok_string))
                # 处理嵌套结构的起始和结束标记
                if nest_count and tok_type == tokenize.OP and tok_string == nest_type:
                    nest_count += 1
                elif nest_count and tok_type == tokenize.OP and tok_string == unnest_type:
                    nest_count -= 1
                    if not nest_count:
                        nest_type = unnest_type = None
                elif not nest_count and tok_type == tokenize.OP and tok_string in ('(', '[', '{'):
                    nest_type = tok_string
                    nest_count = 1
                    unnest_type = {'(': ')', '[': ']', '{': '}'}[nest_type]
    # 返回处理后的参数列表、变量参数、变量关键字参数和默认参数字典
    return sig_args, var_arg, var_kw, defaults
# 定义函数，从字符串中分离出特定位置的表达式并返回
def isolate_expression(string, start_pos, end_pos):
    # 获取起始位置的行和列
    srow, scol = start_pos
    srow -= 1  # 将行数调整为从零开始索引
    erow, ecol = end_pos
    erow -= 1  # 将行数调整为从零开始索引
    # 将字符串按行分割成列表
    lines = string.splitlines(True)
    if srow == erow:
        # 如果起始和结束在同一行，直接返回该行中指定列范围的内容
        return lines[srow][scol:ecol]
    # 否则，构建包含起始行到结束行所有内容的列表
    parts = [lines[srow][scol:]]  # 添加起始行的部分内容
    parts.extend(lines[srow+1:erow])  # 添加起始行到结束行之间的所有内容
    if erow < len(lines):
        # 如果结束行在行列表内，添加结束行的部分内容
        parts.append(lines[erow][:ecol])
    # 将所有部分合并为一个字符串并返回
    return ''.join(parts)

# 填充命令的使用说明文本
_fill_command_usage = """\
%prog [OPTIONS] TEMPLATE arg=value

Use py:arg=value to set a Python value; otherwise all values are
strings.
"""

# 定义填充命令的函数
def fill_command(args=None):
    import sys
    import optparse
    import pkg_resources
    import os
    if args is None:
        args = sys.argv[1:]  # 获取命令行参数，排除脚本名称
    dist = pkg_resources.get_distribution('Paste')  # 获取 'Paste' 软件包的分发信息
    # 创建命令行选项解析器
    parser = optparse.OptionParser(
        version=coerce_text(dist),  # 设置版本信息
        usage=_fill_command_usage)  # 设置使用说明文本
    # 添加命令行选项：输出文件名
    parser.add_option(
        '-o', '--output',
        dest='output',
        metavar="FILENAME",
        help="File to write output to (default stdout)")
    # 添加命令行选项：使用环境变量作为顶层变量
    parser.add_option(
        '--env',
        dest='use_env',
        action='store_true',
        help="Put the environment in as top-level variables")
    # 解析命令行参数
    options, args = parser.parse_args(args)
    if len(args) < 1:
        # 如果没有提供模板文件名，输出错误信息并退出
        print('You must give a template filename')
        sys.exit(2)
    template_name = args[0]  # 第一个参数为模板文件名
    args = args[1:]  # 剩余参数为模板变量
    vars = {}  # 初始化变量字典
    if options.use_env:
        vars.update(os.environ)  # 如果指定了使用环境变量，更新变量字典
    for value in args:
        if '=' not in value:
            # 如果参数格式不正确，输出错误信息并退出
            print('Bad argument: %r' % value)
            sys.exit(2)
        name, value = value.split('=', 1)  # 解析参数名和值
        if name.startswith('py:'):
            name = name[3:]  # 如果参数名以 'py:' 开头，去除 'py:' 前缀
            value = eval(value)  # 将值解析为 Python 表达式的结果
        vars[name] = value  # 将参数名和值加入变量字典
    if template_name == '-':
        # 如果模板名为 '-'，从标准输入读取模板内容
        template_content = sys.stdin.read()
        template_name = '<stdin>'
    else:
        # 否则，从文件中读取模板内容
        with open(template_name, 'rb') as f:
            template_content = f.read()
    # 使用模板内容和变量字典创建模板对象
    template = Template(template_content, name=template_name)
    # 对模板进行变量替换得到结果
    result = template.substitute(vars)
    if options.output:
        # 如果指定了输出文件，将结果写入文件
        with open(options.output, 'wb') as f:
            f.write(result)
    else:
        # 否则，将结果写入标准输出
        sys.stdout.write(result)

# 如果脚本直接运行，则调用 fill_command 函数
if __name__ == '__main__':
    fill_command()
```