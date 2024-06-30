# `D:\src\scipysrc\sympy\sympy\polys\polyoptions.py`

```
"""Options manager for :class:`~.Poly` and public API functions. """

from __future__ import annotations

__all__ = ["Options"]

from sympy.core import Basic, sympify
from sympy.polys.polyerrors import GeneratorsError, OptionError, FlagError
from sympy.utilities import numbered_symbols, topological_sort, public
from sympy.utilities.iterables import has_dups, is_sequence

import sympy.polys

import re

class Option:
    """Base class for all kinds of options. """

    option: str | None = None  # 选项名称，可以是字符串或空值

    is_Flag = False  # 标记是否为标志选项，默认为False

    requires: list[str] = []  # 必须与该选项同时使用的选项列表
    excludes: list[str] = []  # 与该选项互斥的选项列表

    after: list[str] = []  # 在该选项之后应用的选项列表
    before: list[str] = []  # 在该选项之前应用的选项列表

    @classmethod
    def default(cls):
        return None  # 默认的选项值，默认情况下返回None

    @classmethod
    def preprocess(cls, option):
        return None  # 预处理选项值的方法，默认情况下返回None

    @classmethod
    def postprocess(cls, options):
        pass  # 后处理选项的方法，无操作的占位符


class Flag(Option):
    """Base class for all kinds of flags. """

    is_Flag = True  # 标识这是一个标志选项


class BooleanOption(Option):
    """An option that must have a boolean value or equivalent assigned. """

    @classmethod
    def preprocess(cls, value):
        if value in [True, False]:
            return bool(value)  # 如果值为True或False，则返回其布尔值
        else:
            raise OptionError("'%s' must have a boolean value assigned, got %s" % (cls.option, value))  # 抛出选项错误，要求布尔值


class OptionType(type):
    """Base type for all options that does registers options. """

    def __init__(cls, *args, **kwargs):
        @property
        def getter(self):
            try:
                return self[cls.option]  # 获取实例中特定选项的值
            except KeyError:
                return cls.default()  # 如果选项不存在，返回默认值

        setattr(Options, cls.option, getter)  # 将getter方法设置为Options类的属性
        Options.__options__[cls.option] = cls  # 将该选项类注册到Options类的选项字典中


@public
class Options(dict):
    """
    Options manager for polynomial manipulation module.

    Examples
    ========

    >>> from sympy.polys.polyoptions import Options
    >>> from sympy.polys.polyoptions import build_options

    >>> from sympy.abc import x, y, z

    >>> Options((x, y, z), {'domain': 'ZZ'})
    {'auto': False, 'domain': ZZ, 'gens': (x, y, z)}

    >>> build_options((x, y, z), {'domain': 'ZZ'})
    {'auto': False, 'domain': ZZ, 'gens': (x, y, z)}

    **Options**

    * Expand --- boolean option
    * Gens --- option
    * Wrt --- option
    * Sort --- option
    * Order --- option
    * Field --- boolean option
    * Greedy --- boolean option
    * Domain --- option
    * Split --- boolean option
    * Gaussian --- boolean option
    * Extension --- option
    * Modulus --- option
    * Symmetric --- boolean option
    * Strict --- boolean option

    **Flags**

    * Auto --- boolean flag
    * Frac --- boolean flag
    * Formal --- boolean flag
    * Polys --- boolean flag
    * Include --- boolean flag
    * All --- boolean flag
    * Gen --- flag
    * Series --- boolean flag

    """

    __order__ = None  # 选项的顺序，目前未指定
    __options__: dict[str, type[Option]] = {}  # 存储所有选项类的字典
    def __init__(self, gens, args, flags=None, strict=False):
        # 调用基类的初始化方法，初始化一个空的字典
        dict.__init__(self)

        # 检查是否同时提供了 '*gens' 和关键字参数 'gens'，若是则抛出选项错误异常
        if gens and args.get('gens', ()):
            raise OptionError(
                "both '*gens' and keyword argument 'gens' supplied")
        elif gens:
            # 如果只提供了 'gens' 参数，则将其添加到 args 字典中
            args = dict(args)
            args['gens'] = gens

        # 从参数中移除 'defaults' 键，如果存在的话，其值为空字典
        defaults = args.pop('defaults', {})

        # 定义内部函数 preprocess_options，用于预处理参数
        def preprocess_options(args):
            for option, value in args.items():
                try:
                    # 尝试获取选项对应的类
                    cls = self.__options__[option]
                except KeyError:
                    raise OptionError("'%s' is not a valid option" % option)

                # 如果选项的类是 Flag 的子类
                if issubclass(cls, Flag):
                    # 如果未提供 flags 或者该选项不在 flags 中
                    if flags is None or option not in flags:
                        # 如果 strict 为 True，则抛出选项错误异常
                        if strict:
                            raise OptionError("'%s' flag is not allowed in this context" % option)

                # 如果值不为 None，则将其存储到当前对象的字典中经过预处理的值
                if value is not None:
                    self[option] = cls.preprocess(value)

        # 调用 preprocess_options 函数处理 args 参数
        preprocess_options(args)

        # 遍历默认参数字典 defaults
        for key in dict(defaults):
            if key in self:
                # 如果当前对象已存在该键，则从 defaults 中删除该键
                del defaults[key]
            else:
                for option in self.keys():
                    cls = self.__options__[option]

                    # 如果 key 在当前选项类的 excludes 列表中
                    if key in cls.excludes:
                        # 则从 defaults 中删除该键，并跳出循环
                        del defaults[key]
                        break

        # 再次调用 preprocess_options 函数处理剩余的 defaults 参数
        preprocess_options(defaults)

        # 遍历当前对象的所有选项
        for option in self.keys():
            cls = self.__options__[option]

            # 检查当前选项所需的其他选项是否都存在
            for require_option in cls.requires:
                if self.get(require_option) is None:
                    # 若存在未满足的依赖选项，则抛出选项错误异常
                    raise OptionError("'%s' option is only allowed together with '%s'" % (option, require_option))

            # 检查当前选项是否与其排斥的选项同时存在
            for exclude_option in cls.excludes:
                if self.get(exclude_option) is not None:
                    # 若存在排斥的选项同时存在，则抛出选项错误异常
                    raise OptionError("'%s' option is not allowed together with '%s'" % (option, exclude_option))

        # 根据指定的处理顺序 self.__order__，对当前对象的选项进行后处理
        for option in self.__order__:
            self.__options__[option].postprocess(self)

    @classmethod
    def _init_dependencies_order(cls):
        """Resolve the order of options' processing. """
        # 解析选项处理的顺序，构建一个拓扑排序的顺序列表
        if cls.__order__ is None:
            vertices, edges = [], set()

            # 遍历所有选项，并添加到顶点列表中
            for name, option in cls.__options__.items():
                vertices.append(name)

                # 将选项的 before 和 after 属性中的依赖关系添加到边集合中
                edges.update((_name, name) for _name in option.after)
                edges.update((name, _name) for _name in option.before)

            try:
                # 尝试进行拓扑排序
                cls.__order__ = topological_sort((vertices, list(edges)))
            except ValueError:
                # 若出现循环依赖，则抛出运行时异常
                raise RuntimeError(
                    "cycle detected in sympy.polys options framework")

    def clone(self, updates={}):
        """Clone ``self`` and update specified options. """
        # 克隆当前对象，并更新指定的选项
        obj = dict.__new__(self.__class__)

        # 复制当前对象的所有选项到新对象
        for option, value in self.items():
            obj[option] = value

        # 将 updates 参数中的选项更新到新对象中
        for option, value in updates.items():
            obj[option] = value

        return obj
    # 定义一个特殊的方法，用于设置对象的属性
    def __setattr__(self, attr, value):
        # 如果属性在预定义的选项列表中
        if attr in self.__options__:
            # 直接设置对象的属性值
            self[attr] = value
        else:
            # 否则调用父类的设置属性方法
            super().__setattr__(attr, value)
    
    # 定义一个属性方法，返回对象的参数字典，不包括特定的选项 'gens'
    @property
    def args(self):
        args = {}
        
        # 遍历对象的每个选项和对应的值
        for option, value in self.items():
            # 如果值不为 None 并且选项不是 'gens'
            if value is not None and option != 'gens':
                # 获取选项对应的类
                cls = self.__options__[option]
                
                # 如果该类不是 Flag 的子类
                if not issubclass(cls, Flag):
                    # 将选项及其值添加到 args 字典中
                    args[option] = value
        
        return args
    
    # 定义一个属性方法，返回对象的选项字典
    @property
    def options(self):
        options = {}
        
        # 遍历对象的选项及其对应的类
        for option, cls in self.__options__.items():
            # 如果该类不是 Flag 的子类
            if not issubclass(cls, Flag):
                # 获取对象的选项值，并添加到 options 字典中
                options[option] = getattr(self, option)
        
        return options
    
    # 定义一个属性方法，返回对象的标志字典
    @property
    def flags(self):
        flags = {}
        
        # 遍历对象的选项及其对应的类
        for option, cls in self.__options__.items():
            # 如果该类是 Flag 的子类
            if issubclass(cls, Flag):
                # 获取对象的选项值，并添加到 flags 字典中
                flags[option] = getattr(self, option)
        
        return flags
class Expand(BooleanOption, metaclass=OptionType):
    """``expand`` option to polynomial manipulation functions. """

    option = 'expand'  # 设置选项名称为 'expand'

    requires: list[str] = []  # 不需要任何其他选项
    excludes: list[str] = []  # 不排除任何其他选项

    @classmethod
    def default(cls):
        return True  # 默认情况下，选项为 True


class Gens(Option, metaclass=OptionType):
    """``gens`` option to polynomial manipulation functions. """

    option = 'gens'  # 设置选项名称为 'gens'

    requires: list[str] = []  # 不需要任何其他选项
    excludes: list[str] = []  # 不排除任何其他选项

    @classmethod
    def default(cls):
        return ()  # 默认情况下，返回空元组作为生成器

    @classmethod
    def preprocess(cls, gens):
        if isinstance(gens, Basic):
            gens = (gens,)  # 如果 gens 是 Basic 类型，则转换为单元素元组
        elif len(gens) == 1 and is_sequence(gens[0]):
            gens = gens[0]  # 如果 gens 是单元素序列，则直接使用该序列

        if gens == (None,):
            gens = ()  # 如果 gens 是 (None,)，则转换为空元组
        elif has_dups(gens):
            raise GeneratorsError("duplicated generators: %s" % str(gens))  # 如果 gens 中有重复的生成器，则引发异常
        elif any(gen.is_commutative is False for gen in gens):
            raise GeneratorsError("non-commutative generators: %s" % str(gens))  # 如果 gens 中有非交换生成器，则引发异常

        return tuple(gens)  # 返回处理后的生成器元组


class Wrt(Option, metaclass=OptionType):
    """``wrt`` option to polynomial manipulation functions. """

    option = 'wrt'  # 设置选项名称为 'wrt'

    requires: list[str] = []  # 不需要任何其他选项
    excludes: list[str] = []  # 不排除任何其他选项

    _re_split = re.compile(r"\s*,\s*|\s+")  # 编译用于分割字符串的正则表达式

    @classmethod
    def preprocess(cls, wrt):
        if isinstance(wrt, Basic):
            return [str(wrt)]  # 如果 wrt 是 Basic 类型，则返回包含其字符串表示的列表
        elif isinstance(wrt, str):
            wrt = wrt.strip()  # 去除首尾空格
            if wrt.endswith(','):
                raise OptionError('Bad input: missing parameter.')  # 如果末尾有逗号，则引发异常
            if not wrt:
                return []  # 如果为空字符串，则返回空列表
            return list(cls._re_split.split(wrt))  # 使用正则表达式分割 wrt，并返回列表
        elif hasattr(wrt, '__getitem__'):
            return list(map(str, wrt))  # 如果 wrt 是可索引的对象，则返回其所有元素的字符串表示列表
        else:
            raise OptionError("invalid argument for 'wrt' option")  # 其它情况下，引发 'wrt' 选项的无效参数异常


class Sort(Option, metaclass=OptionType):
    """``sort`` option to polynomial manipulation functions. """

    option = 'sort'  # 设置选项名称为 'sort'

    requires: list[str] = []  # 不需要任何其他选项
    excludes: list[str] = []  # 不排除任何其他选项

    @classmethod
    def default(cls):
        return []  # 默认情况下，返回空列表作为排序选项

    @classmethod
    def preprocess(cls, sort):
        if isinstance(sort, str):
            return [ gen.strip() for gen in sort.split('>') ]  # 如果 sort 是字符串，则按 '>' 分割并去除空格后返回列表
        elif hasattr(sort, '__getitem__'):
            return list(map(str, sort))  # 如果 sort 是可索引的对象，则返回其所有元素的字符串表示列表
        else:
            raise OptionError("invalid argument for 'sort' option")  # 其它情况下，引发 'sort' 选项的无效参数异常


class Order(Option, metaclass=OptionType):
    """``order`` option to polynomial manipulation functions. """

    option = 'order'  # 设置选项名称为 'order'

    requires: list[str] = []  # 不需要任何其他选项
    excludes: list[str] = []  # 不排除任何其他选项

    @classmethod
    def default(cls):
        return sympy.polys.orderings.lex  # 默认情况下，使用 sympy 默认的词法排序方式

    @classmethod
    def preprocess(cls, order):
        return sympy.polys.orderings.monomial_key(order)  # 根据给定的 order 参数返回其对应的单项式排序键


class Field(BooleanOption, metaclass=OptionType):
    """``field`` option to polynomial manipulation functions. """

    option = 'field'  # 设置选项名称为 'field'

    requires: list[str] = []  # 不需要任何其他选项
    excludes = ['domain', 'split', 'gaussian']  # 与 'field' 选项互斥的选项有 'domain', 'split', 'gaussian'
class Greedy(BooleanOption, metaclass=OptionType):
    """``greedy`` option to polynomial manipulation functions. """

    option = 'greedy'  # 设置选项名称为 'greedy'

    requires: list[str] = []  # 无需额外选项
    excludes = ['domain', 'split', 'gaussian', 'extension', 'modulus', 'symmetric']  # 与其他选项互斥


class Composite(BooleanOption, metaclass=OptionType):
    """``composite`` option to polynomial manipulation functions. """

    option = 'composite'  # 设置选项名称为 'composite'

    @classmethod
    def default(cls):
        return None  # 默认情况下返回空

    requires: list[str] = []  # 无需额外选项
    excludes = ['domain', 'split', 'gaussian', 'extension', 'modulus', 'symmetric']  # 与其他选项互斥


class Domain(Option, metaclass=OptionType):
    """``domain`` option to polynomial manipulation functions. """

    option = 'domain'  # 设置选项名称为 'domain'

    requires: list[str] = []  # 无需额外选项
    excludes = ['field', 'greedy', 'split', 'gaussian', 'extension']  # 与其他选项互斥

    after = ['gens']  # 依赖于 'gens' 选项之后

    _re_realfield = re.compile(r"^(R|RR)(_(\d+))?$")  # 匹配实数域的正则表达式
    _re_complexfield = re.compile(r"^(C|CC)(_(\d+))?$")  # 匹配复数域的正则表达式
    _re_finitefield = re.compile(r"^(FF|GF)\((\d+)\)$")  # 匹配有限域的正则表达式
    _re_polynomial = re.compile(r"^(Z|ZZ|Q|QQ|ZZ_I|QQ_I|R|RR|C|CC)\[(.+)\]$")  # 匹配多项式的正则表达式
    _re_fraction = re.compile(r"^(Z|ZZ|Q|QQ)\((.+)\)$")  # 匹配分数的正则表达式
    _re_algebraic = re.compile(r"^(Q|QQ)\<(.+)\>$")  # 匹配代数的正则表达式

    @classmethod
    # 类方法，用于预处理给定的域（domain），返回相应的 sympy.polys.domains.Domain 对象
    def preprocess(cls, domain):
        # 如果 domain 是 sympy.polys.domains.Domain 的实例，则直接返回
        if isinstance(domain, sympy.polys.domains.Domain):
            return domain
        # 如果 domain 具有 to_domain 方法，则调用该方法并返回结果
        elif hasattr(domain, 'to_domain'):
            return domain.to_domain()
        # 如果 domain 是字符串类型
        elif isinstance(domain, str):
            # 检查 domain 是否为整数环的别名 'Z' 或 'ZZ'
            if domain in ['Z', 'ZZ']:
                return sympy.polys.domains.ZZ
            # 检查 domain 是否为有理数域的别名 'Q' 或 'QQ'
            elif domain in ['Q', 'QQ']:
                return sympy.polys.domains.QQ
            # 检查 domain 是否为整数环四元数 'ZZ_I'
            elif domain == 'ZZ_I':
                return sympy.polys.domains.ZZ_I
            # 检查 domain 是否为有理数域四元数 'QQ_I'
            elif domain == 'QQ_I':
                return sympy.polys.domains.QQ_I
            # 检查 domain 是否为表达式域 'EX'
            elif domain == 'EX':
                return sympy.polys.domains.EX

            # 使用正则表达式匹配 domain 是否为实数或复数域的特定精度表示
            r = cls._re_realfield.match(domain)
            if r is not None:
                _, _, prec = r.groups()
                # 如果精度为空，则返回实数域 RR
                if prec is None:
                    return sympy.polys.domains.RR
                # 否则，返回指定精度的实数域 RealField
                else:
                    return sympy.polys.domains.RealField(int(prec))

            # 使用正则表达式匹配 domain 是否为复数域的特定精度表示
            r = cls._re_complexfield.match(domain)
            if r is not None:
                _, _, prec = r.groups()
                # 如果精度为空，则返回复数域 CC
                if prec is None:
                    return sympy.polys.domains.CC
                # 否则，返回指定精度的复数域 ComplexField
                else:
                    return sympy.polys.domains.ComplexField(int(prec))

            # 使用正则表达式匹配 domain 是否为有限域的表示
            r = cls._re_finitefield.match(domain)
            if r is not None:
                # 返回对应有限域 FF 的实例
                return sympy.polys.domains.FF(int(r.groups()[1]))

            # 使用正则表达式匹配 domain 是否为多项式环的表示
            r = cls._re_polynomial.match(domain)
            if r is not None:
                ground, gens = r.groups()
                # 将生成器列表解析为 sympify 对象列表
                gens = list(map(sympify, gens.split(',')))
                # 根据基础环类型返回对应的多项式环对象
                if ground in ['Z', 'ZZ']:
                    return sympy.polys.domains.ZZ.poly_ring(*gens)
                elif ground in ['Q', 'QQ']:
                    return sympy.polys.domains.QQ.poly_ring(*gens)
                elif ground in ['R', 'RR']:
                    return sympy.polys.domains.RR.poly_ring(*gens)
                elif ground == 'ZZ_I':
                    return sympy.polys.domains.ZZ_I.poly_ring(*gens)
                elif ground == 'QQ_I':
                    return sympy.polys.domains.QQ_I.poly_ring(*gens)
                else:
                    return sympy.polys.domains.CC.poly_ring(*gens)

            # 使用正则表达式匹配 domain 是否为分式域的表示
            r = cls._re_fraction.match(domain)
            if r is not None:
                ground, gens = r.groups()
                # 将生成器列表解析为 sympify 对象列表
                gens = list(map(sympify, gens.split(',')))
                # 根据基础环类型返回对应的分式域对象
                if ground in ['Z', 'ZZ']:
                    return sympy.polys.domains.ZZ.frac_field(*gens)
                else:
                    return sympy.polys.domains.QQ.frac_field(*gens)

            # 使用正则表达式匹配 domain 是否为代数数域的表示
            r = cls._re_algebraic.match(domain)
            if r is not None:
                # 将生成器列表解析为 sympify 对象列表
                gens = list(map(sympify, r.groups()[1].split(',')))
                # 返回对应的代数数域对象
                return sympy.polys.domains.QQ.algebraic_field(*gens)

        # 如果未能匹配到合适的 domain 类型，则抛出选项错误异常
        raise OptionError('expected a valid domain specification, got %s' % domain)

    @classmethod
    # 定义类方法 postprocess，接受 cls 和 options 两个参数
    def postprocess(cls, options):
        # 检查 options 中是否包含 'gens' 和 'domain'，以及 options['domain'] 是否为 Composite 类型，
        # 同时检查 options['domain'].symbols 和 options['gens'] 是否有交集
        if 'gens' in options and 'domain' in options and options['domain'].is_Composite and \
                (set(options['domain'].symbols) & set(options['gens'])):
            # 如果条件满足，抛出 GeneratorsError 异常，说明生成器和域存在干扰
            raise GeneratorsError(
                "ground domain and generators interfere together")
        # 如果不满足上述条件，进一步检查
        elif ('gens' not in options or not options['gens']) and \
                'domain' in options and options['domain'] == sympy.polys.domains.EX:
            # 如果没有提供生成器并且请求的是 EX 域，抛出 GeneratorsError 异常，提示需要提供生成器
            raise GeneratorsError("you have to provide generators because EX domain was requested")
class Split(BooleanOption, metaclass=OptionType):
    """``split`` option to polynomial manipulation functions. """

    option = 'split'

    # 无需任何额外的参数
    requires: list[str] = []
    # 排除这些选项与``split``选项冲突
    excludes = ['field', 'greedy', 'domain', 'gaussian', 'extension',
        'modulus', 'symmetric']

    @classmethod
    def postprocess(cls, options):
        # 如果用户请求``split``选项，但这个功能尚未实现，则抛出错误
        if 'split' in options:
            raise NotImplementedError("'split' option is not implemented yet")


class Gaussian(BooleanOption, metaclass=OptionType):
    """``gaussian`` option to polynomial manipulation functions. """

    option = 'gaussian'

    # 无需任何额外的参数
    requires: list[str] = []
    # 排除这些选项与``gaussian``选项冲突
    excludes = ['field', 'greedy', 'domain', 'split', 'extension',
        'modulus', 'symmetric']

    @classmethod
    def postprocess(cls, options):
        # 如果用户请求``gaussian``选项并设置为True，则设定域为``QQ_I``
        if 'gaussian' in options and options['gaussian'] is True:
            options['domain'] = sympy.polys.domains.QQ_I
            # 处理``Extension``选项
            Extension.postprocess(options)


class Extension(Option, metaclass=OptionType):
    """``extension`` option to polynomial manipulation functions. """

    option = 'extension'

    # 无需任何额外的参数
    requires: list[str] = []
    # 排除这些选项与``extension``选项冲突
    excludes = ['greedy', 'domain', 'split', 'gaussian', 'modulus',
        'symmetric']

    @classmethod
    def preprocess(cls, extension):
        # 处理传入的扩展参数，确保它符合预期的格式
        if extension == 1:
            return bool(extension)
        elif extension == 0:
            raise OptionError("'False' is an invalid argument for 'extension'")
        else:
            if not hasattr(extension, '__iter__'):
                extension = {extension}
            else:
                if not extension:
                    extension = None
                else:
                    extension = set(extension)

            return extension

    @classmethod
    def postprocess(cls, options):
        # 如果``extension``选项在选项中，并且不是True，则使用``algebraic_field``创建域
        if 'extension' in options and options['extension'] is not True:
            options['domain'] = sympy.polys.domains.QQ.algebraic_field(
                *options['extension'])


class Modulus(Option, metaclass=OptionType):
    """``modulus`` option to polynomial manipulation functions. """

    option = 'modulus'

    # 无需任何额外的参数
    requires: list[str] = []
    # 排除这些选项与``modulus``选项冲突
    excludes = ['greedy', 'split', 'domain', 'gaussian', 'extension']

    @classmethod
    def preprocess(cls, modulus):
        # 处理传入的模数参数，确保它是正整数
        modulus = sympify(modulus)

        if modulus.is_Integer and modulus > 0:
            return int(modulus)
        else:
            raise OptionError(
                "'modulus' must a positive integer, got %s" % modulus)

    @classmethod
    def postprocess(cls, options):
        # 如果``modulus``选项在选项中，则使用``FF``创建域
        if 'modulus' in options:
            modulus = options['modulus']
            symmetric = options.get('symmetric', True)
            options['domain'] = sympy.polys.domains.FF(modulus, symmetric)


class Symmetric(BooleanOption, metaclass=OptionType):
    """``symmetric`` option to polynomial manipulation functions. """

    option = 'symmetric'

    # 需要``modulus``选项
    requires = ['modulus']
    # 排除这些选项与``symmetric``选项冲突
    excludes = ['greedy', 'domain', 'split', 'gaussian', 'extension']
class Strict(BooleanOption, metaclass=OptionType):
    """``strict`` option to polynomial manipulation functions. """

    # 定义选项名称为 'strict' 的布尔类型选项，作为多项式操作函数的严格模式选项
    option = 'strict'

    @classmethod
    def default(cls):
        # 默认情况下，返回 True
        return True


class Auto(BooleanOption, Flag, metaclass=OptionType):
    """``auto`` flag to polynomial manipulation functions. """

    # 定义选项名称为 'auto' 的布尔类型和标志类型选项，作为多项式操作函数的自动模式标志选项
    option = 'auto'

    # 指定 'after' 列表，定义此选项在哪些其他选项之后应用
    after = ['field', 'domain', 'extension', 'gaussian']

    @classmethod
    def default(cls):
        # 默认情况下，返回 True
        return True

    @classmethod
    def postprocess(cls, options):
        # 如果选项中包含 'domain' 或 'field'，并且没有 'auto' 选项，则将 'auto' 设置为 False
        if ('domain' in options or 'field' in options) and 'auto' not in options:
            options['auto'] = False


class Frac(BooleanOption, Flag, metaclass=OptionType):
    """``auto`` option to polynomial manipulation functions. """

    # 定义选项名称为 'frac' 的布尔类型和标志类型选项，作为多项式操作函数的分数选项
    option = 'frac'

    @classmethod
    def default(cls):
        # 默认情况下，返回 False
        return False


class Formal(BooleanOption, Flag, metaclass=OptionType):
    """``formal`` flag to polynomial manipulation functions. """

    # 定义选项名称为 'formal' 的布尔类型和标志类型选项，作为多项式操作函数的正式模式标志选项
    option = 'formal'

    @classmethod
    def default(cls):
        # 默认情况下，返回 False
        return False


class Polys(BooleanOption, Flag, metaclass=OptionType):
    """``polys`` flag to polynomial manipulation functions. """

    # 定义选项名称为 'polys' 的布尔类型和标志类型选项，作为多项式操作函数的多项式标志选项
    option = 'polys'


class Include(BooleanOption, Flag, metaclass=OptionType):
    """``include`` flag to polynomial manipulation functions. """

    # 定义选项名称为 'include' 的布尔类型和标志类型选项，作为多项式操作函数的包含标志选项
    option = 'include'

    @classmethod
    def default(cls):
        # 默认情况下，返回 False
        return False


class All(BooleanOption, Flag, metaclass=OptionType):
    """``all`` flag to polynomial manipulation functions. """

    # 定义选项名称为 'all' 的布尔类型和标志类型选项，作为多项式操作函数的全部标志选项
    option = 'all'

    @classmethod
    def default(cls):
        # 默认情况下，返回 False
        return False


class Gen(Flag, metaclass=OptionType):
    """``gen`` flag to polynomial manipulation functions. """

    # 定义选项名称为 'gen' 的标志类型选项，作为多项式操作函数的生成器选项
    option = 'gen'

    @classmethod
    def default(cls):
        # 默认情况下，返回 0
        return 0

    @classmethod
    def preprocess(cls, gen):
        # 如果 gen 是 Basic 或者整数类型，则返回 gen；否则抛出 OptionError 异常
        if isinstance(gen, (Basic, int)):
            return gen
        else:
            raise OptionError("invalid argument for 'gen' option")


class Series(BooleanOption, Flag, metaclass=OptionType):
    """``series`` flag to polynomial manipulation functions. """

    # 定义选项名称为 'series' 的布尔类型和标志类型选项，作为多项式操作函数的级数标志选项
    option = 'series'

    @classmethod
    def default(cls):
        # 默认情况下，返回 False
        return False


class Symbols(Flag, metaclass=OptionType):
    """``symbols`` flag to polynomial manipulation functions. """

    # 定义选项名称为 'symbols' 的标志类型选项，作为多项式操作函数的符号选项
    option = 'symbols'

    @classmethod
    def default(cls):
        # 默认情况下，返回以 's' 开头的编号符号序列
        return numbered_symbols('s', start=1)

    @classmethod
    def preprocess(cls, symbols):
        # 如果 symbols 有 '__iter__' 属性，则返回其迭代器；否则抛出 OptionError 异常
        if hasattr(symbols, '__iter__'):
            return iter(symbols)
        else:
            raise OptionError("expected an iterator or iterable container, got %s" % symbols)


class Method(Flag, metaclass=OptionType):
    """``method`` flag to polynomial manipulation functions. """

    # 定义选项名称为 'method' 的标志类型选项，作为多项式操作函数的方法选项
    option = 'method'

    @classmethod
    def preprocess(cls, method):
        # 如果 method 是字符串类型，则返回其小写形式；否则抛出 OptionError 异常
        if isinstance(method, str):
            return method.lower()
        else:
            raise OptionError("expected a string, got %s" % method)
# 构建选项函数，根据提供的生成器和参数构造选项。
def build_options(gens, args=None):
    """Construct options from keyword arguments or ... options. """
    # 如果参数 args 为 None，则将 gens 视为 args，并清空 gens
    if args is None:
        gens, args = (), gens

    # 如果参数 args 的长度不为 1，或者不包含 'opt' 键，或者 gens 非空，则返回生成的 Options 对象
    if len(args) != 1 or 'opt' not in args or gens:
        return Options(gens, args)
    else:
        # 否则返回 args 字典中的 'opt' 键对应的值
        return args['opt']


# 允许指定的标志在给定上下文中使用的函数。
def allowed_flags(args, flags):
    """
    Allow specified flags to be used in the given context.

    Examples
    ========

    >>> from sympy.polys.polyoptions import allowed_flags
    >>> from sympy.polys.domains import ZZ

    >>> allowed_flags({'domain': ZZ}, [])

    >>> allowed_flags({'domain': ZZ, 'frac': True}, [])
    Traceback (most recent call last):
    ...
    FlagError: 'frac' flag is not allowed in this context

    >>> allowed_flags({'domain': ZZ, 'frac': True}, ['frac'])

    """
    # 将 flags 转换为集合
    flags = set(flags)

    # 遍历参数 args 中的每一个键
    for arg in args.keys():
        try:
            # 如果 Options.__options__[arg] 存在且是一个标志，并且 arg 不在 flags 中，则抛出 FlagError 异常
            if Options.__options__[arg].is_Flag and arg not in flags:
                raise FlagError(
                    "'%s' flag is not allowed in this context" % arg)
        except KeyError:
            # 如果 KeyError 异常被抛出，表示 arg 不是一个有效的选项，抛出 OptionError 异常
            raise OptionError("'%s' is not a valid option" % arg)


# 更新选项参数 options 的默认值函数。
def set_defaults(options, **defaults):
    """Update options with default values. """
    # 如果选项参数 options 中不存在 'defaults' 键，则将 options 转换为字典并加入 'defaults' 键和默认值 defaults
    if 'defaults' not in options:
        options = dict(options)
        options['defaults'] = defaults

    # 返回更新后的 options 参数
    return options

# 初始化 Options 的依赖顺序
Options._init_dependencies_order()
```