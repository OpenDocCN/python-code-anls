# `D:\src\scipysrc\scikit-learn\sklearn\utils\_pprint.py`

```
"""This module contains the _EstimatorPrettyPrinter class used in
BaseEstimator.__repr__ for pretty-printing estimators"""

# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 Python Software Foundation;
# All Rights Reserved

# Authors: Fred L. Drake, Jr. <fdrake@acm.org> (built-in CPython pprint module)
#          Nicolas Hug (scikit-learn specific changes)

# License: PSF License version 2 (see below)

# PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
# --------------------------------------------

# 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"),
# and the Individual or Organization ("Licensee") accessing and otherwise
# using this software ("Python") in source or binary form and its associated
# documentation.

# 2. Subject to the terms and conditions of this License Agreement, PSF hereby
# grants Licensee a nonexclusive, royalty-free, world-wide license to
# reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use Python alone or in any
# derivative version, provided, however, that PSF's License Agreement and
# PSF's notice of copyright, i.e., "Copyright (c) 2001, 2002, 2003, 2004,
# 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
# 2017, 2018 Python Software Foundation; All Rights Reserved" are retained in
# Python alone or in any derivative version prepared by Licensee.

# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates Python or any part thereof, and wants to make the derivative
# work available to others as provided herein, then Licensee hereby agrees to
# include in any such work a brief summary of the changes made to Python.

# 4. PSF is making Python available to Licensee on an "AS IS" basis. PSF MAKES
# NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT
# NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR WARRANTY OF
# MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF
# PYTHON WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

# 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON FOR ANY
# INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
# MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON, OR ANY DERIVATIVE
# THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.

# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between PSF and
# Licensee. This License Agreement does not grant permission to use PSF
# trademarks or trade name in a trademark sense to endorse or promote products
# or services of Licensee, or any third party.

# 8. By copying, installing or otherwise using Python, Licensee agrees to be
# bound by the terms and conditions of this License Agreement.

# No specific code statements to annotate.
# 导入必要的模块和类
import inspect  # 导入 inspect 模块，用于获取对象信息
import pprint  # 导入 pprint 模块，用于美化打印输出

from .._config import get_config  # 从相对路径导入 get_config 函数
from ..base import BaseEstimator  # 从相对路径导入 BaseEstimator 类
from ._missing import is_scalar_nan  # 从相对路径导入 is_scalar_nan 函数

# 定义一个继承自 tuple 的类 KeyValTuple，用于正确渲染字典中的键值对元组
class KeyValTuple(tuple):
    """Dummy class for correctly rendering key-value tuples from dicts."""

    def __repr__(self):
        # 保证不覆盖 tuple.__repr__()，以便于 _dispatch[tuple.__repr__] 正确工作
        return super().__repr__()

# 定义一个继承自 KeyValTuple 的类 KeyValTupleParam，用于正确渲染参数中的键值对元组
class KeyValTupleParam(KeyValTuple):
    """Dummy class for correctly rendering key-value tuples from parameters."""
    
    pass  # 无需额外方法或属性，因此保持简单的占位类结构

# 定义函数 _changed_params(estimator)，返回给定估计器对象的非默认参数字典
def _changed_params(estimator):
    """Return dict (param_name: value) of parameters that were given to
    estimator with non-default values."""
    
    params = estimator.get_params(deep=False)  # 获取估计器对象的参数字典
    init_func = getattr(estimator.__init__, "deprecated_original", estimator.__init__)
    init_params = inspect.signature(init_func).parameters
    init_params = {name: param.default for name, param in init_params.items()}  # 获取构造函数的参数默认值

    def has_changed(k, v):
        if k not in init_params:  # 如果 k 是 **kwargs 的一部分则直接返回 True
            return True
        if init_params[k] == inspect._empty:  # 如果 k 没有默认值则直接返回 True
            return True
        # 尝试避免对嵌套估计器调用 repr
        if isinstance(v, BaseEstimator) and v.__class__ != init_params[k].__class__:  # 如果 v 是 BaseEstimator 的实例且类型与默认值不同，则返回 True
            return True
        # 最后一种手段使用 repr 进行比较，可能代价较高
        if repr(v) != repr(init_params[k]) and not (
            is_scalar_nan(init_params[k]) and is_scalar_nan(v)
        ):
            return True
        return False

    return {k: v for k, v in params.items() if has_changed(k, v)}  # 返回变更的参数字典

# 定义一个继承自 pprint.PrettyPrinter 的类 _EstimatorPrettyPrinter，用于美化打印估计器对象
class _EstimatorPrettyPrinter(pprint.PrettyPrinter):
    """Pretty Printer class for estimator objects.

    This extends the pprint.PrettyPrinter class, because:
    - we need estimators to be printed with their parameters, e.g.
      Estimator(param1=value1, ...) which is not supported by default.
    - the 'compact' parameter of PrettyPrinter is ignored for dicts, which
      may lead to very long representations that we want to avoid.

    Quick overview of pprint.PrettyPrinter (see also
    https://stackoverflow.com/questions/49565047/pprint-with-hex-numbers):

    - the entry point is the _format() method which calls format() (overridden
      here)
    - format() directly calls _safe_repr() for a first try at rendering the
      object
    - _safe_repr formats the whole object recursively, only calling itself,
      not caring about line length or anything
    """

    # 扩展了 PrettyPrinter 类，以支持估计器对象的美化打印，包括参数
    pass  # 无需额外方法或属性，因此保持简单的占位类结构
    # PrettyPrinter 类的初始化方法，用于设置打印输出的格式和参数
    def __init__(
        self,
        indent=1,  # 缩进级别，默认为1
        width=80,  # 每行最大宽度，默认为80
        depth=None,  # 打印的最大递归深度，默认为None（不限制）
        stream=None,  # 输出流，默认为None（标准输出）
        *,
        compact=False,  # 是否使用紧凑模式，默认为False
        indent_at_name=True,  # 是否在名称处缩进，默认为True
        n_max_elements_to_show=None,  # 列表、字典、元组中最大显示的元素数量，默认为None
    ):
        # 调用父类 PrettyPrinter 的初始化方法，设置基本的打印参数
        super().__init__(indent, width, depth, stream, compact=compact)
        self._indent_at_name = indent_at_name  # 设置是否在名称处缩进
        if self._indent_at_name:
            self._indent_per_level = 1  # 如果在名称处缩进，则每级缩进为1，忽略 indent 参数
        self._changed_only = get_config()["print_changed_only"]  # 获取配置中的是否仅打印更改内容的选项
        self.n_max_elements_to_show = n_max_elements_to_show  # 设置最大显示元素数量
    
    # 覆盖父类的 format() 方法，支持 changed_only 参数，用于生成对象的字符串表示
    def format(self, object, context, maxlevels, level):
        return _safe_repr(
            object, context, maxlevels, level, changed_only=self._changed_only
        )
    # 将机器学习估算器对象的类名写入输出流
    def _pprint_estimator(self, object, stream, indent, allowance, context, level):
        stream.write(object.__class__.__name__ + "(")
        # 如果设置了在类名处缩进，则增加缩进量
        if self._indent_at_name:
            indent += len(object.__class__.__name__)

        # 如果仅打印已更改的参数，则获取已更改的参数列表
        if self._changed_only:
            params = _changed_params(object)
        else:
            # 否则获取所有参数的列表（不包括深层参数）
            params = object.get_params(deep=False)

        # 格式化参数列表或字典项，并写入输出流
        self._format_params(
            sorted(params.items()), stream, indent, allowance + 1, context, level
        )
        stream.write(")")

    # 格式化字典项的输出，写入输出流
    def _format_dict_items(self, items, stream, indent, allowance, context, level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=True
        )

    # 格式化参数的输出，写入输出流
    def _format_params(self, items, stream, indent, allowance, context, level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=False
        )

    # 根据参数或字典项格式化输出，写入输出流
    def _format_params_or_dict_items(
        self, object, stream, indent, allowance, context, level, is_dict
    ):
        """Format dict items or parameters respecting the compact=True
        parameter. For some reason, the builtin rendering of dict items doesn't
        respect compact=True and will use one line per key-value if all cannot
        fit in a single line.
        Dict items will be rendered as <'key': value> while params will be
        rendered as <key=value>. The implementation is mostly copy/pasting from
        the builtin _format_items().
        This also adds ellipsis if the number of items is greater than
        self.n_max_elements_to_show.
        """
        # 将写入操作绑定到流的写方法
        write = stream.write
        # 增加缩进以适应嵌套级别
        indent += self._indent_per_level
        # 分隔符，表示每行结束后的内容
        delimnl = ",\n" + " " * indent
        # 初始化为空字符串
        delim = ""
        # 设置最大宽度和当前宽度
        width = max_width = self._width - indent + 1
        # 使用迭代器遍历对象
        it = iter(object)
        try:
            # 获取下一个条目
            next_ent = next(it)
        except StopIteration:
            # 如果迭代结束，则直接返回
            return
        # 是否到达最后一个条目的标志
        last = False
        # 记录已处理的条目数量
        n_items = 0
        while not last:
            # 如果已显示条目数达到设定的最大显示条目数，则添加省略号并结束
            if n_items == self.n_max_elements_to_show:
                write(", ...")
                break
            n_items += 1
            # 获取当前条目
            ent = next_ent
            try:
                # 获取下一个条目
                next_ent = next(it)
            except StopIteration:
                # 如果没有下一个条目，将最大宽度和当前宽度减去容错值
                last = True
                max_width -= allowance
                width -= allowance
            # 如果是紧凑模式，处理键值对或参数
            if self._compact:
                k, v = ent
                # 获取键的表示形式和值的表示形式
                krepr = self._repr(k, context, level)
                vrepr = self._repr(v, context, level)
                # 如果不是字典，去除键的单引号
                if not is_dict:
                    krepr = krepr.strip("'")
                # 根据是否是字典，设置中间的分隔符
                middle = ": " if is_dict else "="
                # 组合成表示条目的字符串
                rep = krepr + middle + vrepr
                # 计算字符串长度
                w = len(rep) + 2
                if width < w:
                    # 如果当前行容纳不下该条目，重置当前宽度为最大宽度
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    # 如果当前行能够容纳该条目，将条目写入流中
                    width -= w
                    write(delim)
                    delim = ", "
                    write(rep)
                    continue
            # 在非紧凑模式下，添加适当的分隔符并写入流
            write(delim)
            delim = delimnl
            # 根据是否是字典选择合适的类来处理条目
            class_ = KeyValTuple if is_dict else KeyValTupleParam
            # 调用递归处理方法处理条目
            self._format(
                class_(ent), stream, indent, allowance if last else 1, context, level
            )
    def _format_items(self, items, stream, indent, allowance, context, level):
        """Format the items of an iterable (list, tuple...). Same as the
        built-in _format_items, with support for ellipsis if the number of
        elements is greater than self.n_max_elements_to_show.
        """
        # 简化输出流的写入操作
        write = stream.write
        # 增加缩进
        indent += self._indent_per_level
        # 如果缩进量大于1，则在缩进前输出空格
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * " ")
        # 定义逗号和换行的分隔符
        delimnl = ",\n" + " " * indent
        # 初始化空字符串的分隔符
        delim = ""
        # 设置最大宽度和当前行的宽度
        width = max_width = self._width - indent + 1
        # 获取可迭代对象的迭代器
        it = iter(items)
        try:
            # 获取下一个元素
            next_ent = next(it)
        except StopIteration:
            return
        # 初始化最后一个元素标志为False
        last = False
        # 初始化元素计数器
        n_items = 0
        # 迭代处理每个元素
        while not last:
            # 如果元素数量达到最大显示数目
            if n_items == self.n_max_elements_to_show:
                write(", ...")
                break
            # 增加元素计数
            n_items += 1
            # 获取当前元素
            ent = next_ent
            try:
                # 尝试获取下一个元素
                next_ent = next(it)
            except StopIteration:
                # 如果没有下一个元素，则将最大宽度和当前行宽度减去额外的空间
                last = True
                max_width -= allowance
                width -= allowance
            # 如果紧凑模式被启用
            if self._compact:
                # 获取当前元素的字符串表示
                rep = self._repr(ent, context, level)
                # 计算当前字符串的宽度
                w = len(rep) + 2
                # 如果当前行宽度小于当前字符串宽度
                if width < w:
                    width = max_width
                    # 如果已经有分隔符存在，则在当前行末尾追加分隔符
                    if delim:
                        delim = delimnl
                # 如果当前行宽度大于等于当前字符串宽度
                if width >= w:
                    width -= w
                    # 写入当前分隔符
                    write(delim)
                    delim = ", "
                    # 写入当前元素的字符串表示
                    write(rep)
                    continue
            # 写入当前分隔符
            write(delim)
            delim = delimnl
            # 递归调用 _format 方法处理当前元素
            self._format(ent, stream, indent, allowance if last else 1, context, level)

    def _pprint_key_val_tuple(self, object, stream, indent, allowance, context, level):
        """Pretty printing for key-value tuples from dict or parameters."""
        # 解构键值对元组
        k, v = object
        # 获取键的字符串表示
        rep = self._repr(k, context, level)
        # 如果对象是 KeyValTupleParam 类型，则去除单引号
        if isinstance(object, KeyValTupleParam):
            rep = rep.strip("'")
            middle = "="
        else:
            middle = ": "
        # 将键的字符串表示写入输出流
        stream.write(rep)
        # 将分隔符（等号或冒号）写入输出流
        stream.write(middle)
        # 递归调用 _format 方法处理值
        self._format(
            v, stream, indent + len(rep) + len(middle), allowance, context, level
        )

    # 注意：需要复制 _dispatch 方法以防止内置 PrettyPrinter 类的实例调用 _EstimatorPrettyPrinter 的方法（参见问题 12906）
    # mypy 错误：Type[PrettyPrinter] 没有属性 "_dispatch"
    # 复制 _dispatch 方法以确保正确的调用绑定
    _dispatch = pprint.PrettyPrinter._dispatch.copy()  # type: ignore
    # 将自定义的打印方法与相应的对象类型关联
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator
    _dispatch[KeyValTuple.__repr__] = _pprint_key_val_tuple
def _safe_repr(object, context, maxlevels, level, changed_only=False):
    """Same as the builtin _safe_repr, with added support for Estimator
    objects."""
    # 获取对象的类型
    typ = type(object)

    # 如果对象类型是内置标量类型，则直接返回其字符串表示和相应的标志
    if typ in pprint._builtin_scalars:
        return repr(object), True, False

    # 获取对象类型的 __repr__ 方法
    r = getattr(typ, "__repr__", None)

    # 如果对象是字典并且其 __repr__ 方法与 dict.__repr__ 方法相同
    if issubclass(typ, dict) and r is dict.__repr__:
        # 如果字典为空，则返回空字典的字符串表示
        if not object:
            return "{}", True, False
        objid = id(object)
        # 如果超过最大递归层数，则返回省略号
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        # 检查是否已经在上下文中存在该对象的引用
        if objid in context:
            return pprint._recursion(object), False, True
        # 将对象添加到上下文中，表示正在处理该对象
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        # 对字典中的每个键值对进行递归安全的表示
        items = sorted(object.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only
            )
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only
            )
            append("%s: %s" % (krepr, vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        # 返回格式化后的字典表示，以及是否可读和是否递归的标志
        return "{%s}" % ", ".join(components), readable, recursive

    # 如果对象是列表并且其 __repr__ 方法与 list.__repr__ 方法相同，或者是元组并且与 tuple.__repr__ 方法相同
    if (issubclass(typ, list) and r is list.__repr__) or (
        issubclass(typ, tuple) and r is tuple.__repr__
    ):
        # 如果是空列表或元组，则返回其空表示
        if issubclass(typ, list):
            if not object:
                return "[]", True, False
            format = "[%s]"
        elif len(object) == 1:
            format = "(%s,)"
        else:
            if not object:
                return "()", True, False
            format = "(%s)"
        objid = id(object)
        # 如果超过最大递归层数，则返回省略号
        if maxlevels and level >= maxlevels:
            return format % "...", False, objid in context
        # 检查是否已经在上下文中存在该对象的引用
        if objid in context:
            return pprint._recursion(object), False, True
        # 将对象添加到上下文中，表示正在处理该对象
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        # 对列表或元组中的每个元素进行递归安全的表示
        for o in object:
            orepr, oreadable, orecur = _safe_repr(
                o, context, maxlevels, level, changed_only=changed_only
            )
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        # 返回格式化后的列表或元组表示，以及是否可读和是否递归的标志
        return format % ", ".join(components), readable, recursive
    # 检查类型是否为 BaseEstimator 的子类
    if issubclass(typ, BaseEstimator):
        # 获取对象的唯一标识符
        objid = id(object)
        # 如果设置了最大递归深度并且当前递归层级超过最大深度，则返回省略符号、不可读标记、以及对象是否在上下文中的布尔值
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        # 如果对象标识符已经在上下文中，则返回递归对象的安全表示形式、不可读标记、以及True表示已递归
        if objid in context:
            return pprint._recursion(object), False, True
        # 将对象标识符添加到上下文中，表示该对象已经处理过一次
        context[objid] = 1
        readable = True
        recursive = False
        # 如果只需要改变的参数，则获取变更后的参数列表
        if changed_only:
            params = _changed_params(object)
        else:
            # 否则获取对象的参数列表（不深度复制）
            params = object.get_params(deep=False)
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        # 对参数列表按照安全元组排序后处理
        items = sorted(params.items(), key=pprint._safe_tuple)
        # 遍历参数列表的键值对
        for k, v in items:
            # 对键和值进行安全表示，获取其安全表示形式、可读性、递归性
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only
            )
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only
            )
            # 将键值对的安全表示形式添加到组件列表中
            append("%s=%s" % (krepr.strip("'"), vrepr))
            # 更新整体可读性，如果任意键或值不可读或递归，则设置递归标记为True
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        # 处理完毕后从上下文中删除该对象的标识符
        del context[objid]
        # 返回格式化后的类型名和组件列表，以及总体可读性和是否递归的标记
        return ("%s(%s)" % (typ.__name__, ", ".join(components)), readable, recursive)

    # 对于非 BaseEstimator 的类型，返回对象的标准表示形式、对象是否不以"<"开头的布尔值，以及不递归的标记
    rep = repr(object)
    return rep, (rep and not rep.startswith("<")), False
```