# `D:\src\scipysrc\sympy\doc\ext\docscrape_sphinx.py`

```
# 导入系统模块
import sys
# 导入正则表达式模块
import re
# 导入检查模块
import inspect
# 导入文本包装模块
import textwrap
# 导入文档处理模块
import pydoc
# 导入 Sphinx 文档生成模块
import sphinx
# 导入集合模块
import collections

# 从 docscrape 模块中导入 NumpyDocString、FunctionDoc、ClassDoc 类
from docscrape import NumpyDocString, FunctionDoc, ClassDoc


class SphinxDocString(NumpyDocString):
    # 初始化 SphinxDocString 类，继承自 NumpyDocString
    def __init__(self, docstring, config={}):
        # 调用父类 NumpyDocString 的初始化方法
        NumpyDocString.__init__(self, docstring, config=config)
        # 载入配置信息
        self.load_config(config)

    # 载入配置信息的方法
    def load_config(self, config):
        # 设置是否使用图表的配置选项
        self.use_plots = config.get('use_plots', False)
        # 设置是否包含类成员 ToC 树的配置选项
        self.class_members_toctree = config.get('class_members_toctree', True)

    # 字符串转换工具方法：生成标题行的列表
    def _str_header(self, name, symbol='`'):
        return ['.. rubric:: ' + name, '']

    # 字符串转换工具方法：生成字段列表的列表
    def _str_field_list(self, name):
        return [':' + name + ':']

    # 字符串转换工具方法：缩进文本并返回列表
    def _str_indent(self, doc, indent=4):
        out = []
        for line in doc:
            out += [' '*indent + line]
        return out

    # 字符串转换工具方法：生成签名部分的列表
    def _str_signature(self):
        return ['']
        if self['Signature']:
            return ['``%s``' % self['Signature']] + ['']
        else:
            return ['']

    # 字符串转换工具方法：生成摘要部分的列表
    def _str_summary(self):
        return self['Summary'] + ['']

    # 字符串转换工具方法：生成扩展摘要部分的列表
    def _str_extended_summary(self):
        return self['Extended Summary'] + ['']

    # 字符串转换工具方法：生成返回值部分的列表
    def _str_returns(self, name='Returns'):
        out = []
        if self[name]:
            out += self._str_field_list(name)
            out += ['']
            for param, param_type, desc in self[name]:
                if param_type:
                    out += self._str_indent(['**{}** : {}'.format(param.strip(),
                                                                  param_type)])
                else:
                    out += self._str_indent([param.strip()])
                if desc:
                    out += ['']
                    out += self._str_indent(desc, 8)
                out += ['']
        return out

    # 字符串转换工具方法：生成参数列表部分的列表
    def _str_param_list(self, name):
        out = []
        if self[name]:
            out += self._str_field_list(name)
            out += ['']
            for param, param_type, desc in self[name]:
                if param_type:
                    out += self._str_indent(['**{}** : {}'.format(param.strip(),
                                                                  param_type)])
                else:
                    out += self._str_indent(['**%s**' % param.strip()])
                if desc:
                    out += ['']
                    out += self._str_indent(desc, 8)
                out += ['']
        return out

    # 属性装饰器：获取对象的属性
    @property
    def _obj(self):
        if hasattr(self, '_cls'):
            return self._cls
        elif hasattr(self, '_f'):
            return self._f
        return None
    def _str_member_list(self, name):
        """
        Generate a member listing, autosummary:: table where possible,
        and a table where not.

        """
        # 初始化一个空列表用于存储输出内容
        out = []
        # 检查指定名称是否在当前对象中存在
        if self[name]:
            # 添加名称作为 rubric 标题
            out += ['.. rubric:: %s' % name, '']
            # 获取可能存在的对象前缀
            prefix = getattr(self, '_name', '')

            if prefix:
                # 如果存在前缀，使用特定格式 '~%s.' % prefix
                prefix = '~%s.' % prefix

            # Lines that are commented out are used to make the
            # autosummary:: table. Since SymPy does not use the
            # autosummary:: functionality, it is easiest to just comment it
            # out.
            # autosum = []
            others = []
            # 遍历给定名称的参数、参数类型和描述
            for param, param_type, desc in self[name]:
                # 清除参数两侧的空格
                param = param.strip()

                # 检查被引用的成员是否可以有文档字符串
                param_obj = getattr(self._obj, param, None)
                if not (callable(param_obj)
                        or isinstance(param_obj, property)
                        or inspect.isgetsetdescriptor(param_obj)):
                    param_obj = None

                # if param_obj and (pydoc.getdoc(param_obj) or not desc):
                #     # Referenced object has a docstring
                #     autosum += ["   %s%s" % (prefix, param)]
                # else:
                # 若成员没有文档字符串或不是函数/属性，则添加到 others 列表中
                others.append((param, param_type, desc))

            # if autosum:
            #     out += ['.. autosummary::']
            #     if self.class_members_toctree:
            #         out += ['   :toctree:']
            #     out += [''] + autosum

            # 如果存在未归纳的成员，则生成表头和内容
            if others:
                # 计算参数名称的最大长度，最小为3
                maxlen_0 = max(3, max(len(x[0]) for x in others))
                # 根据最大长度生成表头格式
                hdr = "="*maxlen_0 + "  " + "="*10
                fmt = '%%%ds  %%s  ' % (maxlen_0,)
                # 添加表头和每个成员的格式化描述
                out += ['', '', hdr]
                for param, param_type, desc in others:
                    # 清理和格式化描述内容
                    desc = " ".join(x.strip() for x in desc).strip()
                    if param_type:
                        desc = "({}) {}".format(param_type, desc)
                    out += [fmt % (param.strip(), desc)]
                # 添加表尾
                out += [hdr]
            # 添加空行
            out += ['']
        # 返回生成的输出列表
        return out

    def _str_section(self, name):
        out = []
        # 检查指定名称的内容是否存在
        if self[name]:
            # 生成指定名称的标题和空行
            out += self._str_header(name)
            out += ['']
            # 使用 textwrap.dedent 处理内容，并按行拆分
            content = textwrap.dedent("\n".join(self[name])).split("\n")
            # 将处理后的内容添加到输出列表中
            out += content
            # 添加额外的空行
            out += ['']
        # 返回生成的输出列表
        return out

    def _str_see_also(self, func_role):
        out = []
        # 检查 'See Also' 部分是否存在内容
        if self['See Also']:
            # 调用父类方法获取 'See Also' 部分的内容
            see_also = super()._str_see_also(func_role)
            # 添加 'seealso::' 标题和空行
            out = ['.. seealso::', '']
            # 将内容缩进后添加到输出列表中
            out += self._str_indent(see_also[2:])
        # 返回生成的输出列表
        return out

    def _str_warnings(self):
        out = []
        # 检查 'Warnings' 部分是否存在内容
        if self['Warnings']:
            # 添加 'warning::' 标题和空行
            out = ['.. warning::', '']
            # 将 'Warnings' 部分内容缩进后添加到输出列表中
            out += self._str_indent(self['Warnings'])
        # 返回生成的输出列表
        return out
    # 返回对象索引的字符串表示，格式为 ".. index::" 后接索引的默认值，以及每个部分对应的引用列表
    def _str_index(self):
        # 获取索引数据
        idx = self['index']
        out = []
        # 如果索引为空，则返回空列表
        if len(idx) == 0:
            return out

        # 添加默认索引项到输出列表
        out += ['.. index:: %s' % idx.get('default', '')]
        # 遍历索引中的每个部分及其对应的引用列表
        for section, references in idx.items():
            if section == 'default':
                continue
            elif section == 'refguide':
                # 对于 refguide 部分，格式化单独引用条目
                out += ['   single: %s' % (', '.join(references))]
            else:
                # 对于其他部分，格式化为 "<部分名>: 引用1, 引用2, ..."
                out += ['   {}: {}'.format(section, ','.join(references))]
        return out

    # 返回对象的引用部分的字符串表示
    def _str_references(self):
        out = []
        # 如果有引用部分
        if self['References']:
            # 添加 References 部分的标题
            out += self._str_header('References')
            # 如果 References 是字符串，则转换为列表
            if isinstance(self['References'], str):
                self['References'] = [self['References']]
            # 添加所有引用条目
            out.extend(self['References'])
            out += ['']
            # 对于 Latex，将所有引用放入单独的参考文献中，因此需要插入到这里的链接
            if sphinx.__version__ >= "0.6":
                out += ['.. only:: latex', '']
            else:
                out += ['.. latexonly::', '']
            items = []
            # 提取所有引用的标签
            for line in self['References']:
                m = re.match(r'.. \[([a-z0-9._-]+)\]', line, re.I)
                if m:
                    items.append(m.group(1))
            # 格式化引用链接
            out += ['   ' + ", ".join(["[%s]_" % item for item in items]), '']
        return out

    # 返回对象的示例部分的字符串表示
    def _str_examples(self):
        examples_str = "\n".join(self['Examples'])

        # 如果使用绘图且示例中包含 'import matplotlib' 且未包含 'plot::' 则添加绘图命令
        if (self.use_plots and 'import matplotlib' in examples_str
                and 'plot::' not in examples_str):
            out = []
            # 添加示例部分的标题
            out += self._str_header('Examples')
            # 添加绘图命令
            out += ['.. plot::', '']
            # 添加示例内容并缩进
            out += self._str_indent(self['Examples'])
            out += ['']
            return out
        else:
            # 否则返回示例部分的默认表示
            return self._str_section('Examples')

    # 返回对象的字符串表示，可以指定缩进和功能角色
    def __str__(self, indent=0, func_role="obj"):
        out = []
        # 添加对象的签名部分
        out += self._str_signature()
        # 添加对象的索引部分
        out += self._str_index() + ['']
        # 添加对象的摘要部分
        out += self._str_summary()
        # 添加对象的扩展摘要部分
        out += self._str_extended_summary()
        # 添加对象的参数列表部分
        out += self._str_param_list('Parameters')
        # 添加对象的返回值部分
        out += self._str_returns('Returns')
        # 添加对象的生成值部分
        out += self._str_returns('Yields')
        # 添加其他参数、异常和警告部分
        for param_list in ('Other Parameters', 'Raises', 'Warns'):
            out += self._str_param_list(param_list)
        # 添加对象的警告部分
        out += self._str_warnings()
        # 添加所有其他自定义部分的内容
        for s in self._other_keys:
            out += self._str_section(s)
        # 添加对象的相关链接部分
        out += self._str_see_also(func_role)
        # 添加对象的引用部分
        out += self._str_references()
        # 添加对象的成员列表部分
        out += self._str_member_list('Attributes')
        # 对输出进行缩进处理
        out = self._str_indent(out, indent)
        return '\n'.join(out)
# 定义一个继承自SphinxDocString和FunctionDoc的类SphinxFunctionDoc，用于处理函数文档
class SphinxFunctionDoc(SphinxDocString, FunctionDoc):
    # 初始化方法，接受obj、doc和config参数，并加载配置
    def __init__(self, obj, doc=None, config={}):
        self.load_config(config)  # 调用父类方法加载配置
        FunctionDoc.__init__(self, obj, doc=doc, config=config)  # 调用父类构造函数初始化

# 定义一个继承自SphinxDocString和ClassDoc的类SphinxClassDoc，用于处理类文档
class SphinxClassDoc(SphinxDocString, ClassDoc):
    # 初始化方法，接受obj、doc、func_doc和config参数，并加载配置
    def __init__(self, obj, doc=None, func_doc=None, config={}):
        self.load_config(config)  # 调用父类方法加载配置
        ClassDoc.__init__(self, obj, doc=doc, func_doc=None, config=config)  # 调用父类构造函数初始化

# 定义一个继承自SphinxDocString的类SphinxObjDoc，用于处理对象文档
class SphinxObjDoc(SphinxDocString):
    # 初始化方法，接受obj、doc和config参数，并加载配置
    def __init__(self, obj, doc=None, config={}):
        self._f = obj  # 将传入的obj赋值给成员变量_f
        self.load_config(config)  # 调用父类方法加载配置
        SphinxDocString.__init__(self, doc, config=config)  # 调用父类构造函数初始化

# 定义一个函数get_doc_object，用于根据传入的obj、what、doc和config参数获取文档对象
def get_doc_object(obj, what=None, doc=None, config={}):
    # 根据obj的类型确定what的值，分别为'class'、'module'、'function'或'object'
    if inspect.isclass(obj):
        what = 'class'
    elif inspect.ismodule(obj):
        what = 'module'
    elif callable(obj):
        what = 'function'
    else:
        what = 'object'

    # 根据what的值选择返回不同类型的文档对象
    if what == 'class':
        return SphinxClassDoc(obj, func_doc=SphinxFunctionDoc, doc=doc,
                              config=config)  # 返回一个处理类文档的SphinxClassDoc对象
    elif what in ('function', 'method'):
        return SphinxFunctionDoc(obj, doc=doc, config=config)  # 返回一个处理函数文档的SphinxFunctionDoc对象
    else:
        if doc is None:
            doc = pydoc.getdoc(obj)  # 如果没有传入doc参数，则使用pydoc.getdoc获取对象的文档字符串
        return SphinxObjDoc(obj, doc, config=config)  # 返回一个处理对象文档的SphinxObjDoc对象
```