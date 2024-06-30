# `D:\src\scipysrc\sympy\doc\ext\numpydoc.py`

```
"""
========
numpydoc
========

Sphinx extension that handles docstrings in the Numpy standard format. [1]

It will:

- Convert Parameters etc. sections to field lists.
- Convert See Also section to a See also entry.
- Renumber references.
- Extract the signature from the docstring, if it can't be determined
  otherwise.

.. [1] https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

"""

# 导入必要的模块
import sys                      # 导入sys模块，用于系统相关操作
import re                       # 导入re模块，用于正则表达式操作
import pydoc                    # 导入pydoc模块，用于文档处理
import sphinx                   # 导入sphinx模块，Sphinx文档生成工具
import inspect                  # 导入inspect模块，用于检查对象
from collections.abc import Callable  # 导入Callable抽象基类

# 检查Sphinx版本是否符合要求，否则抛出运行时错误
if sphinx.__version__ < '1.0.1':
    raise RuntimeError("Sphinx 1.0.1 or newer is required")

# 导入docscrape_sphinx模块中的函数和类
from docscrape_sphinx import get_doc_object, SphinxDocString


# 定义函数mangle_docstrings，处理文档字符串的格式
def mangle_docstrings(app, what, name, obj, options, lines,
                      reference_offset=[0]):
    # 配置参数字典，从应用的配置中获取
    cfg = {'use_plots': app.config.numpydoc_use_plots,
           'show_class_members': app.config.numpydoc_show_class_members,
           'show_inherited_class_members':
           app.config.numpydoc_show_inherited_class_members,
           'class_members_toctree': app.config.numpydoc_class_members_toctree}

    u_NL = '\n'  # 定义换行符变量

    # 如果处理的是模块文档
    if what == 'module':
        # 正则表达式模式，用于匹配和删除顶部标题
        pattern = '^\\s*[#*=]{4,}\\n[a-z0-9 -]+\\n[#*=]{4,}\\s*'
        title_re = re.compile(pattern, re.I | re.S)
        # 删除顶部标题行，并重新分割lines列表
        lines[:] = title_re.sub('', u_NL.join(lines)).split(u_NL)
    else:
        # 获取对象的文档对象，根据配置参数进行解析
        doc = get_doc_object(obj, what, u_NL.join(lines), config=cfg)
        # 根据Python版本转换文档对象为字符串
        if sys.version_info[0] >= 3:
            doc = str(doc)
        else:
            doc = unicode(doc)  # 在Python 2中处理Unicode

        lines[:] = doc.split(u_NL)  # 将处理后的文档分割成行列表

    # 如果配置要求生成编辑链接，并且对象有__name__属性
    if (app.config.numpydoc_edit_link and hasattr(obj, '__name__') and
            obj.__name__):
        if hasattr(obj, '__module__'):
            v = {"full_name": "{}.{}".format(obj.__module__, obj.__name__)}
        else:
            v = {"full_name": obj.__name__}
        # 添加HTML条件注释和编辑链接到lines列表
        lines += ['', '.. htmlonly::', '']
        lines += ['    %s' % x for x in
                  (app.config.numpydoc_edit_link % v).split("\n")]

    # 替换文档中的引用编号，以避免重复
    references = []
    for line in lines:
        line = line.strip()
        m = re.match('^.. \\[([a-z0-9_.-])\\]', line, re.I)
        if m:
            references.append(m.group(1))

    # 按引用编号长度降序排序，避免覆盖部分
    references.sort(key=lambda x: -len(x))
    if references:
        for i, line in enumerate(lines):
            for r in references:
                if re.match('^\\d+$', r):
                    new_r = "R%d" % (reference_offset[0] + int(r))
                else:
                    new_r = "%s%d" % (r, reference_offset[0])
                # 替换引用编号为新的编号格式
                lines[i] = lines[i].replace('[%s]_' % r,
                                            '[%s]_' % new_r)
                lines[i] = lines[i].replace('.. [%s]' % r,
                                            '.. [%s]' % new_r)

    reference_offset[0] += len(references)  # 更新引用偏移量


# 定义函数mangle_signature，处理对象的签名
def mangle_signature(app, what, name, obj, options, sig, retann):
    # 如果对象是一个类，并且这个类没有定义`__init__`方法，或者`__init__`方法的文档字符串中包含'initializes x; see '，
    # 则不要尝试检查它
    if (inspect.isclass(obj) and
        (not hasattr(obj, '__init__') or
            'initializes x; see ' in pydoc.getdoc(obj.__init__))):
        # 返回空字符串，表示没有找到合适的文档和签名
        return '', ''
    
    # 如果对象既不是可调用的，也没有`__argspec_is_invalid_`属性，则直接返回
    if not (isinstance(obj, Callable) or
            hasattr(obj, '__argspec_is_invalid_')):
        return
    
    # 如果对象没有`__doc__`属性，则直接返回
    if not hasattr(obj, '__doc__'):
        return
    
    # 使用 SphinxDocString 类处理对象的文档字符串
    doc = SphinxDocString(pydoc.getdoc(obj))
    # 如果文档字符串中有签名信息，则提取签名
    if doc['Signature']:
        # 用正则表达式删除签名中的参数之前的部分，只留下参数部分
        sig = re.sub("^[^(]*", "", doc['Signature'])
        # 返回提取的签名和空字符串，表示没有额外的信息
        return sig, ''
def setup(app, get_doc_object_=get_doc_object):
    # 检查是否存在 'add_config_value' 方法，若不存在则返回，可能由 nose 调用，最好退出
    if not hasattr(app, 'add_config_value'):
        return  # probably called by nose, better bail out

    # 设置全局的 get_doc_object 函数
    global get_doc_object
    get_doc_object = get_doc_object_

    # 连接 'autodoc-process-docstring' 事件到 mangle_docstrings 函数
    app.connect('autodoc-process-docstring', mangle_docstrings)
    # 连接 'autodoc-process-signature' 事件到 mangle_signature 函数
    app.connect('autodoc-process-signature', mangle_signature)

    # 添加配置值 'numpydoc_edit_link'，初始值为 None，不可更改
    app.add_config_value('numpydoc_edit_link', None, False)
    # 添加配置值 'numpydoc_use_plots'，初始值为 None，不可更改
    app.add_config_value('numpydoc_use_plots', None, False)
    # 添加配置值 'numpydoc_show_class_members'，初始值为 True，可更改
    app.add_config_value('numpydoc_show_class_members', True, True)
    # 添加配置值 'numpydoc_show_inherited_class_members'，初始值为 True，可更改
    app.add_config_value('numpydoc_show_inherited_class_members', True, True)
    # 添加配置值 'numpydoc_class_members_toctree'，初始值为 True，可更改
    app.add_config_value('numpydoc_class_members_toctree', True, True)

    # 添加额外的文档处理域 NumpyPythonDomain
    app.add_domain(NumpyPythonDomain)
    # 添加额外的文档处理域 NumpyCDomain
    app.add_domain(NumpyCDomain)
```