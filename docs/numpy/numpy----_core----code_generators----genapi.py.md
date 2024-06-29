# `.\numpy\numpy\_core\code_generators\genapi.py`

```
"""
Get API information encoded in C files.

See ``find_function`` for how functions should be formatted, and
``read_order`` for how the order of the functions should be
specified.

"""
# 导入必要的库和模块
import hashlib  # 导入用于哈希计算的模块
import io  # 导入用于处理IO操作的模块
import os  # 导入操作系统相关功能的模块
import re  # 导入正则表达式模块
import sys  # 导入系统相关的参数和功能
import importlib.util  # 导入用于动态导入模块的工具
import textwrap  # 导入用于文本格式化的模块

from os.path import join  # 从os.path模块中导入join函数，用于路径拼接


def get_processor():
    # 由于无法直接从numpy.distutils导入（因为numpy尚未构建），因此采用以下复杂的方式
    # 构造conv_template.py文件的路径
    conv_template_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'distutils', 'conv_template.py'
    )
    # 使用importlib动态导入conv_template.py文件作为模块
    spec = importlib.util.spec_from_file_location(
        'conv_template', conv_template_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 返回处理文件的函数对象
    return mod.process_file


process_c_file = get_processor()

__docformat__ = 'restructuredtext'

# The files under src/ that are scanned for API functions
# 在src/目录下扫描以获取API函数信息的文件
# 定义包含多个文件路径的列表，这些路径指向多维数组和数学运算的源文件
API_FILES = [join('multiarray', 'alloc.c'),
             join('multiarray', 'abstractdtypes.c'),
             join('multiarray', 'arrayfunction_override.c'),
             join('multiarray', 'array_assign_array.c'),
             join('multiarray', 'array_assign_scalar.c'),
             join('multiarray', 'array_coercion.c'),
             join('multiarray', 'array_converter.c'),
             join('multiarray', 'array_method.c'),
             join('multiarray', 'arrayobject.c'),
             join('multiarray', 'arraytypes.c.src'),
             join('multiarray', 'buffer.c'),
             join('multiarray', 'calculation.c'),
             join('multiarray', 'common_dtype.c'),
             join('multiarray', 'conversion_utils.c'),
             join('multiarray', 'convert.c'),
             join('multiarray', 'convert_datatype.c'),
             join('multiarray', 'ctors.c'),
             join('multiarray', 'datetime.c'),
             join('multiarray', 'datetime_busday.c'),
             join('multiarray', 'datetime_busdaycal.c'),
             join('multiarray', 'datetime_strings.c'),
             join('multiarray', 'descriptor.c'),
             join('multiarray', 'dlpack.c'),
             join('multiarray', 'dtypemeta.c'),
             join('multiarray', 'einsum.c.src'),
             join('multiarray', 'public_dtype_api.c'),
             join('multiarray', 'flagsobject.c'),
             join('multiarray', 'getset.c'),
             join('multiarray', 'item_selection.c'),
             join('multiarray', 'iterators.c'),
             join('multiarray', 'mapping.c'),
             join('multiarray', 'methods.c'),
             join('multiarray', 'multiarraymodule.c'),
             join('multiarray', 'nditer_api.c'),
             join('multiarray', 'nditer_constr.c'),
             join('multiarray', 'nditer_pywrap.c'),
             join('multiarray', 'nditer_templ.c.src'),
             join('multiarray', 'number.c'),
             join('multiarray', 'refcount.c'),
             join('multiarray', 'scalartypes.c.src'),
             join('multiarray', 'scalarapi.c'),
             join('multiarray', 'sequence.c'),
             join('multiarray', 'shape.c'),
             join('multiarray', 'stringdtype', 'static_string.c'),
             join('multiarray', 'strfuncs.c'),
             join('multiarray', 'usertypes.c'),
             join('umath', 'dispatching.c'),
             join('umath', 'extobj.c'),
             join('umath', 'loops.c.src'),
             join('umath', 'reduction.c'),
             join('umath', 'ufunc_object.c'),
             join('umath', 'ufunc_type_resolution.c'),
             join('umath', 'wrapping_array_method.c'),
            ]

# 获取当前文件的路径
THIS_DIR = os.path.dirname(__file__)

# 将 API_FILES 列表中的每个文件路径与当前文件的相对路径拼接，更新 API_FILES
API_FILES = [os.path.join(THIS_DIR, '..', 'src', a) for a in API_FILES]

# 返回给定文件名在当前目录中的完整路径
def file_in_this_dir(filename):
    return os.path.join(THIS_DIR, filename)

# 移除字符串中的空白字符并返回结果
def remove_whitespace(s):
    return ''.join(s.split())

# 替换字符串中的 'Bool' 为 'npy_bool' 并返回结果
def _repl(str):
    return str.replace('Bool', 'npy_bool')
# 定义一个表示最小版本的类
class MinVersion:
    def __init__(self, version):
        """ Version should be the normal NumPy version, e.g. "1.25" """
        # 将版本号按照点号分割为主版本号和次版本号
        major, minor = version.split(".")
        # 根据版本号创建一个带有前缀的版本字符串
        self.version = f"NPY_{major}_{minor}_API_VERSION"

    def __str__(self):
        # 用于版本哈希计算的方法，返回版本字符串
        return self.version

    def add_guard(self, name, normal_define):
        """Wrap a definition behind a version guard"""
        # 创建一个带有版本保护的定义包装
        wrap = textwrap.dedent(f"""
            #if NPY_FEATURE_VERSION >= {self.version}
            {{define}}
            #endif""")
        
        # 将`define`后插入以避免混淆 dedent 处理：
        return wrap.format(define=normal_define)


# 定义一个类用于偷取引用
class StealRef:
    def __init__(self, arg):
        # 记录引用的参数位置（从1开始计数）
        self.arg = arg  # counting from 1

    def __str__(self):
        try:
            # 将参数位置转换为对应的偷取引用宏格式
            return ' '.join('NPY_STEALS_REF_TO_ARG(%d)' % x for x in self.arg)
        except TypeError:
            # 如果参数位置不是可迭代的，直接转换为对应的偷取引用宏格式
            return 'NPY_STEALS_REF_TO_ARG(%d)' % self.arg


# 定义一个表示函数的类
class Function:
    def __init__(self, name, return_type, args, doc=''):
        # 函数名
        self.name = name
        # 返回类型，使用内部方法替换处理
        self.return_type = _repl(return_type)
        # 函数参数列表
        self.args = args
        # 函数文档字符串
        self.doc = doc

    def _format_arg(self, typename, name):
        # 格式化函数参数为指定格式
        if typename.endswith('*'):
            return typename + name
        else:
            return typename + ' ' + name

    def __str__(self):
        # 将函数转换为字符串表示形式
        argstr = ', '.join([self._format_arg(*a) for a in self.args])
        if self.doc:
            doccomment = '/* %s */\n' % self.doc
        else:
            doccomment = ''
        return '%s%s %s(%s)' % (doccomment, self.return_type, self.name, argstr)

    def api_hash(self):
        # 计算函数的哈希值
        m = hashlib.md5()
        m.update(remove_whitespace(self.return_type))
        m.update('\000')
        m.update(self.name)
        m.update('\000')
        for typename, name in self.args:
            m.update(remove_whitespace(typename))
            m.update('\000')
        return m.hexdigest()[:8]


# 定义一个自定义异常类，用于解析错误
class ParseError(Exception):
    def __init__(self, filename, lineno, msg):
        # 异常初始化，记录文件名、行号和错误消息
        self.filename = filename
        self.lineno = lineno
        self.msg = msg

    def __str__(self):
        # 将异常信息转换为字符串形式输出
        return '%s:%s:%s' % (self.filename, self.lineno, self.msg)


# 定义一个函数，用于跳过括号内的内容并返回位置
def skip_brackets(s, lbrac, rbrac):
    # 统计括号嵌套层数，返回闭合括号后的位置
    count = 0
    for i, c in enumerate(s):
        if c == lbrac:
            count += 1
        elif c == rbrac:
            count -= 1
        if count == 0:
            return i
    # 若没有找到对应的闭合括号，抛出值错误异常
    raise ValueError("no match '%s' for '%s' (%r)" % (lbrac, rbrac, s))


# 定义一个函数，用于分割函数参数字符串为参数列表
def split_arguments(argstr):
    arguments = []
    current_argument = []
    i = 0

    def finish_arg():
        # 完成当前参数的处理，将其格式化并添加到参数列表中
        if current_argument:
            argstr = ''.join(current_argument).strip()
            m = re.match(r'(.*(\s+|\*))(\w+)$', argstr)
            if m:
                typename = m.group(1).strip()
                name = m.group(3)
            else:
                typename = argstr
                name = ''
            arguments.append((typename, name))
            del current_argument[:]
    # 当前索引 i 小于参数字符串 argstr 的长度时，执行循环
    while i < len(argstr):
        # 获取当前字符 c
        c = argstr[i]
        # 如果当前字符是逗号 ','，则完成当前参数的处理
        if c == ',':
            finish_arg()
        # 如果当前字符是左括号 '('，则跳过括号内的内容，并将整体作为当前参数的一部分
        elif c == '(':
            # 调用 skip_brackets 函数跳过括号内的内容，并将结果追加到当前参数中
            p = skip_brackets(argstr[i:], '(', ')')
            current_argument += argstr[i:i+p]
            # 更新索引 i，跳过括号内的内容
            i += p - 1
        # 如果当前字符既不是逗号也不是左括号，则将其添加到当前参数中
        else:
            current_argument += c
        # 更新索引 i，处理下一个字符
        i += 1
    # 处理最后一个参数
    finish_arg()
    # 返回处理完成的参数列表
    return arguments
# 定义一个函数，用于在指定文件中查找带有标记的函数
def find_functions(filename, tag='API'):
    """
    Scan the file, looking for tagged functions.

    Assuming ``tag=='API'``, a tagged function looks like::

        /*API*/
        static returntype*
        function_name(argtype1 arg1, argtype2 arg2)
        {
        }

    where the return type must be on a separate line, the function
    name must start the line, and the opening ``{`` must start the line.

    An optional documentation comment in ReST format may follow the tag,
    as in::

        /*API
          This function does foo...
         */
    """
    # 根据文件扩展名判断是否为特定类型的文件，然后打开文件或者处理后打开文件
    if filename.endswith(('.c.src', '.h.src')):
        fo = io.StringIO(process_c_file(filename))
    else:
        fo = open(filename, 'r')
    
    # 初始化存储函数信息的列表和其他变量
    functions = []
    return_type = None
    function_name = None
    function_args = []
    doclist = []
    
    # 定义状态常量
    SCANNING, STATE_DOC, STATE_RETTYPE, STATE_NAME, STATE_ARGS = list(range(5))
    state = SCANNING
    
    # 构造标记注释字符串
    tagcomment = '/*' + tag
    # 遍历文件对象 fo 的每一行，并同时获取行号和行内容
    for lineno, line in enumerate(fo):
        try:
            # 去除每行行尾的换行符和空格
            line = line.strip()
            # 如果当前状态是 SCANNING
            if state == SCANNING:
                # 如果该行以 tagcomment 开头
                if line.startswith(tagcomment):
                    # 如果该行以 '*/' 结尾，则切换到 STATE_RETTYPE 状态
                    if line.endswith('*/'):
                        state = STATE_RETTYPE
                    else:
                        # 否则切换到 STATE_DOC 状态
                        state = STATE_DOC
            # 如果当前状态是 STATE_DOC
            elif state == STATE_DOC:
                # 如果该行以 '*/' 开头，则切换到 STATE_RETTYPE 状态
                if line.startswith('*/'):
                    state = STATE_RETTYPE
                else:
                    # 否则将行首的 ' *' 去除后加入到 doclist 中
                    line = line.lstrip(' *')
                    doclist.append(line)
            # 如果当前状态是 STATE_RETTYPE
            elif state == STATE_RETTYPE:
                # 匹配以 'NPY_NO_EXPORT ' 开头的内容，并去除该部分后得到返回类型
                m = re.match(r'NPY_NO_EXPORT\s+(.*)$', line)
                if m:
                    line = m.group(1)
                return_type = line
                # 切换到 STATE_NAME 状态，准备获取函数名
                state = STATE_NAME
            # 如果当前状态是 STATE_NAME
            elif state == STATE_NAME:
                # 匹配以字母或下划线开头，后跟任意个字符再加上 ' ('
                m = re.match(r'(\w+)\s*\(', line)
                if m:
                    function_name = m.group(1)
                else:
                    # 如果未匹配到函数名，则抛出 ParseError 异常
                    raise ParseError(filename, lineno+1,
                                     'could not find function name')
                # 将函数参数部分存入 function_args
                function_args.append(line[m.end():])
                # 切换到 STATE_ARGS 状态，准备获取函数参数
                state = STATE_ARGS
            # 如果当前状态是 STATE_ARGS
            elif state == STATE_ARGS:
                # 如果该行以 '{' 开头，则表示函数参数获取完毕
                if line.startswith('{'):
                    # 清除尾部的空格和闭合括号 '}'，并拆分成参数列表
                    fargs_str = ' '.join(function_args).rstrip()[:-1].rstrip()
                    fargs = split_arguments(fargs_str)
                    # 创建 Function 对象并加入 functions 列表
                    f = Function(function_name, return_type, fargs,
                                 '\n'.join(doclist))
                    functions.append(f)
                    # 重置状态和相关变量
                    return_type = None
                    function_name = None
                    function_args = []
                    doclist = []
                    state = SCANNING
                else:
                    # 否则继续添加函数参数
                    function_args.append(line)
        except ParseError:
            # 如果捕获到 ParseError 异常，则直接抛出
            raise
        except Exception as e:
            # 捕获其它异常，抛出 ParseError 异常并附加详细信息
            msg = "see chained exception for details"
            raise ParseError(filename, lineno + 1, msg) from e
    # 关闭文件对象 fo
    fo.close()
    # 返回解析得到的 functions 列表
    return functions
# 将数据写入指定文件名的文件中
def write_file(filename, data):
    # 检查文件是否存在
    if os.path.exists(filename):
        # 如果文件存在，打开文件并读取内容
        with open(filename) as f:
            # 比较文件内容与要写入的数据是否相同，如果相同则不更新
            if data == f.read():
                return

    # 打开文件并写入数据
    with open(filename, 'w') as fid:
        fid.write(data)


# 这些 *Api 类的实例知道如何为生成的代码输出字符串
class TypeApi:
    def __init__(self, name, index, ptr_cast, api_name, internal_type=None):
        # 初始化 TypeApi 实例，设置名称、索引、指针转换、API 名称和内部类型（可选）
        self.index = index
        self.name = name
        self.ptr_cast = ptr_cast
        self.api_name = api_name
        self.internal_type = internal_type  # 内部类型，如果为 None，则与导出类型相同（ptr_cast）

    # 生成从数组 API 字符串定义的宏
    def define_from_array_api_string(self):
        return "#define %s (*(%s *)%s[%d])" % (self.name,
                                               self.ptr_cast,
                                               self.api_name,
                                               self.index)

    # 返回数组 API 定义的字符串
    def array_api_define(self):
        return "        (void *) &%s" % self.name

    # 生成内部定义的字符串
    def internal_define(self):
        if self.internal_type is None:
            # 如果内部类型为 None，则生成外部定义
            return f"extern NPY_NO_EXPORT {self.ptr_cast} {self.name};\n"

        # 如果存在内部类型，进行类型名称重整
        mangled_name = f"{self.name}Full"
        astr = (
            # 创建重整后的类型名称
            f"extern NPY_NO_EXPORT {self.internal_type} {mangled_name};\n"
            # 定义名称为 (*(type *)(&mangled_name))
            f"#define {self.name} (*({self.ptr_cast} *)(&{mangled_name}))\n"
        )
        return astr

class GlobalVarApi:
    def __init__(self, name, index, type, api_name):
        # 初始化 GlobalVarApi 实例，设置名称、索引、类型和 API 名称
        self.name = name
        self.index = index
        self.type = type
        self.api_name = api_name

    # 生成从数组 API 字符串定义的宏
    def define_from_array_api_string(self):
        return "#define %s (*(%s *)%s[%d])" % (self.name,
                                               self.type,
                                               self.api_name,
                                               self.index)

    # 返回数组 API 定义的字符串
    def array_api_define(self):
        return "        (%s *) &%s" % (self.type, self.name)

    # 生成内部定义的字符串
    def internal_define(self):
        astr = """\
extern NPY_NO_EXPORT %(type)s %(name)s;
""" % {'type': self.type, 'name': self.name}
        return astr

# 虚拟类，用于一致地使用 *Api 实例处理数组 API 中的所有项目
class BoolValuesApi:
    def __init__(self, name, index, api_name):
        # 初始化 BoolValuesApi 实例，设置名称、索引、类型和 API 名称
        self.name = name
        self.index = index
        self.type = 'PyBoolScalarObject'
        self.api_name = api_name
    # 定义一个函数，用于生成从数组API字符串到C语言宏定义的字符串
    def define_from_array_api_string(self):
        # 返回一个格式化后的字符串，包含宏定义格式，使用实例的名称、类型、API名称和索引
        return "#define %s ((%s *)%s[%d])" % (self.name,
                                              self.type,
                                              self.api_name,
                                              self.index)

    # 定义一个函数，生成用于数组API的C语言宏定义字符串
    def array_api_define(self):
        # 返回一个格式化后的字符串，指向实例名称的指针
        return "        (void *) &%s" % self.name

    # 定义一个函数，生成内部使用的C语言宏定义字符串
    def internal_define(self):
        # astr字符串的多行文本，用于嵌入到其他文本中
        astr = """\
"""
定义一个名为 _PyArrayScalar_BoolValues 的数组，其中包含两个 PyBoolScalarObject 对象。
"""
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];

class FunctionApi:
    def __init__(self, name, index, annotations, return_type, args, api_name):
        self.name = name  # 函数名称
        self.index = index  # 函数在 API 中的索引位置

        self.min_version = None  # 最小版本号
        self.annotations = []
        for annotation in annotations:
            # 检查注解类型，如果是 "StealRef" 则加入注解列表
            if type(annotation).__name__ == "StealRef":
                self.annotations.append(annotation)
            # 如果是 "MinVersion" 则设置最小版本号，确保只设置一次
            elif type(annotation).__name__ == "MinVersion":
                if self.min_version is not None:
                    raise ValueError("Two minimum versions specified!")
                self.min_version = annotation
            else:
                raise ValueError(f"unknown annotation {annotation}")

        self.return_type = return_type  # 函数返回类型
        self.args = args  # 函数参数列表
        self.api_name = api_name  # 函数在 API 中的名称

    def _argtypes_string(self):
        """
        返回参数类型字符串，如果没有参数则返回 'void'，否则返回逗号分隔的参数类型列表。
        """
        if not self.args:
            return 'void'
        argstr = ', '.join([_repl(a[0]) for a in self.args])
        return argstr

    def define_from_array_api_string(self):
        """
        根据函数的参数和返回类型生成宏定义字符串。
        """
        arguments = self._argtypes_string()
        define = textwrap.dedent(f"""\
            #define {self.name} \\
                    (*({self.return_type} (*)({arguments})) \\
                {self.api_name}[{self.index}])""")
        
        if self.min_version is not None:
            define = self.min_version.add_guard(self.name, define)
        return define

    def array_api_define(self):
        """
        返回函数在数组 API 中的定义字符串。
        """
        return "        (void *) %s" % self.name

    def internal_define(self):
        """
        返回函数的内部定义字符串，包括注解、返回类型、名称和参数类型。
        """
        annstr = [str(a) for a in self.annotations]
        annstr = ' '.join(annstr)
        astr = """\
NPY_NO_EXPORT %s %s %s \\\n       (%s);""" % (annstr, self.return_type,
                                              self.name,
                                              self._argtypes_string())
        return astr

def order_dict(d):
    """
    对字典按值排序，并返回排序后的列表。
    """
    o = list(d.items())
    def _key(x):
        return x[1] + (x[0],)
    return sorted(o, key=_key)

def merge_api_dicts(dicts):
    """
    合并多个 API 字典为一个字典。
    """
    ret = {}
    for d in dicts:
        for k, v in d.items():
            ret[k] = v
    return ret

def check_api_dict(d):
    """
    检查 API 字典是否有效，确保索引不重复，并移除 "__unused_indices__" 字段。
    """
    # 移除 "__unused_indices__" 字段并获取其值，这些值表示未使用的索引
    removed = set(d.pop("__unused_indices__", []))
    # 从每个键值对中提取索引和第一个值，构建新的字典
    index_d = {k: v[0] for k, v in d.items()}

    # 如果有相同的索引被使用超过一次，抛出异常
    revert_dict = {v: k for k, v in index_d.items()}
    # 检查反转字典的长度是否与索引字典的长度相等，若不相等则抛出异常
    if not len(revert_dict) == len(index_d):
        # 创建一个字典，将索引映射到其关联项目的列表
        doubled = {}
        # 遍历索引字典中的每个名称和索引对
        for name, index in index_d.items():
            try:
                # 尝试将名称添加到对应索引的列表中
                doubled[index].append(name)
            except KeyError:
                # 若索引不存在则创建一个新的列表
                doubled[index] = [name]
        # 格式化错误消息模板，指出重复使用相同索引的情况
        fmt = "Same index has been used twice in api definition: {}"
        # 构造详细错误消息，列出重复使用索引及其对应的名称
        val = ''.join(
            '\n\tindex {} -> {}'.format(index, names)
            for index, names in doubled.items() if len(names) != 1
        )
        # 抛出值错误异常，包含详细错误消息
        raise ValueError(fmt.format(val))

    # 确保索引中没有“空洞”，并且从索引 0 开始连续使用
    indexes = set(index_d.values())
    # 预期的索引集合应为当前索引集合与已移除索引的并集
    expected = set(range(len(indexes) + len(removed)))
    # 如果索引集合与已移除索引有交集，则抛出值错误异常
    if not indexes.isdisjoint(removed):
        raise ValueError("API index used but marked unused: "
                         f"{indexes.intersection(removed)}")
    # 如果索引集合与已移除索引的并集不等于预期的索引集合，则抛出值错误异常
    if indexes.union(removed) != expected:
        # 计算预期索引集合与当前索引集合的对称差异，构造错误消息
        diff = expected.symmetric_difference(indexes.union(removed))
        msg = "There are some holes in the API indexing: " \
              "(symmetric diff is %s)" % diff
        # 抛出值错误异常，指示存在API索引中的“空洞”
        raise ValueError(msg)
# 解析源文件以获取由给定标记标记的函数列表
def get_api_functions(tagname, api_dict):
    """Parse source files to get functions tagged by the given tag."""
    functions = []
    # 遍历 API_FILES 中的每个文件，并找到标记为 tagname 的函数列表
    for f in API_FILES:
        functions.extend(find_functions(f, tagname))
    # 根据函数名排序函数列表，构建 (api_dict[函数名][0], 函数对象) 的元组列表
    dfunctions = [(api_dict[func.name][0], func) for func in functions]
    # 按照 api_dict 中定义的顺序排序函数列表
    dfunctions.sort()
    # 返回排序后的函数对象列表
    return [a[1] for a in dfunctions]

# 给定定义 numpy C API 的 api dicts 列表，计算 API 项目列表的校验和（作为字符串）
def fullapi_hash(api_dicts):
    """Given a list of api dicts defining the numpy C API, compute a checksum
    of the list of items in the API (as a string)."""
    a = []
    # 遍历每个 api_dict
    for d in api_dicts:
        d = d.copy()
        # 移除键为 "__unused_indices__" 的条目
        d.pop("__unused_indices__", None)
        # 按顺序处理字典中的每个条目，将条目的名称和数据转换成字符串并加入到列表 a 中
        for name, data in order_dict(d):
            a.extend(name)
            a.extend(','.join(map(str, data)))
    # 计算列表 a 的 MD5 校验和并返回其十六进制表示
    return hashlib.md5(''.join(a).encode('ascii')).hexdigest()

# 用于解析类似于 'hex = checksum' 的字符串，其中 hex 是类似 0x1234567F 的十六进制数，
# checksum 是一个 128 位的 MD5 校验和（同样是十六进制格式）
VERRE = re.compile(r'(^0x[\da-f]{8})\s*=\s*([\da-f]{32})')

# 从文件 'cversions.txt' 中获取版本号和对应的 MD5 校验和，返回以版本号为键，校验和为值的字典
def get_versions_hash():
    d = []
    # 拼接文件路径
    file = os.path.join(os.path.dirname(__file__), 'cversions.txt')
    # 打开文件并逐行读取
    with open(file) as fid:
        for line in fid:
            # 使用正则表达式 VERRE 匹配每一行
            m = VERRE.match(line)
            if m:
                # 如果匹配成功，将 (版本号, 校验和) 组成的元组加入列表 d
                d.append((int(m.group(1), 16), m.group(2)))
    # 返回版本号和校验和组成的字典
    return dict(d)

# 主函数，程序的入口点
def main():
    # 从命令行参数获取标记名和 order 文件名
    tagname = sys.argv[1]
    order_file = sys.argv[2]
    # 调用 get_api_functions 函数获取标记为 tagname 的函数列表
    functions = get_api_functions(tagname, order_file)
    # 创建一个 hashlib.md5 对象 m，用于计算 MD5 校验和
    m = hashlib.md5(tagname)
    # 遍历函数列表中的每个函数对象
    for func in functions:
        # 打印函数对象的信息
        print(func)
        # 调用函数对象的 api_hash 方法获取其校验和
        ah = func.api_hash()
        # 更新 m 的状态以包含函数的校验和信息
        m.update(ah)
        # 打印校验和的十六进制表示
        print(hex(int(ah, 16)))
    # 打印 MD5 校验和的前 8 个字符的十六进制表示
    print(hex(int(m.hexdigest()[:8], 16)))

if __name__ == '__main__':
    main()
```