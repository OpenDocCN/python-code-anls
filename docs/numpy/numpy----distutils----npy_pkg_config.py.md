# `.\numpy\numpy\distutils\npy_pkg_config.py`

```
import sys
import re
import os

from configparser import RawConfigParser

__all__ = ['FormatError', 'PkgNotFound', 'LibraryInfo', 'VariableSet',
        'read_config', 'parse_flags']

# 正则表达式模式，用于匹配形如 "${...}" 的变量格式
_VAR = re.compile(r'\$\{([a-zA-Z0-9_-]+)\}')

class FormatError(OSError):
    """
    Exception thrown when there is a problem parsing a configuration file.
    
    Attributes:
        msg (str): Error message describing the problem.
    """
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class PkgNotFound(OSError):
    """
    Exception raised when a package cannot be located.
    
    Attributes:
        msg (str): Error message indicating which package could not be found.
    """
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

def parse_flags(line):
    """
    Parse a line from a config file containing compile flags.

    Parameters
    ----------
    line : str
        A single line containing one or more compile flags.

    Returns
    -------
    d : dict
        Dictionary of parsed flags, split into relevant categories.
        These categories are the keys of `d`:

        * 'include_dirs'
        * 'library_dirs'
        * 'libraries'
        * 'macros'
        * 'ignored'

    """
    d = {'include_dirs': [], 'library_dirs': [], 'libraries': [],
         'macros': [], 'ignored': []}

    # Split the line by ' -' to isolate individual flags
    flags = (' ' + line).split(' -')
    for flag in flags:
        flag = '-' + flag
        if len(flag) > 0:
            # Categorize flags based on their type
            if flag.startswith('-I'):
                d['include_dirs'].append(flag[2:].strip())
            elif flag.startswith('-L'):
                d['library_dirs'].append(flag[2:].strip())
            elif flag.startswith('-l'):
                d['libraries'].append(flag[2:].strip())
            elif flag.startswith('-D'):
                d['macros'].append(flag[2:].strip())
            else:
                d['ignored'].append(flag)

    return d

def _escape_backslash(val):
    """
    Escape backslashes in the given string.

    Parameters
    ----------
    val : str
        Input string where backslashes need to be escaped.

    Returns
    -------
    str
        String with backslashes escaped as '\\\\'.
    """
    return val.replace('\\', '\\\\')

class LibraryInfo:
    """
    Object containing build information about a library.

    Parameters
    ----------
    name : str
        The library name.
    description : str
        Description of the library.
    version : str
        Version string.
    sections : dict
        The sections of the configuration file for the library. The keys are
        the section headers, the values the text under each header.
    vars : class instance
        A `VariableSet` instance, which contains ``(name, value)`` pairs for
        variables defined in the configuration file for the library.
    requires : sequence, optional
        The required libraries for the library to be installed.

    Notes
    -----
    All input parameters (except "sections" which is a method) are available as
    attributes of the same name.

    """
    # 初始化函数，用于创建一个配置对象
    def __init__(self, name, description, version, sections, vars, requires=None):
        # 设置配置对象的名称
        self.name = name
        # 设置配置对象的描述
        self.description = description
        # 如果指定了requires，则使用指定的requires，否则使用空列表
        if requires:
            self.requires = requires
        else:
            self.requires = []
        # 设置配置对象的版本号
        self.version = version
        # 设置配置对象的私有属性_sections，用于存储配置文件的各个部分
        self._sections = sections
        # 设置配置对象的vars，用于变量插值
        self.vars = vars

    # 返回配置文件中所有部分的头信息（section headers）
    def sections(self):
        """
        Return the section headers of the config file.

        Parameters
        ----------
        None

        Returns
        -------
        keys : list of str
            The list of section headers.

        """
        # 返回所有部分的键（即部分头信息）
        return list(self._sections.keys())

    # 获取指定部分（section）的cflags配置，进行变量插值后返回
    def cflags(self, section="default"):
        # 获取指定部分的cflags配置值，并进行变量插值
        val = self.vars.interpolate(self._sections[section]['cflags'])
        # 对返回的值进行反斜杠转义处理
        return _escape_backslash(val)

    # 获取指定部分（section）的libs配置，进行变量插值后返回
    def libs(self, section="default"):
        # 获取指定部分的libs配置值，并进行变量插值
        val = self.vars.interpolate(self._sections[section]['libs'])
        # 对返回的值进行反斜杠转义处理
        return _escape_backslash(val)

    # 返回配置对象的字符串表示形式，包括名称、描述、依赖和版本信息
    def __str__(self):
        # 初始化信息列表
        m = ['Name: %s' % self.name, 'Description: %s' % self.description]
        # 如果有依赖信息，则加入依赖信息到列表
        if self.requires:
            m.append('Requires:')
        else:
            # 如果没有依赖信息，则加入空的Requires信息到列表
            m.append('Requires: %s' % ",".join(self.requires))
        # 加入版本信息到列表
        m.append('Version: %s' % self.version)

        # 返回以换行符连接的信息列表作为对象的字符串表示形式
        return "\n".join(m)
class VariableSet:
    """
    Container object for the variables defined in a config file.

    `VariableSet` can be used as a plain dictionary, with the variable names
    as keys.

    Parameters
    ----------
    d : dict
        Dict of items in the "variables" section of the configuration file.

    """
    def __init__(self, d):
        # Initialize with a copy of the provided dictionary
        self._raw_data = dict([(k, v) for k, v in d.items()])

        # Initialize dictionaries for regular expressions and substitution values
        self._re = {}
        self._re_sub = {}

        # Call the initialization method to parse variables
        self._init_parse()

    def _init_parse(self):
        # Iterate over each key-value pair in the raw data dictionary
        for k, v in self._raw_data.items():
            # Initialize a regular expression for variable interpolation
            self._init_parse_var(k, v)

    def _init_parse_var(self, name, value):
        # Compile a regular expression for ${name} and store the substitution value
        self._re[name] = re.compile(r'\$\{%s\}' % name)
        self._re_sub[name] = value

    def interpolate(self, value):
        # Brute force interpolation method to substitute variables in 'value'
        def _interpolate(value):
            for k in self._re.keys():
                value = self._re[k].sub(self._re_sub[k], value)
            return value
        
        # Continue interpolating until no more ${var} patterns are found or until stable
        while _VAR.search(value):
            nvalue = _interpolate(value)
            if nvalue == value:
                break
            value = nvalue

        return value

    def variables(self):
        """
        Return the list of variable names.

        Parameters
        ----------
        None

        Returns
        -------
        names : list of str
            The names of all variables in the `VariableSet` instance.

        """
        return list(self._raw_data.keys())

    # Emulate a dict to set/get variables values
    def __getitem__(self, name):
        return self._raw_data[name]

    def __setitem__(self, name, value):
        # Set a variable value and reinitialize its regular expression and substitution
        self._raw_data[name] = value
        self._init_parse_var(name, value)

def parse_meta(config):
    # Check if 'meta' section exists in the configuration
    if not config.has_section('meta'):
        raise FormatError("No meta section found !")

    # Create a dictionary from items in the 'meta' section
    d = dict(config.items('meta'))

    # Ensure mandatory keys ('name', 'description', 'version') exist in 'meta'
    for k in ['name', 'description', 'version']:
        if not k in d:
            raise FormatError("Option %s (section [meta]) is mandatory, "
                "but not found" % k)

    # If 'requires' key is missing, initialize it as an empty list
    if not 'requires' in d:
        d['requires'] = []

    return d

def parse_variables(config):
    # Check if 'variables' section exists in the configuration
    if not config.has_section('variables'):
        raise FormatError("No variables section found !")

    # Initialize an empty dictionary for variables
    d = {}

    # Populate the dictionary with items from the 'variables' section
    for name, value in config.items("variables"):
        d[name] = value

    # Return a VariableSet object initialized with the variables dictionary
    return VariableSet(d)

def parse_sections(config):
    return meta_d, r

def pkg_to_filename(pkg_name):
    # Convert package name to a filename by appending '.ini'
    return "%s.ini" % pkg_name

def parse_config(filename, dirs=None):
    # If directories are provided, create a list of filenames with path
    if dirs:
        filenames = [os.path.join(d, filename) for d in dirs]
    else:
        filenames = [filename]

    # Initialize a RawConfigParser object
    config = RawConfigParser()

    # Read configuration files specified in 'filenames'
    n = config.read(filenames)

    # If no files were successfully read, raise PkgNotFound exception
    if not len(n) >= 1:
        raise PkgNotFound("Could not find file(s) %s" % str(filenames))

    # Parse 'meta' section of the configuration
    meta = parse_meta(config)

    # Initialize an empty dictionary for variables
    vars = {}
    # 检查配置文件中是否存在 'variables' 这个部分
    if config.has_section('variables'):
        # 遍历 'variables' 部分的每个键值对，将值经过 _escape_backslash 函数处理后存入 vars 字典中
        for name, value in config.items("variables"):
            vars[name] = _escape_backslash(value)

    # 解析除了 'meta' 和 'variables' 以外的普通部分
    secs = [s for s in config.sections() if not s in ['meta', 'variables']]
    sections = {}

    # 创建一个空的 requires 字典，用于存储每个部分可能的 'requires' 值
    requires = {}

    # 遍历每个普通部分（不包括 'meta' 和 'variables'）
    for s in secs:
        d = {}

        # 如果当前部分有 'requires' 选项，则将其存入 requires 字典中
        if config.has_option(s, "requires"):
            requires[s] = config.get(s, 'requires')

        # 遍历当前部分的所有键值对，存入字典 d 中
        for name, value in config.items(s):
            d[name] = value

        # 将当前部分的字典 d 存入 sections 字典中
        sections[s] = d

    # 返回解析得到的 meta 信息、vars 变量字典、sections 部分字典和 requires 要求字典
    return meta, vars, sections, requires
def _read_config_imp(filenames, dirs=None):
    # 定义内部函数 _read_config，用于读取配置文件并返回元数据、变量、部分和要求的字典
    def _read_config(f):
        # 解析配置文件 f，获取元数据、变量、部分和要求的字典
        meta, vars, sections, reqs = parse_config(f, dirs)
        
        # 递归添加所需库的部分和变量
        for rname, rvalue in reqs.items():
            # 递归调用 _read_config 获取所需库的元数据、变量、部分和要求的字典
            nmeta, nvars, nsections, nreqs = _read_config(pkg_to_filename(rvalue))

            # 更新变量字典，添加不在 'top' 配置文件中的变量
            for k, v in nvars.items():
                if k not in vars:
                    vars[k] = v

            # 更新部分字典
            for oname, ovalue in nsections[rname].items():
                if ovalue:
                    sections[rname][oname] += ' %s' % ovalue

        # 返回解析得到的元数据、变量、部分和要求的字典
        return meta, vars, sections, reqs

    # 调用内部函数 _read_config 解析配置文件列表 filenames，并获取元数据、变量、部分和要求的字典
    meta, vars, sections, reqs = _read_config(filenames)

    # FIXME: document this. If pkgname is defined in the variables section, and
    # there is no pkgdir variable defined, pkgdir is automatically defined to
    # the path of pkgname. This requires the package to be imported to work
    # 如果在变量部分定义了 pkgname，并且没有定义 pkgdir 变量，则自动将 pkgdir 定义为 pkgname 的路径。
    # 这要求导入该包才能正常工作。
    if 'pkgdir' not in vars and "pkgname" in vars:
        pkgname = vars["pkgname"]
        # 如果 pkgname 没有在 sys.modules 中注册（未被导入）
        if pkgname not in sys.modules:
            # 抛出值错误异常，要求导入该包以获取信息
            raise ValueError("You should import %s to get information on %s" %
                             (pkgname, meta["name"]))

        # 获取 pkgname 对应的模块对象
        mod = sys.modules[pkgname]
        # 设置 vars 字典中的 pkgdir 变量为模块文件所在目录的转义反斜杠路径
        vars["pkgdir"] = _escape_backslash(os.path.dirname(mod.__file__))

    # 返回 LibraryInfo 类的实例，包含元数据的名称、描述、版本、部分和变量集
    return LibraryInfo(name=meta["name"], description=meta["description"],
            version=meta["version"], sections=sections, vars=VariableSet(vars))

# 简单缓存，用于缓存 LibraryInfo 实例的创建。为了真正高效，缓存应该在 read_config 中处理，
# 因为同一个文件可以在 LibraryInfo 创建之外多次解析，但我怀疑在实践中这不会是问题
_CACHE = {}
def read_config(pkgname, dirs=None):
    """
    从配置文件中返回包的库信息。

    Parameters
    ----------
    pkgname : str
        包的名称（应该与 .ini 文件的名称匹配，不包括扩展名，例如 foo 对应 foo.ini）。
    dirs : sequence, optional
        如果给定，应该是一个目录序列 - 通常包括 NumPy 基础目录 - 用于查找 npy-pkg-config 文件。

    Returns
    -------
    pkginfo : class instance
        包含构建信息的 `LibraryInfo` 实例。

    Raises
    ------
    PkgNotFound
        如果找不到包。

    See Also
    --------
    misc_util.get_info, misc_util.get_pkg_info

    Examples
    --------
    >>> npymath_info = np.distutils.npy_pkg_config.read_config('npymath')
    >>> type(npymath_info)
    <class 'numpy.distutils.npy_pkg_config.LibraryInfo'>
    >>> print(npymath_info)
    Name: npymath
    Description: Portable, core math library implementing C99 standard
    Requires:
    Version: 0.1  #random

    """
    try:
        return _CACHE[pkgname]
    # 如果发生 KeyError 异常，则执行以下操作
    v = _read_config_imp(pkg_to_filename(pkgname), dirs)
    # 将返回的配置信息存入缓存字典中，键为 pkgname
    _CACHE[pkgname] = v
    # 返回获取的配置信息
    return v
# TODO:
#   - implements version comparison (modversion + atleast)

# pkg-config simple emulator - useful for debugging, and maybe later to query
# the system
if __name__ == '__main__':
    from optparse import OptionParser  # 导入选项解析器类
    import glob  # 导入用于文件路径模式匹配的 glob 模块

    parser = OptionParser()  # 创建选项解析器对象
    parser.add_option("--cflags", dest="cflags", action="store_true",
                      help="output all preprocessor and compiler flags")  # 添加选项：输出所有预处理器和编译器标志
    parser.add_option("--libs", dest="libs", action="store_true",
                      help="output all linker flags")  # 添加选项：输出所有链接器标志
    parser.add_option("--use-section", dest="section",
                      help="use this section instead of default for options")  # 添加选项：使用指定的部分而不是默认部分
    parser.add_option("--version", dest="version", action="store_true",
                      help="output version")  # 添加选项：输出版本号
    parser.add_option("--atleast-version", dest="min_version",
                      help="Minimal version")  # 添加选项：至少指定的版本号
    parser.add_option("--list-all", dest="list_all", action="store_true",
                      help="Minimal version")  # 添加选项：列出所有信息
    parser.add_option("--define-variable", dest="define_variable",
                      help="Replace variable with the given value")  # 添加选项：用给定值替换变量

    (options, args) = parser.parse_args(sys.argv)  # 解析命令行参数

    if len(args) < 2:
        raise ValueError("Expect package name on the command line:")  # 如果命令行参数少于两个，抛出值错误异常

    if options.list_all:
        files = glob.glob("*.ini")  # 获取当前目录下所有以 .ini 结尾的文件列表
        for f in files:
            info = read_config(f)  # 读取配置文件信息
            print("%s\t%s - %s" % (info.name, info.name, info.description))  # 打印格式化输出：名称、名称和描述

    pkg_name = args[1]  # 获取命令行中的第二个参数作为包名
    d = os.environ.get('NPY_PKG_CONFIG_PATH')  # 获取环境变量 NPY_PKG_CONFIG_PATH 的值
    if d:
        info = read_config(
            pkg_name, ['numpy/_core/lib/npy-pkg-config', '.', d]
        )  # 读取配置信息，优先使用指定路径
    else:
        info = read_config(
            pkg_name, ['numpy/_core/lib/npy-pkg-config', '.']
        )  # 读取配置信息，默认搜索当前目录

    if options.section:
        section = options.section  # 如果指定了 --use-section 选项，则使用指定的部分
    else:
        section = "default"  # 否则使用默认部分

    if options.define_variable:
        m = re.search(r'([\S]+)=([\S]+)', options.define_variable)  # 匹配 --define-variable 选项的值格式
        if not m:
            raise ValueError("--define-variable option should be of "
                             "the form --define-variable=foo=bar")  # 如果格式不正确，抛出值错误异常
        else:
            name = m.group(1)  # 获取变量名
            value = m.group(2)  # 获取变量值
        info.vars[name] = value  # 将变量名和值存入配置信息的变量字典中

    if options.cflags:
        print(info.cflags(section))  # 如果指定了 --cflags 选项，则打印预处理器和编译器标志
    if options.libs:
        print(info.libs(section))  # 如果指定了 --libs 选项，则打印链接器标志
    if options.version:
        print(info.version)  # 如果指定了 --version 选项，则打印版本号
    if options.min_version:
        print(info.version >= options.min_version)  # 如果指定了 --atleast-version 选项，则打印是否当前版本号不小于指定版本号
```