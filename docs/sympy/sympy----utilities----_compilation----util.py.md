# `D:\src\scipysrc\sympy\sympy\utilities\_compilation\util.py`

```
python
# 导入命名元组类
from collections import namedtuple
# 导入 SHA256 哈希算法
from hashlib import sha256
# 导入操作系统相关功能
import os
# 导入文件和目录操作函数
import shutil
# 导入系统相关功能
import sys
# 导入文件名匹配函数
import fnmatch

# 导入 XFAIL 装饰器
from sympy.testing.pytest import XFAIL


def may_xfail(func):
    # 如果运行在 macOS 或 Windows 平台上，使用 XFAIL 装饰器标记测试函数
    if sys.platform.lower() == 'darwin' or os.name == 'nt':
        # 对 sympy.utilities._compilation 在 Windows 和 macOS 上的支持需要更多测试
        # 一旦这两个平台可靠支持，可以移除这个 xfail 装饰器
        return XFAIL(func)
    else:
        # 否则直接返回原始函数
        return func


# 自定义文件未找到错误异常类，继承自 FileNotFoundError
class CompilerNotFoundError(FileNotFoundError):
    pass


# 自定义编译错误异常类，继承自 Exception
class CompileError(Exception):
    """Failure to compile one or more C/C++ source files."""


def get_abspath(path, cwd='.'):
    """ 返回给定相对路径的绝对路径。

    Parameters
    ==========

    path : str
        (相对) 路径。
    cwd : str
        相对路径的根目录。
    """
    if os.path.isabs(path):
        return path
    else:
        if not os.path.isabs(cwd):
            cwd = os.path.abspath(cwd)
        return os.path.abspath(
            os.path.join(cwd, path)
        )


def make_dirs(path):
    """ 创建目录，相当于命令 ``mkdir -p``。 """
    if path[-1] == '/':
        parent = os.path.dirname(path[:-1])
    else:
        parent = os.path.dirname(path)

    if len(parent) > 0:
        if not os.path.exists(parent):
            make_dirs(parent)

    if not os.path.exists(path):
        os.mkdir(path, 0o777)
    else:
        assert os.path.isdir(path)


def missing_or_other_newer(path, other_path, cwd=None):
    """
    检查路径是否不存在或者比提供的参考路径更新。

    Parameters
    ==========

    path: string
        要检查的路径，可能不存在或者比参考路径旧。
    other_path: string
        参考路径。
    cwd: string
        工作目录（相对路径的根目录）。

    Returns
    =======

    如果路径不存在或者比参考路径旧，则返回 True。
    """
    cwd = cwd or '.'
    path = get_abspath(path, cwd=cwd)
    other_path = get_abspath(other_path, cwd=cwd)
    if not os.path.exists(path):
        return True
    if os.path.getmtime(other_path) - 1e-6 >= os.path.getmtime(path):
        # 1e-6 是因为需要微小的差异以防止 http://stackoverflow.com/questions/17086426/
        return True
    return False


def copy(src, dst, only_update=False, copystat=True, cwd=None,
         dest_is_dir=False, create_dest_dirs=False):
    """ 带有额外选项的 ``shutil.copy`` 变体。

    Parameters
    ==========

    src : str
        源文件路径。
    dst : str
        目标路径。
    only_update : bool
        仅在源文件更新时复制（如果源文件更新，则返回 None），默认：``False``。
    copystat : bool
        是否复制状态信息。默认：``True``。
    cwd : str
        工作目录（相对路径的根目录）。
    dest_is_dir : bool
        确保目标路径被视为目录。默认：``False``。
    create_dest_dirs : bool
        如果需要，创建目标路径的目录。

    Returns
    =======

    """
    Path to the copied file.

    """
    # 如果提供了工作目录，则处理源文件和目标文件的相对路径问题
    if cwd:  # Handle working directory
        # 如果源文件路径不是绝对路径，则拼接工作目录和源文件路径
        if not os.path.isabs(src):
            src = os.path.join(cwd, src)
        # 如果目标文件路径不是绝对路径，则拼接工作目录和目标文件路径
        if not os.path.isabs(dst):
            dst = os.path.join(cwd, dst)

    # 确保源文件存在
    if not os.path.exists(src):  # Make sure source file exists
        raise FileNotFoundError("Source: `{}` does not exist".format(src))

    # 根据目标是否为目录来进行不同的处理
    # 如果目标是目录，确保目录路径以斜杠结尾
    # 如果目标不是目录但是存在且为目录，则将目标标记为目录
    if dest_is_dir:
        if not dst[-1] == '/':
            dst = dst+'/'
    else:
        if os.path.exists(dst) and os.path.isdir(dst):
            dest_is_dir = True

    # 如果目标是目录，则设置目标目录和目标文件名
    # 否则，获取目标文件所在的目录
    if dest_is_dir:
        dest_dir = dst
        dest_fname = os.path.basename(src)
        dst = os.path.join(dest_dir, dest_fname)
    else:
        dest_dir = os.path.dirname(dst)

    # 确保目标目录存在，如果不存在则根据 create_dest_dirs 参数决定是否创建
    if not os.path.exists(dest_dir):
        if create_dest_dirs:
            make_dirs(dest_dir)
        else:
            raise FileNotFoundError("You must create directory first.")

    # 如果 only_update 为 True，则检查目标文件是否需要更新，如果不需要则直接返回
    if only_update:
        if not missing_or_other_newer(dst, src):
            return

    # 如果目标文件是一个符号链接，则获取其真实路径
    if os.path.islink(dst):
        dst = os.path.abspath(os.path.realpath(dst), cwd=cwd)

    # 使用 shutil 模块复制源文件到目标文件
    shutil.copy(src, dst)
    # 如果需要复制文件的元数据，则使用 shutil.copystat 复制
    if copystat:
        shutil.copystat(src, dst)

    # 返回复制后的目标文件路径
    return dst
# 创建命名元组 `Glob`，其中包含一个字段 `pathname`
# 创建命名元组 `ArbitraryDepthGlob`，其中包含一个字段 `filename`
Glob = namedtuple('Glob', 'pathname')
ArbitraryDepthGlob = namedtuple('ArbitraryDepthGlob', 'filename')

def glob_at_depth(filename_glob, cwd=None):
    # 如果 cwd 不为 None，则将其设为当前目录 '.'
    if cwd is not None:
        cwd = '.'
    # 初始化一个空列表用于存储匹配到的文件路径
    globbed = []
    # 遍历当前目录及其子目录下的所有文件和文件夹
    for root, dirs, filenames in os.walk(cwd):
        # 遍历当前目录下的文件名
        for fn in filenames:
            # 检查文件名是否符合指定的 glob 模式 filename_glob
            if fnmatch.fnmatch(fn, filename_glob):
                # 将匹配到的文件路径添加到 globbed 列表中
                globbed.append(os.path.join(root, fn))
    return globbed

def sha256_of_file(path, nblocks=128):
    """ Computes the SHA256 hash of a file.

    Parameters
    ==========

    path : string
        Path to file to compute hash of.
    nblocks : int
        Number of blocks to read per iteration.

    Returns
    =======

    hashlib sha256 hash object. Use ``.digest()`` or ``.hexdigest()``
    on returned object to get binary or hex encoded string.
    """
    # 创建一个 SHA256 哈希对象
    sh = sha256()
    # 打开文件 path 并按照 nblocks*sh.block_size 大小的块读取文件内容并更新哈希对象
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(nblocks*sh.block_size), b''):
            sh.update(chunk)
    return sh

def sha256_of_string(string):
    """ Computes the SHA256 hash of a string. """
    # 创建一个 SHA256 哈希对象并更新其内容为输入字符串 string 的字节表示
    sh = sha256()
    sh.update(string)
    return sh

def pyx_is_cplus(path):
    """
    Inspect a Cython source file (.pyx) and look for comment line like:

    # distutils: language = c++

    Returns True if such a file is present in the file, else False.
    """
    # 打开指定路径的文件进行逐行检查
    with open(path) as fh:
        for line in fh:
            # 如果当前行以 '#' 开头且包含 '=' 符号
            if line.startswith('#') and '=' in line:
                # 分割行内容为左右两部分，并检查是否有且仅有两部分
                splitted = line.split('=')
                if len(splitted) != 2:
                    continue
                lhs, rhs = splitted
                # 如果左侧部分最后一个单词是 'language' 且右侧第一个单词是 'c++'，则返回 True
                if lhs.strip().split()[-1].lower() == 'language' and \
                       rhs.strip().split()[0].lower() == 'c++':
                            return True
    # 如果未找到符合条件的行，则返回 False
    return False

def import_module_from_file(filename, only_if_newer_than=None):
    """ Imports Python extension (from shared object file)

    Provide a list of paths in `only_if_newer_than` to check
    timestamps of dependencies. import_ raises an ImportError
    if any is newer.

    Word of warning: The OS may cache shared objects which makes
    reimporting same path of an shared object file very problematic.

    It will not detect the new time stamp, nor new checksum, but will
    instead silently use old module. Use unique names for this reason.

    Parameters
    ==========

    filename : str
        Path to shared object.
    only_if_newer_than : iterable of strings
        Paths to dependencies of the shared object.

    Raises
    ======

    ``ImportError`` if any of the files specified in ``only_if_newer_than`` are newer
    than the file given by filename.
    """
    # 将文件名分解为路径和文件名及扩展名
    path, name = os.path.split(filename)
    name, ext = os.path.splitext(name)
    name = name.split('.')[0]
    # 检查 Python 版本，如果是 Python 2.x 执行以下代码块，否则执行另一段代码
    if sys.version_info[0] == 2:
        # 从 imp 模块导入 find_module 和 load_module 函数
        from imp import find_module, load_module
        # 使用 find_module 函数查找指定模块的文件
        fobj, filename, data = find_module(name, [path])
        # 如果指定了依赖文件列表 only_if_newer_than，检查模块文件是否比依赖文件更新
        if only_if_newer_than:
            for dep in only_if_newer_than:
                # 检查模块文件的修改时间是否比依赖文件新
                if os.path.getmtime(filename) < os.path.getmtime(dep):
                    # 抛出 ImportError 异常，提示依赖文件比模块文件新
                    raise ImportError("{} is newer than {}".format(dep, filename))
        # 使用 load_module 函数加载模块并返回模块对象
        mod = load_module(name, fobj, filename, data)
    else:
        # 导入 importlib.util 模块
        import importlib.util
        # 使用 spec_from_file_location 函数创建一个模块规范对象
        spec = importlib.util.spec_from_file_location(name, filename)
        # 如果创建的规范对象为空，抛出 ImportError 异常，提示导入失败
        if spec is None:
            raise ImportError("Failed to import: '%s'" % filename)
        # 使用 module_from_spec 函数根据规范对象创建模块对象
        mod = importlib.util.module_from_spec(spec)
        # 使用 exec_module 方法执行模块对象中的代码
        spec.loader.exec_module(mod)
    # 返回加载或执行后的模块对象
    return mod
# 在给定的候选命令列表中查找第一个匹配的可执行文件路径。

# 调用shutils中的`which`函数，用于查找给定候选命令的可执行文件路径，并返回第一个找到的路径。

# Parameters参数：
# candidates : 可迭代的字符串列表
#     候选命令的名称列表

# Raises抛出：
# CompilerNotFoundError 如果没有找到任何候选命令的可执行文件路径。

def find_binary_of_command(candidates):
    """ Finds binary first matching name among candidates.

    Calls ``which`` from shutils for provided candidates and returns
    first hit.

    Parameters
    ==========

    candidates : iterable of str
        Names of candidate commands

    Raises
    ======

    CompilerNotFoundError if no candidates match.
    """
    from shutil import which
    # 遍历候选命令列表
    for c in candidates:
        # 使用`which`函数查找命令的可执行文件路径
        binary_path = which(c)
        # 如果找到了可执行文件路径，则返回命令名称和路径
        if c and binary_path:
            return c, binary_path

    # 如果没有找到任何候选命令的可执行文件路径，则抛出异常
    raise CompilerNotFoundError('No binary located for candidates: {}'.format(candidates))


# 返回一个去重后的列表（跳过重复项）。

# Parameters参数：
# l : list
#     输入的列表

def unique_list(l):
    """ Uniquify a list (skip duplicate items). """
    result = []
    # 遍历输入的列表
    for x in l:
        # 如果当前元素不在结果列表中，则将其添加到结果列表中
        if x not in result:
            result.append(x)
    # 返回去重后的结果列表
    return result
```