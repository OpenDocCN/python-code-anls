# `.\numpy\numpy\distutils\exec_command.py`

```
"""
exec_command

Implements exec_command function that is (almost) equivalent to
commands.getstatusoutput function but on NT, DOS systems the
returned status is actually correct (though, the returned status
values may be different by a factor). In addition, exec_command
takes keyword arguments for (re-)defining environment variables.

Provides functions:

  exec_command  --- execute command in a specified directory and
                    in the modified environment.
  find_executable --- locate a command using info from environment
                    variable PATH. Equivalent to posix `which`
                    command.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: 11 January 2003

Requires: Python 2.x

Successfully tested on:

========  ============  =================================================
os.name   sys.platform  comments
========  ============  =================================================
posix     linux2        Debian (sid) Linux, Python 2.1.3+, 2.2.3+, 2.3.3
                        PyCrust 0.9.3, Idle 1.0.2
posix     linux2        Red Hat 9 Linux, Python 2.1.3, 2.2.2, 2.3.2
posix     sunos5        SunOS 5.9, Python 2.2, 2.3.2
posix     darwin        Darwin 7.2.0, Python 2.3
nt        win32         Windows Me
                        Python 2.3(EE), Idle 1.0, PyCrust 0.7.2
                        Python 2.1.1 Idle 0.8
nt        win32         Windows 98, Python 2.1.1. Idle 0.8
nt        win32         Cygwin 98-4.10, Python 2.1.1(MSC) - echo tests
                        fail i.e. redefining environment variables may
                        not work. FIXED: don't use cygwin echo!
                        Comment: also `cmd /c echo` will not work
                        but redefining environment variables do work.
posix     cygwin        Cygwin 98-4.10, Python 2.3.3(cygming special)
nt        win32         Windows XP, Python 2.3.3
========  ============  =================================================

Known bugs:

* Tests, that send messages to stderr, fail when executed from MSYS prompt
  because the messages are lost at some point.

"""
__all__ = ['exec_command', 'find_executable']

import os  # 导入操作系统相关的功能
import sys  # 导入系统相关的功能
import subprocess  # 导入子进程管理的功能
import locale  # 导入处理本地化的功能
import warnings  # 导入警告处理的功能

from numpy.distutils.misc_util import is_sequence, make_temp_file  # 导入 numpy 相关的功能
from numpy.distutils import log  # 导入 numpy 相关的日志记录功能

def filepath_from_subprocess_output(output):
    """
    Convert `bytes` in the encoding used by a subprocess into a filesystem-appropriate `str`.

    Inherited from `exec_command`, and possibly incorrect.
    """
    mylocale = locale.getpreferredencoding(False)  # 获取首选编码
    if mylocale is None:
        mylocale = 'ascii'
    output = output.decode(mylocale, errors='replace')  # 解码 subprocess 输出为字符串
    output = output.replace('\r\n', '\n')  # 替换换行符格式
    # Another historical oddity
    if output[-1:] == '\n':  # 移除末尾可能存在的多余换行符
        output = output[:-1]
    return output

def forward_bytes_to_stdout(val):
    """
    Forward bytes from a subprocess call to the console, without attempting to
    decode them.
    """
    # 此函数用于将子进程调用的字节流直接输出到控制台，不进行解码操作
    # 假设 subprocess 调用已经返回了适当编码的字节流数据。
    if hasattr(sys.stdout, 'buffer'):
        # 如果 sys.stdout 有 'buffer' 属性，直接使用底层的二进制输出
        sys.stdout.buffer.write(val)
    elif hasattr(sys.stdout, 'encoding'):
        # 如果 sys.stdout 有 'encoding' 属性，尝试使用相同的编码进行解码和再编码
        sys.stdout.write(val.decode(sys.stdout.encoding))
    else:
        # 如果无法确定编码，使用 UTF-8 编码进行解码，替换错误的字符
        sys.stdout.write(val.decode('utf8', errors='replace'))
def temp_file_name():
    # 发出警告，提示 temp_file_name 函数在 NumPy v1.17 版本后已弃用，建议使用 tempfile.mkstemp 替代
    warnings.warn('temp_file_name is deprecated since NumPy v1.17, use '
                  'tempfile.mkstemp instead', DeprecationWarning, stacklevel=1)
    # 调用 make_temp_file 函数创建临时文件，返回文件对象和文件名
    fo, name = make_temp_file()
    # 关闭临时文件对象
    fo.close()
    # 返回临时文件名
    return name

def get_pythonexe():
    # 获取当前 Python 解释器的可执行文件路径
    pythonexe = sys.executable
    # 如果操作系统为 Windows 平台
    if os.name in ['nt', 'dos']:
        # 分离文件路径和文件名
        fdir, fn = os.path.split(pythonexe)
        # 将文件名转换为大写并替换 'PYTHONW' 为 'PYTHON'
        fn = fn.upper().replace('PYTHONW', 'PYTHON')
        # 拼接路径和修改后的文件名
        pythonexe = os.path.join(fdir, fn)
        # 断言文件路径存在且是一个文件
        assert os.path.isfile(pythonexe), '%r is not a file' % (pythonexe,)
    # 返回 Python 解释器的可执行文件路径
    return pythonexe

def find_executable(exe, path=None, _cache={}):
    """Return full path of a executable or None.

    Symbolic links are not followed.
    """
    # 以 (exe, path) 为键从缓存中查找路径
    key = exe, path
    try:
        return _cache[key]
    except KeyError:
        pass
    # 记录调试信息，查找可执行文件
    log.debug('find_executable(%r)' % exe)
    # 保存原始的可执行文件名
    orig_exe = exe

    # 如果路径为空，则使用环境变量 PATH 或者默认路径
    if path is None:
        path = os.environ.get('PATH', os.defpath)
    # 根据操作系统类型选择 realpath 函数
    if os.name=='posix':
        realpath = os.path.realpath
    else:
        realpath = lambda a:a

    # 如果可执行文件名以双引号开头，则去除双引号
    if exe.startswith('"'):
        exe = exe[1:-1]

    # Windows 平台的可执行文件后缀列表
    suffixes = ['']
    if os.name in ['nt', 'dos', 'os2']:
        fn, ext = os.path.splitext(exe)
        extra_suffixes = ['.exe', '.com', '.bat']
        # 如果文件名后缀不在附加后缀列表中，则使用附加后缀
        if ext.lower() not in extra_suffixes:
            suffixes = extra_suffixes

    # 如果可执行文件名是绝对路径
    if os.path.isabs(exe):
        paths = ['']
    else:
        # 将 PATH 分割为绝对路径列表
        paths = [ os.path.abspath(p) for p in path.split(os.pathsep) ]

    # 遍历所有可能的路径和后缀
    for path in paths:
        fn = os.path.join(path, exe)
        for s in suffixes:
            f_ext = fn+s
            # 如果文件不是符号链接，则获取其真实路径
            if not os.path.islink(f_ext):
                f_ext = realpath(f_ext)
            # 如果文件存在且可执行，则返回其路径
            if os.path.isfile(f_ext) and os.access(f_ext, os.X_OK):
                log.info('Found executable %s' % f_ext)
                _cache[key] = f_ext
                return f_ext

    # 记录警告信息，未找到可执行文件
    log.warn('Could not locate executable %s' % orig_exe)
    return None

############################################################

def _preserve_environment( names ):
    # 记录调试信息，保留环境变量
    log.debug('_preserve_environment(%r)' % (names))
    # 创建包含指定环境变量名及其当前值的字典
    env = {name: os.environ.get(name) for name in names}
    # 返回环境变量字典
    return env

def _update_environment( **env ):
    # 记录调试信息，更新环境变量
    log.debug('_update_environment(...)')
    # 遍历环境变量字典，更新当前进程的环境变量
    for name, value in env.items():
        os.environ[name] = value or ''

def exec_command(command, execute_in='', use_shell=None, use_tee=None,
                 _with_python = 1, **env ):
    """
    Return (status,output) of executed command.

    .. deprecated:: 1.17
        Use subprocess.Popen instead

    Parameters
    ----------
    command : str
        A concatenated string of executable and arguments.
    execute_in : str
        Before running command ``cd execute_in`` and after ``cd -``.
    use_shell : {bool, None}, optional
        If True, execute ``sh -c command``. Default None (True)
    use_tee : {bool, None}, optional
        If True use tee. Default None (True)


    Returns
    -------
    """
    res : str
        Both stdout and stderr messages.

    Notes
    -----
    On NT, DOS systems the returned status is correct for external commands.
    Wild cards will not work for non-posix systems or when use_shell=0.

    """
    # 发出警告，提示 exec_command 函数自 NumPy v1.17 起已废弃，请使用 subprocess.Popen 替代
    warnings.warn('exec_command is deprecated since NumPy v1.17, use '
                  'subprocess.Popen instead', DeprecationWarning, stacklevel=1)
    # 记录调试信息，显示执行的命令和环境变量
    log.debug('exec_command(%r,%s)' % (command,
         ','.join(['%s=%r'%kv for kv in env.items()])))

    # 根据操作系统类型决定是否使用 tee 命令
    if use_tee is None:
        use_tee = os.name=='posix'
    # 根据操作系统类型决定是否使用 shell
    if use_shell is None:
        use_shell = os.name=='posix'
    # 将执行路径转换为绝对路径
    execute_in = os.path.abspath(execute_in)
    # 获取当前工作目录的绝对路径
    oldcwd = os.path.abspath(os.getcwd())

    # 根据文件名是否包含 'exec_command' 来决定执行目录的设定
    if __name__[-12:] == 'exec_command':
        exec_dir = os.path.dirname(os.path.abspath(__file__))
    elif os.path.isfile('exec_command.py'):
        exec_dir = os.path.abspath('.')
    else:
        exec_dir = os.path.abspath(sys.argv[0])
        if os.path.isfile(exec_dir):
            exec_dir = os.path.dirname(exec_dir)

    # 如果旧工作目录与执行目录不同，则切换工作目录到执行目录，并记录调试信息
    if oldcwd!=execute_in:
        os.chdir(execute_in)
        log.debug('New cwd: %s' % execute_in)
    else:
        # 如果旧工作目录与执行目录相同，则记录调试信息
        log.debug('Retaining cwd: %s' % oldcwd)

    # 保存当前环境的关键部分，更新环境变量
    oldenv = _preserve_environment( list(env.keys()) )
    _update_environment( **env )

    try:
        # 执行命令，获取执行状态 st
        st = _exec_command(command,
                           use_shell=use_shell,
                           use_tee=use_tee,
                           **env)
    finally:
        # 如果旧工作目录与执行目录不同，则恢复工作目录到旧工作目录，并记录调试信息
        if oldcwd!=execute_in:
            os.chdir(oldcwd)
            log.debug('Restored cwd to %s' % oldcwd)
        # 恢复旧环境变量
        _update_environment(**oldenv)

    # 返回执行状态 st
    return st
# 执行给定命令的内部工作函数，用于 exec_command()。
def _exec_command(command, use_shell=None, use_tee=None, **env):
    # 如果 use_shell 未指定，则根据操作系统确定是否使用 shell
    if use_shell is None:
        use_shell = os.name == 'posix'
    # 如果 use_tee 未指定，则根据操作系统确定是否使用 tee
    if use_tee is None:
        use_tee = os.name == 'posix'

    # 在 POSIX 系统上且需要使用 shell 时，覆盖 subprocess 默认的 /bin/sh
    if os.name == 'posix' and use_shell:
        sh = os.environ.get('SHELL', '/bin/sh')
        # 如果命令是一个序列，则创建一个新的命令序列
        if is_sequence(command):
            command = [sh, '-c', ' '.join(command)]
        else:
            command = [sh, '-c', command]
        use_shell = False

    # 在 Windows 上且命令是一个序列时，手动拼接字符串以用于 CreateProcess()
    elif os.name == 'nt' and is_sequence(command):
        command = ' '.join(_quote_arg(arg) for arg in command)

    # 默认情况下继承环境变量
    env = env or None
    try:
        # 启动子进程执行命令，text 设为 False 以便 communicate() 返回字节流，
        # 我们需要手动解码输出，以避免遇到无效字符时引发 UnicodeDecodeError
        proc = subprocess.Popen(command, shell=use_shell, env=env, text=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    except OSError:
        # 如果启动子进程失败，返回 127，并返回空字符串
        return 127, ''

    # 读取子进程的输出和错误信息
    text, err = proc.communicate()
    
    # 获取系统首选编码，用于解码子进程输出
    mylocale = locale.getpreferredencoding(False)
    if mylocale is None:
        mylocale = 'ascii'
    # 使用替换错误模式解码文本，将 '\r\n' 替换为 '\n'
    text = text.decode(mylocale, errors='replace')
    text = text.replace('\r\n', '\n')

    # 去除末尾的换行符（历史遗留问题）
    if text[-1:] == '\n':
        text = text[:-1]

    # 如果需要使用 tee，并且输出文本不为空，则打印输出文本
    if use_tee and text:
        print(text)

    # 返回子进程的返回码和处理后的输出文本
    return proc.returncode, text


# 将参数 arg 安全地引用为可以在 shell 命令行中使用的形式
def _quote_arg(arg):
    # 如果字符串中不包含双引号且包含空格，则将整个字符串用双引号引起来
    if '"' not in arg and ' ' in arg:
        return '"%s"' % arg
    return arg
```