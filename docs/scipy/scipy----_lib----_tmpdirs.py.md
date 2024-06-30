# `D:\src\scipysrc\scipy\scipy\_lib\_tmpdirs.py`

```
# 导入所需的模块
import os  # 导入操作系统接口模块
from contextlib import contextmanager  # 导入上下文管理器模块
from shutil import rmtree  # 导入递归删除目录模块
from tempfile import mkdtemp  # 导入创建临时目录模块

# 定义一个上下文管理器，创建并返回一个临时目录
@contextmanager
def tempdir():
    """Create and return a temporary directory. This has the same
    behavior as mkdtemp but can be used as a context manager.

    Upon exiting the context, the directory and everything contained
    in it are removed.

    Examples
    --------
    >>> import os
    >>> with tempdir() as tmpdir:
    ...     fname = os.path.join(tmpdir, 'example_file.txt')
    ...     with open(fname, 'wt') as fobj:
    ...         _ = fobj.write('a string\\n')
    >>> os.path.exists(tmpdir)
    False
    """
    # 创建临时目录
    d = mkdtemp()
    yield d  # 返回临时目录路径并等待被使用
    # 在退出上下文时，递归删除临时目录及其内容
    rmtree(d)


# 定义一个上下文管理器，创建临时目录并切换当前工作目录到该目录
@contextmanager
def in_tempdir():
    ''' Create, return, and change directory to a temporary directory

    Examples
    --------
    >>> import os
    >>> my_cwd = os.getcwd()
    >>> with in_tempdir() as tmpdir:
    ...     _ = open('test.txt', 'wt').write('some text')
    ...     assert os.path.isfile('test.txt')
    ...     assert os.path.isfile(os.path.join(tmpdir, 'test.txt'))
    >>> os.path.exists(tmpdir)
    False
    >>> os.getcwd() == my_cwd
    True
    '''
    # 保存当前工作目录路径
    pwd = os.getcwd()
    # 创建临时目录
    d = mkdtemp()
    # 切换当前工作目录到临时目录
    os.chdir(d)
    yield d  # 返回临时目录路径并等待被使用
    # 恢复初始工作目录，递归删除临时目录及其内容
    os.chdir(pwd)
    rmtree(d)


# 定义一个上下文管理器，切换当前工作目录到指定目录
@contextmanager
def in_dir(dir=None):
    """ Change directory to given directory for duration of ``with`` block

    Useful when you want to use `in_tempdir` for the final test, but
    you are still debugging. For example, you may want to do this in the end:

    >>> with in_tempdir() as tmpdir:
    ...     # do something complicated which might break
    ...     pass

    But, indeed, the complicated thing does break, and meanwhile, the
    ``in_tempdir`` context manager wiped out the directory with the
    temporary files that you wanted for debugging. So, while debugging, you
    replace with something like:

    >>> with in_dir() as tmpdir: # Use working directory by default
    ...     # do something complicated which might break
    ...     pass

    You can then look at the temporary file outputs to debug what is happening,
    fix, and finally replace ``in_dir`` with ``in_tempdir`` again.
    """
    # 保存当前工作目录路径
    cwd = os.getcwd()
    if dir is None:
        yield cwd  # 返回当前工作目录路径并等待被使用
        return
    # 切换当前工作目录到指定目录
    os.chdir(dir)
    yield dir  # 返回指定目录路径并等待被使用
    # 恢复初始工作目录
    os.chdir(cwd)
```