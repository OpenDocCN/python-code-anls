# `D:\src\scipysrc\scipy\scipy\_lib\tests\test_tmpdirs.py`

```
""" Test tmpdirs module """
# 导入所需的模块和函数
from os import getcwd  # 导入当前工作目录函数
from os.path import realpath, abspath, dirname, isfile, join as pjoin, exists  # 导入路径处理函数

from scipy._lib._tmpdirs import tempdir, in_tempdir, in_dir  # 导入临时目录管理函数

from numpy.testing import assert_, assert_equal  # 导入断言函数

MY_PATH = abspath(__file__)  # 获取当前文件的绝对路径
MY_DIR = dirname(MY_PATH)  # 获取当前文件所在目录的路径


def test_tempdir():
    # 测试临时目录管理器 tempdir()
    with tempdir() as tmpdir:
        fname = pjoin(tmpdir, 'example_file.txt')  # 在临时目录中创建文件路径
        with open(fname, "w") as fobj:
            fobj.write('a string\\n')  # 向文件中写入内容
    assert_(not exists(tmpdir))  # 断言临时目录已被删除


def test_in_tempdir():
    my_cwd = getcwd()  # 获取当前工作目录
    with in_tempdir() as tmpdir:
        with open('test.txt', "w") as f:
            f.write('some text')  # 在临时目录中创建文件并写入内容
        assert_(isfile('test.txt'))  # 断言当前目录中存在文件 test.txt
        assert_(isfile(pjoin(tmpdir, 'test.txt')))  # 断言临时目录中存在文件 test.txt
    assert_(not exists(tmpdir))  # 断言临时目录已被删除
    assert_equal(getcwd(), my_cwd)  # 断言当前工作目录与原工作目录相同


def test_given_directory():
    # 测试 in_dir() 函数
    cwd = getcwd()  # 获取当前工作目录
    with in_dir() as tmpdir:
        assert_equal(tmpdir, abspath(cwd))  # 断言临时目录与当前工作目录路径相同
        assert_equal(tmpdir, abspath(getcwd()))  # 断言临时目录与当前工作目录路径相同
    with in_dir(MY_DIR) as tmpdir:
        assert_equal(tmpdir, MY_DIR)  # 断言临时目录与当前文件所在目录路径相同
        assert_equal(realpath(MY_DIR), realpath(abspath(getcwd())))  # 断言临时目录的真实路径与当前工作目录的真实路径相同
    # 检查当前文件是否存在
    assert_(isfile(MY_PATH))  # 断言当前文件存在
```