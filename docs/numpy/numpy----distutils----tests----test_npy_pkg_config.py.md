# `.\numpy\numpy\distutils\tests\test_npy_pkg_config.py`

```
import os  # 导入标准库 os

from numpy.distutils.npy_pkg_config import read_config, parse_flags  # 导入 numpy.distutils.npy_pkg_config 中的 read_config 和 parse_flags 函数
from numpy.testing import temppath, assert_  # 导入 numpy.testing 中的 temppath 和 assert_

simple = """\
[meta]
Name = foo
Description = foo lib
Version = 0.1

[default]
cflags = -I/usr/include
libs = -L/usr/lib
"""
simple_d = {'cflags': '-I/usr/include', 'libflags': '-L/usr/lib',
            'version': '0.1', 'name': 'foo'}  # 定义一个字典 simple_d 包含名称、描述、版本和标志的键值对

simple_variable = """\
[meta]
Name = foo
Description = foo lib
Version = 0.1

[variables]
prefix = /foo/bar
libdir = ${prefix}/lib
includedir = ${prefix}/include

[default]
cflags = -I${includedir}
libs = -L${libdir}
"""
simple_variable_d = {'cflags': '-I/foo/bar/include', 'libflags': '-L/foo/bar/lib',
                     'version': '0.1', 'name': 'foo'}  # 定义一个字典 simple_variable_d 包含变量替换后的 cflags 和 libs 的值，以及名称和版本信息

class TestLibraryInfo:  # 定义一个测试类 TestLibraryInfo
    def test_simple(self):  # 定义测试方法 test_simple
        with temppath('foo.ini') as path:  # 使用临时路径 'foo.ini'，作为 path
            with open(path,  'w') as f:  # 打开 path 对应的文件 f 以写入模式
                f.write(simple)  # 将 simple 内容写入文件 f
            pkg = os.path.splitext(path)[0]  # 获取文件名，并去掉扩展名，得到 pkg
            out = read_config(pkg)  # 调用 read_config 函数读取配置信息，返回 out

        assert_(out.cflags() == simple_d['cflags'])  # 断言读取出的 cflags 和预期的值相等
        assert_(out.libs() == simple_d['libflags'])  # 断言读取出的 libs 和预期的值相等
        assert_(out.name == simple_d['name'])  # 断言读取出的 name 和预期的值相等
        assert_(out.version == simple_d['version'])  # 断言读取出的 version 和预期的值相等

    def test_simple_variable(self):  # 定义测试方法 test_simple_variable
        with temppath('foo.ini') as path:  # 使用临时路径 'foo.ini'，作为 path
            with open(path,  'w') as f:  # 打开 path 对应的文件 f 以写入模式
                f.write(simple_variable)  # 将 simple_variable 内容写入文件 f
            pkg = os.path.splitext(path)[0]  # 获取文件名，并去掉扩展名，得到 pkg
            out = read_config(pkg)  # 调用 read_config 函数读取配置信息，返回 out

        assert_(out.cflags() == simple_variable_d['cflags'])  # 断言读取出的 cflags 和预期的值相等
        assert_(out.libs() == simple_variable_d['libflags'])  # 断言读取出的 libs 和预期的值相等
        assert_(out.name == simple_variable_d['name'])  # 断言读取出的 name 和预期的值相等
        assert_(out.version == simple_variable_d['version'])  # 断言读取出的 version 和预期的值相等
        out.vars['prefix'] = '/Users/david'  # 修改 out 对象的 prefix 变量为 '/Users/david'
        assert_(out.cflags() == '-I/Users/david/include')  # 断言读取出的 cflags 经修改后的值正确

class TestParseFlags:  # 定义一个测试类 TestParseFlags
    def test_simple_cflags(self):  # 定义测试方法 test_simple_cflags
        d = parse_flags("-I/usr/include")  # 调用 parse_flags 函数解析参数，返回字典 d
        assert_(d['include_dirs'] == ['/usr/include'])  # 断言解析结果中 include_dirs 键对应的值正确

        d = parse_flags("-I/usr/include -DFOO")  # 再次调用 parse_flags 函数解析参数，返回字典 d
        assert_(d['include_dirs'] == ['/usr/include'])  # 断言解析结果中 include_dirs 键对应的值正确
        assert_(d['macros'] == ['FOO'])  # 断言解析结果中 macros 键对应的值正确

        d = parse_flags("-I /usr/include -DFOO")  # 第三次调用 parse_flags 函数解析参数，返回字典 d
        assert_(d['include_dirs'] == ['/usr/include'])  # 断言解析结果中 include_dirs 键对应的值正确
        assert_(d['macros'] == ['FOO'])  # 断言解析结果中 macros 键对应的值正确

    def test_simple_lflags(self):  # 定义测试方法 test_simple_lflags
        d = parse_flags("-L/usr/lib -lfoo -L/usr/lib -lbar")  # 调用 parse_flags 函数解析参数，返回字典 d
        assert_(d['library_dirs'] == ['/usr/lib', '/usr/lib'])  # 断言解析结果中 library_dirs 键对应的值正确
        assert_(d['libraries'] == ['foo', 'bar'])  # 断言解析结果中 libraries 键对应的值正确

        d = parse_flags("-L /usr/lib -lfoo -L/usr/lib -lbar")  # 再次调用 parse_flags 函数解析参数，返回字典 d
        assert_(d['library_dirs'] == ['/usr/lib', '/usr/lib'])  # 断言解析结果中 library_dirs 键对应的值正确
        assert_(d['libraries'] == ['foo', 'bar'])  # 断言解析结果中 libraries 键对应的值正确
```