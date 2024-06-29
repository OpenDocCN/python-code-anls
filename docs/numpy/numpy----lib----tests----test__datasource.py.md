# `.\numpy\numpy\lib\tests\test__datasource.py`

```
import os  # 导入操作系统模块
import pytest  # 导入 pytest 测试框架
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile  # 导入临时文件和目录创建相关的函数
from shutil import rmtree  # 导入删除目录的函数

import numpy.lib._datasource as datasource  # 导入 numpy 的数据源模块
from numpy.testing import assert_, assert_equal, assert_raises  # 导入 numpy 测试相关的断言函数

import urllib.request as urllib_request  # 导入 urllib 的请求模块
from urllib.parse import urlparse  # 导入解析 URL 的函数
from urllib.error import URLError  # 导入处理 URL 错误的异常类


def urlopen_stub(url, data=None):
    '''Stub to replace urlopen for testing.'''
    if url == valid_httpurl():
        tmpfile = NamedTemporaryFile(prefix='urltmp_')
        return tmpfile
    else:
        raise URLError('Name or service not known')

# setup and teardown
old_urlopen = None


def setup_module():
    global old_urlopen

    old_urlopen = urllib_request.urlopen
    urllib_request.urlopen = urlopen_stub


def teardown_module():
    urllib_request.urlopen = old_urlopen

# A valid website for more robust testing
http_path = 'http://www.google.com/'  # 定义一个有效的 HTTP 网址
http_file = 'index.html'  # 定义一个 HTTP 文件名

http_fakepath = 'http://fake.abc.web/site/'  # 定义一个无效的 HTTP 网址
http_fakefile = 'fake.txt'  # 定义一个无效的 HTTP 文件名

malicious_files = ['/etc/shadow', '../../shadow',
                   '..\\system.dat', 'c:\\windows\\system.dat']  # 定义一些恶意文件路径

magic_line = b'three is the magic number'  # 定义一个神奇的字节序列


# Utility functions used by many tests
def valid_textfile(filedir):
    # Generate and return a valid temporary file.
    fd, path = mkstemp(suffix='.txt', prefix='dstmp_', dir=filedir, text=True)
    os.close(fd)
    return path


def invalid_textfile(filedir):
    # Generate and return an invalid filename.
    fd, path = mkstemp(suffix='.txt', prefix='dstmp_', dir=filedir)
    os.close(fd)
    os.remove(path)
    return path


def valid_httpurl():
    return http_path+http_file


def invalid_httpurl():
    return http_fakepath+http_fakefile


def valid_baseurl():
    return http_path


def invalid_baseurl():
    return http_fakepath


def valid_httpfile():
    return http_file


def invalid_httpfile():
    return http_fakefile


class TestDataSourceOpen:
    def setup_method(self):
        self.tmpdir = mkdtemp()  # 创建临时目录
        self.ds = datasource.DataSource(self.tmpdir)  # 初始化数据源对象

    def teardown_method(self):
        rmtree(self.tmpdir)  # 删除临时目录
        del self.ds  # 删除数据源对象

    def test_ValidHTTP(self):
        fh = self.ds.open(valid_httpurl())  # 打开一个有效的 HTTP 资源
        assert_(fh)  # 断言文件句柄有效
        fh.close()  # 关闭文件句柄

    def test_InvalidHTTP(self):
        url = invalid_httpurl()
        assert_raises(OSError, self.ds.open, url)  # 断言打开无效 HTTP 资源时抛出 OSError 异常
        try:
            self.ds.open(url)
        except OSError as e:
            # Regression test for bug fixed in r4342.
            assert_(e.errno is None)  # 断言异常中的错误号为 None

    def test_InvalidHTTPCacheURLError(self):
        assert_raises(URLError, self.ds._cache, invalid_httpurl())  # 断言在缓存无效 HTTP 资源时抛出 URLError 异常

    def test_ValidFile(self):
        local_file = valid_textfile(self.tmpdir)  # 创建一个有效的本地文本文件
        fh = self.ds.open(local_file)  # 打开该本地文件
        assert_(fh)  # 断言文件句柄有效
        fh.close()  # 关闭文件句柄

    def test_InvalidFile(self):
        invalid_file = invalid_textfile(self.tmpdir)  # 创建一个无效的本地文件名
        assert_raises(OSError, self.ds.open, invalid_file)  # 断言打开无效文件时抛出 OSError 异常
    # 定义测试函数，用于验证处理有效的 Gzip 文件的功能
    def test_ValidGzipFile(self):
        try:
            import gzip
        except ImportError:
            # 如果导入 gzip 失败，跳过测试
            pytest.skip()
        # 设置测试文件路径为临时目录下的 foobar.txt.gz
        filepath = os.path.join(self.tmpdir, 'foobar.txt.gz')
        # 打开文件准备写入 gzip 数据
        fp = gzip.open(filepath, 'w')
        # 写入预设的 magic_line 到 gzip 文件中
        fp.write(magic_line)
        # 关闭文件
        fp.close()
        # 调用被测试的数据源对象的 open 方法打开文件
        fp = self.ds.open(filepath)
        # 从文件中读取一行数据
        result = fp.readline()
        # 关闭文件
        fp.close()
        # 断言读取的结果与预设的 magic_line 相等
        assert_equal(magic_line, result)

    # 定义测试函数，用于验证处理有效的 BZip2 文件的功能
    def test_ValidBz2File(self):
        try:
            import bz2
        except ImportError:
            # 如果导入 bz2 失败，跳过测试
            pytest.skip()
        # 设置测试文件路径为临时目录下的 foobar.txt.bz2
        filepath = os.path.join(self.tmpdir, 'foobar.txt.bz2')
        # 打开文件准备写入 BZip2 数据
        fp = bz2.BZ2File(filepath, 'w')
        # 写入预设的 magic_line 到 BZip2 文件中
        fp.write(magic_line)
        # 关闭文件
        fp.close()
        # 调用被测试的数据源对象的 open 方法打开文件
        fp = self.ds.open(filepath)
        # 从文件中读取一行数据
        result = fp.readline()
        # 关闭文件
        fp.close()
        # 断言读取的结果与预设的 magic_line 相等
        assert_equal(magic_line, result)
class TestDataSourceExists:
    # 测试数据源存在性的测试类

    def setup_method(self):
        # 每个测试方法执行前的设置方法
        self.tmpdir = mkdtemp()
        self.ds = datasource.DataSource(self.tmpdir)

    def teardown_method(self):
        # 每个测试方法执行后的清理方法
        rmtree(self.tmpdir)
        del self.ds

    def test_ValidHTTP(self):
        # 测试有效的 HTTP 路径
        assert_(self.ds.exists(valid_httpurl()))

    def test_InvalidHTTP(self):
        # 测试无效的 HTTP 路径
        assert_equal(self.ds.exists(invalid_httpurl()), False)

    def test_ValidFile(self):
        # 测试存在于目标路径中的有效文件
        tmpfile = valid_textfile(self.tmpdir)
        assert_(self.ds.exists(tmpfile))
        # 测试不在目标路径中的本地有效文件
        localdir = mkdtemp()
        tmpfile = valid_textfile(localdir)
        assert_(self.ds.exists(tmpfile))
        rmtree(localdir)

    def test_InvalidFile(self):
        # 测试无效的文件
        tmpfile = invalid_textfile(self.tmpdir)
        assert_equal(self.ds.exists(tmpfile), False)


class TestDataSourceAbspath:
    # 测试数据源绝对路径的测试类

    def setup_method(self):
        # 每个测试方法执行前的设置方法
        self.tmpdir = os.path.abspath(mkdtemp())
        self.ds = datasource.DataSource(self.tmpdir)

    def teardown_method(self):
        # 每个测试方法执行后的清理方法
        rmtree(self.tmpdir)
        del self.ds

    def test_ValidHTTP(self):
        # 测试有效的 HTTP 路径
        scheme, netloc, upath, pms, qry, frg = urlparse(valid_httpurl())
        local_path = os.path.join(self.tmpdir, netloc,
                                  upath.strip(os.sep).strip('/'))
        assert_equal(local_path, self.ds.abspath(valid_httpurl()))

    def test_ValidFile(self):
        # 测试有效的文件路径
        tmpfile = valid_textfile(self.tmpdir)
        tmpfilename = os.path.split(tmpfile)[-1]
        # 测试仅使用文件名的情况
        assert_equal(tmpfile, self.ds.abspath(tmpfilename))
        # 测试包含完整路径的文件名
        assert_equal(tmpfile, self.ds.abspath(tmpfile))

    def test_InvalidHTTP(self):
        # 测试无效的 HTTP 路径
        scheme, netloc, upath, pms, qry, frg = urlparse(invalid_httpurl())
        invalidhttp = os.path.join(self.tmpdir, netloc,
                                   upath.strip(os.sep).strip('/'))
        assert_(invalidhttp != self.ds.abspath(valid_httpurl()))

    def test_InvalidFile(self):
        # 测试无效的文件路径
        invalidfile = valid_textfile(self.tmpdir)
        tmpfile = valid_textfile(self.tmpdir)
        tmpfilename = os.path.split(tmpfile)[-1]
        # 测试仅使用文件名的情况
        assert_(invalidfile != self.ds.abspath(tmpfilename))
        # 测试包含完整路径的文件名
        assert_(invalidfile != self.ds.abspath(tmpfile))
    # 测试函数：测试沙盒环境限制

    # 创建一个有效的文本文件并返回其路径
    tmpfile = valid_textfile(self.tmpdir)
    # 获取临时文件名
    tmpfilename = os.path.split(tmpfile)[-1]

    # 定义一个临时路径的lambda函数，将输入路径转换为绝对路径并返回
    tmp_path = lambda x: os.path.abspath(self.ds.abspath(x))

    # 断言：验证有效 HTTP URL 转换后的路径是否以 self.tmpdir 开头
    assert_(tmp_path(valid_httpurl()).startswith(self.tmpdir))
    # 断言：验证无效 HTTP URL 转换后的路径是否以 self.tmpdir 开头
    assert_(tmp_path(invalid_httpurl()).startswith(self.tmpdir))
    # 断言：验证临时文件路径转换后是否以 self.tmpdir 开头
    assert_(tmp_path(tmpfile).startswith(self.tmpdir))
    # 断言：验证临时文件名路径转换后是否以 self.tmpdir 开头
    assert_(tmp_path(tmpfilename).startswith(self.tmpdir))

    # 遍历恶意文件列表，验证连接到 HTTP 路径的恶意文件是否以 self.tmpdir 开头
    for fn in malicious_files:
        assert_(tmp_path(http_path+fn).startswith(self.tmpdir))
        # 断言：验证恶意文件路径是否以 self.tmpdir 开头
        assert_(tmp_path(fn).startswith(self.tmpdir))


    # 测试函数：测试 Windows 系统路径分隔符

    # 保存原始的系统路径分隔符
    orig_os_sep = os.sep
    try:
        # 设置系统路径分隔符为反斜杠
        os.sep = '\\'
        # 执行以下测试函数
        self.test_ValidHTTP()
        self.test_ValidFile()
        self.test_InvalidHTTP()
        self.test_InvalidFile()
        self.test_sandboxing()
    finally:
        # 恢复原始的系统路径分隔符
        os.sep = orig_os_sep
# 定义一个名为TestRepositoryAbspath的类
class TestRepositoryAbspath:
    # 定义初始化方法
    def setup_method(self):
        # 创建临时目录，并获取其绝对路径
        self.tmpdir = os.path.abspath(mkdtemp())
        # 创建数据源，使用有效的基本URL和临时目录
        self.repos = datasource.Repository(valid_baseurl(), self.tmpdir)

    # 定义清理方法
    def teardown_method(self):
        # 递归删除临时目录及其内容
        rmtree(self.tmpdir)
        # 删除数据源
        del self.repos

    # 定义测试方法，测试有效的HTTP地址
    def test_ValidHTTP(self):
        # 解析有效的HTTP URL，获取各部分信息
        scheme, netloc, upath, pms, qry, frg = urlparse(valid_httpurl())
        # 拼接本地路径
        local_path = os.path.join(self.repos._destpath, netloc, upath.strip(os.sep).strip('/'))
        # 获取HTTP文件的绝对路径
        filepath = self.repos.abspath(valid_httpfile())
        # 断言本地路径和文件路径相等
        assert_equal(local_path, filepath)

    # 定义测试方法，测试沙盒功能
    def test_sandboxing(self):
        # 使用lambda函数获取临时路径，并断言其以临时目录开头
        tmp_path = lambda x: os.path.abspath(self.repos.abspath(x))
        assert_(tmp_path(valid_httpfile()).startswith(self.tmpdir))
        # 遍历恶意文件列表，断言其绝对路径以临时目录开头
        for fn in malicious_files:
            assert_(tmp_path(http_path+fn).startswith(self.tmpdir))
            assert_(tmp_path(fn).startswith(self.tmpdir))

    # 定义测试方法，测试Windows系统下路径分隔符
    def test_windows_os_sep(self):
        # 保存原始路径分隔符
        orig_os_sep = os.sep
        try:
            # 修改路径分隔符为反斜杠
            os.sep = '\\'
            # 执行ValidHTTP测试
            self.test_ValidHTTP()
            # 执行sandboxing测试
            self.test_sandboxing()
        finally:
            # 恢复原始路径分隔符
            os.sep = orig_os_sep


# 定义一个名为TestRepositoryExists的类
class TestRepositoryExists:
    # 定义初始化方法
    def setup_method(self):
        # 创建临时目录
        self.tmpdir = mkdtemp()
        # 创建数据源，使用有效的基本URL和临时目录
        self.repos = datasource.Repository(valid_baseurl(), self.tmpdir)

    # 定义清理方法
    def teardown_method(self):
        # 递归删除临时目录及其内容
        rmtree(self.tmpdir)
        # 删除数据源
        del self.repos

    # 定义测试方法，测试有效文件是否存在
    def test_ValidFile(self):
        # 创建本地临时文件
        tmpfile = valid_textfile(self.tmpdir)
        # 断言数据源中存在该文件
        assert_(self.repos.exists(tmpfile))

    # 定义测试方法，测试无效文件是否存在
    def test_InvalidFile(self):
        # 创建无效的本地临时文件
        tmpfile = invalid_textfile(self.tmpdir)
        # 断言数据源中不存在该文件
        assert_equal(self.repos.exists(tmpfile), False)

    # 定义测试方法，测试移除HTTP文件
    def test_RemoveHTTPFile(self):
        # 断言数据源中存在有效的HTTP文件
        assert_(self.repos.exists(valid_httpurl()))

    # 定义测试方法，测试缓存的HTTP文件是否存在
    def test_CachedHTTPFile(self):
        # 获取有效的HTTP URL
        localfile = valid_httpurl()
        # 创建一个具有URL目录结构的本地缓存临时文件，类似于Repository.open的操作
        scheme, netloc, upath, pms, qry, frg = urlparse(localfile)
        local_path = os.path.join(self.repos._destpath, netloc)
        os.mkdir(local_path, 0o0700)
        tmpfile = valid_textfile(local_path)
        # 断言数据源中存在该缓存文件
        assert_(self.repos.exists(tmpfile))


# 定义一个名为TestOpenFunc的类
class TestOpenFunc:
    # 定义初始化方法
    def setup_method(self):
        # 创建临时目录
        self.tmpdir = mkdtemp()

    # 定义清理方法
    def teardown_method(self):
        # 递归删除临时目录及其内容
        rmtree(self.tmpdir)

    # 定义测试方法，测试DataSource的打开操作
    def test_DataSourceOpen(self):
        # 创建本地临时文件
        local_file = valid_textfile(self.tmpdir)
        # 测试传入目标路径的情况
        fp = datasource.open(local_file, destpath=self.tmpdir)
        assert_(fp)
        fp.close()
        # 测试使用默认目标路径的情况
        fp = datasource.open(local_file)
        assert_(fp)
        fp.close()

# 定义测试删除属性处理的函数
def test_del_attr_handling():
    # 数据源__del__可能会被调用，即使在初始化失败时（被调用的异常对象被捕获）
    # 就像在refguide_check的is_deprecated()函数中发生的情况一样
    ds = datasource.DataSource()
    # 创建一个名为 ds 的 DataSource 对象实例
    
    # 模拟由于删除 __init__ 中产生的关键属性而导致的初始化失败
    del ds._istmpdest
    
    # 调用 __del__ 方法来确保在初始化失败时也不会触发 AttributeError
    ds.__del__()
    # 执行对象的析构函数，清理资源或执行必要的清理操作
```