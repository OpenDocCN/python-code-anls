# `.\numpy\numpy\distutils\tests\test_system_info.py`

```py
import os  # 导入操作系统接口模块
import shutil  # 导入高级文件操作模块
import pytest  # 导入用于编写简单有效的单元测试的模块
from tempfile import mkstemp, mkdtemp  # 导入创建临时文件和目录的函数
from subprocess import Popen, PIPE  # 导入用于执行子进程的模块和相关功能
import importlib.metadata  # 导入用于获取模块元数据的模块
from distutils.errors import DistutilsError  # 导入Distutils错误模块

from numpy.testing import assert_, assert_equal, assert_raises  # 导入NumPy测试断言函数
from numpy.distutils import ccompiler, customized_ccompiler  # 导入NumPy的C编译器和自定义C编译器
from numpy.distutils.system_info import (  # 导入NumPy系统信息模块中的多个符号
    system_info, ConfigParser, mkl_info, AliasedOptionError
)
from numpy.distutils.system_info import default_lib_dirs, default_include_dirs  # 导入NumPy系统信息模块中的默认库和包含目录
from numpy.distutils import _shell_utils  # 导入NumPy的Shell工具函数

try:
    if importlib.metadata.version('setuptools') >= '60':  # 尝试获取并检查setuptools的版本
        # 如果setuptools版本大于等于60，跳过测试并显示信息
        pytest.skip("setuptools is too new", allow_module_level=True)
except importlib.metadata.PackageNotFoundError:
    # 如果未找到setuptools包，则继续执行
    pass


def get_class(name, notfound_action=1):
    """
    根据名称获取类对象

    notfound_action:
      0 - 什么都不做
      1 - 显示警告消息
      2 - 抛出错误
    """
    cl = {'temp1': Temp1Info,  # 根据名称映射到对应的类对象
          'temp2': Temp2Info,
          'duplicate_options': DuplicateOptionInfo,
          }.get(name.lower(), _system_info)
    return cl()


simple_site = """
[ALL]
library_dirs = {dir1:s}{pathsep:s}{dir2:s}
libraries = {lib1:s},{lib2:s}
extra_compile_args = -I/fake/directory -I"/path with/spaces" -Os
runtime_library_dirs = {dir1:s}

[temp1]
library_dirs = {dir1:s}
libraries = {lib1:s}
runtime_library_dirs = {dir1:s}

[temp2]
library_dirs = {dir2:s}
libraries = {lib2:s}
extra_link_args = -Wl,-rpath={lib2_escaped:s}
rpath = {dir2:s}

[duplicate_options]
mylib_libs = {lib1:s}
libraries = {lib2:s}
"""

site_cfg = simple_site  # 将simple_site赋值给site_cfg变量

fakelib_c_text = """
/* This file is generated from numpy/distutils/testing/test_system_info.py */
#include<stdio.h>
void foo(void) {
   printf("Hello foo");
}
void bar(void) {
   printf("Hello bar");
}
"""

def have_compiler():
    """ 返回True如果存在可执行的编译器 """
    compiler = customized_ccompiler()  # 获取自定义C编译器对象
    try:
        cmd = compiler.compiler  # 尝试获取Unix编译器的命令
    except AttributeError:
        try:
            if not compiler.initialized:
                compiler.initialize()  # MSVC的初始化操作
        except (DistutilsError, ValueError):
            return False
        cmd = [compiler.cc]  # 获取MSVC编译器的命令
    try:
        p = Popen(cmd, stdout=PIPE, stderr=PIPE)  # 执行编译器命令
        p.stdout.close()  # 关闭标准输出流
        p.stderr.close()  # 关闭标准错误流
        p.wait()  # 等待进程执行完成
    except OSError:
        return False
    return True  # 返回编译器是否可用的结果

HAVE_COMPILER = have_compiler()  # 检查编译器是否可用的全局变量

class _system_info(system_info):
    # 这里应该有_system_info类的定义，但由于代码截断，未能提供完整定义。
    # _system_info 类继承自 system_info 类，可能包含特定系统信息的定制行为或配置。
    # 初始化函数，设置默认的库目录、包含目录和详细程度参数
    def __init__(self,
                 default_lib_dirs=default_lib_dirs,
                 default_include_dirs=default_include_dirs,
                 verbosity=1,
                 ):
        # 设置类变量 info 为空字典
        self.__class__.info = {}
        # 初始化本地前缀列表为空
        self.local_prefixes = []
        # 定义默认配置字典
        defaults = {'library_dirs': '',
                    'include_dirs': '',
                    'runtime_library_dirs': '',
                    'rpath': '',
                    'src_dirs': '',
                    'search_static_first': "0",
                    'extra_compile_args': '',
                    'extra_link_args': ''}
        # 使用默认配置初始化 ConfigParser 对象
        self.cp = ConfigParser(defaults)
        # 我们必须在稍后解析配置文件，
        # 以便有一个一致的临时文件路径

    # 检查库函数，覆盖 _check_libs 函数以返回所有目录信息
    def _check_libs(self, lib_dirs, libs, opt_libs, exts):
        """Override _check_libs to return with all dirs """
        # 创建包含库和库目录信息的字典
        info = {'libraries': libs, 'library_dirs': lib_dirs}
        # 返回信息字典
        return info
class Temp1Info(_system_info):
    """For testing purposes"""
    section = 'temp1'

class Temp2Info(_system_info):
    """For testing purposes"""
    section = 'temp2'

class DuplicateOptionInfo(_system_info):
    """For testing purposes"""
    section = 'duplicate_options'

class TestSystemInfoReading:
    
    def setup_method(self):
        """设置测试环境"""
        # 创建临时目录和文件用于测试
        self._dir1 = mkdtemp()  # 创建临时目录1
        self._src1 = os.path.join(self._dir1, 'foo.c')  # 创建源文件路径1
        self._lib1 = os.path.join(self._dir1, 'libfoo.so')  # 创建库文件路径1
        self._dir2 = mkdtemp()  # 创建临时目录2
        self._src2 = os.path.join(self._dir2, 'bar.c')  # 创建源文件路径2
        self._lib2 = os.path.join(self._dir2, 'libbar.so')  # 创建库文件路径2
        
        # 更新本地 site.cfg 配置文件内容
        global simple_site, site_cfg
        site_cfg = simple_site.format(**{
            'dir1': self._dir1,
            'lib1': self._lib1,
            'dir2': self._dir2,
            'lib2': self._lib2,
            'pathsep': os.pathsep,
            'lib2_escaped': _shell_utils.NativeParser.join([self._lib2])
        })
        
        # 写入 site.cfg 文件
        fd, self._sitecfg = mkstemp()  # 创建临时文件
        os.close(fd)
        with open(self._sitecfg, 'w') as fd:
            fd.write(site_cfg)  # 将 site_cfg 内容写入临时文件
        
        # 写入源文件内容
        with open(self._src1, 'w') as fd:
            fd.write(fakelib_c_text)  # 写入虚拟库源文件内容
        with open(self._src2, 'w') as fd:
            fd.write(fakelib_c_text)  # 写入虚拟库源文件内容
        
        # 创建所有类的实例并解析配置文件
        def site_and_parse(c, site_cfg):
            c.files = [site_cfg]
            c.parse_config_files()
            return c
        
        self.c_default = site_and_parse(get_class('default'), self._sitecfg)  # 默认类实例
        self.c_temp1 = site_and_parse(get_class('temp1'), self._sitecfg)  # temp1 类实例
        self.c_temp2 = site_and_parse(get_class('temp2'), self._sitecfg)  # temp2 类实例
        self.c_dup_options = site_and_parse(get_class('duplicate_options'), self._sitecfg)  # duplicate_options 类实例

    def teardown_method(self):
        # 清理测试环境
        try:
            shutil.rmtree(self._dir1)  # 删除临时目录1
        except Exception:
            pass
        try:
            shutil.rmtree(self._dir2)  # 删除临时目录2
        except Exception:
            pass
        try:
            os.remove(self._sitecfg)  # 删除临时文件 site.cfg
        except Exception:
            pass

    def test_all(self):
        # 测试读取 ALL 块中的所有信息
        tsi = self.c_default
        assert_equal(tsi.get_lib_dirs(), [self._dir1, self._dir2])  # 检查获取的库目录列表是否正确
        assert_equal(tsi.get_libraries(), [self._lib1, self._lib2])  # 检查获取的库文件列表是否正确
        assert_equal(tsi.get_runtime_lib_dirs(), [self._dir1])  # 检查获取的运行时库目录列表是否正确
        extra = tsi.calc_extra_info()
        assert_equal(extra['extra_compile_args'], ['-I/fake/directory', '-I/path with/spaces', '-Os'])  # 检查额外的编译参数是否正确
    def test_temp1(self):
        # Read in all information in the temp1 block
        # 获取 self.c_temp1 对象的引用
        tsi = self.c_temp1
        # 断言获取的库目录与预期相符
        assert_equal(tsi.get_lib_dirs(), [self._dir1])
        # 断言获取的库名称与预期相符
        assert_equal(tsi.get_libraries(), [self._lib1])
        # 断言获取的运行时库目录与预期相符
        assert_equal(tsi.get_runtime_lib_dirs(), [self._dir1])

    def test_temp2(self):
        # Read in all information in the temp2 block
        # 获取 self.c_temp2 对象的引用
        tsi = self.c_temp2
        # 断言获取的库目录与预期相符
        assert_equal(tsi.get_lib_dirs(), [self._dir2])
        # 断言获取的库名称与预期相符
        assert_equal(tsi.get_libraries(), [self._lib2])
        # 使用 'rpath' 而不是 runtime_library_dirs 获取运行时库目录
        assert_equal(tsi.get_runtime_lib_dirs(key='rpath'), [self._dir2])
        # 计算额外信息
        extra = tsi.calc_extra_info()
        # 断言额外的链接参数与预期相符
        assert_equal(extra['extra_link_args'], ['-Wl,-rpath=' + self._lib2])

    def test_duplicate_options(self):
        # Ensure that duplicates are raising an AliasedOptionError
        # 确保重复选项会引发 AliasedOptionError 异常
        tsi = self.c_dup_options
        # 断言调用 get_option_single 方法时会抛出 AliasedOptionError 异常
        assert_raises(AliasedOptionError, tsi.get_option_single, "mylib_libs", "libraries")
        # 断言调用 get_libs 方法获取到的库与预期相符
        assert_equal(tsi.get_libs("mylib_libs", [self._lib1]), [self._lib1])
        # 断言调用 get_libs 方法获取到的库与预期相符
        assert_equal(tsi.get_libs("libraries", [self._lib2]), [self._lib2])

    @pytest.mark.skipif(not HAVE_COMPILER, reason="Missing compiler")
    def test_compile1(self):
        # Compile source and link the first source
        # 编译源码并链接第一个源文件
        c = customized_ccompiler()
        previousDir = os.getcwd()
        try:
            # 切换目录以避免影响其他目录
            os.chdir(self._dir1)
            # 编译源文件 self._src1 的基本文件名，并输出到 self._dir1 目录
            c.compile([os.path.basename(self._src1)], output_dir=self._dir1)
            # 确保对象文件存在
            assert_(os.path.isfile(self._src1.replace('.c', '.o')) or
                    os.path.isfile(self._src1.replace('.c', '.obj')))
        finally:
            # 恢复之前的工作目录
            os.chdir(previousDir)

    @pytest.mark.skipif(not HAVE_COMPILER, reason="Missing compiler")
    @pytest.mark.skipif('msvc' in repr(ccompiler.new_compiler()),
                         reason="Fails with MSVC compiler ")
    def test_compile2(self):
        # Compile source and link the second source
        # 编译源码并链接第二个源文件
        tsi = self.c_temp2
        c = customized_ccompiler()
        # 获取额外的链接参数
        extra_link_args = tsi.calc_extra_info()['extra_link_args']
        previousDir = os.getcwd()
        try:
            # 切换目录以避免影响其他目录
            os.chdir(self._dir2)
            # 编译源文件 self._src2 的基本文件名，并输出到 self._dir2 目录
            c.compile([os.path.basename(self._src2)], output_dir=self._dir2,
                      extra_postargs=extra_link_args)
            # 确保对象文件存在
            assert_(os.path.isfile(self._src2.replace('.c', '.o')))
        finally:
            # 恢复之前的工作目录
            os.chdir(previousDir)

    HAS_MKL = "mkl_rt" in mkl_info().calc_libraries_info().get("libraries", [])

    @pytest.mark.xfail(HAS_MKL, reason=("`[DEFAULT]` override doesn't work if "
                                        "numpy is built with MKL support"))
    def test_overrides(self):
        # 保存当前工作目录
        previousDir = os.getcwd()
        # 创建 site.cfg 的完整路径
        cfg = os.path.join(self._dir1, 'site.cfg')
        # 复制 self._sitecfg 到 site.cfg
        shutil.copy(self._sitecfg, cfg)
        try:
            # 切换到 self._dir1 目录
            os.chdir(self._dir1)

            # 检查 '[ALL]' 部分是否覆盖了其他部分缺失的值
            info = mkl_info()
            # 获取 '[ALL]' 部分的 library_dirs，按分隔符分割为列表
            lib_dirs = info.cp['ALL']['library_dirs'].split(os.pathsep)
            # 断言 info.get_lib_dirs() 不等于 lib_dirs
            assert info.get_lib_dirs() != lib_dirs

            # 将值复制到 '[mkl]' 部分后，值应该是正确的
            with open(cfg) as fid:
                # 替换第一个 '[ALL]' 为 '[mkl]'
                mkl = fid.read().replace('[ALL]', '[mkl]', 1)
            with open(cfg, 'w') as fid:
                # 写入替换后的内容到文件
                fid.write(mkl)
            info = mkl_info()
            # 断言 info.get_lib_dirs() 等于 lib_dirs
            assert info.get_lib_dirs() == lib_dirs

            # 同样，值也可以从名为 '[DEFAULT]' 的部分中获取
            with open(cfg) as fid:
                # 替换第一个 '[mkl]' 为 '[DEFAULT]'
                dflt = fid.read().replace('[mkl]', '[DEFAULT]', 1)
            with open(cfg, 'w') as fid:
                # 写入替换后的内容到文件
                fid.write(dflt)
            info = mkl_info()
            # 断言 info.get_lib_dirs() 等于 lib_dirs
            assert info.get_lib_dirs() == lib_dirs

        finally:
            # 恢复到之前保存的工作目录
            os.chdir(previousDir)
# 定义测试函数，用于测试环境变量解析顺序的功能
def test_distutils_parse_env_order(monkeypatch):
    # 导入需要测试的函数 _parse_env_order
    from numpy.distutils.system_info import _parse_env_order
    # 定义环境变量的名称
    env = 'NPY_TESTS_DISTUTILS_PARSE_ENV_ORDER'

    # 定义基础顺序列表
    base_order = list('abcdef')

    # 设置环境变量值为 'b,i,e,f'，模拟环境变量设置
    monkeypatch.setenv(env, 'b,i,e,f')
    # 调用 _parse_env_order 函数解析环境变量，返回顺序列表和未知元素列表
    order, unknown = _parse_env_order(base_order, env)
    # 断言返回的顺序列表长度为3
    assert len(order) == 3
    # 断言返回的顺序列表内容符合预期
    assert order == list('bef')
    # 断言未知元素列表长度为1
    assert len(unknown) == 1

    # 当 LAPACK/BLAS 优化被禁用时的情况
    # 清空环境变量值
    monkeypatch.setenv(env, '')
    # 再次调用 _parse_env_order 函数解析空环境变量，返回顺序列表和未知元素列表
    order, unknown = _parse_env_order(base_order, env)
    # 断言顺序列表为空
    assert len(order) == 0
    # 断言未知元素列表也为空
    assert len(unknown) == 0

    # 测试以 '^!' 开头的情况
    for prefix in '^!':
        # 设置环境变量值为 '^b,i,e' 或 '!b,i,e'，模拟环境变量设置
        monkeypatch.setenv(env, f'{prefix}b,i,e')
        # 再次调用 _parse_env_order 函数解析环境变量，返回顺序列表和未知元素列表
        order, unknown = _parse_env_order(base_order, env)
        # 断言顺序列表长度为4
        assert len(order) == 4
        # 断言顺序列表内容符合预期
        assert order == list('acdf')
        # 断言未知元素列表长度为1
        assert len(unknown) == 1

    # 测试引发 ValueError 的情况
    with pytest.raises(ValueError):
        # 设置环境变量值为 'b,^e,i'，预期会引发 ValueError
        monkeypatch.setenv(env, 'b,^e,i')
        _parse_env_order(base_order, env)

    with pytest.raises(ValueError):
        # 设置环境变量值为 '!b,^e,i'，预期会引发 ValueError
        monkeypatch.setenv(env, '!b,^e,i')
        _parse_env_order(base_order, env)
```