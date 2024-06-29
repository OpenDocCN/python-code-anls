# `.\numpy\numpy\distutils\tests\test_exec_command.py`

```
# 导入所需模块
import os              # 提供与操作系统交互的功能
import pytest          # 提供用于编写和运行测试的框架
import sys             # 提供对Python解释器的访问
from tempfile import TemporaryFile  # 提供临时文件和目录的支持

from numpy.distutils import exec_command  # 导入执行命令的相关功能
from numpy.distutils.exec_command import get_pythonexe  # 导入获取Python解释器路径的函数
from numpy.testing import tempdir, assert_, assert_warns, IS_WASM  # 导入测试相关的函数和变量

# 在Python 3中，stdout和stderr是文本设备，为了模拟它们的行为，从io模块中导入StringIO
from io import StringIO  # 提供在内存中操作文本的功能


class redirect_stdout:
    """上下文管理器，用于重定向stdout，用于exec_command测试。"""
    def __init__(self, stdout=None):
        self._stdout = stdout or sys.stdout

    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = self._stdout

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        sys.stdout = self.old_stdout
        # 注意：关闭sys.stdout不会真正关闭它。
        self._stdout.close()


class redirect_stderr:
    """上下文管理器，用于重定向stderr，用于exec_command测试。"""
    def __init__(self, stderr=None):
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stderr.flush()
        sys.stderr = self.old_stderr
        # 注意：关闭sys.stderr不会真正关闭它。
        self._stderr.close()


class emulate_nonposix:
    """上下文管理器，用于模拟非posix系统的行为。"""
    def __init__(self, osname='non-posix'):
        self._new_name = osname

    def __enter__(self):
        self._old_name = os.name
        os.name = self._new_name

    def __exit__(self, exc_type, exc_value, traceback):
        os.name = self._old_name


def test_exec_command_stdout():
    # gh-2999和gh-2915的回归测试。
    # 有几个包（nose、scipy.weave.inline、Sage inline Fortran）会替换stdout，此时它不具有fileno方法。
    # 在这里进行测试，用一个无操作的命令来测试exec_command中对fileno()的假设是否失败。

    # 代码针对posix系统有一个特殊情况，因此如果在posix系统上，测试特殊情况和通用代码两种情况。

    # 测试posix版本：
    with redirect_stdout(StringIO()):
        with redirect_stderr(TemporaryFile()):
            with assert_warns(DeprecationWarning):
                exec_command.exec_command("cd '.'")

    if os.name == 'posix':
        # 测试通用（非posix）版本：
        with emulate_nonposix():
            with redirect_stdout(StringIO()):
                with redirect_stderr(TemporaryFile()):
                    with assert_warns(DeprecationWarning):
                        exec_command.exec_command("cd '.'")


def test_exec_command_stderr():
    # 测试posix版本：
    with redirect_stdout(TemporaryFile(mode='w+')):
        with redirect_stderr(StringIO()):
            with assert_warns(DeprecationWarning):
                exec_command.exec_command("cd '.'")
    # 检查操作系统类型是否为 POSIX
    if os.name == 'posix':
        # 在非 POSIX 环境下进行通用版本测试:
        with emulate_nonposix():  # 模拟非 POSIX 环境
            with redirect_stdout(TemporaryFile()):  # 重定向标准输出到临时文件
                with redirect_stderr(StringIO()):  # 重定向标准错误到字符串流
                    with assert_warns(DeprecationWarning):  # 断言会发出 DeprecationWarning 警告
                        exec_command.exec_command("cd '.'")  # 执行命令 "cd '.'"
# 使用 pytest 的装饰器标记此测试类，如果在 WebAssembly 环境下则跳过测试
@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
class TestExecCommand:
    # 设置每个测试方法的前置条件，在每次测试方法运行前获取 Python 可执行文件路径
    def setup_method(self):
        self.pyexe = get_pythonexe()

    # 检查在 Windows 环境下执行命令的方法
    def check_nt(self, **kws):
        # 执行命令 'cmd /C echo path=%path%'，检查输出状态和内容
        s, o = exec_command.exec_command('cmd /C echo path=%path%')
        assert_(s == 0)
        assert_(o != '')

        # 执行命令 '"%s" -c "import sys;sys.stderr.write(sys.platform)"' % self.pyexe，在 Windows 下检查输出状态和内容
        s, o = exec_command.exec_command(
         '"%s" -c "import sys;sys.stderr.write(sys.platform)"' % self.pyexe)
        assert_(s == 0)
        assert_(o == 'win32')

    # 检查在类 Unix 环境下执行命令的方法
    def check_posix(self, **kws):
        # 执行命令 'echo Hello'，检查输出状态和内容
        s, o = exec_command.exec_command("echo Hello", **kws)
        assert_(s == 0)
        assert_(o == 'Hello')

        # 执行命令 'echo $AAA'，检查输出状态和内容
        s, o = exec_command.exec_command('echo $AAA', **kws)
        assert_(s == 0)
        assert_(o == '')

        # 执行命令 'echo "$AAA"'，检查输出状态和内容，设置环境变量 AAA='Tere'
        s, o = exec_command.exec_command('echo "$AAA"', AAA='Tere', **kws)
        assert_(s == 0)
        assert_(o == 'Tere')

        # 执行命令 'echo "$AAA"'，检查输出状态和内容
        s, o = exec_command.exec_command('echo "$AAA"', **kws)
        assert_(s == 0)
        assert_(o == '')

        # 如果环境变量 BBB 不在 os.environ 中，则设置它并执行相应命令
        if 'BBB' not in os.environ:
            os.environ['BBB'] = 'Hi'
            # 执行命令 'echo "$BBB"'，检查输出状态和内容
            s, o = exec_command.exec_command('echo "$BBB"', **kws)
            assert_(s == 0)
            assert_(o == 'Hi')

            # 执行命令 'echo "$BBB"'，检查输出状态和内容，设置环境变量 BBB='Hey'
            s, o = exec_command.exec_command('echo "$BBB"', BBB='Hey', **kws)
            assert_(s == 0)
            assert_(o == 'Hey')

            # 执行命令 'echo "$BBB"'，检查输出状态和内容
            s, o = exec_command.exec_command('echo "$BBB"', **kws)
            assert_(s == 0)
            assert_(o == 'Hi')

            # 删除环境变量 BBB 并再次检查执行命令 'echo "$BBB"' 的状态和输出内容
            del os.environ['BBB']
            s, o = exec_command.exec_command('echo "$BBB"', **kws)
            assert_(s == 0)
            assert_(o == '')

        # 执行一个不存在的命令 'this_is_not_a_command'，检查其输出状态和内容
        s, o = exec_command.exec_command('this_is_not_a_command', **kws)
        assert_(s != 0)
        assert_(o != '')

        # 执行命令 'echo path=$PATH'，检查输出状态和内容
        s, o = exec_command.exec_command('echo path=$PATH', **kws)
        assert_(s == 0)
        assert_(o != '')

        # 执行命令 '"%s" -c "import sys,os;sys.stderr.write(os.name)"' % self.pyexe，检查输出状态和内容
        s, o = exec_command.exec_command(
             '"%s" -c "import sys,os;sys.stderr.write(os.name)"' %
             self.pyexe, **kws)
        assert_(s == 0)
        assert_(o == 'posix')

    # 检查基本命令执行情况的方法
    def check_basic(self, *kws):
        # 执行命令 '"%s" -c "raise \'Ignore me.\'"' % self.pyexe，检查非零状态和输出内容
        s, o = exec_command.exec_command(
                     '"%s" -c "raise \'Ignore me.\'"' % self.pyexe, **kws)
        assert_(s != 0)
        assert_(o != '')

        # 执行命令 '"%s" -c "import sys;sys.stderr.write(\'0\');sys.stderr.write(\'1\');sys.stderr.write(\'2\')"' % self.pyexe，检查输出状态和内容
        s, o = exec_command.exec_command(
             '"%s" -c "import sys;sys.stderr.write(\'0\');'
             'sys.stderr.write(\'1\');sys.stderr.write(\'2\')"' %
             self.pyexe, **kws)
        assert_(s == 0)
        assert_(o == '012')

        # 执行命令 '"%s" -c "import sys;sys.exit(15)"' % self.pyexe，检查状态码和输出内容
        s, o = exec_command.exec_command(
                 '"%s" -c "import sys;sys.exit(15)"' % self.pyexe, **kws)
        assert_(s == 15)
        assert_(o == '')

        # 执行命令 '"%s" -c "print(\'Heipa\'")' % self.pyexe，检查输出状态和内容
        s, o = exec_command.exec_command(
                     '"%s" -c "print(\'Heipa\'")' % self.pyexe, **kws)
        assert_(s == 0)
        assert_(o == 'Heipa')
    # 在临时目录中执行检查执行命令的函数
    def check_execute_in(self, **kws):
        # 使用临时目录上下文管理器创建临时文件，并写入'Hello'
        with tempdir() as tmpdir:
            # 定义临时文件名和路径
            fn = "file"
            tmpfile = os.path.join(tmpdir, fn)
            # 打开临时文件并写入内容'Hello'
            with open(tmpfile, 'w') as f:
                f.write('Hello')

            # 执行指定的命令，并获取返回的状态码和输出
            s, o = exec_command.exec_command(
                 '"%s" -c "f = open(\'%s\', \'r\'); f.close()"' %
                 (self.pyexe, fn), **kws)
            # 断言状态码不为0，即命令执行不成功
            assert_(s != 0)
            # 断言输出不为空
            assert_(o != '')

            # 在临时目录中执行另一条命令，并获取返回的状态码和输出
            s, o = exec_command.exec_command(
                     '"%s" -c "f = open(\'%s\', \'r\'); print(f.read()); '
                     'f.close()"' % (self.pyexe, fn), execute_in=tmpdir, **kws)
            # 断言状态码为0，即命令执行成功
            assert_(s == 0)
            # 断言输出与预期输出'Hello'一致
            assert_(o == 'Hello')

    # 测试基本功能
    def test_basic(self):
        # 重定向标准输出和标准错误流到StringIO
        with redirect_stdout(StringIO()):
            with redirect_stderr(StringIO()):
                # 检查是否会发出弃用警告
                with assert_warns(DeprecationWarning):
                    # 根据操作系统类型选择执行相应的检查函数
                    if os.name == "posix":
                        self.check_posix(use_tee=0)
                        self.check_posix(use_tee=1)
                    elif os.name == "nt":
                        self.check_nt(use_tee=0)
                        self.check_nt(use_tee=1)
                    # 执行检查执行函数，分别测试不同参数情况下的执行
                    self.check_execute_in(use_tee=0)
                    self.check_execute_in(use_tee=1)
```