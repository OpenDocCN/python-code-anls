# `D:\src\scipysrc\scipy\dev.py`

```
```python`
import os
import subprocess
import sys
import warnings
import shutil
import json
import datetime
import time
import importlib
import importlib.util
import errno
import contextlib
from sysconfig import get_path
import math
import traceback
from concurrent.futures.process import _MAX_WINDOWS_WORKERS

from pathlib import Path
from collections import namedtuple
from types import ModuleType as new_module
from dataclasses import dataclass

import click
from click import Option, Argument
from doit.cmd_base import ModuleTaskLoader
from doit.reporter import ZeroReporter
from doit.exceptions import TaskError
from doit.api import run_tasks
from doit import task_params
from pydevtool.cli import UnifiedContext, CliGroup, Task
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from rich_click import rich_click
# 定义全局配置字典，包含程序运行时的详细度和最小版本要求
DOIT_CONFIG = {
    'verbosity': 2,
    'minversion': '0.36.0',
}

# 定义控制台主题样式，用于丰富命令行界面的视觉效果
console_theme = Theme({
    "cmd": "italic gray50",
})

# 根据操作系统平台设置表情符号类，Windows下为大于号，其他系统为电脑图标
if sys.platform == 'win32':
    class EMOJI:
        cmd = ">"
else:
    class EMOJI:
        cmd = ":computer:"

# 设置富文本命令行库的静态错误建议样式
rich_click.STYLE_ERRORS_SUGGESTION = "yellow italic"

# 设置富文本命令行库显示参数的选项
rich_click.SHOW_ARGUMENTS = True

# 设置富文本命令行库不使用参数分组的选项
rich_click.GROUP_ARGUMENTS_OPTIONS = False

# 设置富文本命令行库显示参数元数据列的选项
rich_click.SHOW_METAVARS_COLUMN = True

# 设置富文本命令行库使用Markdown语法的选项
rich_click.USE_MARKDOWN = True

# 设置富文本命令行库的选项分组
rich_click.OPTION_GROUPS = {
    "dev.py": [
        {
            "name": "Options",
            "options": [
                "--help", "--build-dir", "--no-build", "--install-prefix"],
        },
    ],

    "dev.py test": [
        {
            "name": "Options",
            "options": ["--help", "--verbose", "--parallel", "--coverage",
                        "--durations"],
        },
        {
            "name": "Options: test selection",
            "options": ["--submodule", "--tests", "--mode"],
        },
    ],
}

# 设置富文本命令行库的命令分组
rich_click.COMMAND_GROUPS = {
    "dev.py": [
        {
            "name": "build & testing",
            "commands": ["build", "test"],
        },
        {
            "name": "static checkers",
            "commands": ["lint", "mypy"],
        },
        {
            "name": "environments",
            "commands": ["shell", "python", "ipython", "show_PYTHONPATH"],
        },
        {
            "name": "documentation",
            "commands": ["doc", "refguide-check", "smoke-docs", "smoke-tutorial"],
        },
        {
            "name": "release",
            "commands": ["notes", "authors"],
        },
        {
            "name": "benchmarking",
            "commands": ["bench"],
        },
    ]
}

# 定义自定义的错误仅报告器类，继承自ZeroReporter类
class ErrorOnlyReporter(ZeroReporter):
    desc = """Report errors only"""

    # 定义处理运行时错误的方法，输出错误信息到控制台
    def runtime_error(self, msg):
        console = Console()
        console.print("[red bold] msg")

    # 定义添加任务失败信息的方法，根据情况输出任务错误或者异常信息到控制台
    def add_failure(self, task, fail_info):
        console = Console()
        if isinstance(fail_info, TaskError):
            console.print(f'[red]Task Error - {task.name}'
                          f' => {fail_info.message}')
        if fail_info.traceback:
            console.print(Panel(
                "".join(fail_info.traceback),
                title=f"{task.name}",
                subtitle=fail_info.message,
                border_style="red",
            ))

# 定义统一上下文对象，包含构建目录和相关选项的信息
CONTEXT = UnifiedContext({
    'build_dir': Option(
        ['--build-dir'], metavar='BUILD_DIR',
        default='build', show_default=True,
        help=':wrench: Relative path to the build directory.'),
    'no_build': Option(
        ["--no-build", "-n"], default=False, is_flag=True,
        help=(":wrench: Do not build the project"
              " (note event python only modification require build).")),
    'install_prefix': Option(
        ['--install-prefix'], default=None, metavar='INSTALL_DIR',
        help=(":wrench: Relative path to the install directory."
              " Default is <build-dir>-install.")),
})
# 定义一个函数 run_doit_task，用于运行给定任务集合，并返回执行结果
def run_doit_task(tasks):
    """
      :param tasks: (dict) task_name -> {options}
    """
    # 使用 ModuleTaskLoader 加载全局变量作为任务模块的环境
    loader = ModuleTaskLoader(globals())
    # 配置 doit 的一些选项，包括详细度为 2 和使用 ErrorOnlyReporter 报告错误
    doit_config = {
        'verbosity': 2,
        'reporter': ErrorOnlyReporter,
    }
    # 调用 run_tasks 函数执行任务，传入 loader、tasks 和额外的配置项
    return run_tasks(loader, tasks, extra_config={'GLOBAL': doit_config})


# 定义一个 CLI 类，继承自 CliGroup，用于组织命令行接口
class CLI(CliGroup):
    context = CONTEXT
    run_doit_task = run_doit_task


# 创建命令行接口的主命令组，使用 CLI 类作为命令行接口的基类
@click.group(cls=CLI)
@click.pass_context
def cli(ctx, **kwargs):
    """Developer Tool for SciPy

    \bCommands that require a built/installed instance are marked with :wrench:.


    \b**python dev.py --build-dir my-build test -s stats**

    """
    # 更新命令行上下文 ctx，传入 kwargs 参数
    CLI.update_context(ctx, kwargs)


# 定义项目模块名和项目根文件列表
PROJECT_MODULE = "scipy"
PROJECT_ROOT_FILES = ['scipy', 'LICENSE.txt', 'meson.build']


# 定义 Dirs 类，用于管理项目的各个目录路径
@dataclass
class Dirs:
    """
        root:
            Directory where src, build config and tools are located
            (and this file)
        build:
            Directory where build output files (i.e. *.o) are saved
        install:
            Directory where .so from build and .py from src are put together.
        site:
            Directory where the built SciPy version was installed.
            This is a custom prefix, followed by a relative path matching
            the one the system would use for the site-packages of the active
            Python interpreter.
    """
    # 所有路径均为绝对路径
    root: Path
    build: Path
    installed: Path
    site: Path  # <install>/lib/python<version>/site-packages

    def __init__(self, args=None):
        """:params args: object like Context(build_dir, install_prefix)"""
        # 设置 root 为当前文件所在目录的绝对路径
        self.root = Path(__file__).parent.absolute()
        if not args:
            return

        # 解析并设置 build 目录路径
        self.build = Path(args.build_dir).resolve()
        # 如果指定了 install_prefix，则设置 installed 为其绝对路径，否则在 build 目录下创建以 build 名称加后缀 '-install' 的目录
        if args.install_prefix:
            self.installed = Path(args.install_prefix).resolve()
        else:
            self.installed = self.build.parent / (self.build.stem + "-install")

        # 如果运行环境为 Windows 且 Python 版本低于 3.10，则使用绝对路径处理 pathlib 的 bug
        if sys.platform == 'win32' and sys.version_info < (3, 10):
            self.build = Path(os.path.abspath(self.build))
            self.installed = Path(os.path.abspath(self.installed))

        # 获取 site-packages 目录的相对路径，格式为 'lib/python<version>/site-packages'
        self.site = self.get_site_packages()

    def add_sys_path(self):
        """Add site dir to sys.path / PYTHONPATH"""
        # 将 site 目录路径添加到 sys.path / PYTHONPATH 环境变量中
        site_dir = str(self.site)
        sys.path.insert(0, site_dir)
        os.environ['PYTHONPATH'] = \
            os.pathsep.join((site_dir, os.environ.get('PYTHONPATH', '')))
    # 获取站点包路径，根据当前 Python 版本和操作系统环境的不同选择不同的路径
    def get_site_packages(self):
        """
        Depending on whether we have debian python or not,
        return dist_packages path or site_packages path.
        """
        # 检查 Python 版本是否大于等于 3.12
        if sys.version_info >= (3, 12):
            # 如果是 Python 3.12 及以上版本，使用 platlib 路径
            plat_path = Path(get_path('platlib'))
        else:
            # 对于 Python < 3.12，需要使用 distutils 探测 Meson 安装路径
            # 忽略 DeprecationWarning 警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                from distutils import dist
                from distutils.command.install import INSTALL_SCHEMES
            # 检查是否为 Debian 修改过的 Python
            if 'deb_system' in INSTALL_SCHEMES:
                # 使用 Debian 修改过的 Python 安装路径
                install_cmd = dist.Distribution().get_command_obj('install')
                install_cmd.select_scheme('deb_system')
                install_cmd.finalize_options()
                plat_path = Path(install_cmd.install_platlib)
            else:
                # 默认情况下仍然使用 platlib 路径
                plat_path = Path(get_path('platlib'))
        # 返回安装目录下的 plat_path 相对路径
        return self.installed / plat_path.relative_to(sys.exec_prefix)
@contextlib.contextmanager
def working_dir(new_dir):
    # 获取当前工作目录
    current_dir = os.getcwd()
    try:
        # 切换到新的工作目录
        os.chdir(new_dir)
        # 使用 yield 将控制权交给调用者，执行 with 语句块的内容
        yield
    finally:
        # 最终确保恢复到原始工作目录
        os.chdir(current_dir)


def import_module_from_path(mod_name, mod_path):
    """Import module with name `mod_name` from file path `mod_path`"""
    # 根据模块名和文件路径创建模块的规范对象
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    # 根据规范对象创建并加载模块
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_test_runner(project_module):
    """
    get Test Runner from locally installed/built project
    """
    # 动态导入项目中的模块
    __import__(project_module)
    # 获取测试对象
    test = sys.modules[project_module].test
    # 获取项目模块的版本信息
    version = sys.modules[project_module].__version__
    # 获取项目模块的文件路径，并转换为绝对路径
    mod_path = sys.modules[project_module].__file__
    mod_path = os.path.abspath(os.path.join(os.path.dirname(mod_path)))
    return test, version, mod_path


############

@cli.cls_cmd('build')
class Build(Task):
    """:wrench: Build & install package on path.

    \b
    ```shell-session
    Examples:

    $ python dev.py build --asan ;
        ASAN_OPTIONS=detect_leaks=0:symbolize=1:strict_init_order=true
        LD_PRELOAD=$(gcc --print-file-name=libasan.so)
        python dev.py test -v -t
        ./scipy/ndimage/tests/test_morphology.py -- -s
    ```
    """
    # 使用 CONTEXT 上下文对象初始化上下文
    ctx = CONTEXT

    # 设置编译时的选项
    werror = Option(
        ['--werror'], default=False, is_flag=True,
        help="Treat warnings as errors")
    gcov = Option(
        ['--gcov'], default=False, is_flag=True,
        help="enable C code coverage via gcov (requires GCC)."
             "gcov output goes to build/**/*.gc*")
    asan = Option(
        ['--asan'], default=False, is_flag=True,
        help=("Build and run with AddressSanitizer support. "
              "Note: the build system doesn't check whether "
              "the project is already compiled with ASan. "
              "If not, you need to do a clean build (delete "
              "build and build-install directories)."))
    debug = Option(
        ['--debug', '-d'], default=False, is_flag=True, help="Debug build")
    parallel = Option(
        ['--parallel', '-j'], default=None, metavar='N_JOBS',
        help=("Number of parallel jobs for building. "
              "This defaults to the number of available physical CPU cores"))
    setup_args = Option(
        ['--setup-args', '-C'], default=[], multiple=True,
        help=("Pass along one or more arguments to `meson setup` "
              "Repeat the `-C` in case of multiple arguments."))
    show_build_log = Option(
        ['--show-build-log'], default=False, is_flag=True,
        help="Show build output rather than using a log file")
    with_scipy_openblas = Option(
        ['--with-scipy-openblas'], default=False, is_flag=True,
        help=("If set, use the `scipy-openblas32` wheel installed into the "
              "current environment as the BLAS/LAPACK to build against."))
    # 创建一个带有选项参数的对象，并设置默认值为 False，表示不启用加速
    with_accelerate = Option(
        ['--with-accelerate'], default=False, is_flag=True,
        help=("If set, use `Accelerate` as the BLAS/LAPACK to build against."
              " Takes precedence over -with-scipy-openblas (macOS only)")
    )
    
    # 创建一个带有选项参数的对象，并设置默认值为字符串"runtime,python-runtime,tests,devel"
    tags = Option(
        ['--tags'], default="runtime,python-runtime,tests,devel",
        show_default=True, help="Install tags to be used by meson."
    )
    
    # 定义一个类方法，用于标识后续的方法是一个类方法
    @classmethod
    def setup_build(cls, dirs, args):
        """
        设置 meson-build

        检查项目根目录下是否存在必需的文件，若不存在则打印错误信息并退出
        """
        for fn in PROJECT_ROOT_FILES:
            if not (dirs.root / fn).exists():
                print("To build the project, run dev.py in "
                      "git checkout or unpacked source")
                sys.exit(1)

        # 复制当前环境变量字典
        env = dict(os.environ)
        
        # 设置 meson 命令及其参数
        cmd = ["meson", "setup", dirs.build, "--prefix", dirs.installed]
        build_dir = dirs.build
        run_dir = Path()

        # 如果构建目录已存在且不包含 'meson-info' 文件夹，则抛出运行时错误
        if build_dir.exists() and not (build_dir / 'meson-info').exists():
            if list(build_dir.iterdir()):
                raise RuntimeError("Can't build into non-empty directory "
                                   f"'{build_dir.absolute()}'")

        # 对于 Cygwin 平台，设置特定的 BLAS 和 LAPACK 库选项
        if sys.platform == "cygwin":
            cmd.extend(["-Dlapack=lapack", "-Dblas=blas"])

        # 检查构建选项文件是否存在，并加载其中的选项
        build_options_file = (
            build_dir / "meson-info" / "intro-buildoptions.json")
        if build_options_file.exists():
            with open(build_options_file) as f:
                build_options = json.load(f)
            installdir = None
            for option in build_options:
                if option["name"] == "prefix":
                    installdir = option["value"]
                    break
            # 如果安装目录与预期不符，则重新配置 Meson 并指定安装目录
            if installdir != str(dirs.installed):
                run_dir = build_dir
                cmd = ["meson", "setup", "--reconfigure",
                       "--prefix", str(dirs.installed)]
            else:
                return
        
        # 根据参数设置 Meson 的额外选项
        if args.werror:
            cmd += ["--werror"]
        if args.gcov:
            cmd += ['-Db_coverage=true']
        if args.asan:
            cmd += ['-Db_sanitize=address,undefined']
        if args.setup_args:
            cmd += [str(arg) for arg in args.setup_args]
        
        # 根据参数设置 BLAS 库选项
        if args.with_accelerate:
            cmd += ["-Dblas=accelerate"]
        elif args.with_scipy_openblas:
            # 如果使用 scipy_openblas，则配置相关设置
            cls.configure_scipy_openblas()
            # 更新 PKG_CONFIG_PATH 环境变量
            env['PKG_CONFIG_PATH'] = os.pathsep.join([
                    os.getcwd(),
                    env.get('PKG_CONFIG_PATH', '')
                    ])

        # 输出设置的 Meson 构建命令
        cmd_str = ' '.join([str(p) for p in cmd])
        cls.console.print(f"{EMOJI.cmd} [cmd] {cmd_str}")

        # 执行 Meson 构建命令，并返回执行结果
        ret = subprocess.call(cmd, env=env, cwd=run_dir)
        if ret == 0:
            print("Meson build setup OK")
        else:
            print("Meson build setup failed!")
            sys.exit(1)

        # 返回更新后的环境变量字典
        return env
    # 定义一个类方法，用于构建项目的开发版本
    def build_project(cls, dirs, args, env):
        """
        Build a dev version of the project.
        """
        # 构建 Ninja 命令行
        cmd = ["ninja", "-C", str(dirs.build)]
        
        # 如果未指定并行编译数，则使用物理核心数代替 ninja 默认的 2N+2，
        # 以避免内存不足问题（参见 gh-17941 和 gh-18443）
        if args.parallel is None:
            n_cores = cpu_count(only_physical_cores=True)  # 获取物理核心数
            cmd += [f"-j{n_cores}"]
        else:
            cmd += ["-j", str(args.parallel)]  # 指定并行编译数

        # 构建命令行的字符串表示
        cmd_str = ' '.join([str(p) for p in cmd])
        
        # 打印构建命令行到控制台
        cls.console.print(f"{EMOJI.cmd} [cmd] {cmd_str}")
        
        # 执行构建命令，返回执行结果
        ret = subprocess.call(cmd, env=env, cwd=dirs.root)

        # 根据返回值判断构建是否成功
        if ret == 0:
            print("Build OK")  # 构建成功
        else:
            print("Build failed!")  # 构建失败
            sys.exit(1)  # 退出程序，返回状态码 1（非正常退出）

    @classmethod
    def install_project(cls, dirs, args):
        """
        Installs the project after building.
        """
        # 检查安装目录是否存在
        if dirs.installed.exists():
            # 统计安装目录下非空文件数量
            non_empty = len(os.listdir(dirs.installed))
            # 如果目录非空且目标站点目录不存在，则抛出运行时异常
            if non_empty and not dirs.site.exists():
                raise RuntimeError("Can't install in non-empty directory: "
                                   f"'{dirs.installed}'")
        
        # 构建安装命令
        cmd = ["meson", "install", "-C", args.build_dir,
               "--only-changed", "--tags", args.tags]
        
        # 指定日志文件路径
        log_filename = dirs.root / 'meson-install.log'
        
        # 记录安装开始时间
        start_time = datetime.datetime.now()
        
        # 将命令列表转换为字符串
        cmd_str = ' '.join([str(p) for p in cmd])
        
        # 打印安装命令信息
        cls.console.print(f"{EMOJI.cmd} [cmd] {cmd_str}")
        
        # 如果设置了显示构建日志，则直接调用 subprocess 执行命令
        if args.show_build_log:
            ret = subprocess.call(cmd, cwd=dirs.root)
        else:
            # 否则，打印提示信息，并将输出重定向到日志文件
            print("Installing, see meson-install.log...")
            with open(log_filename, 'w') as log:
                p = subprocess.Popen(cmd, stdout=log, stderr=log,
                                     cwd=dirs.root)

            try:
                # 等待进程完成，并在日志文件增长时定期打印进度信息
                last_blip = time.time()
                last_log_size = os.stat(log_filename).st_size
                while p.poll() is None:
                    time.sleep(0.5)
                    if time.time() - last_blip > 60:
                        log_size = os.stat(log_filename).st_size
                        if log_size > last_log_size:
                            elapsed = datetime.datetime.now() - start_time
                            print(f"    ... installation in progress ({elapsed} "
                                  "elapsed)")
                            last_blip = time.time()
                            last_log_size = log_size

                ret = p.wait()
            except:  # noqa: E722
                # 发生异常时终止进程并抛出异常
                p.terminate()
                raise
        
        # 计算安装总耗时
        elapsed = datetime.datetime.now() - start_time
        
        # 如果安装过程返回非零状态码，则打印日志内容（如果未设置显示构建日志）并退出
        if ret != 0:
            if not args.show_build_log:
                with open(log_filename) as f:
                    print(f.read())
            print(f"Installation failed! ({elapsed} elapsed)")
            sys.exit(1)
        
        # 在安装目录下创建.gitignore文件，忽略所有文件
        with open(dirs.installed / ".gitignore", "w") as f:
            f.write("*")
        
        # 如果运行环境为Cygwin，执行重新定位命令来调整DLL库的地址空间
        if sys.platform == "cygwin":
            rebase_cmd = ["/usr/bin/rebase", "--database", "--oblivious"]
            rebase_cmd.extend(Path(dirs.installed).glob("**/*.dll"))
            subprocess.check_call(rebase_cmd)
        
        # 打印安装成功信息
        print("Installation OK")
        
        # 方法结束，无返回值
        return
    def configure_scipy_openblas(self, blas_variant='32'):
        """Create scipy-openblas.pc and scipy/_distributor_init_local.py

        Requires a pre-installed scipy-openblas32 wheel from PyPI.
        """
        # 获取当前工作目录
        basedir = os.getcwd()
        # 构建 pkg-config 文件的完整路径
        pkg_config_fname = os.path.join(basedir, "scipy-openblas.pc")

        # 如果 pkg-config 文件已存在，则直接返回
        if os.path.exists(pkg_config_fname):
            return None

        # 构建模块名，使用给定的 blas_variant
        module_name = f"scipy_openblas{blas_variant}"
        try:
            # 尝试导入指定的模块
            openblas = importlib.import_module(module_name)
        except ModuleNotFoundError:
            # 抛出异常，如果模块未找到
            raise RuntimeError(f"Importing '{module_name}' failed. "
                               "Make sure it is installed and reachable "
                               "by the current Python executable. You can "
                               f"install it via 'pip install {module_name}'.")

        # 构建 _distributor_init_local.py 文件的完整路径
        local = os.path.join(basedir, "scipy", "_distributor_init_local.py")
        # 写入导入模块的代码到 _distributor_init_local.py 文件中
        with open(local, "w", encoding="utf8") as fid:
            fid.write(f"import {module_name}\n")

        # 将 pkg-config 文件写入内容
        with open(pkg_config_fname, "w", encoding="utf8") as fid:
            fid.write(openblas.get_pkg_config())

    @classmethod
    def run(cls, add_path=False, **kwargs):
        # 更新 kwargs 以获取上下文中的参数
        kwargs.update(cls.ctx.get(kwargs))
        # 使用 namedtuple 创建 Args 对象
        Args = namedtuple('Args', [k for k in kwargs.keys()])
        args = Args(**kwargs)

        # 设置控制台主题
        cls.console = Console(theme=console_theme)
        # 创建 Dirs 对象
        dirs = Dirs(args)
        
        # 如果 args.no_build 为真，则跳过构建步骤
        if args.no_build:
            print("Skipping build")
        else:
            # 设置构建环境
            env = cls.setup_build(dirs, args)
            # 构建项目
            cls.build_project(dirs, args, env)
            # 安装项目
            cls.install_project(dirs, args)

        # 如果 add_path 为真，则将 site 添加到 sys.path 中
        if add_path:
            dirs.add_sys_path()
# 定义名为 Test 的类，继承自 Task 类，作为 CLI 命令 'test' 的处理器
@cli.cls_cmd('test')
class Test(Task):
    """:wrench: Run tests.

    \b
    ```python
    Examples:

    $ python dev.py test -s {SAMPLE_SUBMODULE}
    $ python dev.py test -t scipy.optimize.tests.test_minimize_constrained
    $ python dev.py test -s cluster -m full --durations 20
    $ python dev.py test -s stats -- --tb=line  # `--` passes next args to pytest
    $ python dev.py test -b numpy -b pytorch -s cluster
    ```
    """
    
    # 获取上下文 CONTEXT 的引用
    ctx = CONTEXT

    # 定义 verbose 参数选项，用于控制日志详细程度
    verbose = Option(
        ['--verbose', '-v'], default=False, is_flag=True,
        help="more verbosity")

    # 定义 coverage 参数选项，用于报告项目代码的覆盖率
    coverage = Option(
        ['--coverage', '-c'], default=False, is_flag=True,
        help=("report coverage of project code. "
              "HTML output goes under build/coverage"))

    # 定义 durations 参数选项，用于显示最慢测试的运行时间
    durations = Option(
        ['--durations', '-d'], default=None, metavar="NUM_TESTS",
        help="Show timing for the given number of slowest tests"
    )

    # 定义 submodule 参数选项，用于指定要运行测试的子模块
    submodule = Option(
        ['--submodule', '-s'], default=None, metavar='MODULE_NAME',
        help="Submodule whose tests to run (cluster, constants, ...)"
    )

    # 定义 tests 参数选项，用于指定要运行的具体测试
    tests = Option(
        ['--tests', '-t'], default=None, multiple=True, metavar='TESTS',
        help='Specify tests to run'
    )

    # 定义 mode 参数选项，用于指定测试运行的模式（快速或全面）
    mode = Option(
        ['--mode', '-m'], default='fast', metavar='MODE', show_default=True,
        help=("'fast', 'full', or something that could be passed to "
              "`pytest -m` as a marker expression"))

    # 定义 parallel 参数选项，用于指定测试的并行作业数
    parallel = Option(
        ['--parallel', '-j'], default=1, metavar='N_JOBS',
        help="Number of parallel jobs for testing"
    )

    # 定义 array_api_backend 参数选项，用于指定数组 API 的后端
    array_api_backend = Option(
        ['--array-api-backend', '-b'], default=None, metavar='ARRAY_BACKEND',
        multiple=True,
        help=(
            "Array API backend "
            "('all', 'numpy', 'pytorch', 'cupy', 'array_api_strict', 'jax.numpy')."
        )
    )

    # 定义 pytest_args 参数选项，用于传递给 pytest 的额外参数
    pytest_args = Argument(
        ['pytest_args'], nargs=-1, metavar='PYTEST-ARGS', required=False
    )

    # 定义任务的元数据，包括依赖于 'build' 任务
    TASK_META = {
        'task_dep': ['build'],
    }

    @classmethod
    def scipy_tests(cls, args, pytest_args):
        # 创建Dirs对象，用于管理目录
        dirs = Dirs(args)
        # 将系统路径添加到Dirs对象中
        dirs.add_sys_path()
        # 打印SciPy开发安装路径
        print(f"SciPy from development installed path at: {dirs.site}")

        # FIXME: support pos-args with doit
        # 复制pytest_args以防止修改原始参数列表
        extra_argv = list(pytest_args[:]) if pytest_args else []
        # 如果extra_argv不为空且第一个参数是'--'，则移除第一个参数
        if extra_argv and extra_argv[0] == '--':
            extra_argv = extra_argv[1:]

        # 如果设置了coverage选项
        if args.coverage:
            # 设置coverage输出目录和文件路径
            dst_dir = dirs.root / args.build_dir / 'coverage'
            fn = dst_dir / 'coverage_html.js'
            # 如果目录存在且文件存在，则删除目录
            if dst_dir.is_dir() and fn.is_file():
                shutil.rmtree(dst_dir)
            # 添加coverage报告选项
            extra_argv += ['--cov-report=html:' + str(dst_dir)]
            # 拷贝.coveragerc文件到dirs.site目录
            shutil.copyfile(dirs.root / '.coveragerc',
                            dirs.site / '.coveragerc')

        # 如果设置了durations选项，则添加到extra_argv中
        if args.durations:
            extra_argv += ['--durations', args.durations]

        # 将选项转换为测试选择
        if args.submodule:
            tests = [PROJECT_MODULE + "." + args.submodule]
        elif args.tests:
            tests = args.tests
        else:
            tests = None

        # 如果设置了array_api_backend选项，则设置环境变量SCIPY_ARRAY_API
        if len(args.array_api_backend) != 0:
            os.environ['SCIPY_ARRAY_API'] = json.dumps(list(args.array_api_backend))

        # 获取测试运行器、版本和模块路径
        runner, version, mod_path = get_test_runner(PROJECT_MODULE)
        # FIXME: changing CWD is not a good practice
        # 在dirs.site目录下运行测试
        with working_dir(dirs.site):
            # 打印正在运行的测试信息
            print(f"Running tests for {PROJECT_MODULE} version:{version}, "
                  f"installed at:{mod_path}")
            # 设置runner的verbosity，将bool转换为int
            verbose = int(args.verbose) + 1
            # 运行测试，并返回结果
            result = runner(
                args.mode,
                verbose=verbose,
                extra_argv=extra_argv,
                doctests=False,
                coverage=args.coverage,
                tests=tests,
                parallel=args.parallel)
        return result

    @classmethod
    def run(cls, pytest_args, **kwargs):
        """run unit-tests"""
        # 更新kwargs参数
        kwargs.update(cls.ctx.get())
        # 创建Args命名元组
        Args = namedtuple('Args', [k for k in kwargs.keys()])
        # 使用kwargs创建args对象
        args = Args(**kwargs)
        # 调用scipy_tests方法运行测试，并返回结果
        return cls.scipy_tests(args, pytest_args)
# 定义一个名为 'smoke-docs' 的命令行接口命令，继承自 Task 类
@cli.cls_cmd('smoke-docs')
class SmokeDocs(Task):
    # XXX 这本质上是 Task 类的复制粘贴。考虑消除重复。
    # 设置上下文为全局变量 CONTEXT
    ctx = CONTEXT

    # 定义一个布尔类型选项 verbose，用于控制输出更多详细信息
    verbose = Option(
        ['--verbose', '-v'], default=False, is_flag=True,
        help="more verbosity")

    # 定义一个整数类型选项 durations，用于显示最慢测试的执行时间
    durations = Option(
        ['--durations', '-d'], default=None, metavar="NUM_TESTS",
        help="Show timing for the given number of slowest tests"
    )

    # 定义一个字符串类型选项 submodule，用于指定要运行测试的子模块名称
    submodule = Option(
        ['--submodule', '-s'], default=None, metavar='MODULE_NAME',
        help="Submodule whose tests to run (cluster, constants, ...)"
    )

    # 定义一个列表类型选项 tests，允许指定要运行的具体测试的名称
    tests = Option(
        ['--tests', '-t'], default=None, multiple=True, metavar='TESTS',
        help='Specify tests to run'
    )

    # 定义一个整数类型选项 parallel，用于指定并行测试的作业数量
    parallel = Option(
        ['--parallel', '-j'], default=1, metavar='N_JOBS',
        help="Number of parallel jobs for testing"
    )

    # 定义一个位置参数选项 pytest_args，用于接收 pytest 的额外参数
    # 此选项没有 help 参数，用于消耗所有的 `-- arg1 arg2 arg3` 形式的参数
    pytest_args = Argument(
        ['pytest_args'], nargs=-1, metavar='PYTEST-ARGS', required=False
    )

    # 定义任务的元数据信息
    TASK_META = {
        'task_dep': ['build'],
    }

    @classmethod
    def scipy_tests(cls, args, pytest_args):
        # 创建 Dirs 对象，用于管理目录信息
        dirs = Dirs(args)
        # 将系统路径添加到 Dirs 对象中
        dirs.add_sys_path()
        # 打印 SciPy 开发安装路径
        print(f"SciPy from development installed path at: {dirs.site}")

        # 防止后续出现模糊错误；参考 https://github.com/numpy/numpy/pull/26691/
        if not importlib.util.find_spec("scipy_doctest"):
            raise ModuleNotFoundError("Please install scipy-doctest")

        # FIXME: 支持使用 doit 进行额外参数的处理
        extra_argv = list(pytest_args[:]) if pytest_args else []
        if extra_argv and extra_argv[0] == '--':
            extra_argv = extra_argv[1:]

        # 如果指定了持续时间选项，则加入到额外参数中
        if args.durations:
            extra_argv += ['--durations', args.durations]

        # 将选项转换为测试选择
        if args.submodule:
            tests = [PROJECT_MODULE + "." + args.submodule]
        elif args.tests:
            tests = args.tests
        else:
            tests = None

        # 请求进行文档测试；除非指定具体文件路径，否则使用 strategy=api，并关闭断言重写
        extra_argv += ["--doctest-modules", "--assert=plain"]
        if not args.tests:
            extra_argv += ['--doctest-collect=api']

        # 获取测试运行器和相关信息
        runner, version, mod_path = get_test_runner(PROJECT_MODULE)
        # FIXME: 修改当前工作目录不是良好的实践
        with working_dir(dirs.site):
            # 打印正在运行的项目模块的测试信息
            print(f"Running tests for {PROJECT_MODULE} version:{version}, "
                  f"installed at:{mod_path}")
            # 设置运行器的详细程度 - 将布尔值转换为整数
            verbose = int(args.verbose) + 1
            # 运行测试
            result = runner(
                "fast",
                verbose=verbose,
                extra_argv=extra_argv,
                doctests=True,
                coverage=False,
                tests=tests,
                parallel=args.parallel)
        # 返回测试结果
        return result

    @classmethod
    def run(cls, pytest_args, **kwargs):
        """run unit-tests"""
        # 更新 kwargs 中的上下文信息
        kwargs.update(cls.ctx.get())
        # 创建 Args 命名元组，用于存储 kwargs 中的参数
        Args = namedtuple('Args', [k for k in kwargs.keys()])
        args = Args(**kwargs)
        # 调用 scipy_tests 方法执行测试
        return cls.scipy_tests(args, pytest_args)
@cli.cls_cmd('smoke-tutorials')
class SmokeTutorials(Task):
    """:wrench: Run smoke-tests on tutorial files."""

    # 上下文环境
    ctx = CONTEXT

    # 指定要进行 smoke 测试的 *rst 文件
    tests = Option(
        ['--tests', '-t'], default=None, multiple=True, metavar='TESTS',
        help='Specify *rst files to smoke test')

    # 是否显示详细输出
    verbose = Option(
        ['--verbose', '-v'], default=False, is_flag=True, help="verbosity")

    # 传递给 pytest 的参数
    pytest_args = Argument(
        ['pytest_args'], nargs=-1, metavar='PYTEST-ARGS', required=False
    )

    @classmethod
    def task_meta(cls, **kwargs):
        # 更新 kwargs
        kwargs.update(cls.ctx.get())

        # 创建具名元组 Args
        Args = namedtuple('Args', [k for k in kwargs.keys()])
        args = Args(**kwargs)

        # 创建目录对象
        dirs = Dirs(args)

        # 初始化命令列表
        cmd = ['pytest']

        # 如果指定了测试文件，则加入命令列表
        if args.tests:
            cmd += list(args.tests)
        else:
            # 否则使用默认测试文件路径及选项
            cmd += ['doc/source/tutorial', '--doctest-glob=*rst']

        # 如果启用了详细模式，添加相应选项
        if args.verbose:
            cmd += ['-v']

        # 获取并移除 pytest_args 参数
        pytest_args = kwargs.pop('pytest_args', None)
        extra_argv = list(pytest_args[:]) if pytest_args else []

        # 如果 extra_argv 存在且以 '--' 开头，则移除第一个元素
        if extra_argv and extra_argv[0] == '--':
            extra_argv = extra_argv[1:]

        # 将 extra_argv 添加到命令列表末尾
        cmd += extra_argv

        # 将命令列表转换为字符串
        cmd_str = ' '.join(cmd)

        # 返回任务元数据
        return {
            'actions': [f'env PYTHONPATH={dirs.site} {cmd_str}'],  # 执行命令的动作
            'task_dep': ['build'],  # 依赖的任务
            'io': {'capture': False},  # 输入输出设置
        }


@cli.cls_cmd('bench')
class Bench(Task):
    """:wrench: Run benchmarks.

    \b
    ```python
     Examples:

    $ python dev.py bench -t integrate.SolveBVP
    $ python dev.py bench -t linalg.Norm
    $ python dev.py bench --compare main
    ```
    """

    # 上下文环境
    ctx = CONTEXT

    # 任务元数据
    TASK_META = {
        'task_dep': ['build'],  # 依赖的任务
    }

    # 指定子模块进行测试的选项
    submodule = Option(
        ['--submodule', '-s'], default=None, metavar='SUBMODULE',
        help="Submodule whose tests to run (cluster, constants, ...)")

    # 指定要运行的测试
    tests = Option(
        ['--tests', '-t'], default=None, multiple=True,
        metavar='TESTS', help='Specify tests to run')

    # 比较基准结果的选项
    compare = Option(
        ['--compare', '-c'], default=None, metavar='COMPARE', multiple=True,
        help=(
            "Compare benchmark results of current HEAD to BEFORE. "
            "Use an additional --bench COMMIT to override HEAD with COMMIT. "
            "Note that you need to commit your changes first!"))

    @staticmethod
    def run_asv(dirs, cmd):
        # 额外的路径用于查找ccache和f90cache程序
        EXTRA_PATH = ['/usr/lib/ccache', '/usr/lib/f90cache',
                      '/usr/local/lib/ccache', '/usr/local/lib/f90cache']
        # 拼接出benchmarks目录的路径
        bench_dir = dirs.root / 'benchmarks'
        # 将benchmarks目录添加到系统路径的最前面
        sys.path.insert(0, str(bench_dir))
        
        # 创建一个新的环境变量字典，复制当前环境
        env = dict(os.environ)
        # 将额外路径加入到PATH环境变量中，并保留原有路径
        env['PATH'] = os.pathsep.join(EXTRA_PATH +
                                      env.get('PATH', '').split(os.pathsep))
        
        # 控制BLAS/LAPACK的线程数
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['MKL_NUM_THREADS'] = '1'

        # 限制内存使用
        from benchmarks.common import set_mem_rlimit
        try:
            set_mem_rlimit()  # 调用设置内存限制的函数
        except (ImportError, RuntimeError):
            pass
        
        try:
            # 在benchmarks目录中以env环境变量运行给定的命令
            return subprocess.call(cmd, env=env, cwd=bench_dir)
        except OSError as err:
            # 处理命令不存在的异常
            if err.errno == errno.ENOENT:
                cmd_str = " ".join(cmd)
                # 输出错误消息和建议的安装链接
                print(f"Error when running '{cmd_str}': {err}\n")
                print("You need to install Airspeed Velocity "
                      "(https://airspeed-velocity.github.io/asv/)")
                print("to run Scipy benchmarks")
                return 1
            raise  # 抛出其他OSError异常

    @classmethod
    def scipy_bench(cls, args):
        # 创建Dirs对象，用于管理目录路径
        dirs = Dirs(args)
        # 添加系统路径到Dirs对象中
        dirs.add_sys_path()
        # 打印SciPy开发安装路径
        print(f"SciPy from development installed path at: {dirs.site}")
        # 进入工作目录到dirs.site指定的路径
        with working_dir(dirs.site):
            # 获取测试运行器、版本号和模块路径
            runner, version, mod_path = get_test_runner(PROJECT_MODULE)
            # 初始化额外的命令行参数列表
            extra_argv = []
            # 如果指定了测试名称，添加到额外参数列表
            if args.tests:
                extra_argv.append(args.tests)
            # 如果指定了子模块，添加到额外参数列表
            if args.submodule:
                extra_argv.append([args.submodule])

            # 处理benchmark参数
            bench_args = []
            for a in extra_argv:
                bench_args.extend(['--bench', ' '.join(str(x) for x in a)])
            # 如果不进行比较，运行SciPy版本的基准测试
            if not args.compare:
                print(f"Running benchmarks for Scipy version {version} at {mod_path}")
                # 组装运行基准测试的命令
                cmd = ['asv', 'run', '--dry-run', '--show-stderr',
                       '--python=same', '--quick'] + bench_args
                # 调用类方法运行asv命令，并返回状态码
                retval = cls.run_asv(dirs, cmd)
                # 退出程序，使用asv命令的返回值作为退出码
                sys.exit(retval)
            else:
                # 如果指定了要比较的提交版本
                if len(args.compare) == 1:
                    commit_a = args.compare[0]
                    commit_b = 'HEAD'
                elif len(args.compare) == 2:
                    commit_a, commit_b = args.compare
                else:
                    print("Too many commits to compare benchmarks for")

                # 检查是否有未提交的文件
                if commit_b == 'HEAD':
                    r1 = subprocess.call(['git', 'diff-index', '--quiet',
                                          '--cached', 'HEAD'])
                    r2 = subprocess.call(['git', 'diff-files', '--quiet'])
                    # 如果存在未提交的文件，输出警告信息
                    if r1 != 0 or r2 != 0:
                        print("*" * 80)
                        print("WARNING: you have uncommitted changes --- "
                              "these will NOT be benchmarked!")
                        print("*" * 80)

                # 获取提交版本的commit id（如果commit_b为HEAD，则为当前仓库本地）
                p = subprocess.Popen(['git', 'rev-parse', commit_b],
                                     stdout=subprocess.PIPE)
                out, err = p.communicate()
                commit_b = out.strip()

                p = subprocess.Popen(['git', 'rev-parse', commit_a],
                                     stdout=subprocess.PIPE)
                out, err = p.communicate()
                commit_a = out.strip()

                # 组装比较基准测试的命令
                cmd_compare = [
                    'asv', 'continuous', '--show-stderr', '--factor', '1.05',
                    '--quick', commit_a, commit_b
                ] + bench_args
                # 调用类方法运行asv命令比较基准测试
                cls.run_asv(dirs, cmd_compare)
                # 退出程序，使用1作为退出码
                sys.exit(1)

    @classmethod
    def run(cls, **kwargs):
        """run benchmark"""
        # 更新kwargs，使用cls.ctx.get()的上下文信息
        kwargs.update(cls.ctx.get())
        # 创建命名元组Args，用于传递参数
        Args = namedtuple('Args', [k for k in kwargs.keys()])
        args = Args(**kwargs)
        # 调用scipy_bench方法，传入Args命名元组对象作为参数
        cls.scipy_bench(args)
###################
# linters

# 打印正在运行的命令到标准输出
def emit_cmdstr(cmd):
    """Print the command that's being run to stdout

    Note: cannot use this in the below tasks (yet), because as is these command
    strings are always echoed to the console, even if the command isn't run
    (but for example the `build` command is run).
    """
    console = Console(theme=console_theme)
    # 使用 [cmd] 方括号控制字体样式，通常使用斜体以便与其他标准输出内容区分
    console.print(f"{EMOJI.cmd} [cmd] {cmd}")


@task_params([{"name": "fix", "default": False}])
def task_lint(fix):
    # 仅对自分支分离后修改的文件进行更严格的配置检查
    # emit_cmdstr(os.path.join('tools', 'lint.py') + ' --diff-against main')
    cmd = str(Dirs().root / 'tools' / 'lint.py') + ' --diff-against=main'
    if fix:
        cmd += ' --fix'
    return {
        'basename': 'lint',
        'actions': [cmd],
        'doc': 'Lint only files modified since last commit (stricter rules)',
    }

@task_params([])
def task_check_python_h_first():
    # 仅对自分支分离后修改的文件进行更严格的配置检查
    # emit_cmdstr(os.path.join('tools', 'lint.py') + ' --diff-against main')
    cmd = "{!s} --diff-against=main".format(
        Dirs().root / 'tools' / 'check_python_h_first.py'
    )
    return {
        'basename': 'check_python_h_first',
        'actions': [cmd],
        'doc': (
            'Check Python.h order only files modified since last commit '
            '(stricter rules)'
        ),
    }


def task_unicode_check():
    # emit_cmdstr(os.path.join('tools', 'unicode-check.py'))
    return {
        'basename': 'unicode-check',
        'actions': [str(Dirs().root / 'tools' / 'unicode-check.py')],
        'doc': 'Check for disallowed Unicode characters in the SciPy Python '
               'and Cython source code.',
    }


def task_check_test_name():
    # emit_cmdstr(os.path.join('tools', 'check_test_name.py'))
    return {
        "basename": "check-testname",
        "actions": [str(Dirs().root / "tools" / "check_test_name.py")],
        "doc": "Check tests are correctly named so that pytest runs them."
    }


@cli.cls_cmd('lint')
class Lint:
    """:dash: Run linter on modified files and check for
    disallowed Unicode characters and possibly-invalid test names."""
    fix = Option(
        ['--fix'], default=False, is_flag=True, help='Attempt to auto-fix errors'
    )

    @classmethod
    def run(cls, fix):
        run_doit_task({
            'lint': {'fix': fix},
            'unicode-check': {},
            'check-testname': {},
            'check_python_h_first': {},
        })


@cli.cls_cmd('mypy')
class Mypy(Task):
    """:wrench: Run mypy on the codebase."""
    ctx = CONTEXT

    TASK_META = {
        'task_dep': ['build'],
    }

    @classmethod
    # 定义一个类方法 `run`，接受关键字参数 `kwargs`
    def run(cls, **kwargs):
        # 将类属性 `ctx` 中的内容更新到关键字参数中
        kwargs.update(cls.ctx.get())
        # 创建一个命名元组 `Args`，其字段为关键字参数中所有键的列表
        Args = namedtuple('Args', [k for k in kwargs.keys()])
        # 使用关键字参数初始化 `args` 对象
        args = Args(**kwargs)
        # 根据 `args` 创建 `Dirs` 类实例 `dirs`
        dirs = Dirs(args)

        try:
            # 尝试导入 `mypy.api` 模块
            import mypy.api
        except ImportError as e:
            # 如果导入失败，抛出 `RuntimeError` 异常，并提示安装 Mypy 的方法
            raise RuntimeError(
                "Mypy not found. Please install it by running "
                "pip install -r mypy_requirements.txt from the repo root"
            ) from e

        # 设置 `config` 变量为 `dirs.root / "mypy.ini"`
        config = dirs.root / "mypy.ini"
        # 设置 `check_path` 变量为 `PROJECT_MODULE`

        # 在 `dirs.site` 目录下执行以下代码块
        with working_dir(dirs.site):
            # 设置环境变量 `MYPY_FORCE_COLOR` 为 `'1'`，强制使用彩色输出
            os.environ['MYPY_FORCE_COLOR'] = '1'
            # 执行 `mypy.api.run` 命令，传入配置文件 `config` 和检查路径 `check_path`
            emit_cmdstr(f"mypy.api.run --config-file {config} {check_path}")
            # 运行 `mypy.api.run` 函数，传入以下参数列表
            report, errors, status = mypy.api.run([
                "--config-file",
                str(config),
                check_path,
            ])
        # 打印 `report` 内容，结束符设为空字符串以避免额外换行
        print(report, end='')
        # 将 `errors` 内容输出到标准错误流，保持末尾无额外换行
        print(errors, end='', file=sys.stderr)
        # 返回运行状态，`status == 0` 表示执行成功
        return status == 0
##########################################
# DOC

@cli.cls_cmd('doc')
class Doc(Task):
    """:wrench: Build documentation.

    TARGETS: Sphinx build targets [default: 'html']

    Running `python dev.py doc -j8 html` is equivalent to:
    1. Execute build command (skip by passing the global `-n` option).
    2. Set the PYTHONPATH environment variable
       (query with `python dev.py -n show_PYTHONPATH`).
    3. Run make on `doc/Makefile`, i.e.: `make -C doc -j8 TARGETS`

    To remove all generated documentation do: `python dev.py -n doc clean`
    """
    ctx = CONTEXT

    # 定义参数 args，接收任意数量的目标参数
    args = Argument(['args'], nargs=-1, metavar='TARGETS', required=False)
    
    # 定义选项 list_targets，用于列出文档构建的目标
    list_targets = Option(
        ['--list-targets', '-t'], default=False, is_flag=True,
        help='List doc targets',
    )
    
    # 定义选项 parallel，指定并行作业的数量
    parallel = Option(
        ['--parallel', '-j'], default=1, metavar='N_JOBS',
        help="Number of parallel jobs"
    )
    
    # 定义选项 no_cache，用于强制完全重新构建文档
    no_cache = Option(
        ['--no-cache'], default=False, is_flag=True,
        help="Forces a full rebuild of the docs. Note that this may be " + \
             "needed in order to make docstring changes in C/Cython files " + \
             "show up."
    )

    @classmethod
    def task_meta(cls, list_targets, parallel, no_cache, args, **kwargs):
        # 如果 list_targets 选项为真，则不设置默认目标，不依赖默认任务 'build'
        if list_targets:
            task_dep = []
            targets = ''
        else:
            # 否则设置默认依赖任务 'build'，并获取或设置构建的目标
            task_dep = ['build']
            targets = ' '.join(args) if args else 'html'

        kwargs.update(cls.ctx.get())
        Args = namedtuple('Args', [k for k in kwargs.keys()])
        build_args = Args(**kwargs)
        dirs = Dirs(build_args)

        make_params = [f'PYTHON="{sys.executable}"']
        
        # 根据 parallel 和 no_cache 选项设置 sphinxopts 和 make_params
        if parallel or no_cache:
            sphinxopts = ""
            if parallel:
                sphinxopts += f"-j{parallel} "
            if no_cache:
                sphinxopts += "-E"
            make_params.append(f'SPHINXOPTS="{sphinxopts}"')

        # 返回任务的配置字典
        return {
            'actions': [
                # 进入 doc/ 目录以避免导入本地 scipy
                (f'cd doc; env PYTHONPATH="{dirs.site}" '
                 f'make {" ".join(make_params)} {targets}'),
            ],
            'task_dep': task_dep,
            'io': {'capture': False},
        }


@cli.cls_cmd('refguide-check')
class RefguideCheck(Task):
    """:wrench: Run refguide check."""
    ctx = CONTEXT

    # 定义选项 submodule，指定要运行测试的子模块
    submodule = Option(
        ['--submodule', '-s'], default=None, metavar='SUBMODULE',
        help="Submodule whose tests to run (cluster, constants, ...)")
    
    # 定义选项 verbose，用于控制输出详细程度
    verbose = Option(
        ['--verbose', '-v'], default=False, is_flag=True, help="verbosity")

    @classmethod
    # 定义一个静态方法 `task_meta`，接受 `cls` 和关键字参数 `kwargs`
    def task_meta(cls, **kwargs):
        # 将 `cls.ctx.get()` 返回的内容更新到 `kwargs` 中
        kwargs.update(cls.ctx.get())
        # 使用关键字参数的键创建一个具名元组 `Args`
        Args = namedtuple('Args', [k for k in kwargs.keys()])
        # 使用 `kwargs` 创建 `Args` 元组实例 `args`
        args = Args(**kwargs)
        # 使用 `args` 创建 `Dirs` 类的实例 `dirs`
        dirs = Dirs(args)

        # 创建一个命令列表 `cmd`，包含 Python 解释器路径和要执行的脚本路径
        cmd = [f'{sys.executable}',
               str(dirs.root / 'tools' / 'refguide_check.py')]
        # 如果 `args.verbose` 为真，将 `-vvv` 参数添加到 `cmd` 列表中
        if args.verbose:
            cmd += ['-vvv']
        # 如果 `args.submodule` 存在，将其作为参数添加到 `cmd` 列表中
        if args.submodule:
            cmd += [args.submodule]
        # 将 `cmd` 列表转换为字符串 `cmd_str`，以便后续返回
        cmd_str = ' '.join(cmd)
        # 返回一个包含以下字段的字典
        return {
            'actions': [f'env PYTHONPATH={dirs.site} {cmd_str}'],  # 定义一个包含执行命令的列表 `actions`
            'task_dep': ['build'],  # 定义一个依赖任务列表 `task_dep`，包含 'build'
            'io': {'capture': False},  # 定义一个 I/O 控制字典 `io`，设置捕获输出为假
        }
##########################################
# ENVS

@cli.cls_cmd('python')
class Python:
    """:wrench: Start a Python shell with PYTHONPATH set.

    ARGS: Arguments passed to the Python interpreter.
          If not set, an interactive shell is launched.

    Running `python dev.py shell my_script.py` is equivalent to:
    1. Execute build command (skip by passing the global `-n` option).
    2. Set the PYTHONPATH environment variable
       (query with `python dev.py -n show_PYTHONPATH`).
    3. Run interpreter: `python my_script.py`
    """
    # 设置环境上下文为全局上下文对象
    ctx = CONTEXT
    # PYTHONPATH 选项，用于指定要预置到 PYTHONPATH 的路径
    pythonpath = Option(
        ['--pythonpath', '-p'], metavar='PYTHONPATH', default=None,
        help='Paths to prepend to PYTHONPATH')
    # 额外的命令行参数，传递给 Python 解释器
    extra_argv = Argument(
        ['extra_argv'], nargs=-1, metavar='ARGS', required=False)

    @classmethod
    def _setup(cls, pythonpath, **kwargs):
        # 获取构建选项的默认值并更新为传入的参数
        vals = Build.opt_defaults()
        vals.update(kwargs)
        # 执行构建命令，添加路径
        Build.run(add_path=True, **vals)
        if pythonpath:
            # 如果指定了 PYTHONPATH，则将其逆序添加到 sys.path 中
            for p in reversed(pythonpath.split(os.pathsep)):
                sys.path.insert(0, p)

    @classmethod
    def run(cls, pythonpath, extra_argv=None, **kwargs):
        # 调用 _setup 方法进行环境设置
        cls._setup(pythonpath, **kwargs)
        if extra_argv:
            # 如果存在额外的命令行参数
            # 使用 subprocess 不是一个好选择，因为我们不希望包含当前路径在 PYTHONPATH 中
            sys.argv = extra_argv
            # 打开第一个额外参数指定的脚本文件并读取内容
            with open(extra_argv[0]) as f:
                script = f.read()
            # 创建一个新的模块对象作为 __main__
            sys.modules['__main__'] = new_module('__main__')
            ns = dict(__name__='__main__', __file__=extra_argv[0])
            # 在命名空间 ns 中执行脚本内容
            exec(script, ns)
        else:
            # 如果没有额外的命令行参数，导入 code 模块并启动交互式解释器
            import code
            code.interact()


@cli.cls_cmd('ipython')
class Ipython(Python):
    """:wrench: Start IPython shell with PYTHONPATH set.

    Running `python dev.py ipython` is equivalent to:
    1. Execute build command (skip by passing the global `-n` option).
    2. Set the PYTHONPATH environment variable
       (query with `python dev.py -n show_PYTHONPATH`).
    3. Run the `ipython` interpreter.
    """
    # 设置环境上下文为全局上下文对象
    ctx = CONTEXT
    # 继承自父类 Python 的 PYTHONPATH 选项
    pythonpath = Python.pythonpath

    @classmethod
    def run(cls, pythonpath, **kwargs):
        # 调用 _setup 方法进行环境设置
        cls._setup(pythonpath, **kwargs)
        # 导入 IPython 模块并启动 IPython 交互式环境
        import IPython
        IPython.embed(user_ns={})


@cli.cls_cmd('shell')
class Shell(Python):
    """:wrench: Start Unix shell with PYTHONPATH set.

    Running `python dev.py shell` is equivalent to:
    1. Execute build command (skip by passing the global `-n` option).
    2. Open a new shell.
    3. Set the PYTHONPATH environment variable in shell
       (query with `python dev.py -n show_PYTHONPATH`).
    """
    # 设置环境上下文为全局上下文对象
    ctx = CONTEXT
    # 继承自父类 Python 的 PYTHONPATH 选项
    pythonpath = Python.pythonpath
    # 继承自父类 Python 的额外命令行参数选项
    extra_argv = Python.extra_argv

    @classmethod
    # 定义一个类方法 `run`，接受参数 `cls`, `pythonpath`, `extra_argv` 和关键字参数 `kwargs`
    def run(cls, pythonpath, extra_argv, **kwargs):
        # 调用类方法 `_setup`，设置 Python 路径及其他参数
        cls._setup(pythonpath, **kwargs)
        # 获取环境变量中的 shell，如果不存在则默认为 'sh'
        shell = os.environ.get('SHELL', 'sh')
        # 输出一条信息，指示正在启动一个 Unix shell
        click.echo(f"Spawning a Unix shell '{shell}' ...")
        # 用给定的 shell 执行当前进程，替换当前进程的映像
        os.execv(shell, [shell] + list(extra_argv))
        # 如果 os.execv() 执行失败，则退出当前进程，返回状态码 1
        sys.exit(1)
@cli.cls_cmd('show_PYTHONPATH')
class ShowDirs(Python):
    """:information: Show value of the PYTHONPATH environment variable used in
    this script.

    PYTHONPATH sets the default search path for module files for the
    interpreter. Here, it includes the path to the local SciPy build
    (typically `.../build-install/lib/python3.10/site-packages`).

    Use the global option `-n` to skip the building step, e.g.:
    `python dev.py -n show_PYTHONPATH`
    """
    # 设置上下文为全局CONTEXT
    ctx = CONTEXT
    # 获取Python类的pythonpath属性
    pythonpath = Python.pythonpath
    # 获取Python类的extra_argv属性
    extra_argv = Python.extra_argv

    @classmethod
    def run(cls, pythonpath, extra_argv, **kwargs):
        # 调用_setup方法设置pythonpath和其他参数
        cls._setup(pythonpath, **kwargs)
        # 获取环境变量PYTHONPATH的值
        py_path = os.environ.get('PYTHONPATH', '')
        # 输出PYTHONPATH的值到控制台
        click.echo(f"PYTHONPATH={py_path}")


@cli.command()
@click.argument('version_args', nargs=2)
@click.pass_obj
def notes(ctx_obj, version_args):
    """:ledger: Release notes and log generation.

    \b
    ```python
     Example:

    $ python dev.py notes v1.7.0 v1.8.0
    ```
    """
    # 如果有版本参数，则设置sys.argv为版本参数列表
    if version_args:
        sys.argv = version_args
        # 获取版本日志的起始和结束版本
        log_start = sys.argv[0]
        log_end = sys.argv[1]
    # 构建执行命令字符串
    cmd = f"python tools/write_release_and_log.py {log_start} {log_end}"
    # 输出执行命令字符串到控制台
    click.echo(cmd)
    try:
        # 执行命令并等待完成，检查返回状态
        subprocess.run([cmd], check=True, shell=True)
    except subprocess.CalledProcessError:
        # 捕获并输出错误信息
        print('Error caught: Incorrect log start or log end version')


@cli.command()
@click.argument('revision_args', nargs=2)
@click.pass_obj
def authors(ctx_obj, revision_args):
    """:ledger: Generate list of authors who contributed within revision
    interval.

    \b
    ```python
    Example:

    $ python dev.py authors v1.7.0 v1.8.0
    ```
    """
    # 如果有修订版本参数，则设置sys.argv为修订版本参数列表
    if revision_args:
        sys.argv = revision_args
        # 获取修订版本的起始和结束版本
        start_revision = sys.argv[0]
        end_revision = sys.argv[1]
    # 构建执行命令字符串
    cmd = f"python tools/authors.py {start_revision}..{end_revision}"
    # 输出执行命令字符串到控制台
    click.echo(cmd)
    try:
        # 执行命令并等待完成，检查返回状态
        subprocess.run([cmd], check=True, shell=True)
    except subprocess.CalledProcessError:
        # 捕获并输出错误信息
        print('Error caught: Incorrect revision start or revision end')


# The following CPU core count functions were taken from loky/backend/context.py
# See https://github.com/joblib/loky

# Cache for the number of physical cores to avoid repeating subprocess calls.
# It should not change during the lifetime of the program.
physical_cores_cache = None


def cpu_count(only_physical_cores=False):
    """Return the number of CPUs the current process can use.

    The returned number of CPUs accounts for:
     * the number of CPUs in the system, as given by
       ``multiprocessing.cpu_count``;
     * the CPU affinity settings of the current process
       (available on some Unix systems);
     * Cgroup CPU bandwidth limit (available on Linux only, typically
       set by docker and similar container orchestration systems);
     * the value of the LOKY_MAX_CPU_COUNT environment variable if defined.
    and is given as the minimum of these constraints.
    """
    # 返回当前进程可用CPU核心数的计算说明
    pass
    # 如果 ``only_physical_cores`` 为 True，则返回物理核心数，而不是逻辑核心数（超线程 / SMT）。
    # 注意，如果可用核心数受到其他控制，如进程亲和性、Cgroup 限制的 CPU 带宽或 LOKY_MAX_CPU_COUNT 环境变量，则此选项不起作用。
    # 如果未找到物理核心数，则返回逻辑核心数。
    
    # 注意，在 Windows 上，返回的 CPU 数量不能超过 61（Python < 3.10 为 60），参考：
    # https://bugs.python.org/issue26903。
    
    # 返回的 CPU 数始终大于或等于 1。
    """
    If ``only_physical_cores`` is True, return the number of physical cores
    instead of the number of logical cores (hyperthreading / SMT). Note that
    this option is not enforced if the number of usable cores is controlled in
    any other way such as: process affinity, Cgroup restricted CPU bandwidth
    or the LOKY_MAX_CPU_COUNT environment variable. If the number of physical
    cores is not found, return the number of logical cores.
    
    Note that on Windows, the returned number of CPUs cannot exceed 61 (or 60 for
    Python < 3.10), see:
    https://bugs.python.org/issue26903.
    
    It is also always larger or equal to 1.
    """
    
    # 注意：os.cpu_count() 在其文档字符串中允许返回 None
    os_cpu_count = os.cpu_count() or 1
    
    if sys.platform == "win32":
        # 在 Windows 上，尝试使用超过 61 个 CPU 会导致操作系统级错误。
        # 参考 https://bugs.python.org/issue26903。根据
        # https://learn.microsoft.com/en-us/windows/win32/procthread/processor-groups
        # 可能可以通过大量的额外工作超过这个限制，但这并不容易。
        os_cpu_count = min(os_cpu_count, _MAX_WINDOWS_WORKERS)
    
    # 获取用户定义的 CPU 数量
    cpu_count_user = _cpu_count_user(os_cpu_count)
    
    # 计算聚合的 CPU 数量，取用户定义数与操作系统可用数的交集，并确保至少为 1
    aggregate_cpu_count = max(min(os_cpu_count, cpu_count_user), 1)
    
    if not only_physical_cores:
        return aggregate_cpu_count
    
    if cpu_count_user < os_cpu_count:
        # 尊重用户设置
        return max(cpu_count_user, 1)
    
    # 获取物理核心数及其可能的异常
    cpu_count_physical, exception = _count_physical_cores()
    
    if cpu_count_physical != "not found":
        return cpu_count_physical
    
    # 默认行为回退
    if exception is not None:
        # 仅在第一次发出警告
        warnings.warn(
            "Could not find the number of physical cores for the "
            f"following reason:\n{exception}\n"
            "Returning the number of logical cores instead. You can "
            "silence this warning by setting LOKY_MAX_CPU_COUNT to "
            "the number of cores you want to use.",
            stacklevel=2
        )
        traceback.print_tb(exception.__traceback__)
    
    return aggregate_cpu_count
# 根据操作系统提供的 CPU 数量来确定 Cgroup CPU 限制
def _cpu_count_cgroup(os_cpu_count):
    # Cgroup CPU 最大带宽限制，在 Linux 2.6 内核中可用
    cpu_max_fname = "/sys/fs/cgroup/cpu.max"
    cfs_quota_fname = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    cfs_period_fname = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    
    if os.path.exists(cpu_max_fname):
        # 如果存在 cpu_max_fname 文件，表明是 cgroup v2
        # 参考：https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
        with open(cpu_max_fname) as fh:
            cpu_quota_us, cpu_period_us = fh.read().strip().split()
    elif os.path.exists(cfs_quota_fname) and os.path.exists(cfs_period_fname):
        # 否则如果存在 cfs_quota_fname 和 cfs_period_fname 文件，表明是 cgroup v1
        # 参考：https://www.kernel.org/doc/html/latest/scheduler/sched-bwc.html#management
        with open(cfs_quota_fname) as fh:
            cpu_quota_us = fh.read().strip()
        with open(cfs_period_fname) as fh:
            cpu_period_us = fh.read().strip()
    else:
        # 如果以上文件都不存在，表示没有 Cgroup CPU 带宽限制（例如非 Linux 平台）
        cpu_quota_us = "max"
        cpu_period_us = 100_000  # 未使用，为了与默认值保持一致

    if cpu_quota_us == "max":
        # 在支持 Cgroup 的平台上没有活动的 Cgroup 限额
        return os_cpu_count
    else:
        cpu_quota_us = int(cpu_quota_us)
        cpu_period_us = int(cpu_period_us)
        if cpu_quota_us > 0 and cpu_period_us > 0:
            # 计算 CPU 的有效数量，向上取整到最接近的整数
            return math.ceil(cpu_quota_us / cpu_period_us)
        else:  # pragma: no cover
            # 设置负的 cpu_quota_us 值是禁用 cgroup CPU 带宽限制的有效方法
            return os_cpu_count


def _cpu_count_affinity(os_cpu_count):
    # 根据 CPU 亲和性设置确定可用的 CPU 数量
    if hasattr(os, "sched_getaffinity"):
        try:
            # 尝试获取当前进程的 CPU 亲和性掩码，并返回其长度
            return len(os.sched_getaffinity(0))
        except NotImplementedError:
            pass

    # 在 PyPy 和可能其他平台上，os.sched_getaffinity 不存在或引发 NotImplementedError，尝试使用 psutil
    try:
        import psutil

        p = psutil.Process()
        if hasattr(p, "cpu_affinity"):
            # 如果进程对象支持 cpu_affinity 属性，返回其长度
            return len(p.cpu_affinity())

    except ImportError:  # pragma: no cover
        if (
            sys.platform == "linux"
            and os.environ.get("LOKY_MAX_CPU_COUNT") is None
        ):
            # PyPy 在 Linux 上没有实现 os.sched_getaffinity，可能导致严重的超订问题。在这种特殊的情况下，最好警告用户。
            warnings.warn(
                "Failed to inspect CPU affinity constraints on this system. "
                "Please install psutil or explicitly set LOKY_MAX_CPU_COUNT.",
                stacklevel=4
            )

    # 对于不支持任何类型 CPU 亲和性设置的平台，例如基于 macOS 的平台，返回操作系统提供的 CPU 数量
    return os_cpu_count
    """Number of user defined available CPUs"""
    # 获取用户定义的可用 CPU 数量，通过调用 _cpu_count_affinity 函数
    cpu_count_affinity = _cpu_count_affinity(os_cpu_count)

    # 获取通过 cgroup 控制组定义的可用 CPU 数量，通过调用 _cpu_count_cgroup 函数
    cpu_count_cgroup = _cpu_count_cgroup(os_cpu_count)

    # 从 loky 特定的环境变量中获取用户定义的软限制，如果未设置则使用默认的 os_cpu_count
    cpu_count_loky = int(os.environ.get("LOKY_MAX_CPU_COUNT", os_cpu_count))

    # 返回三者中的最小值作为最终可用 CPU 数量
    return min(cpu_count_affinity, cpu_count_cgroup, cpu_count_loky)
# 返回一个包含物理核心数和异常的元组

def _count_physical_cores():
    exception = None  # 初始化异常变量为 None

    # 首先检查缓存中是否已经有值
    global physical_cores_cache
    if physical_cores_cache is not None:
        return physical_cores_cache, exception  # 如果有缓存值直接返回

    # 如果没有缓存值，则需要查找
    try:
        if sys.platform == "linux":
            # 在 Linux 平台上运行 lscpu 命令获取核心数信息
            cpu_info = subprocess.run(
                "lscpu --parse=core".split(), capture_output=True, text=True
            )
            cpu_info = cpu_info.stdout.splitlines()
            cpu_info = {line for line in cpu_info if not line.startswith("#")}
            cpu_count_physical = len(cpu_info)
        elif sys.platform == "win32":
            # 在 Windows 平台上运行 wmic 命令获取核心数信息
            cpu_info = subprocess.run(
                "wmic CPU Get NumberOfCores /Format:csv".split(),
                capture_output=True,
                text=True,
            )
            cpu_info = cpu_info.stdout.splitlines()
            cpu_info = [
                l.split(",")[1]
                for l in cpu_info
                if (l and l != "Node,NumberOfCores")
            ]
            cpu_count_physical = sum(map(int, cpu_info))
        elif sys.platform == "darwin":
            # 在 macOS 平台上运行 sysctl 命令获取核心数信息
            cpu_info = subprocess.run(
                "sysctl -n hw.physicalcpu".split(),
                capture_output=True,
                text=True,
            )
            cpu_info = cpu_info.stdout
            cpu_count_physical = int(cpu_info)
        else:
            # 对于不支持的平台抛出异常
            raise NotImplementedError(f"unsupported platform: {sys.platform}")

        # 如果获取的物理核心数小于 1，则认为未找到有效值，抛出异常
        if cpu_count_physical < 1:
            raise ValueError(f"found {cpu_count_physical} physical cores < 1")

    except Exception as e:
        exception = e  # 捕获异常并赋值给 exception
        cpu_count_physical = "not found"  # 如果出现异常，设置物理核心数为 "not found"

    # 将结果放入缓存
    physical_cores_cache = cpu_count_physical

    return cpu_count_physical, exception  # 返回物理核心数和异常信息


if __name__ == '__main__':
    cli()  # 执行命令行接口程序
```