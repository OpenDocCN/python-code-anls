# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\__init__.py`

```
"""
Helper functions for testing.
"""
# 导入必要的模块
from pathlib import Path
from tempfile import TemporaryDirectory
import locale
import logging
import os
import subprocess
import sys

import matplotlib as mpl
from matplotlib import _api

# 获取当前模块的日志记录器
_log = logging.getLogger(__name__)


def set_font_settings_for_testing():
    # 设置测试中使用的字体系列为 DejaVu Sans
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    # 关闭文本的字体渲染提示
    mpl.rcParams['text.hinting'] = 'none'
    # 设置文本渲染提示因子
    mpl.rcParams['text.hinting_factor'] = 8


def set_reproducibility_for_testing():
    # 设置 SVG 输出的哈希盐值，以保证可复现性
    mpl.rcParams['svg.hashsalt'] = 'matplotlib'


def setup():
    # 设置当前测试使用的本地化设置
    # 尝试设置本地化为 'en_US.UTF-8'
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    # 如果设置失败，则尝试 'English_United States.1252'
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'English_United States.1252')
        # 如果再次失败，则记录警告
        except locale.Error:
            _log.warning(
                "Could not set locale to English/United States. "
                "Some date-related tests may fail.")

    # 使用 'Agg' 后端以便在无图形界面的环境中生成图像
    mpl.use('Agg')

    # 忽略 matplotlib 的即将弃用警告
    with _api.suppress_matplotlib_deprecation_warning():
        mpl.rcdefaults()  # 恢复默认的 matplotlib 配置

    # 设置测试所需的字体和可复现性参数
    set_font_settings_for_testing()
    set_reproducibility_for_testing()


def subprocess_run_for_testing(command, env=None, timeout=60, stdout=None,
                               stderr=None, check=False, text=True,
                               capture_output=False):
    """
    Create and run a subprocess.

    Thin wrapper around `subprocess.run`, intended for testing.  Will
    mark fork() failures on Cygwin as expected failures: not a
    success, but not indicating a problem with the code either.

    Parameters
    ----------
    args : list of str
        Command to be executed and its arguments.
    env : dict[str, str]
        Optional environment variables for the subprocess.
    timeout : float
        Timeout period for the subprocess execution.
    stdout, stderr : file objects
        File objects for capturing subprocess stdout and stderr.
    check : bool
        If True, raises subprocess.CalledProcessError for non-zero return code.
    text : bool
        If True, return stdout and stderr as str (text mode).
        If False, return as bytes.
    capture_output : bool
        If True, capture stdout and stderr.

    Returns
    -------
    proc : subprocess.Popen
        Completed process object.

    See Also
    --------
    subprocess.run

    Raises
    ------
    pytest.xfail
        If platform is Cygwin and subprocess reports a fork() failure.
    """
    if capture_output:
        stdout = stderr = subprocess.PIPE
    try:
        proc = subprocess.run(
            command, env=env,
            timeout=timeout, check=check,
            stdout=stdout, stderr=stderr,
            text=text
        )
    except BlockingIOError:
        # 在 Cygwin 平台下捕获 fork() 失败的异常
        if sys.platform == "cygwin":
            import pytest
            pytest.xfail("Fork failure")  # 标记为预期失败
        raise  # 重新抛出异常
    return proc


def subprocess_run_helper(func, *args, timeout, extra_env=None):
    # 此函数未完整提供，未包含在示例注释中，因此不进行进一步注释
    pass
    """
    Run a function in a sub-process.
    
    Parameters
    ----------
    func : function
        The function to be run.  It must be in a module that is importable.
    *args : str
        Any additional command line arguments to be passed in
        the first argument to ``subprocess.run``.
    extra_env : dict[str, str]
        Any additional environment variables to be set for the subprocess.
    """
    # 获取要运行的函数的名称
    target = func.__name__
    # 获取函数所在的模块名
    module = func.__module__
    # 获取函数定义所在的文件路径
    file = func.__code__.co_filename
    # 调用 subprocess_run_for_testing 函数来执行子进程
    proc = subprocess_run_for_testing(
        [
            sys.executable,
            "-c",
            # 构建动态执行代码的字符串，导入指定模块并执行目标函数
            f"import importlib.util;"
            f"_spec = importlib.util.spec_from_file_location({module!r}, {file!r});"
            f"_module = importlib.util.module_from_spec(_spec);"
            f"_spec.loader.exec_module(_module);"
            f"_module.{target}()",
            *args
        ],
        # 设置子进程执行环境变量，包括 SOURCE_DATE_EPOCH 和额外的用户指定变量
        env={**os.environ, "SOURCE_DATE_EPOCH": "0", **(extra_env or {})},
        # 设置子进程执行超时时间
        timeout=timeout,
        # 设置子进程执行完毕后检查返回状态
        check=True,
        # 将子进程的标准输出重定向到 PIPE
        stdout=subprocess.PIPE,
        # 将子进程的标准错误重定向到 PIPE
        stderr=subprocess.PIPE,
        # 指定标准输出和标准错误为文本模式
        text=True
    )
    # 返回子进程对象
    return proc
# 检查给定的 TeX 系统是否支持 pgf 包
def _check_for_pgf(texsystem):
    # 使用临时目录作为工作目录
    with TemporaryDirectory() as tmpdir:
        # 创建一个名为 test.tex 的文件路径
        tex_path = Path(tmpdir, "test.tex")
        # 将 LaTeX 源码写入 test.tex 文件
        tex_path.write_text(r"""
            \documentclass{article}
            \usepackage{pgf}
            \begin{document}
            \typeout{pgfversion=\pgfversion}
            \makeatletter
            \@@end
        """, encoding="utf-8")
        # 尝试执行命令检查 TeX 系统是否能编译该 LaTeX 文件
        try:
            subprocess.check_call(
                [texsystem, "-halt-on-error", str(tex_path)], cwd=tmpdir,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 捕获 OSError 或 subprocess.CalledProcessError 异常
        except (OSError, subprocess.CalledProcessError):
            # 如果发生异常则返回 False
            return False
        # 如果一切正常则返回 True
        return True


# 检查是否存在指定名称的 TeX 包
def _has_tex_package(package):
    try:
        # 查找指定名称的 .sty 文件是否存在
        mpl.dviread.find_tex_file(f"{package}.sty")
        # 如果找到文件则返回 True
        return True
    # 捕获 FileNotFoundError 异常
    except FileNotFoundError:
        # 如果文件未找到则返回 False
        return False


# 在子进程中运行 IPython，并检查所使用的后端或 GUI 框架是否符合预期
def ipython_in_subprocess(requested_backend_or_gui_framework, all_expected_backends):
    import pytest
    IPython = pytest.importorskip("IPython")

    # 如果运行平台是 Windows，则跳过测试
    if sys.platform == "win32":
        pytest.skip("Cannot change backend running IPython in subprocess on Windows")

    # 如果 IPython 版本为 8.24.0 且请求使用的后端为 "osx"，则跳过测试
    if (IPython.version_info[:3] == (8, 24, 0) and
            requested_backend_or_gui_framework == "osx"):
        pytest.skip("Bug using macosx backend in IPython 8.24.0 fixed in 8.24.1")

    # 在 IPython 版本大于等于指定版本时，选择预期的后端
    for min_version, backend in all_expected_backends.items():
        if IPython.version_info[:2] >= min_version:
            expected_backend = backend
            break

    # 定义用于在测试中运行 IPython 的代码
    code = ("import matplotlib as mpl, matplotlib.pyplot as plt;"
            "fig, ax=plt.subplots(); ax.plot([1, 3, 2]); mpl.get_backend()")
    
    # 在子进程中运行 IPython 并捕获输出
    proc = subprocess_run_for_testing(
        [
            "ipython",
            "--no-simple-prompt",
            f"--matplotlib={requested_backend_or_gui_framework}",
            "-c", code,
        ],
        check=True,
        capture_output=True,
    )

    # 断言 IPython 的输出以预期的后端名称结尾
    assert proc.stdout.strip().endswith(f"'{expected_backend}'")
```