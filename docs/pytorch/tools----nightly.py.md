# `.\pytorch\tools\nightly.py`

```py
# 指定脚本在 Python 3 环境下执行
#!/usr/bin/env python3
# 导入日志代码，来源于 https://github.com/ezyang/ghstack 的一部分
# 版权归 Edward Z. Yang <ezyang@mit.edu> 所有
# 此脚本检出 PyTorch 的夜间开发版本，并将预构建的二进制文件安装到仓库中。

"""Checks out the nightly development version of PyTorch and installs pre-built
binaries into the repo.

You can use this script to check out a new nightly branch with the following::

    $ ./tools/nightly.py checkout -b my-nightly-branch
    $ conda activate pytorch-deps

Or if you would like to re-use an existing conda environment, you can pass in
the regular environment parameters (--name or --prefix)::

    $ ./tools/nightly.py checkout -b my-nightly-branch -n my-env
    $ conda activate my-env

You can also use this tool to pull the nightly commits into the current branch as
well. This can be done with

    $ ./tools/nightly.py pull -n my-env
    $ conda activate my-env

Pulling will reinstalle the conda dependencies as well as the nightly binaries into
the repo directory.
"""

from __future__ import annotations

# 导入上下文管理器、日期时间处理、函数装饰器、文件模式匹配、JSON处理、系统操作等模块
import contextlib
import datetime
import functools
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
# 从参数解析器导入 ArgumentParser 类
from argparse import ArgumentParser
# 从抽象语法树中导入 literal_eval 函数
from ast import literal_eval
# 导入类型提示相关的泛型和类型变量
from typing import Any, Callable, cast, Generator, Iterable, Iterator, Sequence, TypeVar

# 定义全局变量 LOGGER，用于记录日志，初始值为 None
LOGGER: logging.Logger | None = None
# 定义 URL 格式字符串模板
URL_FORMAT = "{base_url}/{platform}/{dist_name}.tar.bz2"
# 定义日期时间格式字符串模板
DATETIME_FORMAT = "%Y-%m-%d_%Hh%Mm%Ss"
# 定义 SHA1 字符串的正则表达式模式
SHA1_RE = re.compile("([0-9a-fA-F]{40})")
# 定义用户名和密码在 URL 中的正则表达式模式
USERNAME_PASSWORD_RE = re.compile(r":\/\/(.*?)\@")
# 定义日志目录名称的正则表达式模式
LOG_DIRNAME_RE = re.compile(
    r"(\d{4}-\d\d-\d\d_\d\dh\d\dm\d\ds)_" r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}"
)
# 定义要安装的软件包规范列表
SPECS_TO_INSTALL = ("pytorch", "mypy", "pytest", "hypothesis", "ipython", "sphinx")

# 定义 Formatter 类，用于定制日志格式
class Formatter(logging.Formatter):
    redactions: dict[str, str]

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(fmt, datefmt)
        # 初始化需要过滤的敏感信息字典
        self.redactions = {}

    # 从 URL 中移除敏感信息（用户名和密码）
    def _filter(self, s: str) -> str:
        s = USERNAME_PASSWORD_RE.sub(r"://<USERNAME>:<PASSWORD>@", s)
        for needle, replace in self.redactions.items():
            s = s.replace(needle, replace)
        return s

    # 格式化日志消息
    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO or record.levelno == logging.DEBUG:
            # INFO 和 DEBUG 级别的日志直接返回消息内容
            return record.getMessage()
        else:
            # 其他级别的日志使用父类的 formatMessage 方法格式化
            return super().formatMessage(record)

    # 格式化日志记录
    def format(self, record: logging.LogRecord) -> str:
        return self._filter(super().format(record))
    def redact(self, needle: str, replace: str = "<REDACTED>") -> None:
        """Redact specific strings; e.g., authorization tokens.  This won't
        retroactively redact stuff you've already leaked, so make sure
        you redact things as soon as possible.
        """
        # 如果需要替换的字符串为空，则直接返回，不进行任何操作
        if needle == "":
            return
        # 将需要替换的字符串及其替换后的值存储到实例变量 redactions 中
        self.redactions[needle] = replace
# 使用 functools 模块的 lru_cache 装饰器缓存结果，避免重复计算
@functools.lru_cache
# 获取基础日志目录，基于当前工作目录构建
def logging_base_dir() -> str:
    meta_dir = os.getcwd()
    base_dir = os.path.join(meta_dir, "nightly", "log")
    # 创建日志目录，如果目录已存在则不做任何操作
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


# 使用 functools 模块的 lru_cache 装饰器缓存结果，避免重复计算
@functools.lru_cache
# 获取当前日志运行目录，基于 logging_base_dir 返回的结果和当前时间和唯一标识符构建
def logging_run_dir() -> str:
    cur_dir = os.path.join(
        logging_base_dir(),
        f"{datetime.datetime.now().strftime(DATETIME_FORMAT)}_{uuid.uuid1()}",
    )
    # 创建当前日志运行目录，如果目录已存在则不做任何操作
    os.makedirs(cur_dir, exist_ok=True)
    return cur_dir


# 记录当前进程的命令行参数到日志运行目录下的 argv 文件中
def logging_record_argv() -> None:
    s = subprocess.list2cmdline(sys.argv)
    with open(os.path.join(logging_run_dir(), "argv"), "w") as f:
        f.write(s)


# 记录异常信息到日志运行目录下的 exception 文件中
def logging_record_exception(e: BaseException) -> None:
    with open(os.path.join(logging_run_dir(), "exception"), "w") as f:
        f.write(type(e).__name__)


# 清理日志基础目录中超过1000个的旧日志文件
def logging_rotate() -> None:
    log_base = logging_base_dir()
    old_logs = os.listdir(log_base)
    old_logs.sort(reverse=True)
    for stale_log in old_logs[1000:]:
        # 检查旧日志文件名是否符合预期的格式，以确保只删除日志文件
        if LOG_DIRNAME_RE.fullmatch(stale_log) is not None:
            shutil.rmtree(os.path.join(log_base, stale_log))


# 使用 contextlib 模块的 contextmanager 装饰器定义日志管理器上下文
@contextlib.contextmanager
def logging_manager(*, debug: bool = False) -> Generator[logging.Logger, None, None]:
    """设置日志记录。如果从这里开始失败，我们无法以合理的方式保存用户。

    日志结构：有一个记录器（根记录器），处理所有事件。
    有两个处理器：stderr（INFO 级别）和文件处理器（DEBUG 级别）。
    """
    formatter = Formatter(fmt="%(levelname)s: %(message)s", datefmt="")
    root_logger = logging.getLogger("conda-pytorch")
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = os.path.join(logging_run_dir(), "nightly.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    logging_record_argv()

    try:
        logging_rotate()
        print(f"log file: {log_file}")
        yield root_logger
    except Exception as e:
        # 记录异常信息，并将异常信息写入日志运行目录下的 exception 文件中
        logging.exception("Fatal exception")
        logging_record_exception(e)
        print(f"log file: {log_file}")
        sys.exit(1)
    except BaseException as e:
        # 在这里使用 logging.debug 可以完全隐藏回溯，但没有理由向技术上精通的用户隐藏它。
        logging.info("", exc_info=True)
        logging_record_exception(e)
        print(f"log file: {log_file}")
        sys.exit(1)


# 检查当前目录是否在 PyTorch 仓库的根目录中
def check_in_repo() -> str | None:
    """确保我们在 PyTorch 仓库中。"""
    if not os.path.isfile("setup.py"):
        return "Not in root-level PyTorch repo, no setup.py found"
    # 打开文件 "setup.py" 并赋值给变量 f，使用 with 语句确保文件关闭
    with open("setup.py") as f:
        # 读取文件内容并赋值给变量 s
        s = f.read()
    
    # 检查字符串 "PyTorch" 是否不在变量 s 中
    if "PyTorch" not in s:
        # 如果条件满足，返回说明字符串
        return "Not in PyTorch repo, 'PyTorch' not found in setup.py"
    
    # 如果条件不满足，即 "PyTorch" 存在于变量 s 中，返回 None
    return None
# 检查分支名称是否可以检出的函数
def check_branch(subcommand: str, branch: str | None) -> str | None:
    # 如果子命令不是 "checkout"，则返回 None
    if subcommand != "checkout":
        return None
    # 确保提供了实际的分支名称
    if branch is None:
        return "Branch name to checkout must be supplied with '-b' option"
    
    # 检查本地仓库是否干净，即没有未跟踪的文件和未提交的更改
    cmd = ["git", "status", "--untracked-files=no", "--porcelain"]
    # 运行 git 命令获取仓库状态
    p = subprocess.run(
        cmd,
        capture_output=True,  # 捕获命令输出
        check=True,           # 如果命令返回非零退出代码，则引发异常
        text=True,            # 以文本模式处理命令输出
    )
    # 如果有未提交的更改，返回需要清理工作目录的提示和详细信息
    if p.stdout.strip():
        return "Need to have clean working tree to checkout!\n\n" + p.stdout
    
    # 检查分支名称是否已经存在于本地仓库
    cmd = ["git", "show-ref", "--verify", "--quiet", "refs/heads/" + branch]
    # 运行 git 命令检查分支是否存在
    p = subprocess.run(cmd, capture_output=True, check=False)  # type: ignore[assignment]
    # 如果命令返回码为 0，表示分支已经存在，返回提示信息
    if not p.returncode:
        return f"Branch {branch!r} already exists"
    
    # 如果以上条件都不满足，返回 None，表示分支可以被检出
    return None


@contextlib.contextmanager
def timer(logger: logging.Logger, prefix: str) -> Iterator[None]:
    """计时上下文管理器"""
    start_time = time.time()  # 记录开始时间
    yield  # 执行被装饰函数
    logger.info("%s took %.3f [s]", prefix, time.time() - start_time)  # 记录执行时间


F = TypeVar("F", bound=Callable[..., Any])


def timed(prefix: str) -> Callable[[F], F]:
    """用于计时函数执行时间的装饰器"""

    def dec(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            global LOGGER
            logger = cast(logging.Logger, LOGGER)  # 强制类型转换获取全局日志记录器
            logger.info(prefix)  # 记录日志，标记函数开始执行
            with timer(logger, prefix):  # 使用计时器记录函数执行时间
                return f(*args, **kwargs)  # 执行被装饰的函数

        return cast(F, wrapper)

    return dec


def _make_channel_args(
    channels: Iterable[str] = ("pytorch-nightly",),
    override_channels: bool = False,
) -> list[str]:
    """生成 conda 频道参数列表"""
    args = []
    for channel in channels:
        args.append("--channel")  # 添加频道选项
        args.append(channel)  # 添加具体频道名称
    if override_channels:
        args.append("--override-channels")  # 如果需要覆盖默认频道，添加选项
    return args


@timed("Solving conda environment")
def conda_solve(
    name: str | None = None,
    prefix: str | None = None,
    channels: Iterable[str] = ("pytorch-nightly",),
    override_channels: bool = False,
) -> tuple[list[str], str, str, bool, list[str]]:
    """执行 conda 环境解析，分离依赖项和软件包"""
    # 确定要使用的环境
    if prefix is not None:
        existing_env = True
        env_opts = ["--prefix", prefix]  # 使用指定的环境前缀
    elif name is not None:
        existing_env = True
        env_opts = ["--name", name]  # 使用指定的环境名称
    else:
        existing_env = False
        env_opts = ["--name", "pytorch-deps"]  # 创建新环境，使用默认名称

    # 运行 conda 解析命令
    if existing_env:
        cmd = [
            "conda",
            "install",
            "--yes",
            "--dry-run",
            "--json",
        ]
        cmd.extend(env_opts)  # 添加环境选项
    else:
        # 构建 conda 命令列表，创建环境的预演模式，使用 JSON 输出格式
        cmd = [
            "conda",
            "create",
            "--yes",
            "--dry-run",
            "--json",
            "--name",
            "__pytorch__",
        ]
    channel_args = _make_channel_args(
        channels=channels, override_channels=override_channels
    )
    # 将频道参数扩展到命令列表中
    cmd.extend(channel_args)
    # 将要安装的包规格列表扩展到命令列表中
    cmd.extend(SPECS_TO_INSTALL)
    # 运行 subprocess，捕获输出结果，检查执行是否成功
    p = subprocess.run(cmd, capture_output=True, check=True)
    # 解析 subprocess 的标准输出为 JSON 格式
    # 解析解决方案中的链接操作
    solve = json.loads(p.stdout)
    link = solve["actions"]["LINK"]
    deps = []
    # 遍历链接操作中的每个软件包
    for pkg in link:
        # 构建每个软件包的 URL
        url = URL_FORMAT.format(**pkg)
        if pkg["name"] == "pytorch":
            # 如果软件包名称是 'pytorch'，则将其 URL 赋给 pytorch 变量
            pytorch = url
            # 将软件包的平台信息赋给 platform 变量
            platform = pkg["platform"]
        else:
            # 如果软件包名称不是 'pytorch'，则将其 URL 添加到依赖列表中
            deps.append(url)
    # 返回依赖列表、pytorch 变量、platform 变量、现有环境和环境选项
    return deps, pytorch, platform, existing_env, env_opts
# 装饰器函数，用于计时和记录函数执行信息
@timed("Installing dependencies")
# 安装依赖项到指定环境
def deps_install(deps: list[str], existing_env: bool, env_opts: list[str]) -> None:
    """Install dependencies to deps environment"""
    # 如果不存在已有环境，首先移除之前的 pytorch-deps 环境
    if not existing_env:
        cmd = ["conda", "env", "remove", "--yes"] + env_opts
        # 执行命令，并检查执行状态
        p = subprocess.run(cmd, check=True)
    # 安装新的依赖项
    inst_opt = "install" if existing_env else "create"
    cmd = ["conda", inst_opt, "--yes", "--no-deps"] + env_opts + deps
    # 执行命令，并检查执行状态
    p = subprocess.run(cmd, check=True)


# 装饰器函数，用于计时和记录函数执行信息
@timed("Installing pytorch nightly binaries")
# 安装 PyTorch 到临时目录
def pytorch_install(url: str) -> tempfile.TemporaryDirectory[str]:
    """Install pytorch into a temporary directory"""
    # 创建临时目录对象
    pytdir = tempfile.TemporaryDirectory()
    cmd = ["conda", "create", "--yes", "--no-deps", "--prefix", pytdir.name, url]
    # 执行命令，并检查执行状态
    p = subprocess.run(cmd, check=True)
    # 返回临时目录对象
    return pytdir


# 根据操作系统平台和目录名，获取 site-packages 目录路径
def _site_packages(dirname: str, platform: str) -> str:
    if platform.startswith("win"):
        template = os.path.join(dirname, "Lib", "site-packages")
    else:
        template = os.path.join(dirname, "lib", "python*.*", "site-packages")
    # 根据模板路径查找匹配的 site-packages 目录
    spdir = glob.glob(template)[0]
    return spdir


# 确保本地存在指定的 Git 提交
def _ensure_commit(git_sha1: str) -> None:
    """Make sure that we actually have the commit locally"""
    cmd = ["git", "cat-file", "-e", git_sha1 + "^{commit}"]
    # 捕获命令输出，并检查命令执行结果
    p = subprocess.run(cmd, capture_output=True, check=False)
    if p.returncode == 0:
        # 如果本地存在指定的提交，直接返回
        return
    # 如果本地不存在指定的提交，则需要先进行 fetch 操作
    cmd = ["git", "fetch", "https://github.com/pytorch/pytorch.git", git_sha1]
    # 执行命令，并检查执行状态
    p = subprocess.run(cmd, check=True)


# 获取安装模块中的 Git 版本号作为夜间版本号
def _nightly_version(spdir: str) -> str:
    # 获取安装模块中的 Git 版本信息文件路径
    version_fname = os.path.join(spdir, "torch", "version.py")
    # 打开文件并读取所有行
    with open(version_fname) as f:
        lines = f.read().splitlines()
    # 遍历文件的每一行，找到包含 git_version 的行，并解析其值
    for line in lines:
        if not line.startswith("git_version"):
            continue
        git_version = literal_eval(line.partition("=")[2].strip())
        break
    else:
        # 如果未找到 git_version 信息，抛出运行时错误
        raise RuntimeError(f"Could not find git_version in {version_fname}")
    print(f"Found released git version {git_version}")
    # 根据 git_version 查找夜间版本号
    _ensure_commit(git_version)
    cmd = ["git", "show", "--no-patch", "--format=%s", git_version]
    # 执行命令，捕获输出，并检查执行状态
    p = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        text=True,
    )
    # 使用正则表达式从命令输出中匹配夜间版本号
    m = SHA1_RE.search(p.stdout)
    if m is None:
        # 如果未找到夜间版本号，抛出运行时错误
        raise RuntimeError(
            f"Could not find nightly release in git history:\n  {p.stdout}"
        )
    nightly_version = m.group(1)
    print(f"Found nightly release version {nightly_version}")
    # 根据夜间版本号确保本地存在对应的提交
    _ensure_commit(nightly_version)
    return nightly_version


# 装饰器函数，用于计时和记录函数执行信息
@timed("Checking out nightly PyTorch")
# 检出夜间版本的 PyTorch 并切换
def checkout_nightly_version(branch: str, spdir: str) -> None:
    """Get's the nightly version and then checks it out."""
    # 调用函数 `_nightly_version` 获取 nightly 版本号
    nightly_version = _nightly_version(spdir)
    # 构建 git 命令，创建新分支并切换到 nightly 版本
    cmd = ["git", "checkout", "-b", branch, nightly_version]
    # 使用 subprocess 模块执行 git 命令，确保命令执行成功
    p = subprocess.run(cmd, check=True)
# 根据装饰器标记函数为定时器，用于测量函数执行时间
@timed("Pulling nightly PyTorch")
# 从指定目录中拉取每夜版的 PyTorch 并合并
def pull_nightly_version(spdir: str) -> None:
    """Fetches the nightly version and then merges it ."""
    # 获取每夜版的版本号
    nightly_version = _nightly_version(spdir)
    # 构建 git merge 命令
    cmd = ["git", "merge", nightly_version]
    # 执行命令并确保其成功完成
    p = subprocess.run(cmd, check=True)


# 获取在 Linux 平台下的文件列表
def _get_listing_linux(source_dir: str) -> list[str]:
    # 使用 glob 获取所有 .so 结尾的文件
    listing = glob.glob(os.path.join(source_dir, "*.so"))
    # 扩展列表以包含 lib 目录下的 .so 文件
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.so")))
    return listing


# 获取在 macOS 平台下的文件列表
def _get_listing_osx(source_dir: str) -> list[str]:
    # 即使在 macOS 上，这些文件通常也是 .so 类型
    listing = glob.glob(os.path.join(source_dir, "*.so"))
    # 扩展列表以包含 lib 目录下的 .dylib 文件
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.dylib")))
    return listing


# 获取在 Windows 平台下的文件列表
def _get_listing_win(source_dir: str) -> list[str]:
    # 获取所有 .pyd 结尾的文件
    listing = glob.glob(os.path.join(source_dir, "*.pyd"))
    # 扩展列表以包含 lib 目录下的 .lib 和 .dll 文件
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.lib")))
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.dll")))
    return listing


# 在指定目录下，使用递归方式获取所有 .pyi 文件的相对路径集合
def _glob_pyis(d: str) -> set[str]:
    search = os.path.join(d, "**", "*.pyi")
    pyis = {os.path.relpath(p, d) for p in glob.iglob(search)}
    return pyis


# 找出在源目录中存在而在目标目录中不存在的 .pyi 文件的列表
def _find_missing_pyi(source_dir: str, target_dir: str) -> list[str]:
    source_pyis = _glob_pyis(source_dir)
    target_pyis = _glob_pyis(target_dir)
    missing_pyis = [os.path.join(source_dir, p) for p in (source_pyis - target_pyis)]
    missing_pyis.sort()
    return missing_pyis


# 根据平台不同获取源目录和目标目录下的文件列表，并返回完整的列表
def _get_listing(source_dir: str, target_dir: str, platform: str) -> list[str]:
    if platform.startswith("linux"):
        listing = _get_listing_linux(source_dir)
    elif platform.startswith("osx"):
        listing = _get_listing_osx(source_dir)
    elif platform.startswith("win"):
        listing = _get_listing_win(source_dir)
    else:
        raise RuntimeError(f"Platform {platform!r} not recognized")
    # 找出源目录和目标目录中存在差异的 .pyi 文件，并添加到列表中
    listing.extend(_find_missing_pyi(source_dir, target_dir))
    listing.append(os.path.join(source_dir, "version.py"))
    listing.append(os.path.join(source_dir, "testing", "_internal", "generated"))
    listing.append(os.path.join(source_dir, "bin"))
    listing.append(os.path.join(source_dir, "include"))
    return listing


# 如果目标路径 trg 存在，根据 is_dir 判断是删除文件还是目录
def _remove_existing(trg: str, is_dir: bool) -> None:
    if os.path.exists(trg):
        if is_dir:
            # 递归地删除目录 trg
            shutil.rmtree(trg)
        else:
            # 删除文件 trg
            os.remove(trg)


# 将单个文件或目录从源路径 src 移动到目标路径 target_dir 中
def _move_single(
    src: str,
    source_dir: str,
    target_dir: str,
    mover: Callable[[str, str], None],
    verb: str,
) -> None:
    # 判断 src 是否为目录
    is_dir = os.path.isdir(src)
    # 获取 src 相对于 source_dir 的相对路径
    relpath = os.path.relpath(src, source_dir)
    # 构建目标路径 trg
    trg = os.path.join(target_dir, relpath)
    # 删除已存在的目标文件或目录
    _remove_existing(trg, is_dir)
    # 移动新的文件或目录到目标位置
    # 这一步实际的移动操作由传入的 mover 函数完成，此处未提供完整实现
    # 如果目标路径是目录，则递归地复制源路径下的文件和目录结构到目标路径
    if is_dir:
        # 如果目标路径不存在，则创建该目录
        os.makedirs(trg, exist_ok=True)
        # 遍历源路径下的所有子目录和文件
        for root, dirs, files in os.walk(src):
            # 计算当前子目录相对于源路径的相对路径
            relroot = os.path.relpath(root, src)
            # 复制每个文件
            for name in files:
                # 计算当前文件相对于源路径的相对路径
                relname = os.path.join(relroot, name)
                # 构建源文件的完整路径
                s = os.path.join(src, relname)
                # 构建目标文件的完整路径
                t = os.path.join(trg, relname)
                # 打印操作信息，显示复制过程中的源文件和目标文件路径
                print(f"{verb} {s} -> {t}")
                # 调用移动函数将源文件移动到目标文件路径
                mover(s, t)
            # 创建每个子目录
            for name in dirs:
                # 计算当前子目录相对于源路径的相对路径
                relname = os.path.join(relroot, name)
                # 创建目标路径下对应的子目录，如果已存在则忽略
                os.makedirs(os.path.join(trg, relname), exist_ok=True)
    else:
        # 如果目标路径不是目录，则直接复制源文件到目标路径
        print(f"{verb} {src} -> {trg}")
        # 调用移动函数将源文件移动到目标文件路径
        mover(src, trg)
# 将列表中的每个源文件复制到目标目录中，使用 shutil.copy2 函数，显示 "Copying" 作为操作信息
def _copy_files(listing: list[str], source_dir: str, target_dir: str) -> None:
    for src in listing:
        _move_single(src, source_dir, target_dir, shutil.copy2, "Copying")


# 将列表中的每个源文件创建硬链接到目标目录中，使用 os.link 函数，显示 "Linking" 作为操作信息
def _link_files(listing: list[str], source_dir: str, target_dir: str) -> None:
    for src in listing:
        _move_single(src, source_dir, target_dir, os.link, "Linking")


# 装饰器，用于计时函数执行时间，显示 "Moving nightly files into repo" 作为操作信息
@timed("Moving nightly files into repo")
# 将 PyTorch 文件从临时安装位置移动到仓库目录中
def move_nightly_files(spdir: str, platform: str) -> None:
    """Moves PyTorch files from temporary installed location to repo."""
    # 获取源目录和目标目录
    source_dir = os.path.join(spdir, "torch")
    target_dir = os.path.abspath("torch")
    # 获取文件列表
    listing = _get_listing(source_dir, target_dir, platform)
    # 根据平台选择复制或创建链接文件
    if platform.startswith("win"):
        _copy_files(listing, source_dir, target_dir)
    else:
        try:
            _link_files(listing, source_dir, target_dir)
        except Exception:
            _copy_files(listing, source_dir, target_dir)


# 返回当前系统中所有可用的环境及其路径的字典
def _available_envs() -> dict[str, str]:
    cmd = ["conda", "env", "list"]
    # 执行命令获取环境列表
    p = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    # 解析命令输出，获取环境及其路径信息
    lines = p.stdout.splitlines()
    envs = {}
    for line in map(str.strip, lines):
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            # 未命名的环境
            continue
        envs[parts[0]] = parts[-1]
    return envs


# 装饰器，用于计时函数执行时间，显示 "Writing pytorch-nightly.pth" 作为操作信息
@timed("Writing pytorch-nightly.pth")
# 为当前目录写入 Python 路径文件
def write_pth(env_opts: list[str], platform: str) -> None:
    """Writes Python path file for this dir."""
    env_type, env_dir = env_opts
    if env_type == "--name":
        # 根据环境名查找环境路径
        envs = _available_envs()
        env_dir = envs[env_dir]
    # 获取目标目录路径
    spdir = _site_packages(env_dir, platform)
    # 创建路径文件 pytorch-nightly.pth
    pth = os.path.join(spdir, "pytorch-nightly.pth")
    # 写入路径文件内容
    s = (
        "# This file was autogenerated by PyTorch's tools/nightly.py\n"
        "# Please delete this file if you no longer need the following development\n"
        "# version of PyTorch to be importable\n"
        f"{os.getcwd()}\n"
    )
    with open(pth, "w") as f:
        f.write(s)


# 执行 PyTorch 的开发安装过程
def install(
    *,
    logger: logging.Logger,
    subcommand: str = "checkout",
    branch: str | None = None,
    name: str | None = None,
    prefix: str | None = None,
    channels: Iterable[str] = ("pytorch-nightly",),
    override_channels: bool = False,
) -> None:
    """Development install of PyTorch"""
    # 解决安装依赖、获取 PyTorch 版本及环境信息
    deps, pytorch, platform, existing_env, env_opts = conda_solve(
        name=name, prefix=prefix, channels=channels, override_channels=override_channels
    )
    # 如果有依赖需要安装，则执行依赖安装过程
    if deps:
        deps_install(deps, existing_env, env_opts)
    # 安装 PyTorch 到指定目录
    pytdir = pytorch_install(pytorch)
    # 获取安装后的 site-packages 目录
    spdir = _site_packages(pytdir.name, platform)
    # 根据 subcommand 的不同执行不同的操作
    if subcommand == "checkout":
        checkout_nightly_version(cast(str, branch), spdir)
    elif subcommand == "pull":
        pull_nightly_version(spdir)
    else:
        # 如果 subcommand 不是 'checkout' 或 'pull'，则抛出 ValueError 异常
        raise ValueError(f"Subcommand {subcommand} must be one of: checkout, pull.")
    # 移动 nightly 文件到指定目录 spdir 下的 platform 子目录中
    move_nightly_files(spdir, platform)
    # 在指定的环境选项中写入路径设置
    write_pth(env_opts, platform)
    # 清理 Python 环境临时目录
    pytdir.cleanup()
    # 记录信息到日志，提示 PyTorch 开发环境已设置完成
    logger.info(
        "-------\nPyTorch Development Environment set up!\nPlease activate to "
        "enable this environment:\n  $ conda activate %s",
        env_opts[1],
    )
# 创建命令行解析器对象，命名为"nightly"
def make_parser() -> ArgumentParser:
    p = ArgumentParser("nightly")
    
    # 添加子命令解析器集合，用于处理不同的子命令
    subcmd = p.add_subparsers(dest="subcmd", help="subcommand to execute")
    
    # 添加名为"checkout"的子命令解析器，用于执行新分支的检出操作
    co = subcmd.add_parser("checkout", help="checkout a new branch")
    
    # 为"checkout"子命令解析器添加参数
    co.add_argument(
        "-b",
        "--branch",
        help="Branch name to checkout",
        dest="branch",
        default=None,
        metavar="NAME",
    )
    
    # 添加名为"pull"的子命令解析器，用于将每夜构建的提交合并到当前分支
    pull = subcmd.add_parser(
        "pull", help="pulls the nightly commits into the current branch"
    )
    
    # 通用参数设置，适用于"checkout"和"pull"两个子命令解析器
    subps = [co, pull]
    for subp in subps:
        # 添加环境名称参数
        subp.add_argument(
            "-n",
            "--name",
            help="Name of environment",
            dest="name",
            default=None,
            metavar="ENVIRONMENT",
        )
        # 添加环境路径参数
        subp.add_argument(
            "-p",
            "--prefix",
            help="Full path to environment location (i.e. prefix)",
            dest="prefix",
            default=None,
            metavar="PATH",
        )
        # 添加详细调试信息参数
        subp.add_argument(
            "-v",
            "--verbose",
            help="Provide debugging info",
            dest="verbose",
            default=False,
            action="store_true",
        )
        # 添加是否覆盖通道搜索参数
        subp.add_argument(
            "--override-channels",
            help="Do not search default or .condarc channels.",
            dest="override_channels",
            default=False,
            action="store_true",
        )
        # 添加通道搜索参数
        subp.add_argument(
            "-c",
            "--channel",
            help="Additional channel to search for packages. 'pytorch-nightly' will always be prepended to this list.",
            dest="channels",
            action="append",
            metavar="CHANNEL",
        )
    
    # 返回配置好的命令行解析器对象
    return p


def main(args: Sequence[str] | None = None) -> None:
    """程序的主入口点"""
    # 设置全局日志记录器对象
    global LOGGER
    # 创建命令行解析器对象
    p = make_parser()
    # 解析命令行参数并存储在命名空间对象ns中
    ns = p.parse_args(args)
    # 获取分支名称（如果提供的话）
    ns.branch = getattr(ns, "branch", None)
    # 检查代码仓库状态
    status = check_in_repo()
    # 检查分支状态
    status = status or check_branch(ns.subcmd, ns.branch)
    # 如果状态存在，则退出程序
    if status:
        sys.exit(status)
    
    # 设置默认通道列表，包含"pytorch-nightly"
    channels = ["pytorch-nightly"]
    # 如果命令行参数中指定了其他通道，则将其添加到通道列表中
    if ns.channels:
        channels.extend(ns.channels)
    
    # 使用日志管理器创建上下文，并将其赋值给全局日志记录器LOGGER
    with logging_manager(debug=ns.verbose) as logger:
        LOGGER = logger
        # 调用安装函数，传递命令行参数和日志记录器等参数
        install(
            subcommand=ns.subcmd,
            branch=ns.branch,
            name=ns.name,
            prefix=ns.prefix,
            logger=logger,
            channels=channels,
            override_channels=ns.override_channels,
        )


if __name__ == "__main__":
    # 当脚本直接运行时，调用主函数main()
    main()
```