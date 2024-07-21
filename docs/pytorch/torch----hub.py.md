# `.\pytorch\torch\hub.py`

```py
# 声明导入需要的库和模块
import contextlib  # 上下文管理模块，用于创建上下文管理器
import errno  # 错误码模块，处理操作系统相关的错误码
import hashlib  # 哈希算法模块，用于计算数据的哈希值
import json  # JSON 数据格式处理模块
import os  # 操作系统相关的功能模块
import re  # 正则表达式模块，用于处理正则表达式操作
import shutil  # 文件操作模块，提供了高级的文件操作功能
import sys  # 系统相关的参数和功能模块
import tempfile  # 临时文件和目录创建模块
import uuid  # UUID 模块，用于生成唯一标识符
import warnings  # 警告控制模块，用于管理警告信息的显示
import zipfile  # ZIP 文件处理模块
from pathlib import Path  # 文件路径操作模块，提供了处理文件路径的类和方法
from typing import Any, Dict, Optional  # 类型提示模块，用于静态类型检查
from typing_extensions import deprecated  # 弃用类型的支持扩展
from urllib.error import HTTPError, URLError  # URL 相关的异常模块
from urllib.parse import urlparse  # URL 解析模块，用于解析 URL
from urllib.request import Request, urlopen  # URL 请求和打开模块

# 导入 PyTorch 相关模块
import torch
from torch.serialization import MAP_LOCATION  # PyTorch 模型序列化的映射位置

# 定义一个类 _Faketqdm，用于模拟 tqdm 进度条显示
class _Faketqdm:  # type: ignore[no-redef]
    def __init__(self, total=None, disable=False, unit=None, *args, **kwargs):
        self.total = total
        self.disable = disable
        self.n = 0
        # 忽略所有额外的 *args 和 **kwargs，除非要重新实现 tqdm

    def update(self, n):
        if self.disable:
            return

        self.n += n
        if self.total is None:
            sys.stderr.write(f"\r{self.n:.1f} bytes")
        else:
            sys.stderr.write(f"\r{100 * self.n / float(self.total):.1f}%")
        sys.stderr.flush()

    # 不实现具体的方法；如果需要，使用真正的 tqdm
    def set_description(self, *args, **kwargs):
        pass

    def write(self, s):
        sys.stderr.write(f"{s}\n")

    def close(self):
        self.disable = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return

        sys.stderr.write("\n")

# 尝试导入 tqdm，如果导入失败，则使用 _Faketqdm 作为替代
try:
    from tqdm import tqdm  # 如果 tqdm 已安装，使用它来显示进度条
except ImportError:
    tqdm = _Faketqdm

# 定义导出的模块列表
__all__ = [
    "download_url_to_file",
    "get_dir",
    "help",
    "list",
    "load",
    "load_state_dict_from_url",
    "set_dir",
]

# 匹配文件名中的哈希值，例如从 resnet18-bfd8deac.pth 中提取出 bfd8deac
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")

# 受信任的代码库所有者列表
_TRUSTED_REPO_OWNERS = (
    "facebookresearch",
    "facebookincubator",
    "pytorch",
    "fairinternal",
)

# GitHub Token 环境变量名称
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"

# PyTorch 缓存目录环境变量名称
ENV_TORCH_HOME = "TORCH_HOME"

# XDG 缓存目录环境变量名称
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"

# 默认的缓存目录路径
DEFAULT_CACHE_DIR = "~/.cache"

# 变量名称：依赖
VAR_DEPENDENCY = "dependencies"

# 模块名称：hubconf.py
MODULE_HUBCONF = "hubconf.py"

# 读取数据时的块大小
READ_DATA_CHUNK = 128 * 1024

# Hub 目录的路径（可选）
_hub_dir: Optional[str] = None

# 上下文管理器：将指定路径添加到 sys.path 中
@contextlib.contextmanager
def _add_to_sys_path(path):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)

# 从指定路径导入模块
def _import_module(name, path):
    import importlib.util
    from importlib.abc import Loader

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, Loader)
    spec.loader.exec_module(module)
    return module

# 如果指定路径存在，则删除对应的文件或目录
def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

# Git 存档链接生成函数的声明
def _git_archive_link(repo_owner, repo_name, ref):
   `
# 生成 GitHub 仓库的 ZIP 文件下载链接，参考 GitHub API 文档
return f"https://github.com/{repo_owner}/{repo_name}/zipball/{ref}"
# 检查指定模块中是否定义了可调用的函数，如果没有则返回 None
def _load_attr_from_module(module, func_name):
    if func_name not in dir(module):
        return None
    # 返回指定函数名在模块中的实际可调用对象
    return getattr(module, func_name)


# 获取 Torch 的主目录路径
def _get_torch_home():
    # 获取环境变量中定义的 Torch 主目录路径，若未定义则使用默认缓存目录中的 "torch"
    torch_home = os.path.expanduser(
        os.getenv(
            ENV_TORCH_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch"),
        )
    )
    return torch_home


# 解析 GitHub 仓库信息，包括所有者、仓库名及分支或标签（如果指定的话）
def _parse_repo_info(github):
    if ":" in github:
        repo_info, ref = github.split(":")
    else:
        repo_info, ref = github, None
    repo_owner, repo_name = repo_info.split("/")

    if ref is None:
        # 如果用户未指定 ref，尝试确定默认分支：main 或 master
        try:
            with urlopen(f"https://github.com/{repo_owner}/{repo_name}/tree/main/"):
                ref = "main"
        except HTTPError as e:
            if e.code == 404:
                ref = "master"
            else:
                raise
        except URLError as e:
            # 如果无法连接互联网，尝试从缓存中获取信息作为最后手段
            for possible_ref in ("main", "master"):
                if os.path.exists(
                    f"{get_dir()}/{repo_owner}_{repo_name}_{possible_ref}"
                ):
                    ref = possible_ref
                    break
            if ref is None:
                raise RuntimeError(
                    "It looks like there is no internet connection and the "
                    f"repo could not be found in the cache ({get_dir()})"
                ) from e
    return repo_owner, repo_name, ref


# 读取指定 URL 的内容并以 UTF-8 解码返回
def _read_url(url):
    with urlopen(url) as r:
        return r.read().decode(r.headers.get_content_charset("utf-8"))


# 验证指定 GitHub 仓库不是派生库（forked repo）
def _validate_not_a_forked_repo(repo_owner, repo_name, ref):
    # 使用 urlopen 避免依赖本地 git
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get(ENV_GITHUB_TOKEN)
    if token is not None:
        headers["Authorization"] = f"token {token}"
    for url_prefix in (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/branches",
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/tags",
    ):
        page = 0
        while True:
            page += 1
            url = f"{url_prefix}?per_page=100&page={page}"
            response = json.loads(_read_url(Request(url, headers=headers)))
            # 空响应表示无更多数据可处理
            if not response:
                break
            for br in response:
                if br["name"] == ref or br["commit"]["sha"].startswith(ref):
                    return

    # 若未找到指定的 ref，则抛出错误
    raise ValueError(
        f"Cannot find {ref} in https://github.com/{repo_owner}/{repo_name}. "
        "If it's a commit from a forked repo, please call hub.load() with forked repo directly."
    )
# 设置用于保存下载文件的目录
hub_dir = get_dir()
# 如果目录不存在，则创建它
os.makedirs(hub_dir, exist_ok=True)

# 解析 GitHub 仓库信息，获取所有者、仓库名和引用（分支或标签）
repo_owner, repo_name, ref = _parse_repo_info(github)

# GitHub 允许使用斜杠 '/' 作为分支名，但这在 Linux 和 Windows 上会引起路径混淆。
# 由于 GitHub 分支名不允许反斜杠，因此这里无需担心转义问题。
normalized_br = ref.replace("/", "_")

# GitHub 将文件夹 repo-v1.x.x 重命名为 repo-1.x.x
# 在下载 ZIP 文件并检查名称之前，我们不知道仓库的确切名称。
# 为了检查缓存的仓库是否存在，我们需要规范化文件夹名称。
owner_name_branch = "_".join([repo_owner, repo_name, normalized_br])
repo_dir = os.path.join(hub_dir, owner_name_branch)

# 检查仓库是否在受信任的列表中
_check_repo_is_trusted(
    repo_owner,
    repo_name,
    owner_name_branch,
    trust_repo=trust_repo,
    calling_fn=calling_fn,
)

# 根据是否需要强制重新加载和仓库目录是否存在来决定是否使用缓存
use_cache = (not force_reload) and os.path.exists(repo_dir)

# 如果使用缓存，并且 verbose 参数为 True，则输出信息指示使用了缓存
if use_cache:
    if verbose:
        sys.stderr.write(f"Using cache found in {repo_dir}\n")
    else:
        # 如果不是第一种情况，验证标签/分支是否来自原始仓库而不是分叉仓库
        if not skip_validation:
            _validate_not_a_forked_repo(repo_owner, repo_name, ref)

        # 缓存文件路径为将标准化的分支名与 ".zip" 后缀拼接而成
        cached_file = os.path.join(hub_dir, normalized_br + ".zip")
        # 如果存在同名文件则删除它
        _remove_if_exists(cached_file)

        try:
            # 获取 Git 存档链接的 URL
            url = _git_archive_link(repo_owner, repo_name, ref)
            sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
            # 下载 URL 指向的文件到指定路径，关闭进度显示
            download_url_to_file(url, cached_file, progress=False)
        except HTTPError as err:
            if err.code == 300:
                # 收到 300 多重选择错误通常意味着引用既是标签又是分支
                # 这可以通过显式使用 refs/heads/ 或 refs/tags/ 来消除歧义
                # 参见 https://git-scm.com/book/en/v2/Git-Internals-Git-References
                # 在这里，我们与 Git 行为一致：发出警告，并假设用户想要的是分支
                warnings.warn(
                    f"The ref {ref} is ambiguous. Perhaps it is both a tag and a branch in the repo? "
                    "Torchhub will now assume that it's a branch. "
                    "You can disambiguate tags and branches by explicitly passing refs/heads/branch_name or "
                    "refs/tags/tag_name as the ref. That might require using skip_validation=True."
                )
                # 通过增加 refs/heads/ 前缀来消除歧义，并重新获取 Git 存档链接的 URL
                disambiguated_branch_ref = f"refs/heads/{ref}"
                url = _git_archive_link(
                    repo_owner, repo_name, ref=disambiguated_branch_ref
                )
                # 下载 URL 指向的文件到指定路径，关闭进度显示
                download_url_to_file(url, cached_file, progress=False)
            else:
                raise

        # 使用 zipfile 模块打开缓存文件
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            # 获取解压后的仓库目录名
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(hub_dir, extraced_repo_name)
            # 如果已存在同名目录则删除它
            _remove_if_exists(extracted_repo)
            # 解压缩文件到指定目录
            cached_zipfile.extractall(hub_dir)

        # 删除缓存文件
        _remove_if_exists(cached_file)
        # 删除原仓库目录
        _remove_if_exists(repo_dir)
        # 将解压后的仓库目录重命名为原仓库目录名
        shutil.move(extracted_repo, repo_dir)  # rename the repo

    # 返回更新后的仓库目录路径
    return repo_dir
# 检查指定仓库是否被信任
def _check_repo_is_trusted(
    repo_owner,
    repo_name,
    owner_name_branch,
    trust_repo,
    calling_fn="load",
):
    # 获取存储目录路径
    hub_dir = get_dir()
    # 构建 trusted_list 文件路径
    filepath = os.path.join(hub_dir, "trusted_list")

    # 如果 trusted_list 文件不存在，则创建一个空文件
    if not os.path.exists(filepath):
        Path(filepath).touch()
    
    # 读取 trusted_list 文件中的所有行，并去除首尾空白字符，存入元组 trusted_repos
    with open(filepath) as file:
        trusted_repos = tuple(line.strip() for line in file)

    # 获取 hub_dir 目录下所有的目录名，这些目录代表旧版本中已下载的仓库，将其存入元组 trusted_repos_legacy
    trusted_repos_legacy = next(os.walk(hub_dir))[1]

    # 组合 repo_owner 和 repo_name，生成 owner_name，检查仓库是否被信任的逻辑
    owner_name = "_".join([repo_owner, repo_name])
    is_trusted = (
        owner_name in trusted_repos  # 检查 owner_name 是否在 trusted_repos 中
        or owner_name_branch in trusted_repos_legacy  # 检查 owner_name_branch 是否在 trusted_repos_legacy 中
        or repo_owner in _TRUSTED_REPO_OWNERS  # 检查 repo_owner 是否在 _TRUSTED_REPO_OWNERS 中
    )

    # 如果 trust_repo 参数为 None，则发出警告，并返回
    if trust_repo is None:
        if not is_trusted:
            warnings.warn(
                "You are about to download and run code from an untrusted repository. In a future release, this won't "
                "be allowed. To add the repository to your trusted list, change the command to {calling_fn}(..., "
                "trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, "
                f"or {calling_fn}(..., trust_repo=True), which will assume that the prompt is to be answered with "
                f"'yes'. You can also use {calling_fn}(..., trust_repo='check') which will only prompt for "
                f"confirmation if the repo is not already trusted. This will eventually be the default behaviour"
            )
        return

    # 如果 trust_repo 参数为 False 或者为 "check" 且当前仓库未被信任，则需要用户确认
    if (trust_repo is False) or (trust_repo == "check" and not is_trusted):
        response = input(
            f"The repository {owner_name} does not belong to the list of trusted repositories and as such cannot be downloaded. "
            "Do you trust this repository and wish to add it to the trusted list of repositories (y/N)?"
        )
        # 根据用户的响应决定是否信任该仓库
        if response.lower() in ("y", "yes"):
            if is_trusted:
                print("The repository is already trusted.")
        elif response.lower() in ("n", "no", ""):
            raise Exception("Untrusted repository.")  # noqa: TRY002
        else:
            raise ValueError(f"Unrecognized response {response}.")

    # 至此，确认用户信任该仓库或者希望信任它
    # 如果仓库尚未被信任，则将 owner_name 写入 trusted_list 文件
    if not is_trusted:
        with open(filepath, "a") as file:
            file.write(owner_name + "\n")


# 检查指定模块是否存在
def _check_module_exists(name):
    import importlib.util

    # 使用 importlib.util.find_spec 方法检查模块是否存在
    return importlib.util.find_spec(name) is not None


# 检查模块的依赖关系
def _check_dependencies(m):
    dependencies = _load_attr_from_module(m, VAR_DEPENDENCY)
    # 如果依赖列表不为空，则进行以下操作
    if dependencies is not None:
        # 检查依赖中是否有任何一个模块不存在，将不存在的模块组成列表
        missing_deps = [pkg for pkg in dependencies if not _check_module_exists(pkg)]
        # 如果存在缺失的模块
        if len(missing_deps):
            # 抛出运行时异常，指明缺失的依赖模块
            raise RuntimeError(f"Missing dependencies: {', '.join(missing_deps)}")
def _load_entry_from_hubconf(m, model):
    # 检查输入参数是否为字符串，如果不是则抛出数值错误异常
    if not isinstance(model, str):
        raise ValueError("Invalid input: model should be a string of function name")

    # 检查依赖项，确保所需的模块已经导入
    _check_dependencies(m)

    # 从指定模块中加载指定的函数对象
    func = _load_attr_from_module(m, model)

    # 如果加载的函数对象为空或不可调用，抛出运行时错误
    if func is None or not callable(func):
        raise RuntimeError(f"Cannot find callable {model} in hubconf")

    # 返回加载的可调用函数对象
    return func


def get_dir():
    r"""
    获取用于存储下载模型和权重的 Torch Hub 缓存目录。

    如果未调用 :func:`~torch.hub.set_dir`，默认路径为 ``$TORCH_HOME/hub``，
    其中环境变量 ``$TORCH_HOME`` 默认为 ``$XDG_CACHE_HOME/torch``。
    ``$XDG_CACHE_HOME`` 遵循 Linux 文件系统布局的 X Design Group 规范，
    如果环境变量未设置，默认为 ``~/.cache``。
    """
    # 如果设置了旧的环境变量 TORCH_HUB，发出警告建议使用 TORCH_HOME 替代
    if os.getenv("TORCH_HUB"):
        warnings.warn("TORCH_HUB is deprecated, please use env TORCH_HOME instead")

    # 如果 _hub_dir 已经设置，则返回其值作为 Torch Hub 目录
    if _hub_dir is not None:
        return _hub_dir
    # 否则返回默认的 Torch Hub 目录路径
    return os.path.join(_get_torch_home(), "hub")


def set_dir(d):
    r"""
    可选地设置用于保存下载模型和权重的 Torch Hub 目录。

    Args:
        d (str): 本地文件夹路径，用于保存下载的模型和权重。
    """
    # 全局变量 _hub_dir 被设置为扩展用户目录后的输入路径 d
    global _hub_dir
    _hub_dir = os.path.expanduser(d)


def list(
    github,
    force_reload=False,
    skip_validation=False,
    trust_repo=None,
    verbose=True,
):
    r"""
    列出由 github 指定的仓库中所有可调用的入口点。
    """
    # 根据提供的 GitHub 仓库信息下载或获取缓存中的模块目录
    repo_dir = _get_cache_or_reload(
        github,  # GitHub 仓库的所有者、仓库名及可选的引用
        force_reload,  # 是否强制重新加载，即丢弃现有缓存并重新下载
        trust_repo,  # 用于确定是否信任仓库的参数，可以是 "check"、True、False 或 None
        "list",  # 操作类型，这里是获取列表
        verbose=verbose,  # 是否输出详细信息，默认为 True
        skip_validation=skip_validation,  # 是否跳过验证，默认为 False，表示需要验证 GitHub 仓库的归属
    )
    
    # 将模块目录添加到系统路径中，以便 Python 可以找到对应的模块文件
    with _add_to_sys_path(repo_dir):
        # 构建 hubconf 文件的完整路径
        hubconf_path = os.path.join(repo_dir, MODULE_HUBCONF)
        # 根据路径导入模块
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)
    
    # 从导入的 hub 模块中筛选出可调用的入口点函数，排除掉以 '_' 开头的内部辅助函数
    entrypoints = [
        f
        for f in dir(hub_module)
        if callable(getattr(hub_module, f)) and not f.startswith("_")
    ]
    
    # 返回找到的所有可调用的入口点函数列表
    return entrypoints
# 定义函数 help，用于显示指定模型入口点的文档字符串
def help(github, model, force_reload=False, skip_validation=False, trust_repo=None):
    # 获取缓存目录或重新加载指定 github 仓库
    repo_dir = _get_cache_or_reload(
        github,
        force_reload,
        trust_repo,
        "help",
        verbose=True,
        skip_validation=skip_validation,
    )

    # 将 repo_dir 添加到系统路径中，以便导入模块
    with _add_to_sys_path(repo_dir):
        # 构建 hubconf.py 的完整路径
        hubconf_path = os.path.join(repo_dir, MODULE_HUBCONF)
        # 动态导入 hubconf 模块
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)

    # 从 hubconf 模块中加载指定模型的入口点
    entry = _load_entry_from_hubconf(hub_module, model)

    # 返回加载模型入口点的文档字符串
    return entry.__doc__


def load(
    repo_or_dir,
    model,
    *args,
    source="github",
    trust_repo=None,
    force_reload=False,
    verbose=True,
    skip_validation=False,
    **kwargs,
):
    r"""
    从 github 仓库或本地目录加载模型。

    注意：加载模型是典型用例，但也可以用于
    for loading other objects such as tokenizers, loss functions, etc.


    # 用于加载其他对象，例如分词器、损失函数等。



    If ``source`` is 'github', ``repo_or_dir`` is expected to be
    of the form ``repo_owner/repo_name[:ref]`` with an optional
    ref (a tag or a branch).


    # 如果 ``source`` 是 'github'，则 ``repo_or_dir`` 应该是形如
    # ``repo_owner/repo_name[:ref]`` 的格式，其中 ref 是可选的，
    # 可以是标签或分支。



    If ``source`` is 'local', ``repo_or_dir`` is expected to be a
    path to a local directory.


    # 如果 ``source`` 是 'local'，则 ``repo_or_dir`` 应该是本地目录的路径。
    Args:
        repo_or_dir (str): If ``source`` is 'github',
            this should correspond to a github repo with format ``repo_owner/repo_name[:ref]`` with
            an optional ref (tag or branch), for example 'pytorch/vision:0.10'. If ``ref`` is not specified,
            the default branch is assumed to be ``main`` if it exists, and otherwise ``master``.
            If ``source`` is 'local'  then it should be a path to a local directory.
        model (str): the name of a callable (entrypoint) defined in the
            repo/dir's ``hubconf.py``.
        *args (optional): the corresponding args for callable ``model``.
        source (str, optional): 'github' or 'local'. Specifies how
            ``repo_or_dir`` is to be interpreted. Default is 'github'.
        trust_repo (bool, str or None): ``"check"``, ``True``, ``False`` or ``None``.
            This parameter was introduced in v1.12 and helps ensuring that users
            only run code from repos that they trust.

            - If ``False``, a prompt will ask the user whether the repo should
              be trusted.
            - If ``True``, the repo will be added to the trusted list and loaded
              without requiring explicit confirmation.
            - If ``"check"``, the repo will be checked against the list of
              trusted repos in the cache. If it is not present in that list, the
              behaviour will fall back onto the ``trust_repo=False`` option.
            - If ``None``: this will raise a warning, inviting the user to set
              ``trust_repo`` to either ``False``, ``True`` or ``"check"``. This
              is only present for backward compatibility and will be removed in
              v2.0.

            Default is ``None`` and will eventually change to ``"check"`` in v2.0.
        force_reload (bool, optional): whether to force a fresh download of
            the github repo unconditionally. Does not have any effect if
            ``source = 'local'``. Default is ``False``.
        verbose (bool, optional): If ``False``, mute messages about hitting
            local caches. Note that the message about first download cannot be
            muted. Does not have any effect if ``source = 'local'``.
            Default is ``True``.
        skip_validation (bool, optional): if ``False``, torchhub will check that the branch or commit
            specified by the ``github`` argument properly belongs to the repo owner. This will make
            requests to the GitHub API; you can specify a non-default GitHub token by setting the
            ``GITHUB_TOKEN`` environment variable. Default is ``False``.
        **kwargs (optional): the corresponding kwargs for callable ``model``.

    Returns:
        The output of the ``model`` callable when called with the given
        ``*args`` and ``**kwargs``.
    """
    将源字符串转换为小写
    """
    source = source.lower()

    """
    如果源不是 'github' 或 'local'，抛出值错误异常
    """
    if source not in ("github", "local"):
        raise ValueError(
            f'Unknown source: "{source}". Allowed values: "github" | "local".'
        )

    """
    如果源是 'github'，则从缓存中获取或重新加载资源或目录
    """
    if source == "github":
        repo_or_dir = _get_cache_or_reload(
            repo_or_dir,
            force_reload,
            trust_repo,
            "load",
            verbose=verbose,
            skip_validation=skip_validation,
        )

    """
    载入本地资源或目录中的模型
    """
    model = _load_local(repo_or_dir, model, *args, **kwargs)

    """
    返回载入的模型
    """
    return model
# 从本地目录加载模型，使用 hubconf.py 中定义的入口点来实现
def _load_local(hubconf_dir, model, *args, **kwargs):
    # 将 hubconf_dir 添加到系统路径中，以便可以导入其中的模块
    with _add_to_sys_path(hubconf_dir):
        # 构建 hubconf.py 的完整路径并导入其模块
        hubconf_path = os.path.join(hubconf_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)

        # 从 hubconf 模块中加载指定的 model 入口点
        entry = _load_entry_from_hubconf(hub_module, model)
        # 使用加载的 entry 创建模型，传递给定的 args 和 kwargs
        model = entry(*args, **kwargs)

    # 返回加载的模型对象
    return model


# 将指定 URL 的对象下载到本地文件
def download_url_to_file(
    url: str,
    dst: str,
    hash_prefix: Optional[str] = None,
    progress: bool = True,
) -> None:
    # 获取要下载文件的大小信息
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # 将目标路径扩展成用户的绝对路径
    dst = os.path.expanduser(dst)
    # 尝试创建一个唯一的临时文件名来保存下载的文件
    for seq in range(tempfile.TMP_MAX):
        tmp_dst = dst + "." + uuid.uuid4().hex + ".partial"
        try:
            f = open(tmp_dst, "w+b")
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")
    # 尝试执行以下代码块，处理文件写入和哈希计算
    try:
        # 如果指定了哈希前缀，创建 SHA-256 哈希对象
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        
        # 使用 tqdm 创建进度条，显示文件写入进度
        with tqdm(
            total=file_size,  # 总共要写入的字节数
            disable=not progress,  # 是否禁用进度条显示
            unit="B",  # 使用字节作为单位
            unit_scale=True,  # 使用适合的单位进行缩放
            unit_divisor=1024,  # 单位除数，用于划分单位
        ) as pbar:
            # 循环读取数据块并写入文件，直到文件末尾
            while True:
                buffer = u.read(READ_DATA_CHUNK)  # 从数据流中读取一块数据
                if len(buffer) == 0:  # 如果读取到的数据块为空，则跳出循环
                    break
                f.write(buffer)  # 将读取的数据块写入文件  # type: ignore[possibly-undefined]
                
                # 如果指定了哈希前缀，更新 SHA-256 哈希对象
                if hash_prefix is not None:
                    sha256.update(buffer)  # 更新哈希对象  # type: ignore[possibly-undefined]
                
                pbar.update(len(buffer))  # 更新进度条，表示已处理的字节数
        
        f.close()  # 关闭文件
        
        # 如果指定了哈希前缀，检查计算出的哈希值是否符合预期
        if hash_prefix is not None:
            digest = sha256.hexdigest()  # 计算最终的 SHA-256 哈希值  # type: ignore[possibly-undefined]
            if digest[: len(hash_prefix)] != hash_prefix:  # 检查哈希值的前缀是否匹配预期
                raise RuntimeError(
                    f'invalid hash value (expected "{hash_prefix}", got "{digest}")'
                )
        
        # 将临时文件移动到目标位置
        shutil.move(f.name, dst)
    
    finally:
        f.close()  # 在 finally 块中确保文件被关闭
        if os.path.exists(f.name):  # 检查临时文件是否存在
            os.remove(f.name)  # 如果存在，则删除临时文件
# 检查给定的文件名是否符合旧的 ZIP 格式，用于支持手动压缩的 zipfile
def _is_legacy_zip_format(filename: str) -> bool:
    # 判断文件是否为有效的 ZIP 文件
    if zipfile.is_zipfile(filename):
        # 获取 ZIP 文件中的文件列表信息
        infolist = zipfile.ZipFile(filename).infolist()
        # 返回 True 如果 ZIP 文件只包含一个文件且不是目录，否则返回 False
        return len(infolist) == 1 and not infolist[0].is_dir()
    return False


# 加载旧的 ZIP 格式的模型状态字典
@deprecated(
    "Falling back to the old format < 1.6. This support will be "
    "deprecated in favor of default zipfile format introduced in 1.6. "
    "Please redo torch.save() to save it in the new zipfile format.",
    category=FutureWarning,
)
def _legacy_zip_load(
    filename: str,
    model_dir: str,
    map_location: MAP_LOCATION,
    weights_only: bool,
) -> Dict[str, Any]:
    # 提示：extractall() 默认覆盖已存在的文件，不需要事先清理。
    #       这里故意不处理 tarfile，因为我们的旧的序列化格式是 tar。
    #       例如 widely used 的 resnet18-5c106cde.pth。
    # 打开给定的 ZIP 文件
    with zipfile.ZipFile(filename) as f:
        # 获取 ZIP 文件中的成员列表
        members = f.infolist()
        # 如果 ZIP 文件中的成员不只一个，则抛出运行时错误
        if len(members) != 1:
            raise RuntimeError("Only one file(not dir) is allowed in the zipfile")
        # 解压 ZIP 文件中的内容到指定的模型目录
        f.extractall(model_dir)
        # 获取解压后的文件名
        extraced_name = members[0].filename
        # 拼接得到解压后的文件路径
        extracted_file = os.path.join(model_dir, extraced_name)
    # 使用 Torch 加载解压后的文件，返回加载的对象
    return torch.load(
        extracted_file, map_location=map_location, weights_only=weights_only
    )


# 从指定的 URL 加载 Torch 序列化对象的状态字典
def load_state_dict_from_url(
    url: str,
    model_dir: Optional[str] = None,
    map_location: MAP_LOCATION = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
    weights_only: bool = False,
) -> Dict[str, Any]:
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.
    """
    # 如果设置了环境变量 TORCH_MODEL_ZOO，发出警告提示用户使用 TORCH_HOME 替代
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead"
        )

    # 如果未指定模型保存目录，则使用默认目录下的 'checkpoints' 子目录
    if model_dir is None:
        hub_dir = get_dir()  # 获取默认的模型存储目录
        model_dir = os.path.join(hub_dir, "checkpoints")

    # 确保模型保存目录存在，若不存在则创建之
    os.makedirs(model_dir, exist_ok=True)

    # 解析 URL 的各个部分，提取出文件名
    parts = urlparse(url)
    filename = os.path.basename(parts.path)

    # 如果指定了 file_name 参数，则使用指定的文件名
    if file_name is not None:
        filename = file_name

    # 构建本地缓存文件的路径
    cached_file = os.path.join(model_dir, filename)

    # 如果本地缓存文件不存在，则下载文件
    if not os.path.exists(cached_file):
        # 在标准错误输出中显示下载进度信息
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')

        # 如果启用了 check_hash 参数，从文件名中提取哈希前缀
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r 是一个 Optional[Match[str]]
            hash_prefix = r.group(1) if r else None

        # 调用下载函数将文件下载到指定路径
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    # 如果本地缓存文件是旧的 ZIP 格式，则调用 _legacy_zip_load 函数加载
    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location, weights_only)

    # 否则，使用 torch.load 加载模型数据文件
    return torch.load(cached_file, map_location=map_location, weights_only=weights_only)
```