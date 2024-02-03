# `stable-diffusion-webui\modules\extensions.py`

```py
# 导入必要的模块和库
from __future__ import annotations
import configparser
import os
import threading
import re
from modules import shared, errors, cache, scripts
from modules.gitpython_hack import Repo
from modules.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401

# 确保 extensions_dir 目录存在，如果不存在则创建
os.makedirs(extensions_dir, exist_ok=True)

# 定义函数 active，根据条件返回不同的扩展列表
def active():
    # 如果命令行参数中禁用了所有扩展或配置文件中设置禁用了所有扩展，则返回空列表
    if shared.cmd_opts.disable_all_extensions or shared.opts.disable_all_extensions == "all":
        return []
    # 如果命令行参数中禁用了额外扩展或配置文件中设置禁用了额外扩展，则返回内置且启用的扩展列表
    elif shared.cmd_opts.disable_extra_extensions or shared.opts.disable_all_extensions == "extra":
        return [x for x in extensions if x.enabled and x.is_builtin]
    # 否则返回所有启用的扩展列表
    else:
        return [x for x in extensions if x.enabled]

# 定义类 ExtensionMetadata，用于处理扩展元数据
class ExtensionMetadata:
    # 定义类属性
    filename = "metadata.ini"
    config: configparser.ConfigParser
    canonical_name: str
    requires: list

    # 初始化方法，接收路径和规范名称作为参数
    def __init__(self, path, canonical_name):
        # 创建 ConfigParser 对象
        self.config = configparser.ConfigParser()

        # 拼接元数据文件路径
        filepath = os.path.join(path, self.filename)
        # 如果文件存在，则尝试读取配置信息
        if os.path.isfile(filepath):
            try:
                self.config.read(filepath)
            except Exception:
                # 报告读取元数据文件出错的异常信息
                errors.report(f"Error reading {self.filename} for extension {canonical_name}.", exc_info=True)

        # 获取扩展的规范名称，如果不存在则使用默认规范名称
        self.canonical_name = self.config.get("Extension", "Name", fallback=canonical_name)
        self.canonical_name = canonical_name.lower().strip()

        # 获取扩展所需的脚本要求
        self.requires = self.get_script_requirements("Requires", "Extension")
    # 从配置文件中读取要求列表，field 是 ini 文件中字段的名称，如 Requires 或 Before，section 是 ini 文件中 [section] 的名称；
    # 如果指定了 extra_section，则从 [extra_section] 中读取更多要求。
    def get_script_requirements(self, field, section, extra_section=None):
        """reads a list of requirements from the config; field is the name of the field in the ini file,
        like Requires or Before, and section is the name of the [section] in the ini file; additionally,
        reads more requirements from [extra_section] if specified."""

        # 从配置文件中获取指定 section 和 field 的值，如果没有找到则返回空字符串
        x = self.config.get(section, field, fallback='')

        # 如果指定了 extra_section，则从 extra_section 中获取 field 的值，并与之前的值拼接
        if extra_section:
            x = x + ', ' + self.config.get(extra_section, field, fallback='')

        # 调用 parse_list 方法将字符串转换为小写的列表并返回
        return self.parse_list(x.lower())

    # 将配置文件中的一行文本转换为 Python 列表
    def parse_list(self, text):
        """converts a line from config ("ext1 ext2, ext3  ") into a python list (["ext1", "ext2", "ext3"])"""

        # 如果文本为空，则返回空列表
        if not text:
            return []

        # 使用正则表达式将文本按照逗号和空格分割成列表，并去除首尾空格
        # both "," and " " are accepted as separator
        return [x for x in re.split(r"[,\s]+", text.strip()) if x]
# 定义 Extension 类
class Extension:
    # 创建一个线程锁
    lock = threading.Lock()
    # 缓存的字段列表
    cached_fields = ['remote', 'commit_date', 'branch', 'commit_hash', 'version']
    # 扩展元数据
    metadata: ExtensionMetadata

    # 初始化方法，设置扩展的属性
    def __init__(self, name, path, enabled=True, is_builtin=False, metadata=None):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.status = ''
        self.can_update = False
        self.is_builtin = is_builtin
        self.commit_hash = ''
        self.commit_date = None
        self.version = ''
        self.branch = None
        self.remote = None
        self.have_info_from_repo = False
        # 如果有元数据则使用，否则创建一个新的 ExtensionMetadata 对象
        self.metadata = metadata if metadata else ExtensionMetadata(self.path, name.lower())
        self.canonical_name = metadata.canonical_name

    # 将 Extension 对象转换为字典
    def to_dict(self):
        return {x: getattr(self, x) for x in self.cached_fields}

    # 从字典中恢复 Extension 对象的属性
    def from_dict(self, d):
        for field in self.cached_fields:
            setattr(self, field, d[field])

    # 从仓库中读取扩展信息
    def read_info_from_repo(self):
        # 如果是内置扩展或已经获取过信息，则直接返回
        if self.is_builtin or self.have_info_from_repo:
            return

        # 从仓库中读取信息的内部方法
        def read_from_repo():
            with self.lock:
                if self.have_info_from_repo:
                    return

                self.do_read_info_from_repo()

                return self.to_dict()

        try:
            # 从缓存中获取数据，如果不存在则调用 read_from_repo 方法
            d = cache.cached_data_for_file('extensions-git', self.name, os.path.join(self.path, ".git"), read_from_repo)
            self.from_dict(d)
        except FileNotFoundError:
            pass
        # 如果状态为空，则设置为未知状态
        self.status = 'unknown' if self.status == '' else self.status
    # 从仓库中读取信息
    def do_read_info_from_repo(self):
        # 初始化仓库对象为 None
        repo = None
        try:
            # 检查路径下是否存在 .git 文件夹，如果存在则创建 Repo 对象
            if os.path.exists(os.path.join(self.path, ".git")):
                repo = Repo(self.path)
        except Exception:
            # 报告读取仓库信息时的错误
            errors.report(f"Error reading github repository info from {self.path}", exc_info=True)

        # 如果仓库对象为 None 或者是 bare 仓库，则将远程地址设为 None
        if repo is None or repo.bare:
            self.remote = None
        else:
            try:
                # 获取仓库的远程地址
                self.remote = next(repo.remote().urls, None)
                # 获取最新提交的信息
                commit = repo.head.commit
                self.commit_date = commit.committed_date
                # 如果存在活跃分支，则获取分支名
                if repo.active_branch:
                    self.branch = repo.active_branch.name
                # 获取最新提交的哈希值
                self.commit_hash = commit.hexsha
                # 获取版本号（取哈希值的前8位）
                self.version = self.commit_hash[:8]

            except Exception:
                # 报告从 Git 仓库读取扩展数据时的错误
                errors.report(f"Failed reading extension data from Git repository ({self.name})", exc_info=True)
                self.remote = None

        # 标记已经从仓库中获取了信息
        self.have_info_from_repo = True

    # 列出指定子目录下指定扩展名的文件
    def list_files(self, subdir, extension):
        # 构建子目录的完整路径
        dirpath = os.path.join(self.path, subdir)
        # 如果路径不存在或者不是目录，则返回空列表
        if not os.path.isdir(dirpath):
            return []

        # 初始化结果列表
        res = []
        # 遍历子目录下的文件名，创建 ScriptFile 对象并添加到结果列表中
        for filename in sorted(os.listdir(dirpath)):
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename)))

        # 过滤结果列表，只保留指定扩展名的文件并且是文件而不是目录
        res = [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

        # 返回过滤后的结果列表
        return res
    # 检查是否有更新可用
    def check_updates(self):
        # 根据路径创建仓库对象
        repo = Repo(self.path)
        # 遍历远程仓库的更新，dry_run=True表示不执行实际操作
        for fetch in repo.remote().fetch(dry_run=True):
            # 如果有新的提交
            if fetch.flags != fetch.HEAD_UPTODATE:
                self.can_update = True
                self.status = "new commits"
                return

        try:
            # 获取远程仓库的 HEAD 指针
            origin = repo.rev_parse('origin')
            # 如果本地 HEAD 指针落后于远程仓库
            if repo.head.commit != origin:
                self.can_update = True
                self.status = "behind HEAD"
                return
        except Exception:
            # 捕获异常，设置更新状态为未知
            self.can_update = False
            self.status = "unknown (remote error)"
            return

        # 如果没有更新可用
        self.can_update = False
        self.status = "latest"

    # 从远程仓库拉取最新提交并硬重置本地仓库
    def fetch_and_reset_hard(self, commit='origin'):
        # 根据路径创建仓库对象
        repo = Repo(self.path)
        # 修复错误：`error: Your local changes to the following files would be overwritten by merge`
        # 因为 WSL2 Docker 设置了 755 的文件权限而不是 644，导致此错误
        # 拉取所有远程分支的更新
        repo.git.fetch(all=True)
        # 硬重置本地仓库到指定提交
        repo.git.reset(commit, hard=True)
        # 标记未从仓库获取信息
        self.have_info_from_repo = False
# 清空已加载的扩展列表
extensions.clear()

# 检查是否禁用了所有扩展
if shared.cmd_opts.disable_all_extensions:
    print("*** \"--disable-all-extensions\" arg was used, will not load any extensions ***")
elif shared.opts.disable_all_extensions == "all":
    print("*** \"Disable all extensions\" option was set, will not load any extensions ***")
elif shared.cmd_opts.disable_extra_extensions:
    print("*** \"--disable-extra-extensions\" arg was used, will only load built-in extensions ***")
elif shared.opts.disable_all_extensions == "extra":
    print("*** \"Disable all extensions\" option was set, will only load built-in extensions ***")

# 存储已加载的扩展
loaded_extensions = {}

# 遍历扩展目录并加载元数据
for dirname in [extensions_builtin_dir, extensions_dir]:
    if not os.path.isdir(dirname):
        continue

    for extension_dirname in sorted(os.listdir(dirname)):
        path = os.path.join(dirname, extension_dirname)
        if not os.path.isdir(path):
            continue

        canonical_name = extension_dirname
        metadata = ExtensionMetadata(path, canonical_name)

        # 检查重复的 canonical 名称
        already_loaded_extension = loaded_extensions.get(metadata.canonical_name)
        if already_loaded_extension is not None:
            errors.report(f'Duplicate canonical name "{canonical_name}" found in extensions "{extension_dirname}" and "{already_loaded_extension.name}". Former will be discarded.', exc_info=False)
            continue

        is_builtin = dirname == extensions_builtin_dir
        extension = Extension(name=extension_dirname, path=path, enabled=extension_dirname not in shared.opts.disabled_extensions, is_builtin=is_builtin, metadata=metadata)
        extensions.append(extension)
        loaded_extensions[canonical_name] = extension

# 检查扩展的要求
    # 遍历给定的扩展列表
    for extension in extensions:
        # 遍历当前扩展所需的其他扩展列表
        for req in extension.metadata.requires:
            # 获取当前所需扩展是否已加载
            required_extension = loaded_extensions.get(req)
            # 如果所需扩展未加载，则报告错误并继续下一个所需扩展
            if required_extension is None:
                errors.report(f'Extension "{extension.name}" requires "{req}" which is not installed.', exc_info=False)
                continue

            # 如果当前扩展未启用，则报告错误并继续下一个扩展
            if not extension.enabled:
                errors.report(f'Extension "{extension.name}" requires "{required_extension.name}" which is disabled.', exc_info=False)
                continue
# 定义一个空的 Extension 对象列表
extensions: list[Extension] = []
```