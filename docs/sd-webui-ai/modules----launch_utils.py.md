# `stable-diffusion-webui\modules\launch_utils.py`

```py
# 导入必要的库
import logging
import re
import subprocess
import os
import shutil
import sys
import importlib.util
import importlib.metadata
import platform
import json
from functools import lru_cache

# 导入自定义模块
from modules import cmd_args, errors
from modules.paths_internal import script_path, extensions_dir
from modules.timer import startup_timer
from modules import logging_config

# 解析命令行参数
args, _ = cmd_args.parser.parse_known_args()
# 设置日志级别
logging_config.setup_logging(args.loglevel)

# 获取当前 Python 解释器路径
python = sys.executable
# 获取环境变量中的 GIT 路径，如果不存在则使用默认值 "git"
git = os.environ.get('GIT', "git")
# 获取环境变量中的 INDEX_URL，如果不存在则为空字符串
index_url = os.environ.get('INDEX_URL', "")
# 定义存储仓库的目录名
dir_repos = "repositories"

# 是否默认打印命令输出
default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")

# 检查是否启用 Gradio 分析
if 'GRADIO_ANALYTICS_ENABLED' not in os.environ:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# 检查 Python 版本兼容性
def check_python_version():
    # 判断操作系统是否为 Windows
    is_windows = platform.system() == "Windows"
    # 获取 Python 主版本号、次版本号和微版本号
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    # 根据操作系统设置支持的次版本号列表
    if is_windows:
        supported_minors = [10]
    else:
        supported_minors = [7, 8, 9, 10, 11]

    # 如果 Python 版本不在支持的版本范围内，则输出错误信息
    if not (major == 3 and minor in supported_minors):
        import modules.errors

        modules.errors.print_error_explanation(f"""
INCOMPATIBLE PYTHON VERSION

This program is tested with 3.10.6 Python, but you have {major}.{minor}.{micro}.
If you encounter an error with "RuntimeError: Couldn't install torch." message,
or any other error regarding unsuccessful package (library) installation,
please downgrade (or upgrade) to the latest version of 3.10 Python
and delete current Python and "venv" folder in WebUI's directory.

You can download 3.10 Python from here: https://www.python.org/downloads/release/python-3106/

{"Alternatively, use a binary release of WebUI: https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases" if is_windows else ""}
# 使用 --skip-python-version-check 参数来抑制此警告
"""

# 用于缓存结果的装饰器，避免重复计算
@lru_cache()
def commit_hash():
    try:
        # 获取当前代码库的提交哈希值
        return subprocess.check_output([git, "-C", script_path, "rev-parse", "HEAD"], shell=False, encoding='utf8').strip()
    except Exception:
        return "<none>"

# 用于缓存结果的装饰器，避免重复计算
@lru_cache()
def git_tag():
    try:
        # 获取当前代码库的标签
        return subprocess.check_output([git, "-C", script_path, "describe", "--tags"], shell=False, encoding='utf8').strip()
    except Exception:
        try:
            # 尝试从 CHANGELOG.md 文件中获取标签信息
            changelog_md = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CHANGELOG.md")
            with open(changelog_md, "r", encoding="utf-8") as file:
                line = next((line.strip() for line in file if line.strip()), "<none>")
                line = line.replace("## ", "")
                return line
        except Exception:
            return "<none>"

# 运行系统命令，并返回结果
def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        # 打印命令描述信息
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    # 执行系统命令
    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        # 抛出运行时错误
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")

# 检查指定包是否已安装
def is_installed(package):
    try:
        dist = importlib.metadata.distribution(package)
    # 捕获 importlib.metadata.PackageNotFoundError 异常
    except importlib.metadata.PackageNotFoundError:
        # 尝试查找指定包的规范
        try:
            spec = importlib.util.find_spec(package)
        # 捕获 ModuleNotFoundError 异常
        except ModuleNotFoundError:
            # 如果找不到指定模块，则返回 False
            return False

        # 返回规范是否不为 None
        return spec is not None

    # 返回 dist 是否不为 None
    return dist is not None
# 返回指定仓库名称的路径
def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


# 运行 pip 命令安装依赖包
def run_pip(command, desc=None, live=default_command_live):
    # 如果设置了跳过安装参数，则直接返回
    if args.skip_install:
        return

    # 根据是否设置了 index_url 构建 index_url_line
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    # 执行 pip 命令安装依赖包
    return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


# 检查并运行 Python 代码
def check_run_python(code: str) -> bool:
    result = subprocess.run([python, "-c", code], capture_output=True, shell=False)
    return result.returncode == 0


# 修复 Git 仓库工作区
def git_fix_workspace(dir, name):
    # 拉取所有内容并重新索引
    run(f'"{git}" -C "{dir}" fetch --refetch --no-auto-gc', f"Fetching all contents for {name}", f"Couldn't fetch {name}", live=True)
    # 执行 Git 垃圾回收
    run(f'"{git}" -C "{dir}" gc --aggressive --prune=now', f"Pruning {name}", f"Couldn't prune {name}", live=True)
    return


# 运行 Git 命令
def run_git(dir, name, command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live, autofix=True):
    try:
        return run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)
    except RuntimeError:
        if not autofix:
            raise

    # 如果出错且允许自动修复，则尝试自动修复
    print(f"{errdesc}, attempting autofix...")
    git_fix_workspace(dir, name)

    return run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)


# 克隆 Git 仓库
def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful
    # 检查目录是否存在
    if os.path.exists(dir):
        # 如果提交哈希值为 None，则直接返回
        if commithash is None:
            return
    
        # 获取当前目录的提交哈希值
        current_hash = run_git(dir, name, 'rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}", live=False).strip()
        # 如果当前哈希值与给定哈希值相同，则直接返回
        if current_hash == commithash:
            return
    
        # 如果当前目录的远程仓库 URL 与给定 URL 不同，则设置为给定 URL
        if run_git(dir, name, 'config --get remote.origin.url', None, f"Couldn't determine {name}'s origin URL", live=False).strip() != url:
            run_git(dir, name, f'remote set-url origin "{url}"', None, f"Failed to set {name}'s origin URL", live=False)
    
        # 拉取远程仓库的更新
        run_git(dir, name, 'fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}", autofix=False)
    
        # 切换到给定的提交哈希值
        run_git(dir, name, f'checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}", live=True)
    
        return
    
    # 如果目录不存在，则尝试克隆远程仓库
    try:
        run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}", live=True)
    except RuntimeError:
        # 如果克隆失败，则删除目录并抛出异常
        shutil.rmtree(dir, ignore_errors=True)
        raise
    
    # 如果给定了提交哈希值，则在克隆完成后切换到该提交哈希值
    if commithash is not None:
        run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")
# 递归地在指定目录下执行 git pull 操作
def git_pull_recursive(dir):
    # 遍历目录下的所有子目录
    for subdir, _, _ in os.walk(dir):
        # 检查子目录是否包含 .git 文件夹
        if os.path.exists(os.path.join(subdir, '.git')):
            try:
                # 执行 git pull 操作，并自动保存当前工作区的变更
                output = subprocess.check_output([git, '-C', subdir, 'pull', '--autostash'])
                # 打印成功拉取更新的信息
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                # 打印无法执行 git pull 操作的错误信息
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


# 检查提交的版本是否为最新版本
def version_check(commit):
    try:
        import requests
        # 获取 GitHub 上最新提交的信息
        commits = requests.get('https://api.github.com/repos/AUTOMATIC1111/stable-diffusion-webui/branches/master').json()
        # 检查提交的版本是否为最新版本
        if commit != "<none>" and commits['commit']['sha'] != commit:
            # 提示用户当前版本不是最新版本，建议执行 git pull 更新
            print("--------------------------------------------------------")
            print("| You are not up to date with the most recent release. |")
            print("| Consider running `git pull` to update.               |")
            print("--------------------------------------------------------")
        elif commits['commit']['sha'] == commit:
            # 提示用户当前版本为最新版本
            print("You are up to date with the most recent release.")
        else:
            # 提示无法执行版本检查，可能不是一个 git 仓库
            print("Not a git clone, can't perform version check.")
    except Exception as e:
        # 打印版本检查失败的错误信息
        print("version check failed", e)


# 运行扩展安装程序
def run_extension_installer(extension_dir):
    # 获取安装程序的路径
    path_installer = os.path.join(extension_dir, "install.py")
    # 如果安装程序不存在，则直接返回
    if not os.path.isfile(path_installer):
        return

    try:
        # 设置环境变量 PYTHONPATH，添加当前目录到 Python 模块搜索路径中
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{os.path.abspath('.')}{os.pathsep}{env.get('PYTHONPATH', '')}"

        # 运行安装程序，并获取输出结果
        stdout = run(f'"{python}" "{path_installer}"', errdesc=f"Error running install.py for extension {extension_dir}", custom_env=env).strip()
        if stdout:
            # 如果有输出结果，则打印输出
            print(stdout)
    except Exception as e:
        # 报告运行安装程序时的错误信息
        errors.report(str(e))


# 列出扩展列表
def list_extensions(settings_file):
    settings = {}
    # 尝试加载设置文件，如果文件存在则读取其中内容
    try:
        # 检查设置文件是否存在
        if os.path.isfile(settings_file):
            # 以只读方式打开设置文件，使用 UTF-8 编码
            with open(settings_file, "r", encoding="utf8") as file:
                # 从文件中加载 JSON 数据到 settings 变量
                settings = json.load(file)
    # 捕获任何异常
    except Exception:
        # 报告无法加载设置文件的错误信息，包括异常信息
        errors.report("Could not load settings", exc_info=True)

    # 从设置中获取禁用的扩展列表
    disabled_extensions = set(settings.get('disabled_extensions', []))
    # 获取是否禁用所有扩展的设置
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    # 如果禁用所有扩展的设置不是 'none'，或者命令行参数中包含禁用额外扩展的选项，或者禁用所有扩展的选项被设置，或者扩展目录不存在
    if disable_all_extensions != 'none' or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_dir):
        # 返回空列表，表示不加载任何扩展
        return []

    # 返回扩展目录中未禁用的扩展列表
    return [x for x in os.listdir(extensions_dir) if x not in disabled_extensions]
# 运行扩展安装程序，根据设置文件中的信息安装扩展
def run_extensions_installers(settings_file):
    # 如果扩展目录不存在，则直接返回
    if not os.path.isdir(extensions_dir):
        return

    # 在启动计时器中创建一个子类别为"run extensions installers"的计时器
    with startup_timer.subcategory("run extensions installers"):
        # 遍历扩展列表
        for dirname_extension in list_extensions(settings_file):
            # 记录调试信息，安装当前扩展
            logging.debug(f"Installing {dirname_extension}")

            # 获取当前扩展的完整路径
            path = os.path.join(extensions_dir, dirname_extension)

            # 如果路径是一个目录
            if os.path.isdir(path):
                # 运行扩展安装程序
                run_extension_installer(path)
                # 记录启动计时器中的当前扩展
                startup_timer.record(dirname_extension)


# 编译正则表达式，用于解析 requirements.txt 文件中的要求
re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


# 检查是否满足要求，解析 requirements.txt 文件以确定所有要求是否已安装
def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine if all rerqirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    """

    # 导入必要的模块
    import importlib.metadata
    import packaging.version

    # 打开 requirements.txt 文件进行逐行解析
    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            # 如果当前行为空，则继续下一行
            if line.strip() == "":
                continue

            # 使用正则表达式匹配当前行
            m = re.match(re_requirement, line)
            if m is None:
                return False

            # 获取包名和所需版本号
            package = m.group(1).strip()
            version_required = (m.group(2) or "").strip()

            # 如果所需版本号为空，则继续下一行
            if version_required == "":
                continue

            try:
                # 获取已安装的版本号
                version_installed = importlib.metadata.version(package)
            except Exception:
                return False

            # 比较所需版本号和已安装版本号是否一致
            if packaging.version.parse(version_required) != packaging.version.parse(version_installed):
                return False

    return True


# 准备环境变量
def prepare_environment():
    # 获取环境变量中的 torch_index_url，默认为指定的链接
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu118")
    # 获取环境变量中的 torch_command，默认为指定的命令
    torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url {torch_index_url}")
    # 获取环境变量中的 requirements_file，默认为指定的文件名
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
    # 获取环境变量中的 XFORMERS_PACKAGE 值，如果不存在则默认为 'xformers==0.0.20'
    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.20')
    # 获取环境变量中的 CLIP_PACKAGE 值，如果不存在则默认为指定的 GitHub 链接
    clip_package = os.environ.get('CLIP_PACKAGE', "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip")
    # 获取环境变量中的 OPENCLIP_PACKAGE 值，如果不存在则默认为指定的 GitHub 链接
    openclip_package = os.environ.get('OPENCLIP_PACKAGE', "https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip")

    # 获取环境变量中的 STABLE_DIFFUSION_REPO 值，如果不存在则默认为指定的 GitHub 链接
    stable_diffusion_repo = os.environ.get('STABLE_DIFFUSION_REPO', "https://github.com/Stability-AI/stablediffusion.git")
    # 获取环境变量中的 STABLE_DIFFUSION_XL_REPO 值，如果不存在则默认为指定的 GitHub 链接
    stable_diffusion_xl_repo = os.environ.get('STABLE_DIFFUSION_XL_REPO', "https://github.com/Stability-AI/generative-models.git")
    # 获取环境变量中的 K_DIFFUSION_REPO 值，如果不存在则默认为指定的 GitHub 链接
    k_diffusion_repo = os.environ.get('K_DIFFUSION_REPO', 'https://github.com/crowsonkb/k-diffusion.git')
    # 获取环境变量中的 CODEFORMER_REPO 值，如果不存在则默认为指定的 GitHub 链接
    codeformer_repo = os.environ.get('CODEFORMER_REPO', 'https://github.com/sczhou/CodeFormer.git')
    # 获取环境变量中的 BLIP_REPO 值，如果不存在则默认为指定的 GitHub 链接
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')

    # 获取环境变量中的 STABLE_DIFFUSION_COMMIT_HASH 值，如果不存在则默认为指定的提交哈希值
    stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf")
    # 获取环境变量中的 STABLE_DIFFUSION_XL_COMMIT_HASH 值，如果不存在则默认为指定的提交哈希值
    stable_diffusion_xl_commit_hash = os.environ.get('STABLE_DIFFUSION_XL_COMMIT_HASH', "45c443b316737a4ab6e40413d7794a7f5657c19f")
    # 获取环境变量中的 K_DIFFUSION_COMMIT_HASH 值，如果不存在则默认为指定的提交哈希值
    k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "ab527a9a6d347f364e3d185ba6d714e22d80cb3c")
    # 获取环境变量中的 CODEFORMER_COMMIT_HASH 值，如果不存在则默认为指定的提交哈希值
    codeformer_commit_hash = os.environ.get('CODEFORMER_COMMIT_HASH', "c5b4593074ba6214284d6acd5f1719b6c5d739af")
    # 获取环境变量中的 BLIP_COMMIT_HASH 值，如果不存在则默认为指定的提交哈希值
    blip_commit_hash = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")

    try:
        # 尝试删除指定路径下的文件，用于信号通知 webui.sh/bat 在停止执行时需要重新启动 webui
        os.remove(os.path.join(script_path, "tmp", "restart"))
        # 设置环境变量 SD_WEBUI_RESTARTING 为 '1'
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')
    except OSError:
        pass

    # 如果未设置参数 args.skip_python_version_check，则执行检查 Python 版本的函数
    if not args.skip_python_version_check:
        check_python_version()

    # 记录启动时间
    startup_timer.record("checks")

    # 获取当前代码库的提交哈希值
    commit = commit_hash()
    # 获取当前代码库的 Git 标签
    tag = git_tag()
    # 记录启动计时器，记录获取 git 版本信息的时间
    startup_timer.record("git version info")

    # 打印 Python 版本信息、tag 版本号和 commit 哈希值
    print(f"Python {sys.version}")
    print(f"Version: {tag}")
    print(f"Commit hash: {commit}")

    # 如果需要重新安装 torch 或者 torch、torchvision 未安装，则安装 torch 和 torchvision
    if args.reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)
        startup_timer.record("install torch")

    # 如果使用 ipex，则跳过 torch cuda 测试
    if args.use_ipex:
        args.skip_torch_cuda_test = True
    # 如果不跳过 torch cuda 测试且 torch 无法使用 GPU，则抛出异常
    if not args.skip_torch_cuda_test and not check_run_python("import torch; assert torch.cuda.is_available()"):
        raise RuntimeError(
            'Torch is not able to use GPU; '
            'add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'
        )
    startup_timer.record("torch GPU test")

    # 如果 clip 未安装，则安装 clip
    if not is_installed("clip"):
        run_pip(f"install {clip_package}", "clip")
        startup_timer.record("install clip")

    # 如果 open_clip 未安装，则安装 open_clip
    if not is_installed("open_clip"):
        run_pip(f"install {openclip_package}", "open_clip")
        startup_timer.record("install open_clip")

    # 如果 xformers 未安装或需要重新安装且启用 xformers，则安装 xformers
    if (not is_installed("xformers") or args.reinstall_xformers) and args.xformers:
        run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")
        startup_timer.record("install xformers")

    # 如果 ngrok 未安装且启用 ngrok，则安装 ngrok
    if not is_installed("ngrok") and args.ngrok:
        run_pip("install ngrok", "ngrok")
        startup_timer.record("install ngrok")

    # 创建目录，如果目录已存在则不报错
    os.makedirs(os.path.join(script_path, dir_repos), exist_ok=True)

    # 克隆稳定扩散库、稳定扩散 XL 库、K-扩散库和 CodeFormer 库
    git_clone(stable_diffusion_repo, repo_dir('stable-diffusion-stability-ai'), "Stable Diffusion", stable_diffusion_commit_hash)
    git_clone(stable_diffusion_xl_repo, repo_dir('generative-models'), "Stable Diffusion XL", stable_diffusion_xl_commit_hash)
    git_clone(k_diffusion_repo, repo_dir('k-diffusion'), "K-diffusion", k_diffusion_commit_hash)
    git_clone(codeformer_repo, repo_dir('CodeFormer'), "CodeFormer", codeformer_commit_hash)
    # 克隆 BLIP 仓库到指定目录，使用指定的提交哈希
    git_clone(blip_repo, repo_dir('BLIP'), "BLIP", blip_commit_hash)

    # 记录启动计时器，记录“克隆仓库”操作
    startup_timer.record("clone repositores")

    # 如果 lpips 未安装
    if not is_installed("lpips"):
        # 运行 pip 安装 CodeFormer 所需的依赖
        run_pip(f"install -r \"{os.path.join(repo_dir('CodeFormer'), 'requirements.txt')}\"", "requirements for CodeFormer")
        # 记录“安装 CodeFormer 所需依赖”操作
        startup_timer.record("install CodeFormer requirements")

    # 如果 requirements_file 文件不存在
    if not os.path.isfile(requirements_file):
        # 将 requirements_file 设置为脚本路径下的 requirements_file
        requirements_file = os.path.join(script_path, requirements_file)

    # 如果未满足 requirements_file 中的依赖
    if not requirements_met(requirements_file):
        # 运行 pip 安装 requirements_file 中的依赖
        run_pip(f"install -r \"{requirements_file}\"", "requirements")
        # 记录“安装依赖”操作
        startup_timer.record("install requirements")

    # 如果不跳过安装
    if not args.skip_install:
        # 运行扩展安装器，使用指定的设置文件
        run_extensions_installers(settings_file=args.ui_settings_file)

    # 如果启用更新检查
    if args.update_check:
        # 进行版本检查，使用指定的提交哈希
        version_check(commit)
        # 记录“检查版本”操作
        startup_timer.record("check version")

    # 如果启用更新所有扩展
    if args.update_all_extensions:
        # 递归地从扩展目录进行 git 拉取
        git_pull_recursive(extensions_dir)
        # 记录“更新扩展”操作
        startup_timer.record("update extensions")

    # 如果命令行参数中包含 "--exit"
    if "--exit" in sys.argv:
        # 打印信息并退出程序
        print("Exiting because of --exit argument")
        exit(0)
# 为测试配置环境
def configure_for_tests():
    # 如果命令行参数中没有"--api"，则添加"--api"
    if "--api" not in sys.argv:
        sys.argv.append("--api")
    # 如果命令行参数中没有"--ckpt"，则添加"--ckpt"和指定的文件路径
    if "--ckpt" not in sys.argv:
        sys.argv.append("--ckpt")
        sys.argv.append(os.path.join(script_path, "test/test_files/empty.pt"))
    # 如果命令行参数中没有"--skip-torch-cuda-test"，则添加"--skip-torch-cuda-test"
    if "--skip-torch-cuda-test" not in sys.argv:
        sys.argv.append("--skip-torch-cuda-test")
    # 如果命令行参数中没有"--disable-nan-check"，则添加"--disable-nan-check"
    if "--disable-nan-check" not in sys.argv:
        sys.argv.append("--disable-nan-check")

    # 设置环境变量'COMMANDLINE_ARGS'为空字符串
    os.environ['COMMANDLINE_ARGS'] = ""


# 启动函数
def start():
    # 打印启动信息，根据命令行参数判断启动API服务器还是Web UI
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {' '.join(sys.argv[1:])}")
    import webui
    # 如果命令行参数中有'--nowebui'，则只启动API服务器
    if '--nowebui' in sys.argv:
        webui.api_only()
    else:
        webui.webui()


# 导出系统信息到文件
def dump_sysinfo():
    from modules import sysinfo
    import datetime

    # 获取系统信息文本
    text = sysinfo.get()
    # 根据当前时间生成文件名
    filename = f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.json"

    # 将系统信息文本写入文件
    with open(filename, "w", encoding="utf8") as file:
        file.write(text)

    # 返回生成的文件名
    return filename
```