# `.\yolov8\ultralytics\utils\checks.py`

```py
# 导入所需的标准库和第三方库
import contextlib  # 提供了对上下文管理器的支持
import glob  # 文件名匹配库
import inspect  # 检查对象，例如获取函数的源代码
import math  # 数学函数库
import os  # 提供了与操作系统交互的功能
import platform  # 提供了访问平台相关信息的函数
import re  # 正则表达式库
import shutil  # 文件操作工具
import subprocess  # 启动和管理子进程的库
import time  # 提供了各种时间相关的功能
from importlib import metadata  # 用于访问导入的模块元数据
from pathlib import Path  # 提供了处理文件路径的功能
from typing import Optional  # 提供类型提示支持

import cv2  # OpenCV库，用于计算机视觉
import numpy as np  # 数值计算库，支持多维数组和矩阵运算
import requests  # 发送HTTP请求的库
import torch  # PyTorch深度学习框架

from ultralytics.utils import (
    ASSETS,  # 从ultralytics.utils中导入ASSETS常量
    AUTOINSTALL,  # 从ultralytics.utils中导入AUTOINSTALL常量
    IS_COLAB,  # 从ultralytics.utils中导入IS_COLAB常量
    IS_JUPYTER,  # 从ultralytics.utils中导入IS_JUPYTER常量
    IS_KAGGLE,  # 从ultralytics.utils中导入IS_KAGGLE常量
    IS_PIP_PACKAGE,  # 从ultralytics.utils中导入IS_PIP_PACKAGE常量
    LINUX,  # 从ultralytics.utils中导入LINUX常量
    LOGGER,  # 从ultralytics.utils中导入LOGGER常量
    ONLINE,  # 从ultralytics.utils中导入ONLINE常量
    PYTHON_VERSION,  # 从ultralytics.utils中导入PYTHON_VERSION常量
    ROOT,  # 从ultralytics.utils中导入ROOT常量
    TORCHVISION_VERSION,  # 从ultralytics.utils中导入TORCHVISION_VERSION常量
    USER_CONFIG_DIR,  # 从ultralytics.utils中导入USER_CONFIG_DIR常量
    Retry,  # 从ultralytics.utils中导入Retry类
    SimpleNamespace,  # 从ultralytics.utils中导入SimpleNamespace类
    ThreadingLocked,  # 从ultralytics.utils中导入ThreadingLocked类
    TryExcept,  # 从ultralytics.utils中导入TryExcept类
    clean_url,  # 从ultralytics.utils中导入clean_url函数
    colorstr,  # 从ultralytics.utils中导入colorstr函数
    downloads,  # 从ultralytics.utils中导入downloads函数
    emojis,  # 从ultralytics.utils中导入emojis函数
    is_github_action_running,  # 从ultralytics.utils中导入is_github_action_running函数
    url2file,  # 从ultralytics.utils中导入url2file函数
)


def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (Path): Path to the requirements.txt file.
        package (str, optional): Python package to use instead of requirements.txt file, i.e. package='ultralytics'.

    Returns:
        (List[Dict[str, str]]): List of parsed requirements as dictionaries with `name` and `specifier` keys.

    Example:
        ```python
        from ultralytics.utils.checks import parse_requirements

        parse_requirements(package='ultralytics')
        ```py
    """

    if package:
        # 使用元数据获取指定包的依赖信息，排除额外的条件依赖
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        # 读取requirements.txt文件内容并按行分割成列表
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()  # 去除首尾空格
        if line and not line.startswith("#"):
            line = line.split("#")[0].strip()  # 忽略行内注释
            match = re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line)
            if match:
                # 将解析后的依赖信息作为SimpleNamespace对象存入requirements列表
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))

    return requirements


def parse_version(version="0.0.0") -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
    """
    try:
        # 使用正则表达式匹配并提取版本号中的数字部分，转换为整数元组
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        # 如果出现异常，记录警告日志并返回(0, 0, 0)
        LOGGER.warning(f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        (bool): True if the string is composed only of ASCII characters, False otherwise.
    """
    # 将变量 s 转换为字符串形式，无论其原始类型是列表、元组、None 等
    s = str(s)
    
    # 检查字符串 s 是否仅由 ASCII 字符组成
    # 使用 all() 函数和 ord() 函数来检查字符串中的每个字符的 ASCII 编码是否小于 128
    return all(ord(c) < 128 for c in s)
# 确认图像尺寸在每个维度上是否是给定步长的倍数。如果图像尺寸不是步长的倍数，则将其更新为大于或等于给定最小值的最近步长倍数。

def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | cList[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (List[int]): Updated image size.
    """

    # 如果步长是张量，则将其转换为整数
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # 如果图像尺寸是整数，则将其转换为列表
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):  # 例如 '640' 或 '[640,640]'
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )

    # 应用最大维度限制
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]

    # 将图像尺寸调整为步长的倍数
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # 如果图像尺寸已更新，则打印警告信息
    if sz != imgsz:
        LOGGER.warning(f"WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    # 如果需要，添加缺失的维度
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str, optional): Name to be used in warning message.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.
        msg (str, optional): Extra message to display if verbose.
    """
    # 在当前版本和所需版本或范围之间进行检查

    # (此函数中代码已省略，不在要求范围内)
    # 检查版本号是否符合要求的函数
    def check_version(current='', required=''):
        """
        Args:
            current (str): 当前版本号字符串，例如 '22.04'
            required (str): 要求的版本号约束，例如 '==22.04', '>=22.04', '>20.04,<22.04'
    
        Returns:
            (bool): 如果版本号符合要求则返回True，否则返回False.
    
        Example:
            ```python
            # 检查当前版本是否正好是 22.04
            check_version(current='22.04', required='==22.04')
    
            # 检查当前版本是否大于或等于 22.10（假设未指定不等式时，默认为 '>='）
            check_version(current='22.10', required='22.04')
    
            # 检查当前版本是否小于或等于 22.04
            check_version(current='22.04', required='<=22.04')
    
            # 检查当前版本是否在 20.04（包括）与 22.04（不包括）之间
            check_version(current='21.10', required='>20.04,<22.04')
            ```
        """
        if not current:  # 如果当前版本号为空或None
            LOGGER.warning(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
            return True
        elif not current[0].isdigit():  # 如果当前版本号开头不是数字（可能是包名而不是版本号字符串，例如 current='ultralytics'）
            try:
                name = current  # 将包名赋值给 'name' 参数
                current = metadata.version(current)  # 从包名获取版本号字符串
            except metadata.PackageNotFoundError as e:
                if hard:
                    raise ModuleNotFoundError(emojis(f"WARNING ⚠️ {current} package is required but not installed")) from e
                else:
                    return False
    
        if not required:  # 如果要求的版本号约束为空或None，则视为版本号符合要求
            return True
    
        op = ""
        version = ""
        result = True
        c = parse_version(current)  # 将当前版本号字符串解析为版本号元组，例如 '1.2.3' -> (1, 2, 3)
        for r in required.strip(",").split(","):
            op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # 解析版本号约束，例如 '>=22.04' -> ('>=', '22.04')
            v = parse_version(version)  # 将要求的版本号字符串解析为版本号元组，例如 '1.2.3' -> (1, 2, 3)
            if op == "==" and c != v:
                result = False
            elif op == "!=" and c == v:
                result = False
            elif op in {">=", ""} and not (c >= v):  # 如果未指定约束，则默认为 '>=required'
                result = False
            elif op == "<=" and not (c <= v):
                result = False
            elif op == ">" and not (c > v):
                result = False
            elif op == "<" and not (c < v):
                result = False
        if not result:
            warning = f"WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
            if hard:
                raise ModuleNotFoundError(emojis(warning))  # 断言版本要求得到满足
            if verbose:
                LOGGER.warning(warning)
        return result
# 检查最新的 PyPI 包版本，不下载或安装包
def check_latest_pypi_version(package_name="ultralytics"):
    """
    Returns the latest version of a PyPI package without downloading or installing it.

    Parameters:
        package_name (str): The name of the package to find the latest version for.

    Returns:
        (str): The latest version of the package.
    """
    # 禁止 InsecureRequestWarning 警告
    with contextlib.suppress(Exception):
        requests.packages.urllib3.disable_warnings()  # Disable the InsecureRequestWarning
        # 获取包在 PyPI 上的 JSON 信息
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            # 返回包的最新版本号
            return response.json()["info"]["version"]


# 检查 ultralytics 包是否有可用的更新版本
def check_pip_update_available():
    """
    Checks if a new version of the ultralytics package is available on PyPI.

    Returns:
        (bool): True if an update is available, False otherwise.
    """
    if ONLINE and IS_PIP_PACKAGE:
        with contextlib.suppress(Exception):
            from ultralytics import __version__

            # 获取最新的 PyPI 版本号
            latest = check_latest_pypi_version()
            # 检查当前版本是否小于最新版本
            if check_version(__version__, f"<{latest}"):  # check if current version is < latest version
                LOGGER.info(
                    f"New https://pypi.org/project/ultralytics/{latest} available 😃 "
                    f"Update with 'pip install -U ultralytics'"
                )
                return True
    return False


# 使用线程锁检查字体文件是否存在于用户配置目录，不存在则下载
@ThreadingLocked()
def check_font(font="Arial.ttf"):
    """
    Find font locally or download to user's configuration directory if it does not already exist.

    Args:
        font (str): Path or name of font.

    Returns:
        file (Path): Resolved font file path.
    """
    from matplotlib import font_manager

    # 检查用户配置目录是否存在字体文件
    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    # 检查系统中是否存在指定的字体
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]

    # 如果缺失，则从 GitHub 下载到用户配置目录
    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}"
    if downloads.is_url(url, check=True):
        downloads.safe_download(url=url, file=file)
        return file


# 检查当前 Python 版本是否满足指定的最小要求
def check_python(minimum: str = "3.8.0", hard: bool = True) -> bool:
    """
    Check current python version against the required minimum version.

    Args:
        minimum (str): Required minimum version of python.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.

    Returns:
        (bool): Whether the installed Python version meets the minimum constraints.
    """
    return check_version(PYTHON_VERSION, minimum, name="Python", hard=hard)


# 尝试检查安装的依赖项是否满足 YOLOv8 的要求，并尝试自动更新
@TryExcept()
def check_requirements(requirements=ROOT.parent / "requirements.txt", exclude=(), install=True, cmds=""):
    """
    Check if installed dependencies meet YOLOv8 requirements and attempt to auto-update if needed.
    """
    Args:
        requirements (Union[Path, str, List[str]]): Path to a requirements.txt file, a single package requirement as a
            string, or a list of package requirements as strings.
        exclude (Tuple[str]): Tuple of package names to exclude from checking.
        install (bool): If True, attempt to auto-update packages that don't meet requirements.
        cmds (str): Additional commands to pass to the pip install command when auto-updating.

    Example:
        ```py
        from ultralytics.utils.checks import check_requirements

        # Check a requirements.txt file
        check_requirements('path/to/requirements.txt')

        # Check a single package
        check_requirements('ultralytics>=8.0.0')

        # Check multiple packages
        check_requirements(['numpy', 'ultralytics>=8.0.0'])
        ```

    prefix = colorstr("red", "bold", "requirements:")  # 设置带有颜色的输出前缀

    check_python()  # 检查当前 Python 版本是否满足要求
    check_torchvision()  # 检查 torch 和 torchvision 的兼容性

    if isinstance(requirements, Path):  # 如果 requirements 是 Path 对象，代表是一个 requirements.txt 文件
        file = requirements.resolve()  # 获取文件的绝对路径
        assert file.exists(), f"{prefix} {file} not found, check failed."  # 断言文件存在，否则抛出异常
        requirements = [f"{x.name}{x.specifier}" for x in parse_requirements(file) if x.name not in exclude]  # 解析 requirements.txt 中的内容，并排除 exclude 中的包名
    elif isinstance(requirements, str):
        requirements = [requirements]  # 如果 requirements 是字符串，转为包含单个字符串的列表

    pkgs = []
    for r in requirements:
        r_stripped = r.split("/")[-1].replace(".git", "")  # 从 URL 形式的包名中提取出真实的包名
        match = re.match(r"([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)  # 使用正则表达式匹配包名和版本要求
        name, required = match[1], match[2].strip() if match[2] else ""  # 获取包名和版本要求
        try:
            assert check_version(metadata.version(name), required)  # 检查当前安装的包版本是否符合要求，不符合则抛出异常
        except (AssertionError, metadata.PackageNotFoundError):
            pkgs.append(r)  # 将不符合要求的包加入列表中

    @Retry(times=2, delay=1)
    def attempt_install(packages, commands):
        """Attempt pip install command with retries on failure."""
        return subprocess.check_output(f"pip install --no-cache-dir {packages} {commands}", shell=True).decode()
        # 使用带有重试机制的 subprocess 执行 pip install 命令并返回输出结果

    s = " ".join(f'"{x}"' for x in pkgs)  # 构建控制台输出字符串，列出需要更新的包名
    # 如果条件 s 不为空，则进入条件判断
    if s:
        # 如果 install 为真并且 AUTOINSTALL 环境变量为真，则继续执行
        if install and AUTOINSTALL:  # check environment variable
            # 计算需要更新的包的数量
            n = len(pkgs)  # number of packages updates
            # 记录日志信息，指示 Ultralytics 的要求未找到，并尝试自动更新
            LOGGER.info(f"{prefix} Ultralytics requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate...")
            try:
                t = time.time()  # 记录开始时间
                assert ONLINE, "AutoUpdate skipped (offline)"  # 检查是否在线，否则跳过自动更新
                # 执行自动安装操作，并记录日志返回信息
                LOGGER.info(attempt_install(s, cmds))
                dt = time.time() - t  # 计算自动更新所需时间
                # 记录自动更新成功的日志信息，显示安装的包的数量和名称
                LOGGER.info(
                    f"{prefix} AutoUpdate success ✅ {dt:.1f}s, installed {n} package{'s' * (n > 1)}: {pkgs}\n"
                    f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
                )
            except Exception as e:
                # 记录警告日志，指示自动更新失败
                LOGGER.warning(f"{prefix} ❌ {e}")
                # 如果发生异常，返回 False
                return False
        else:
            # 如果不满足自动安装的条件，直接返回 False
            return False

    # 如果条件 s 为空或未满足自动安装条件，则返回 True
    return True
# 检查 PyTorch 和 Torchvision 的兼容性
def check_torchvision():
    """
    Checks the installed versions of PyTorch and Torchvision to ensure they're compatible.

    This function checks the installed versions of PyTorch and Torchvision, and warns if they're incompatible according
    to the provided compatibility table based on:
    https://github.com/pytorch/vision#installation.

    The compatibility table is a dictionary where the keys are PyTorch versions and the values are lists of compatible
    Torchvision versions.
    """

    # 兼容性表
    compatibility_table = {
        "2.3": ["0.18"],
        "2.2": ["0.17"],
        "2.1": ["0.16"],
        "2.0": ["0.15"],
        "1.13": ["0.14"],
        "1.12": ["0.13"],
    }

    # 提取主要和次要版本号
    v_torch = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    # 如果当前 PyTorch 版本在兼容性表中
    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        # 提取当前 Torchvision 的主要和次要版本号
        v_torchvision = ".".join(TORCHVISION_VERSION.split("+")[0].split(".")[:2])
        # 如果当前 Torchvision 版本不在兼容的版本列表中
        if all(v_torchvision != v for v in compatible_versions):
            # 打印警告信息，说明 Torchvision 版本不兼容
            print(
                f"WARNING ⚠️ torchvision=={v_torchvision} is incompatible with torch=={v_torch}.\n"
                f"Run 'pip install torchvision=={compatible_versions[0]}' to fix torchvision or "
                "'pip install -U torch torchvision' to update both.\n"
                "For a full compatibility table see https://github.com/pytorch/vision#installation"
            )


# 检查文件后缀是否符合要求
def check_suffix(file="yolov8n.pt", suffix=".pt", msg=""):
    """Check file(s) for acceptable suffix."""
    # 如果 file 和 suffix 都不为空
    if file and suffix:
        # 如果 suffix 是字符串，转换为元组
        if isinstance(suffix, str):
            suffix = (suffix,)
        # 对于 file 是列表或元组的情况，遍历每个文件名
        for f in file if isinstance(file, (list, tuple)) else [file]:
            # 获取文件的后缀名并转换为小写
            s = Path(f).suffix.lower().strip()  # file suffix
            # 如果后缀名长度大于0
            if len(s):
                # 断言文件后缀在给定的后缀列表中，否则触发 AssertionError
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"


# 检查 YOLOv5u 文件名，并输出警告信息
def check_yolov5u_filename(file: str, verbose: bool = True):
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames."""
    # 检查文件名中是否包含'yolov3'或'yolov5'
    if "yolov3" in file or "yolov5" in file:
        # 如果文件名中包含'u.yaml'，将其替换为'.yaml'
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")  # 例如将'yolov5nu.yaml'替换为'yolov5n.yaml'
        # 如果文件名包含'.pt'且不包含'u'
        elif ".pt" in file and "u" not in file:
            # 保存原始文件名
            original_file = file
            # 使用正则表达式将文件名中的特定模式替换为带'u'后缀的新模式
            file = re.sub(r"(.*yolov5([nsmlx]))\.pt", "\\1u.pt", file)  # 例如将'yolov5n.pt'替换为'yolov5nu.pt'
            file = re.sub(r"(.*yolov5([nsmlx])6)\.pt", "\\1u.pt", file)  # 例如将'yolov5n6.pt'替换为'yolov5n6u.pt'
            file = re.sub(r"(.*yolov3(|-tiny|-spp))\.pt", "\\1u.pt", file)  # 例如将'yolov3-spp.pt'替换为'yolov3-sppu.pt'
            # 如果文件名已被修改且verbose为真，记录日志信息
            if file != original_file and verbose:
                LOGGER.info(
                    f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                    f"trained with https://github.com/ultralytics/ultralytics and feature improved performance vs "
                    f"standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n"
                )
    # 返回处理后的文件名
    return file
# 检查模型文件名是否是有效的模型 stem，并返回一个完整的模型文件名
def check_model_file_from_stem(model="yolov8n"):
    if model and not Path(model).suffix and Path(model).stem in downloads.GITHUB_ASSETS_STEMS:
        # 如果模型名存在且没有后缀，并且模型 stem 在下载的 GitHub 资源中
        return Path(model).with_suffix(".pt")  # 添加后缀，例如 yolov8n -> yolov8n.pt
    else:
        return model  # 否则返回原始模型名


# 搜索/下载文件（如果需要），并返回文件路径
def check_file(file, suffix="", download=True, download_dir=".", hard=True):
    check_suffix(file, suffix)  # 可选步骤，检查文件后缀
    file = str(file).strip()  # 转换为字符串并去除空格
    file = check_yolov5u_filename(file)  # 将 yolov5n 转换为 yolov5nu
    if (
        not file
        or ("://" not in file and Path(file).exists())  # 在 Windows Python<3.10 中需要检查 '://' 的存在
        or file.lower().startswith("grpc://")
    ):  # 文件存在或者是 gRPC Triton 图像
        return file
    elif download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):  # 下载文件
        url = file  # 警告：Pathlib 会将 :// 转换为 :/
        file = Path(download_dir) / url2file(file)  # 将 URL 转换为本地文件路径，处理 %2F 和路径分隔符
        if file.exists():
            LOGGER.info(f"Found {clean_url(url)} locally at {file}")  # 文件已经存在
        else:
            downloads.safe_download(url=url, file=file, unzip=False)  # 安全下载文件
        return str(file)
    else:  # 搜索文件
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))  # 查找文件
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # 返回第一个匹配的文件，如果没有找到则返回空列表


# 搜索/下载 YAML 文件（如果需要），并返回文件路径，同时检查后缀
def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    return check_file(file, suffix, hard=hard)


# 检查解析后的路径是否在预期目录下，防止路径遍历攻击
def check_is_path_safe(basedir, path):
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()

    return path_resolved.exists() and path_resolved.parts[: len(base_dir_resolved.parts)] == base_dir_resolved.parts


# 检查环境是否支持显示图像
def check_imshow(warn=False):
    try:
        if LINUX:
            assert not IS_COLAB and not IS_KAGGLE
            assert "DISPLAY" in os.environ, "The DISPLAY environment variable isn't set."
        cv2.imshow("test", np.zeros((8, 8, 3), dtype=np.uint8))  # 显示一个小的 8x8 RGB 图像
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True  # 返回 True 表示显示正常
    # 捕获所有异常，并将异常信息保存在变量 e 中
    except Exception as e:
        # 如果 warn 参数为真，则记录警告消息，指示环境不支持 cv2.imshow() 或 PIL Image.show()
        LOGGER.warning(f"WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        # 返回 False 表示函数执行失败
        return False
def check_yolo(verbose=True, device=""):
    """Return a human-readable YOLO software and hardware summary."""
    # 导入 psutil 库，用于获取系统信息
    import psutil
    # 从 ultralytics.utils.torch_utils 中导入 select_device 函数
    from ultralytics.utils.torch_utils import select_device

    # 如果运行在 Jupyter 环境下
    if IS_JUPYTER:
        # 检查是否满足使用 wandb，如果不满足，不安装
        if check_requirements("wandb", install=False):
            os.system("pip uninstall -y wandb")  # 卸载 wandb：避免创建不必要的账户并导致无限挂起
        # 如果运行在 Colab 环境下，移除 /sample_data 目录
        if IS_COLAB:
            shutil.rmtree("sample_data", ignore_errors=True)  # 移除 Colab 的 /sample_data 目录

    # 如果 verbose 参数为 True
    if verbose:
        # 计算 GiB 换算的字节数
        gib = 1 << 30  # bytes per GiB
        # 获取系统的内存总量
        ram = psutil.virtual_memory().total
        # 获取根目录 "/" 的磁盘使用情况：总容量、已用容量、空闲容量
        total, used, free = shutil.disk_usage("/")
        # 构建系统信息字符串
        s = f"({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)"
        # 尝试清除 IPython 环境下的显示
        with contextlib.suppress(Exception):  # 如果安装了 ipython，则清除显示
            from IPython import display

            display.clear_output()
    else:
        s = ""

    # 调用 select_device 函数，设置设备
    select_device(device=device, newline=False)
    # 记录日志信息，表示设置完成
    LOGGER.info(f"Setup complete ✅ {s}")


def collect_system_info():
    """Collect and print relevant system information including OS, Python, RAM, CPU, and CUDA."""
    # 导入 psutil 库，用于获取系统信息
    import psutil
    # 从 ultralytics.utils 中导入相关变量和函数：ENVIRONMENT, IS_GIT_DIR
    from ultralytics.utils import ENVIRONMENT, IS_GIT_DIR
    # 从 ultralytics.utils.torch_utils 中导入 get_cpu_info 函数
    from ultralytics.utils.torch_utils import get_cpu_info

    # 计算 RAM 信息，将字节转换为 GB
    ram_info = psutil.virtual_memory().total / (1024**3)  # Convert bytes to GB
    # 调用 check_yolo 函数，执行 YOLO 系统信息的检查
    check_yolo()
    # 记录系统信息到日志中
    LOGGER.info(
        f"\n{'OS':<20}{platform.platform()}\n"
        f"{'Environment':<20}{ENVIRONMENT}\n"
        f"{'Python':<20}{PYTHON_VERSION}\n"
        f"{'Install':<20}{'git' if IS_GIT_DIR else 'pip' if IS_PIP_PACKAGE else 'other'}\n"
        f"{'RAM':<20}{ram_info:.2f} GB\n"
        f"{'CPU':<20}{get_cpu_info()}\n"
        f"{'CUDA':<20}{torch.version.cuda if torch and torch.cuda.is_available() else None}\n"
    )

    # 遍历解析 ultralytics 包的依赖要求
    for r in parse_requirements(package="ultralytics"):
        try:
            # 获取当前包的版本信息
            current = metadata.version(r.name)
            # 检查当前版本是否符合要求，返回对应的标志符号
            is_met = "✅ " if check_version(current, str(r.specifier), hard=True) else "❌ "
        except metadata.PackageNotFoundError:
            # 如果包未安装，标记为未安装
            current = "(not installed)"
            is_met = "❌ "
        # 记录依赖包的信息到日志中
        LOGGER.info(f"{r.name:<20}{is_met}{current}{r.specifier}")

    # 如果正在使用 GitHub Actions
    if is_github_action_running():
        LOGGER.info(
            f"\nRUNNER_OS: {os.getenv('RUNNER_OS')}\n"
            f"GITHUB_EVENT_NAME: {os.getenv('GITHUB_EVENT_NAME')}\n"
            f"GITHUB_WORKFLOW: {os.getenv('GITHUB_WORKFLOW')}\n"
            f"GITHUB_ACTOR: {os.getenv('GITHUB_ACTOR')}\n"
            f"GITHUB_REPOSITORY: {os.getenv('GITHUB_REPOSITORY')}\n"
            f"GITHUB_REPOSITORY_OWNER: {os.getenv('GITHUB_REPOSITORY_OWNER')}\n"
        )


def check_amp(model):
    """
    This function checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLOv8 model. If the checks
    fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP results, so AMP will
    """
    # 这个函数检查 YOLOv8 模型的 PyTorch Automatic Mixed Precision (AMP) 功能
    pass
    def check_amp(model):
        """
        Check if Automatic Mixed Precision (AMP) works correctly with a YOLOv8 model.
    
        Args:
            model (nn.Module): A YOLOv8 model instance.
    
        Example:
            ```py
            from ultralytics import YOLO
            from ultralytics.utils.checks import check_amp
    
            model = YOLO('yolov8n.pt').model.cuda()
            check_amp(model)
            ```
    
        Returns:
            (bool): Returns True if the AMP functionality works correctly with YOLOv8 model, else False.
        """
        from ultralytics.utils.torch_utils import autocast  # Import autocast function from torch_utils
    
        device = next(model.parameters()).device  # Get the device of the model
        if device.type in {"cpu", "mps"}:
            return False  # Return False if AMP is only supported on CUDA devices
    
        def amp_allclose(m, im):
            """All close FP32 vs AMP results."""
            a = m(im, device=device, verbose=False)[0].boxes.data  # Perform FP32 inference
            with autocast(enabled=True):
                b = m(im, device=device, verbose=False)[0].boxes.data  # Perform AMP inference
            del m  # Delete the model instance
            return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # Check if results are close with 0.5 absolute tolerance
    
        im = ASSETS / "bus.jpg"  # Define the path to the image for checking
        prefix = colorstr("AMP: ")  # Add color formatting to log messages
        LOGGER.info(f"{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...")  # Log AMP check initialization
        warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."  # Warning message about AMP usage
        try:
            from ultralytics import YOLO  # Import YOLO class from ultralytics
    
            assert amp_allclose(YOLO("yolov8n.pt"), im)  # Assert if AMP results are close to FP32 results
            LOGGER.info(f"{prefix}checks passed ✅")  # Log that AMP checks passed
        except ConnectionError:
            LOGGER.warning(f"{prefix}checks skipped ⚠️, offline and unable to download YOLOv8n. {warning_msg}")  # Log warning if YOLOv8n download fails
        except (AttributeError, ModuleNotFoundError):
            LOGGER.warning(
                f"{prefix}checks skipped ⚠️. "
                f"Unable to load YOLOv8n due to possible Ultralytics package modifications. {warning_msg}"
            )  # Log warning if YOLOv8n loading fails due to modifications
        except AssertionError:
            LOGGER.warning(
                f"{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to "
                f"NaN losses or zero-mAP results, so AMP will be disabled during training."
            )  # Log if AMP checks fail, indicating potential issues
            return False  # Return False if AMP checks fail
        return True  # Return True if AMP checks pass successfully
def git_describe(path=ROOT):  # path must be a directory
    """Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe."""
    # 尝试执行 git describe 命令获取当前目录下 Git 仓库的描述信息
    with contextlib.suppress(Exception):
        return subprocess.check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
    # 如果执行失败或出现异常，返回空字符串
    return ""


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Print function arguments (optional args dict)."""

    def strip_auth(v):
        """Clean longer Ultralytics HUB URLs by stripping potential authentication information."""
        # 如果 URL 开头为 "http"，长度超过 100，且为字符串类型，则清除可能的认证信息
        return clean_url(v) if (isinstance(v, str) and v.startswith("http") and len(v) > 100) else v

    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        # 如果未传入参数字典，则自动获取当前函数的参数和值
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        # 尝试解析文件路径并相对于根目录确定文件路径或文件名（不带后缀）
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        # 如果解析失败，直接取文件名（不带后缀）
        file = Path(file).stem
    # 构建输出字符串，包括文件名和函数名（根据传入的显示选项）
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    # 使用 LOGGER 记录信息，输出每个参数的名称和经过 strip_auth 处理后的值
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={strip_auth(v)}" for k, v in args.items()))


def cuda_device_count() -> int:
    """
    Get the number of NVIDIA GPUs available in the environment.

    Returns:
        (int): The number of NVIDIA GPUs available.
    """
    try:
        # 运行 nvidia-smi 命令并捕获其输出
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"], encoding="utf-8"
        )

        # 取输出的第一行并去除首尾空白字符
        first_line = output.strip().split("\n")[0]

        # 将第一行的内容转换为整数并返回
        return int(first_line)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # 如果命令执行失败，nvidia-smi 未找到，或输出无法转换为整数，则假定没有可用的 GPU
        return 0


def cuda_is_available() -> bool:
    """
    Check if CUDA is available in the environment.

    Returns:
        (bool): True if one or more NVIDIA GPUs are available, False otherwise.
    """
    # 检查是否有可用的 NVIDIA GPU，返回结果为布尔值
    return cuda_device_count() > 0


# Define constants
IS_PYTHON_MINIMUM_3_10 = check_python("3.10", hard=False)
IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")
```