# `.\pytorch\tools\linter\adapters\s3_init.py`

```py
# 导入必要的库
import argparse                 # 解析命令行参数的库
import hashlib                  # 提供多种哈希算法的库
import json                     # 处理 JSON 数据的库
import logging                  # 记录日志的库
import os                       # 提供了访问操作系统功能的库
import platform                 # 提供了访问平台相关信息的库
import stat                     # 提供了处理文件状态的库
import subprocess               # 运行子进程的库
import sys                      # 提供了对 Python 运行时环境的访问
import urllib.error             # 处理 URL 异常的库
import urllib.request           # 发送 HTTP 请求的库
from pathlib import Path        # 提供了处理文件路径的类

# String representing the host platform (e.g. Linux, Darwin).
HOST_PLATFORM = platform.system()                # 获取当前系统名称，例如 Linux 或 Darwin
HOST_PLATFORM_ARCH = platform.system() + "-" + platform.processor()  # 获取当前系统和处理器信息的组合

# PyTorch directory root
try:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        check=True,
    )
    PYTORCH_ROOT = result.stdout.decode("utf-8").strip()  # 获取 PyTorch 代码库的根目录路径
except subprocess.CalledProcessError:
    # If git is not installed, compute repo root as 3 folders up from this file
    path_ = os.path.abspath(__file__)
    for _ in range(4):
        path_ = os.path.dirname(path_)   # 获取当前文件所在目录的父级目录路径
    PYTORCH_ROOT = path_

DRY_RUN = False    # 是否处于 dry run 模式的标志，默认为 False


def compute_file_sha256(path: str) -> str:
    """Compute the SHA256 hash of a file and return it as a hex string."""
    # 如果文件不存在，则返回空字符串
    if not os.path.exists(path):
        return ""

    hash = hashlib.sha256()    # 创建 SHA256 哈希对象

    # 以二进制模式打开文件并计算哈希值
    with open(path, "rb") as f:
        for b in f:
            hash.update(b)    # 更新哈希对象

    # 返回哈希值的十六进制字符串表示
    return hash.hexdigest()


def report_download_progress(
    chunk_number: int, chunk_size: int, file_size: int
) -> None:
    """
    Pretty printer for file download progress.
    """
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write(f"\r0% |{bar:<64}| {int(percent * 100)}%")  # 在命令行中展示下载进度条


def check(binary_path: Path, reference_hash: str) -> bool:
    """Check whether the binary exists and is the right one.

    If there is hash difference, delete the actual binary.
    """
    if not binary_path.exists():   # 检查二进制文件是否存在
        logging.info("%s does not exist.", binary_path)  # 记录信息日志，显示文件不存在
        return False

    existing_binary_hash = compute_file_sha256(str(binary_path))  # 计算当前二进制文件的 SHA256 哈希值
    if existing_binary_hash == reference_hash:   # 检查当前哈希值是否与参考哈希值一致
        return True

    logging.warning(
        """\
Found binary hash does not match reference!

Found hash: %s
Reference hash: %s

Deleting %s just to be safe.
""",
        existing_binary_hash,
        reference_hash,
        binary_path,
    )
    if DRY_RUN:
        logging.critical(
            "In dry run mode, so not actually deleting the binary. But consider deleting it ASAP!"
        )   # 如果处于 dry run 模式，则记录临界信息，显示不会真正删除文件，但建议尽快删除
        return False

    try:
        binary_path.unlink()   # 删除二进制文件
    except OSError as e:
        logging.critical("Failed to delete binary: %s", e)   # 记录临界日志，显示删除文件失败原因
        logging.critical(
            "Delete this binary as soon as possible and do not execute it!"
        )   # 记录临界日志，显示尽快删除二进制文件并禁止执行

    return False


def download(
    name: str,
    output_dir: str,
    url: str,
    reference_bin_hash: str,
) -> bool:
    """
    Download a platform-appropriate binary if one doesn't already exist at the expected location and verifies
    """
    that it is the right binary by checking its SHA256 hash against the expected hash.
    """
    # 首先检查是否需要执行任何操作
    binary_path = Path(output_dir, name)
    # 检查生成的二进制文件是否已经存在且正确
    if check(binary_path, reference_bin_hash):
        logging.info("Correct binary already exists at %s. Exiting.", binary_path)
        return True

    # 创建输出文件夹（如果不存在则创建）
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    # 下载二进制文件
    logging.info("Downloading %s to %s", url, binary_path)

    # 如果处于干运行模式，则直接退出
    if DRY_RUN:
        logging.info("Exiting as there is nothing left to do in dry run mode")
        return True

    # 使用 urllib 请求下载文件，并显示下载进度（如果在终端中）
    urllib.request.urlretrieve(
        url,
        binary_path,
        reporthook=report_download_progress if sys.stdout.isatty() else None,
    )

    logging.info("Downloaded %s successfully.", name)

    # 检查下载的二进制文件的完整性
    if not check(binary_path, reference_bin_hash):
        logging.critical("Downloaded binary %s failed its hash check", name)
        return False

    # 确保下载的文件具有执行权限
    mode = os.stat(binary_path).st_mode
    mode |= stat.S_IXUSR
    os.chmod(binary_path, mode)

    # 记录所使用的二进制文件的路径
    logging.info("Using %s located at %s", name, binary_path)
    return True
# 如果脚本被直接执行而不是作为模块导入，则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser(
        description="downloads and checks binaries from s3",  # 设置参数解析器的描述信息
    )
    
    # 添加参数 `--config-json`，必填，指定配置文件的路径
    parser.add_argument(
        "--config-json",
        required=True,
        help="Path to config json that describes where to find binaries and hashes",
    )
    
    # 添加参数 `--linter`，必填，指定从配置文件中初始化哪个代码检查工具
    parser.add_argument(
        "--linter",
        required=True,
        help="Which linter to initialize from the config json",
    )
    
    # 添加参数 `--output-dir`，必填，指定二进制文件的输出目录
    parser.add_argument(
        "--output-dir",
        required=True,
        help="place to put the binary",
    )
    
    # 添加参数 `--output-name`，必填，指定生成的二进制文件的名称
    parser.add_argument(
        "--output-name",
        required=True,
        help="name of binary",
    )
    
    # 添加参数 `--dry-run`，默认为 False，如果设置为 "0" 则表示不执行下载操作，只输出操作信息
    parser.add_argument(
        "--dry-run",
        default=False,
        help="do not download, just print what would be done",
    )

    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据 `--dry-run` 参数设置 DRY_RUN 变量，True 表示执行干运行，False 表示执行实际下载
    if args.dry_run == "0":
        DRY_RUN = False
    else:
        DRY_RUN = True

    # 根据 DRY_RUN 的值设置日志的格式和输出级别
    logging.basicConfig(
        format="[DRY_RUN] %(levelname)s: %(message)s" if DRY_RUN else "%(levelname)s: %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    # 加载配置文件中的 JSON 数据
    config = json.load(open(args.config_json))
    # 根据 `--linter` 参数从配置文件中获取相应的配置信息
    config = config[args.linter]

    # 确定主机平台，通常是根据环境变量或系统信息确定的
    host_platform = HOST_PLATFORM if HOST_PLATFORM in config else HOST_PLATFORM_ARCH
    # 如果主机平台不在配置文件中，则打印错误信息并退出程序
    if host_platform not in config:
        logging.error("Unsupported platform: %s/%s", HOST_PLATFORM, HOST_PLATFORM_ARCH)
        sys.exit(1)

    # 从配置文件中获取下载 URL 和哈希值
    url = config[host_platform]["download_url"]
    hash = config[host_platform]["hash"]

    # 调用 download 函数下载指定的二进制文件，并返回操作是否成功的布尔值
    ok = download(args.output_name, args.output_dir, url, hash)
    # 如果下载不成功，则记录严重级别的日志并退出程序
    if not ok:
        logging.critical("Unable to initialize %s", args.linter)
        sys.exit(1)
```