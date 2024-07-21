# `.\pytorch\tools\download_mnist.py`

```
import argparse  # 导入命令行参数解析模块
import gzip  # 导入 gzip 压缩文件处理模块
import os  # 导入操作系统功能模块
import sys  # 导入系统相关模块
from urllib.error import URLError  # 从 urllib.error 导入 URLError 异常类
from urllib.request import urlretrieve  # 从 urllib.request 导入 urlretrieve 函数


MIRRORS = [
    "http://yann.lecun.com/exdb/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

RESOURCES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def report_download_progress(
    chunk_number: int,
    chunk_size: int,
    file_size: int,
) -> None:
    # 如果文件大小不为 -1，则计算下载进度百分比并显示进度条
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write(f"\r0% |{bar:<64}| {int(percent * 100)}%")


def download(destination_path: str, resource: str, quiet: bool) -> None:
    # 检查目标路径是否已存在文件，如果存在且不静默模式，则打印跳过信息
    if os.path.exists(destination_path):
        if not quiet:
            print(f"{destination_path} already exists, skipping ...")
    else:
        # 遍历镜像列表尝试下载资源文件
        for mirror in MIRRORS:
            url = mirror + resource
            print(f"Downloading {url} ...")
            try:
                # 如果不是静默模式，设置下载进度回调函数为 report_download_progress
                hook = None if quiet else report_download_progress
                urlretrieve(url, destination_path, reporthook=hook)
            except (URLError, ConnectionError) as e:
                # 下载失败时输出错误信息并尝试下一个镜像源
                print(f"Failed to download (trying next):\n{e}")
                continue
            finally:
                if not quiet:
                    print()  # 输出空行，仅作为换行符
            break
        else:
            raise RuntimeError("Error downloading resource!")


def unzip(zipped_path: str, quiet: bool) -> None:
    # 根据压缩文件路径确定解压后的文件路径
    unzipped_path = os.path.splitext(zipped_path)[0]
    if os.path.exists(unzipped_path):
        if not quiet:
            print(f"{unzipped_path} already exists, skipping ... ")
        return
    # 使用 gzip 模块解压文件
    with gzip.open(zipped_path, "rb") as zipped_file:
        with open(unzipped_path, "wb") as unzipped_file:
            unzipped_file.write(zipped_file.read())
            if not quiet:
                print(f"Unzipped {zipped_path} ...")


def main() -> None:
    # 解析命令行参数，设置程序描述和选项
    parser = argparse.ArgumentParser(
        description="Download the MNIST dataset from the internet"
    )
    parser.add_argument(
        "-d", "--destination", default=".", help="Destination directory"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Don't report about progress"
    )
    options = parser.parse_args()

    # 如果目标目录不存在，则创建它
    if not os.path.exists(options.destination):
        os.makedirs(options.destination)

    try:
        # 遍历资源列表，下载并解压每个资源文件
        for resource in RESOURCES:
            path = os.path.join(options.destination, resource)
            download(path, resource, options.quiet)
            unzip(path, options.quiet)
    except KeyboardInterrupt:
        print("Interrupted")


if __name__ == "__main__":
    main()
```