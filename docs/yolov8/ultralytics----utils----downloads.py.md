# `.\yolov8\ultralytics\utils\downloads.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

# 导入必要的库
import contextlib  # 提供上下文管理工具的标准库
import re  # 提供正则表达式操作的模块
import shutil  # 提供高级文件操作的模块
import subprocess  # 提供运行外部命令的功能
from itertools import repeat  # 提供迭代工具函数
from multiprocessing.pool import ThreadPool  # 提供多线程池的功能
from pathlib import Path  # 提供处理文件路径的类和函数
from urllib import parse, request  # 提供处理 URL 相关的模块

import requests  # 提供进行 HTTP 请求的模块
import torch  # PyTorch 深度学习框架

# 从 Ultralytics 的 utils 模块中导入特定函数和类
from ultralytics.utils import LOGGER, TQDM, checks, clean_url, emojis, is_online, url2file

# 定义 Ultralytics GitHub 上的资源仓库和文件名列表
GITHUB_ASSETS_REPO = "ultralytics/assets"
GITHUB_ASSETS_NAMES = (
    [f"yolov8{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb", "-oiv7")]
    + [f"yolov5{k}{resolution}u.pt" for k in "nsmlx" for resolution in ("", "6")]
    + [f"yolov3{k}u.pt" for k in ("", "-spp", "-tiny")]
    + [f"yolov8{k}-world.pt" for k in "smlx"]
    + [f"yolov8{k}-worldv2.pt" for k in "smlx"]
    + [f"yolov9{k}.pt" for k in "tsmce"]
    + [f"yolov10{k}.pt" for k in "nsmblx"]
    + [f"yolo_nas_{k}.pt" for k in "sml"]
    + [f"sam_{k}.pt" for k in "bl"]
    + [f"FastSAM-{k}.pt" for k in "sx"]
    + [f"rtdetr-{k}.pt" for k in "lx"]
    + ["mobile_sam.pt"]
    + ["calibration_image_sample_data_20x128x128x3_float32.npy.zip"]
)
GITHUB_ASSETS_STEMS = [Path(k).stem for k in GITHUB_ASSETS_NAMES]


def is_url(url, check=False):
    """
    验证给定的字符串是否为 URL，并可选择检查该 URL 是否在线可用。

    Args:
        url (str): 要验证为 URL 的字符串。
        check (bool, optional): 如果为 True，则额外检查 URL 是否在线可用。默认为 True。

    Returns:
        bool: 如果是有效的 URL 返回 True。如果 'check' 为 True，则同时检查 URL 在线是否可用。否则返回 False。

    Example:
        ```py
        valid = is_url("https://www.example.com")
        ```py
    """
    with contextlib.suppress(Exception):
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # 检查是否为 URL
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200  # 检查是否在线可用
        return True
    return False


def delete_dsstore(path, files_to_delete=(".DS_Store", "__MACOSX")):
    """
    删除指定目录下的所有 ".DS_Store" 文件。

    Args:
        path (str, optional): 应删除 ".DS_Store" 文件的目录路径。
        files_to_delete (tuple): 要删除的文件列表。

    Example:
        ```py
        from ultralytics.utils.downloads import delete_dsstore

        delete_dsstore('path/to/dir')
        ```py

    Note:
        ".DS_Store" 文件由苹果操作系统创建，包含关于文件和文件夹的元数据。它们是隐藏的系统文件，在不同操作系统间传输文件时可能会引起问题。
    """
    # 遍历需要删除的文件列表
    for file in files_to_delete:
        # 使用路径对象查找所有匹配指定文件名的文件
        matches = list(Path(path).rglob(file))
        # 记录日志信息，指示正在删除哪些文件
        LOGGER.info(f"Deleting {file} files: {matches}")
        # 遍历每一个找到的文件路径，并删除文件
        for f in matches:
            f.unlink()
# 解压缩一个 ZIP 文件到指定路径，排除在排除列表中的文件
def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX"), exist_ok=False, progress=True):
    """
    Unzips a *.zip file to the specified path, excluding files containing strings in the exclude list.

    If the zipfile does not contain a single top-level directory, the function will create a new
    directory with the same name as the zipfile (without the extension) to extract its contents.
    If a path is not provided, the function will use the parent directory of the zipfile as the default path.

    Args:
        file (str): The path to the zipfile to be extracted.
        path (str, optional): The path to extract the zipfile to. Defaults to None.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist. Defaults to False.
        progress (bool, optional): Whether to display a progress bar. Defaults to True.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.

    Example:
        ```py
        from ultralytics.utils.downloads import unzip_file

        dir = unzip_file('path/to/file.zip')
        ```py
    """
    from zipfile import ZipFile, BadZipFile
    from pathlib import Path

    # 删除目录中的 .DS_Store 文件
    delete_dsstore(directory)
    # 转换输入的路径为 Path 对象
    directory = Path(directory)
    # 如果目录不存在，则抛出 FileNotFoundError 异常
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # 查找目录下所有不在排除列表中的文件并压缩
    files_to_zip = [f for f in directory.rglob("*") if f.is_file() and all(x not in f.name for x in exclude)]
    # 设定压缩后的文件名为目录名加 .zip 后缀
    zip_file = directory.with_suffix(".zip")
    # 设定压缩方式，根据 compress 参数选择 ZIP_DEFLATED 或 ZIP_STORED
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    # 使用 ZipFile 对象打开 zip_file，以写入模式创建压缩文件
    with ZipFile(zip_file, "w", compression) as f:
        # 使用 TQDM 显示压缩进度条，遍历 files_to_zip 列表中的文件
        for file in TQDM(files_to_zip, desc=f"Zipping {directory} to {zip_file}...", unit="file", disable=not progress):
            # 将文件写入压缩文件中，文件的相对路径以目录为基准
            f.write(file, file.relative_to(directory))

    # 返回压缩文件的路径
    return zip_file  # return path to zip file
    from zipfile import BadZipFile, ZipFile, is_zipfile

    # 检查文件是否存在且为有效的 ZIP 文件
    if not (Path(file).exists() and is_zipfile(file)):
        # 如果文件不存在或者不是有效的 ZIP 文件，则抛出异常
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    
    if path is None:
        path = Path(file).parent  # 默认路径为文件所在目录

    # 解压缩文件内容
    with ZipFile(file) as zipObj:
        # 从所有文件中筛选出不包含指定排除项的文件
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        
        # 获取顶层目录列表
        top_level_dirs = {Path(f).parts[0] for f in files}

        # 决定是直接解压缩还是解压缩到一个目录
        unzip_as_dir = len(top_level_dirs) == 1  # 判断是否只有一个顶层目录
        if unzip_as_dir:
            # 若 ZIP 文件只有一个顶层目录，则解压到指定的路径下
            extract_path = path
            path = Path(path) / list(top_level_dirs)[0]  # 将顶层目录添加到路径中
        else:
            # 若 ZIP 文件有多个文件在顶层，则解压缩到单独的子目录中
            path = extract_path = Path(path) / Path(file).stem  # 创建一个新的子目录

        # 检查目标目录是否已经存在且不为空，如果不允许覆盖，则直接返回目录路径
        if path.exists() and any(path.iterdir()) and not exist_ok:
            LOGGER.warning(f"WARNING ⚠️ Skipping {file} unzip as destination directory {path} is not empty.")
            return path

        # 遍历文件列表，逐个解压文件
        for f in TQDM(files, desc=f"Unzipping {file} to {Path(path).resolve()}...", unit="file", disable=not progress):
            # 确保文件路径在指定的解压路径内，避免路径遍历安全漏洞
            if ".." in Path(f).parts:
                LOGGER.warning(f"Potentially insecure file path: {f}, skipping extraction.")
                continue
            zipObj.extract(f, extract_path)

    return path  # 返回解压后的目录路径
# 根据给定的 URL 获取文件的头部信息
try:
    r = requests.head(url)  # 发起 HEAD 请求获取文件信息
    assert r.status_code < 400, f"URL error for {url}: {r.status_code} {r.reason}"  # 检查响应状态码
except Exception:
    return True  # 请求出现问题，默认返回 True

# 计算每个 GiB（2^30 字节）
gib = 1 << 30  # 每个 GiB 的字节数
# 计算要下载文件的大小（GB）
data = int(r.headers.get("Content-Length", 0)) / gib  # 文件大小（GB）

# 获取指定路径的磁盘使用情况
total, used, free = (x / gib for x in shutil.disk_usage(path))  # 总空间、已用空间、剩余空间（GB）

# 检查剩余空间是否足够
if data * sf < free:
    return True  # 空间足够

# 磁盘空间不足的情况
text = (
    f"WARNING ⚠️ Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, "
    f"Please free {data * sf - free:.1f} GB additional disk space and try again."
)
if hard:
    raise MemoryError(text)  # 抛出内存错误异常
LOGGER.warning(text)  # 记录警告日志
return False  # 返回空间不足
    # 使用 requests 库创建一个会话对象
    with requests.Session() as session:
        # 发送 GET 请求到指定的 Google Drive URL，并允许流式传输
        response = session.get(drive_url, stream=True)
        
        # 检查响应内容是否包含 "quota exceeded"，如果是则抛出连接错误异常
        if "quota exceeded" in str(response.content.lower()):
            raise ConnectionError(
                emojis(
                    f"❌  Google Drive file download quota exceeded. "
                    f"Please try again later or download this file manually at {link}."
                )
            )
        
        # 遍历响应中的 cookies
        for k, v in response.cookies.items():
            # 如果 cookie 的键以 "download_warning" 开头，将 token 添加到 drive_url 中
            if k.startswith("download_warning"):
                drive_url += f"&confirm={v}"  # v 是 token
        
        # 获取响应头中的 content-disposition 属性
        cd = response.headers.get("content-disposition")
        
        # 如果 content-disposition 存在
        if cd:
            # 使用正则表达式解析出文件名
            filename = re.findall('filename="(.+)"', cd)[0]
    
    # 返回更新后的 drive_url 和解析出的文件名 filename
    return drive_url, filename
# 定义一个安全下载函数，从指定的 URL 下载文件，支持多种选项如重试、解压和删除已下载文件等

def safe_download(
    url,
    file=None,
    dir=None,
    unzip=True,
    delete=False,
    curl=False,
    retry=3,
    min_bytes=1e0,
    exist_ok=False,
    progress=True,
):
    """
    Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir (str, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
        curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
        retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.
        progress (bool, optional): Whether to display a progress bar during the download. Default: True.

    Example:
        ```py
        from ultralytics.utils.downloads import safe_download

        link = "https://ultralytics.com/assets/bus.jpg"
        path = safe_download(link)
        ```py
    """

    gdrive = url.startswith("https://drive.google.com/")  # 检查 URL 是否是谷歌驱动器的链接
    if gdrive:
        url, file = get_google_drive_file_info(url)  # 如果是谷歌驱动器链接，获取文件信息

    f = Path(dir or ".") / (file or url2file(url))  # 构造文件路径，默认在当前目录下生成或指定目录
    if "://" not in str(url) and Path(url).is_file():  # 检查 URL 是否存在（在 Windows Python<3.10 中需要检查 '://'）
        f = Path(url)  # 如果 URL 是一个文件路径，则直接使用该路径作为文件名
    elif not f.is_file():  # 如果 URL 或文件不存在
        uri = (url if gdrive else clean_url(url)).replace(  # 清理和替换的 URL
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/",
            "https://ultralytics.com/assets/",  # 替换为的 URL 别名
        )
        desc = f"Downloading {uri} to '{f}'"  # 下载描述信息
        LOGGER.info(f"{desc}...")  # 记录下载信息到日志
        f.parent.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）
        check_disk_space(url, path=f.parent)  # 检查磁盘空间是否足够
        for i in range(retry + 1):  # 重试下载的次数范围
            try:
                if curl or i > 0:  # 使用 curl 下载并支持重试
                    s = "sS" * (not progress)  # 是否静默下载
                    r = subprocess.run(["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-"]).returncode  # 执行 curl 命令下载文件
                    assert r == 0, f"Curl return value {r}"  # 确保 curl 命令返回值为 0，表示下载成功
                else:  # 使用 urllib 下载
                    method = "torch"
                    if method == "torch":
                        torch.hub.download_url_to_file(url, f, progress=progress)  # 使用 torch 模块下载文件到指定路径
                    else:
                        with request.urlopen(url) as response, TQDM(  # 使用 urllib 打开 URL 并显示下载进度
                            total=int(response.getheader("Content-Length", 0)),
                            desc=desc,
                            disable=not progress,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as pbar:
                            with open(f, "wb") as f_opened:  # 打开文件并写入下载的数据
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))  # 更新下载进度条

                if f.exists():  # 如果文件存在
                    if f.stat().st_size > min_bytes:  # 如果文件大小大于指定的最小字节数
                        break  # 成功下载，退出循环
                    f.unlink()  # 删除部分下载的文件
            except Exception as e:
                if i == 0 and not is_online():  # 如果是第一次尝试且未联网
                    raise ConnectionError(emojis(f"❌  Download failure for {uri}. Environment is not online.")) from e  # 抛出连接错误异常
                elif i >= retry:  # 如果重试次数超过设定的值
                    raise ConnectionError(emojis(f"❌  Download failure for {uri}. Retry limit reached.")) from e  # 抛出连接错误异常
                LOGGER.warning(f"⚠️ Download failure, retrying {i + 1}/{retry} {uri}...")  # 记录下载失败并重试的警告信息

    if unzip and f.exists() and f.suffix in {"", ".zip", ".tar", ".gz"}:  # 如果需要解压且文件存在且文件后缀合法
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve()  # 如果提供了目录则解压到指定目录，否则解压到文件所在目录
        if is_zipfile(f):  # 如果是 ZIP 文件
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)  # 解压 ZIP 文件
        elif f.suffix in {".tar", ".gz"}:  # 如果是 .tar 或 .gz 文件
            LOGGER.info(f"Unzipping {f} to {unzip_dir}...")  # 记录解压信息到日志
            subprocess.run(["tar", "xf" if f.suffix == ".tar" else "xfz", f, "--directory", unzip_dir], check=True)  # 使用 tar 命令解压文件
        if delete:
            f.unlink()  # 删除原始压缩文件
        return unzip_dir  # 返回解压后的目录路径
# 从 GitHub 仓库中获取指定版本的标签和资产列表。如果未指定版本，则获取最新发布的资产。
def get_github_assets(repo="ultralytics/assets", version="latest", retry=False):
    # 如果版本不是最新，将版本号格式化为 'tags/version'，例如 'tags/v6.2'
    if version != "latest":
        version = f"tags/{version}"
    # 构建 GitHub API 请求的 URL
    url = f"https://api.github.com/repos/{repo}/releases/{version}"
    # 发送 GET 请求获取数据
    r = requests.get(url)  # github api
    # 如果请求失败且不是因为 403 状态码限制，并且设置了重试标志，则再次尝试请求
    if r.status_code != 200 and r.reason != "rate limit exceeded" and retry:
        r = requests.get(url)  # try again
    # 如果请求仍然失败，记录警告日志并返回空字符串和空列表
    if r.status_code != 200:
        LOGGER.warning(f"⚠️ GitHub assets check failure for {url}: {r.status_code} {r.reason}")
        return "", []
    # 解析 JSON 数据，返回标签名和资产名称列表
    data = r.json()
    return data["tag_name"], [x["name"] for x in data["assets"]]  # tag, assets i.e. ['yolov8n.pt', 'yolov8s.pt', ...]


# 尝试从 GitHub 发布资产中下载文件，如果本地不存在。首先检查本地文件，然后尝试从指定的 GitHub 仓库版本下载。
def attempt_download_asset(file, repo="ultralytics/assets", release="v8.2.0", **kwargs):
    from ultralytics.utils import SETTINGS  # 用于解决循环导入问题的局部引入

    # 对文件名进行 YOLOv5u 文件名检查和更新
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ""))
    # 如果文件存在于本地，直接返回文件路径
    if file.exists():
        return str(file)
    # 如果文件存在于设置中指定的权重目录中，直接返回文件路径
    elif (SETTINGS["weights_dir"] / file).exists():
        return str(SETTINGS["weights_dir"] / file)
    else:
        # 如果不是本地文件路径，则是URL
        name = Path(parse.unquote(str(file))).name  # 解码文件路径中的特殊字符，如 '%2F' 解码为 '/'
        download_url = f"https://github.com/{repo}/releases/download"
        
        if str(file).startswith(("http:/", "https:/")):  # 如果是以 http:/ 或 https:/ 开头的URL，则下载文件
            url = str(file).replace(":/", "://")  # 修正URL格式，Pathlib 会将 :// 转换为 :/
            file = url2file(name)  # 解析URL中的认证信息，例如 https://url.com/file.txt?auth...
            
            if Path(file).is_file():
                LOGGER.info(f"Found {clean_url(url)} locally at {file}")  # 文件已存在于本地
            else:
                safe_download(url=url, file=file, min_bytes=1e5, **kwargs)  # 安全下载文件

        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            # 如果是 GitHub 的资源仓库且文件名在预定义的资源名称列表中，则安全下载
            safe_download(url=f"{download_url}/{release}/{name}", file=file, min_bytes=1e5, **kwargs)

        else:
            # 否则，获取指定仓库和发布版本的 GitHub 资源标签和文件列表
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)  # 获取最新的发布版本
            if name in assets:
                # 如果文件名在资源列表中，则安全下载对应文件
                safe_download(url=f"{download_url}/{tag}/{name}", file=file, min_bytes=1e5, **kwargs)

        return str(file)  # 返回文件路径（本地文件或下载后的文件路径）
# 定义了一个下载函数，用于从指定的 URL 下载文件到指定目录。支持并发下载如果指定了多个线程。
def download(url, dir=Path.cwd(), unzip=True, delete=False, curl=False, threads=1, retry=3, exist_ok=False):
    """
    Downloads files from specified URLs to a given directory. Supports concurrent downloads if multiple threads are
    specified.

    Args:
        url (str | list): The URL or list of URLs of the files to be downloaded.
        dir (Path, optional): The directory where the files will be saved. Defaults to the current working directory.
        unzip (bool, optional): Flag to unzip the files after downloading. Defaults to True.
        delete (bool, optional): Flag to delete the zip files after extraction. Defaults to False.
        curl (bool, optional): Flag to use curl for downloading. Defaults to False.
        threads (int, optional): Number of threads to use for concurrent downloads. Defaults to 1.
        retry (int, optional): Number of retries in case of download failure. Defaults to 3.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.

    Example:
        ```py
        download('https://ultralytics.com/assets/example.zip', dir='path/to/dir', unzip=True)
        ```py
    """
    dir = Path(dir)  # 将目录参数转换为 Path 对象
    dir.mkdir(parents=True, exist_ok=True)  # 创建目录，如果目录不存在则递归创建

    if threads > 1:
        # 如果指定了多个线程，则使用线程池并发下载
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0],  # 单个文件的下载 URL
                    dir=x[1],  # 下载文件保存的目录
                    unzip=unzip,  # 是否解压缩
                    delete=delete,  # 是否删除压缩文件
                    curl=curl,  # 是否使用 curl 下载
                    retry=retry,  # 下载失败时的重试次数
                    exist_ok=exist_ok,  # 是否覆盖已存在的文件
                    progress=threads <= 1,  # 是否显示下载进度
                ),
                zip(url, repeat(dir)),  # 将 URL 和目录参数进行组合
            )
            pool.close()  # 关闭线程池
            pool.join()  # 等待所有线程任务完成
    else:
        # 如果只有单个线程，顺序下载每个文件
        for u in [url] if isinstance(url, (str, Path)) else url:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry, exist_ok=exist_ok)
```