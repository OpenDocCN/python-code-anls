# `D:\src\scipysrc\scikit-learn\build_tools\circle\list_versions.py`

```
#!/usr/bin/env python3

# Write the available versions page (--rst) and the version switcher JSON (--json).
# Version switcher see:
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/announcements.html#announcement-banners

import argparse   # 导入命令行参数解析模块
import json       # 导入处理 JSON 的模块
import re         # 导入正则表达式模块
import sys        # 导入系统相关模块
from urllib.request import urlopen   # 从 urllib 库中导入 urlopen 函数

from sklearn.utils.fixes import parse_version   # 从 scikit-learn 库中导入版本解析函数


def json_urlread(url):
    try:
        return json.loads(urlopen(url).read().decode("utf8"))   # 读取指定 URL 的 JSON 数据并解析
    except Exception:
        print("Error reading", url, file=sys.stderr)   # 若出现异常，则输出错误信息
        raise


def human_readable_data_quantity(quantity, multiple=1024):
    # https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    if quantity == 0:
        quantity = +0
    SUFFIXES = ["B"] + [i + {1000: "B", 1024: "iB"}[multiple] for i in "KMGTPEZY"]   # 定义数据单位后缀
    for suffix in SUFFIXES:
        if quantity < multiple or suffix == SUFFIXES[-1]:   # 根据数据量确定使用的最佳单位后缀
            if suffix == SUFFIXES[0]:
                return "%d %s" % (quantity, suffix)   # 返回格式化后的数据量字符串
            else:
                return "%.1f %s" % (quantity, suffix)   # 返回格式化后带有小数的数据量字符串
        else:
            quantity /= multiple


def get_file_extension(version):
    if "dev" in version:
        # The 'dev' branch should be explicitly handled
        return "zip"   # 如果版本号中包含 'dev'，返回 'zip' 文件扩展名

    current_version = parse_version(version)   # 解析当前版本号
    min_zip_version = parse_version("0.24")   # 定义最低支持的版本号

    return "zip" if current_version >= min_zip_version else "pdf"   # 根据版本号确定文件扩展名


def get_file_size(version):
    api_url = ROOT_URL + "%s/_downloads" % version   # 构建 API 请求 URL
    for path_details in json_urlread(api_url):
        file_extension = get_file_extension(version)   # 获取文件扩展名
        file_path = f"scikit-learn-docs.{file_extension}"   # 构建文件路径
        if path_details["name"] == file_path:
            return human_readable_data_quantity(path_details["size"], 1000)   # 返回人类可读的文件大小


parser = argparse.ArgumentParser()   # 创建命令行参数解析器
parser.add_argument("--rst", type=str, required=True)   # 添加 --rst 参数，必需
parser.add_argument("--json", type=str, required=True)   # 添加 --json 参数，必需
args = parser.parse_args()   # 解析命令行参数

heading = "Available documentation for scikit-learn"   # 定义页面标题
json_content = []   # 初始化 JSON 内容列表
rst_content = [
    ":orphan:\n",   # RST 文件的特殊标记，避免无法解析
    heading,
    "=" * len(heading) + "\n",   # 添加标题下的分隔线
    "Web-based documentation is available for versions listed below:\n",   # 添加文本说明
]

ROOT_URL = (
    "https://api.github.com/repos/scikit-learn/scikit-learn.github.io/contents/"  # GitHub API URL
)
RAW_FMT = "https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/master/%s/index.html"  # GitHub Raw 文件 URL
VERSION_RE = re.compile(r"scikit-learn ([\w\.\-]+) documentation</title>")   # 版本号的正则表达式
NAMED_DIRS = ["dev", "stable"]   # 定义特定名称的版本目录

# Gather data for each version directory, including symlinks
dirs = {}   # 初始化版本目录字典
symlinks = {}   # 初始化符号链接字典
root_listing = json_urlread(ROOT_URL)   # 获取根目录内容的 JSON 数据
for path_details in root_listing:
    name = path_details["name"]   # 获取目录或文件名
    if not (name[:1].isdigit() or name in NAMED_DIRS):   # 检查是否是以数字开头或包含在 NAMED_DIRS 中
        continue   # 跳过非版本目录和特定名称目录的处理
    # 如果路径详情中的类型是目录
    if path_details["type"] == "dir":
        # 打开指定URL并读取HTML内容，解码为UTF-8格式
        html = urlopen(RAW_FMT % name).read().decode("utf8")
        # 从HTML内容中匹配版本号并获取第一个匹配项
        version_num = VERSION_RE.search(html).group(1)
        # 获取文件大小
        file_size = get_file_size(name)
        # 将目录名作为键，版本号和文件大小作为值存入dirs字典
        dirs[name] = (version_num, file_size)
    
    # 如果路径详情中的类型是符号链接
    if path_details["type"] == "symlink":
        # 从路径详情中的链接属性中获取目标链接的URL，并读取返回的JSON数据，获取目标链接
        symlinks[name] = json_urlread(path_details["_links"]["self"])["target"]
# 对于符号链接（symlinks），应确保其数据与目标相同
for src, dst in symlinks.items():
    # 如果目标在dirs中存在，则将源对应的数据设置为目标对应的数据
    if dst in dirs:
        dirs[src] = dirs[dst]

# 按顺序输出：dev, stable，以及其他版本按降序排列
seen = set()
for i, name in enumerate(
    NAMED_DIRS
    + sorted((k for k in dirs if k[:1].isdigit()), key=parse_version, reverse=True)
):
    # 获取版本号和文件大小
    version_num, file_size = dirs[name]
    # 如果版本号已经存在于seen集合中，则跳过当前循环
    if version_num in seen:
        # 如果已经见过该版本号，则跳过当前迭代
        continue
    else:
        seen.add(version_num)

    # 构建完整的版本名，如果name以数字开头则直接使用版本号，否则加上名称
    full_name = f"{version_num}" if name[:1].isdigit() else f"{version_num} ({name})"
    # 构建版本的URL路径
    path = f"https://scikit-learn.org/{name}/"

    # 更新版本切换器的JSON数据；仅保留最新的8个版本以避免版本切换器下拉框的过载
    if i < 8:
        # 构建版本信息字典
        info = {"name": full_name, "version": version_num, "url": path}
        if name == "stable":
            info["preferred"] = True  # 如果版本名为"stable"，标记为首选版本
        json_content.append(info)

    # 用于历史版本页面的打印输出
    out = f"* `scikit-learn {full_name} documentation <{path}>`_"
    if file_size is not None:
        # 获取文件扩展名
        file_extension = get_file_extension(version_num)
        # 添加文件下载链接到输出内容中
        out += (
            f" (`{file_extension.upper()} {file_size} <{path}/"
            f"_downloads/scikit-learn-docs.{file_extension}>`_)"
        )
    rst_content.append(out)

# 将rst_content列表中的内容写入到指定的args.rst文件中
with open(args.rst, "w", encoding="utf-8") as f:
    f.write("\n".join(rst_content) + "\n")
# 打印写入的文件名和操作信息
print(f"Written {args.rst}")

# 将json_content列表中的内容写入到指定的args.json文件中
with open(args.json, "w", encoding="utf-8") as f:
    json.dump(json_content, f, indent=2)
# 打印写入的文件名和操作信息
print(f"Written {args.json}")
```