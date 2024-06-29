# `D:\src\scipysrc\matplotlib\tools\cache_zenodo_svg.py`

```
# 导入 urllib.request 模块，用于处理 URL 相关操作
import urllib.request
# 导入 BytesIO 类，用于在内存中操作二进制数据
from io import BytesIO
# 导入 os 模块，提供操作系统相关的功能
import os
# 导入 Path 类，用于操作文件和目录路径
from pathlib import Path


def download_or_cache(url, version):
    """
    从给定的 URL 或本地缓存获取字节数据。

    Parameters
    ----------
    url : str
        要下载的 URL。
    version : str
        文件的版本标识。

    Returns
    -------
    BytesIO
        加载到内存中的文件。
    """
    # 获取 XDG 缓存目录路径
    cache_dir = _get_xdg_cache_dir()

    # 尝试从缓存中读取数据
    if cache_dir is not None:
        try:
            data = (cache_dir / version).read_bytes()
        except OSError:
            pass
        else:
            return BytesIO(data)

    # 如果缓存中没有找到数据，则从给定的 URL 下载
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": ""})
    ) as req:
        data = req.read()

    # 尝试将下载的文件缓存到本地
    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_dir / version, "xb") as fout:
                fout.write(data)
        except OSError:
            pass

    # 返回加载到内存中的文件数据
    return BytesIO(data)


def _get_xdg_cache_dir():
    """
    返回 XDG 缓存目录。

    See
    https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
    """
    # 获取环境变量 XDG_CACHE_HOME 的值作为缓存目录
    cache_dir = os.environ.get("XDG_CACHE_HOME")
    # 如果 XDG_CACHE_HOME 未设置，则使用默认的缓存目录 ~/.cache
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache")
        # 如果路径扩展失败，则返回 None
        if cache_dir.startswith("~/"):  # Expansion failed.
            return None
    # 返回带有 "matplotlib" 子目录的 Path 对象
    return Path(cache_dir, "matplotlib")


if __name__ == "__main__":
    data = {
        "v3.9.0": "11201097",  # 版本号 "v3.9.0" 对应的 DOI 编号
        "v3.8.4": "10916799",  # 版本号 "v3.8.4" 对应的 DOI 编号
        "v3.8.3": "10661079",  # 版本号 "v3.8.3" 对应的 DOI 编号
        "v3.8.2": "10150955",  # 版本号 "v3.8.2" 对应的 DOI 编号
        "v3.8.1": "10059757",  # 版本号 "v3.8.1" 对应的 DOI 编号
        "v3.8.0": "8347255",   # 版本号 "v3.8.0" 对应的 DOI 编号
        "v3.7.3": "8336761",   # 版本号 "v3.7.3" 对应的 DOI 编号
        "v3.7.2": "8118151",   # 版本号 "v3.7.2" 对应的 DOI 编号
        "v3.7.1": "7697899",   # 版本号 "v3.7.1" 对应的 DOI 编号
        "v3.7.0": "7637593",   # 版本号 "v3.7.0" 对应的 DOI 编号
        "v3.6.3": "7527665",   # 版本号 "v3.6.3" 对应的 DOI 编号
        "v3.6.2": "7275322",   # 版本号 "v3.6.2" 对应的 DOI 编号
        "v3.6.1": "7162185",   # 版本号 "v3.6.1" 对应的 DOI 编号
        "v3.6.0": "7084615",   # 版本号 "v3.6.0" 对应的 DOI 编号
        "v3.5.3": "6982547",   # 版本号 "v3.5.3" 对应的 DOI 编号
        "v3.5.2": "6513224",   # 版本号 "v3.5.2" 对应的 DOI 编号
        "v3.5.1": "5773480",   # 版本号 "v3.5.1" 对应的 DOI 编号
        "v3.5.0": "5706396",   # 版本号 "v3.5.0" 对应的 DOI 编号
        "v3.4.3": "5194481",   # 版本号 "v3.4.3" 对应的 DOI 编号
        "v3.4.2": "4743323",   # 版本号 "v3.4.2" 对应的 DOI 编号
        "v3.4.1": "4649959",   # 版本号 "v3.4.1" 对应的 DOI 编号
        "v3.4.0": "4638398",   # 版本号 "v3.4.0" 对应的 DOI 编号
        "v3.3.4": "4475376",   # 版本号 "v3.3.4" 对应的 DOI 编号
        "v3.3.3": "4268928",   # 版本号 "v3.3.3" 对应的 DOI 编号
        "v3.3.2": "4030140",   # 版本号 "v3.3.2" 对应的 DOI 编号
        "v3.3.1": "3984190",   # 版本号 "v3.3.1" 对应的 DOI 编号
        "v3.3.0": "3948793",   # 版本号 "v3.3.0" 对应的 DOI 编号
        "v3.2.2": "3898017",   # 版本号 "v3.2.2" 对应的 DOI 编号
        "v3.2.1": "3714460",   # 版本号 "v3.2.1" 对应的 DOI 编号
        "v3.2.0": "3695547",   # 版本号 "v3.2.0" 对应的 DOI 编号
        "v3.1.3": "3633844",   # 版本号 "v3.1.3" 对应的 DOI 编号
        "v3.1.2": "3563226",   # 版本号 "v3.1.2" 对应的 DOI 编号
        "v3.1.1": "3264781",   # 版本号 "v3.1.1" 对应的 DOI 编号
        "v3.1.0": "2893252",   # 版本号 "v3.1.0" 对应的 DOI 编号
        "v3.0.3": "2577644",   # 版本号 "v3.0.3" 对应的 DOI 编号
        "v3.0.2": "1482099",   # 版本号 "v3.0.2" 对应的 DOI 编号
        "v3.0.1": "1482098",   # 版本号 "v3.0.1" 对应的 DOI 编号
        "v2.2.5": "3633833",   # 版本号 "v2.2.5" 对应的 DOI 编号
        "v3.0.0": "1420605",   # 版本号 "v3.0.0" 对应的 DOI 编号
        "v2.2.4": "2669103",   # 版本号 "v2.2.4" 对应的 DOI 编号
        "v2.2.3": "1343133",   # 版本号 "v2.2.3" 对应的 DOI 编号
        "v2.2.2": "1202077",   # 版本号 "v2.2.2" 对应的 DOI 编号
        "v2.2.1": "1202050",   # 版本号 "v2.2.1" 对应的 DOI 编号
        "v2.2.0": "1189358",   # 版本号 "v2.2.0" 对应的 DOI 编号
        "v2.1.2": "1154287",   # 版本号 "v2.1.2" 对应的 DOI 编号
        "v2.1.1": "1098480",   # 版本号 "v2.1.1" 对应的 DOI 编号
        "v2.1.0": "1004650",   # 版本号 "v2.1.0" 对应的 DOI 编号
        "v2.0.2": "573577",    # 版本号 "v2.0.2" 对应的 DOI 编号
        "v2.0.1": "570311",    # 版本号 "v2.0.1" 对应的 DOI 编号
        "v2.0.0": "248351",    # 版本号 "v2.0.0" 对应的 DOI 编号
        "v1.5.3": "61948",     # 版本号 "v1.5.3" 对应的 DOI 编号
        "v1.5.2": "56926",     # 版本号 "v1.5.2" 对应的 DOI 编号
        "v1.5.1": "44579",     # 版本号 "v1.5.1" 对应的 DOI 编号
        "v1.5.0": "32914",     # 版本号 "v1.5.0" 对应的 DOI 编号
        "v1.4.3": "15423",     # 版本号 "v1.4.3" 对应的 DOI 编号
        "v1.4.2": "12400",     # 版本号 "v1.4.2" 对应的 DOI 编号
        "v1.4.1": "12287",     # 版本号 "v1.4.1" 对应的 DOI 编
{version}
   .. image:: ../_static/zenodo_cache/{doi}.svg
      :target:  https://doi.org/10.5281/zenodo.{doi}"""
            )
        fout.write("\n\n")
        fout.write("\n".join(footer))
        fout.write("\n")



{version}
   # 在文档中插入版本号
   .. image:: ../_static/zenodo_cache/{doi}.svg
      # 插入来自指定 DOI 的 SVG 图像
      :target:  https://doi.org/10.5281/zenodo.{doi}
            )
        # 写入两个换行符到输出文件
        fout.write("\n\n")
        # 将页脚列表转换成换行分隔的字符串并写入输出文件
        fout.write("\n".join(footer))
        # 写入最后一个换行符到输出文件
        fout.write("\n")
```