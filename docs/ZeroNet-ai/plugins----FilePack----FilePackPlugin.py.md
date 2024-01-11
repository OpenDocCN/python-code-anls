# `ZeroNet\plugins\FilePack\FilePackPlugin.py`

```
# 导入所需的模块
import os
import re

# 导入 gevent 模块，用于实现协程
import gevent

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager

# 从 Config 模块中导入 config 变量
from Config import config

# 从 Debug 模块中导入 Debug 类
from Debug import Debug

# 用于缓存已打开的归档文件，以提高对大型站点的响应速度
archive_cache = {}

# 关闭指定归档文件的函数
def closeArchive(archive_path):
    if archive_path in archive_cache:
        del archive_cache[archive_path]

# 打开指定归档文件的函数
def openArchive(archive_path, file_obj=None):
    if archive_path not in archive_cache:
        # 如果归档文件以 "tar.gz" 结尾，则使用 tarfile 模块打开
        if archive_path.endswith("tar.gz"):
            import tarfile
            archive_cache[archive_path] = tarfile.open(archive_path, fileobj=file_obj, mode="r:gz")
        # 否则使用 zipfile 模块打开
        else:
            import zipfile
            archive_cache[archive_path] = zipfile.ZipFile(file_obj or archive_path)
        # 5 秒后关闭归档文件
        gevent.spawn_later(5, lambda: closeArchive(archive_path))

    # 返回打开的归档文件对象
    archive = archive_cache[archive_path]
    return archive

# 打开归档文件中的指定文件的函数
def openArchiveFile(archive_path, path_within, file_obj=None):
    # 调用 openArchive 函数打开归档文件
    archive = openArchive(archive_path, file_obj=file_obj)
    # 如果归档文件以 ".zip" 结尾，则使用 zipfile 模块打开指定文件
    if archive_path.endswith(".zip"):
        return archive.open(path_within)
    # 否则使用 extractfile 方法提取指定文件
    else:
        return archive.extractfile(path_within)

# 将 UiRequest 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 读取文件的函数
    def streamFile(self, file):
        # 最多读取 6MB 的数据
        for i in range(100):
            try:
                # 每次读取 60KB 的数据
                block = file.read(60 * 1024)
                if block:
                    # 返回读取的数据块
                    yield block
                else:
                    # 如果没有数据可读，则抛出 StopIteration 异常
                    raise StopIteration
            except StopIteration:
                # 关闭文件
                file.close()
                break

# 将 SiteStorage 类注册到 PluginManager 的 SiteStorage 插件中
@PluginManager.registerTo("SiteStorage")
class SiteStoragePlugin(object):
    # 判断指定路径是否为文件的函数
    def isFile(self, inner_path):
        # 如果路径中包含 ".zip/" 或 ".tar.gz/"，则处理归档文件中的文件
        if ".zip/" in inner_path or ".tar.gz/" in inner_path:
            # 使用正则表达式匹配归档文件路径和内部路径
            match = re.match("^(.*\.(?:tar.gz|zip))/(.*)", inner_path)
            archive_inner_path, path_within = match.groups()
            # 调用父类的 isFile 方法判断归档文件中的文件是否存在
            return super(SiteStoragePlugin, self).isFile(archive_inner_path)
        else:
            # 否则调用父类的 isFile 方法判断普通文件是否存在
            return super(SiteStoragePlugin, self).isFile(inner_path)
    # 打开存档文件并返回存档对象
    def openArchive(self, inner_path):
        # 获取存档文件的路径
        archive_path = self.getPath(inner_path)
        file_obj = None
        # 如果存档路径不在缓存中
        if archive_path not in archive_cache:
            # 如果存档文件不存在
            if not os.path.isfile(archive_path):
                # 从网站下载文件
                result = self.site.needFile(inner_path, priority=10)
                # 更新 WebSocket，标记文件下载完成
                self.site.updateWebsocket(file_done=inner_path)
                # 如果下载失败，则抛出异常
                if not result:
                    raise Exception("Unable to download file")
            # 打开大文件
            file_obj = self.site.storage.openBigfile(inner_path)
            # 如果打开文件失败，则置空文件对象
            if file_obj == False:
                file_obj = None

        try:
            # 打开存档文件
            archive = openArchive(archive_path, file_obj=file_obj)
        except Exception as err:
            # 如果打开存档文件失败，则抛出异常
            raise Exception("Unable to download file: %s" % Debug.formatException(err))

        # 返回存档对象
        return archive

    # 遍历存档文件内部的文件路径
    def walk(self, inner_path, *args, **kwags):
        # 如果文件路径包含".zip"或".tar.gz"
        if ".zip" in inner_path or ".tar.gz" in inner_path:
            # 匹配存档内部路径和存档内部文件路径
            match = re.match("^(.*\.(?:tar.gz|zip))(.*)", inner_path)
            archive_inner_path, path_within = match.groups()
            # 打开存档文件
            archive = self.openArchive(archive_inner_path)
            path_within = path_within.lstrip("/")

            # 如果存档内部路径以".zip"结尾
            if archive_inner_path.endswith(".zip"):
                # 获取存档内部文件列表
                namelist = [name for name in archive.namelist() if not name.endswith("/")]
            else:
                # 获取存档内部文件列表
                namelist = [item.name for item in archive.getmembers() if not item.isdir()]

            namelist_relative = []
            # 遍历存档内部文件列表
            for name in namelist:
                # 如果文件路径不以指定路径开头，则跳过
                if not name.startswith(path_within):
                    continue
                # 获取相对路径
                name_relative = name.replace(path_within, "", 1).rstrip("/")
                namelist_relative.append(name_relative)

            # 返回相对路径列表
            return namelist_relative

        else:
            # 如果文件路径不包含".zip"或".tar.gz"，则调用父类的walk方法
            return super(SiteStoragePlugin, self).walk(inner_path, *args, **kwags)
    # 定义一个方法用于列出指定路径下的文件名
    def list(self, inner_path, *args, **kwags):
        # 如果路径是一个压缩文件（.zip 或 .tar.gz）
        if ".zip" in inner_path or ".tar.gz" in inner_path:
            # 用正则表达式匹配压缩文件名和路径
            match = re.match("^(.*\.(?:tar.gz|zip))(.*)", inner_path)
            archive_inner_path, path_within = match.groups()
            # 打开压缩文件
            archive = self.openArchive(archive_inner_path)
            path_within = path_within.lstrip("/")

            # 如果是 ZIP 文件
            if archive_inner_path.endswith(".zip"):
                # 获取 ZIP 文件中的文件名列表
                namelist = [name for name in archive.namelist()]
            else:
                # 获取压缩文件中的文件名列表
                namelist = [item.name for item in archive.getmembers()]

            # 将文件名转换为相对路径
            namelist_relative = []
            for name in namelist:
                # 如果文件名不是以指定路径开头，则跳过
                if not name.startswith(path_within):
                    continue
                # 去除指定路径部分，得到相对路径
                name_relative = name.replace(path_within, "", 1).rstrip("/")

                # 如果文件在子目录中，则跳过
                if "/" in name_relative:
                    continue

                # 将相对路径添加到列表中
                namelist_relative.append(name_relative)
            # 返回相对路径列表
            return namelist_relative

        else:
            # 如果不是压缩文件，则调用父类的方法
            return super(SiteStoragePlugin, self).list(inner_path, *args, **kwags)

    # 定义一个方法用于读取指定路径下的文件内容
    def read(self, inner_path, mode="rb", **kwargs):
        # 如果路径是一个压缩文件中的文件
        if ".zip/" in inner_path or ".tar.gz/" in inner_path:
            # 用正则表达式匹配压缩文件名和路径
            match = re.match("^(.*\.(?:tar.gz|zip))(.*)", inner_path)
            archive_inner_path, path_within = match.groups()
            # 打开压缩文件
            archive = self.openArchive(archive_inner_path)
            path_within = path_within.lstrip("/")

            # 如果是 ZIP 文件
            if archive_inner_path.endswith(".zip"):
                # 读取 ZIP 文件中指定文件的内容
                return archive.open(path_within).read()
            else:
                # 读取压缩文件中指定文件的内容
                return archive.extractfile(path_within).read()

        else:
            # 如果不是压缩文件中的文件，则调用父类的方法
            return super(SiteStoragePlugin, self).read(inner_path, mode, **kwargs)
```