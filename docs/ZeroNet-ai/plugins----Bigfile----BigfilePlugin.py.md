# `ZeroNet\plugins\Bigfile\BigfilePlugin.py`

```
# 导入时间模块
import time
# 导入操作系统模块
import os
# 导入子进程管理模块
import subprocess
# 导入文件操作模块
import shutil
# 导入集合模块
import collections
# 导入数学模块
import math
# 导入警告模块
import warnings
# 导入base64编解码模块
import base64
# 导入二进制数据和ASCII字符互转模块
import binascii
# 导入JSON模块
import json

# 导入协程模块
import gevent
# 导入协程锁模块
import gevent.lock

# 导入自定义插件管理模块
from Plugin import PluginManager
# 导入调试模块
from Debug import Debug
# 导入加密哈希模块
from Crypt import CryptHash
# 忽略警告
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")  # 忽略缺少sha3警告
    # 导入默克尔树模块
    import merkletools

# 导入自定义工具模块
from util import helper
# 导入消息打包模块
from util import Msgpack
# 导入标志模块
from util.Flag import flag
# 导入工具模块
import util
# 导入大文件分片字段模块
from .BigfilePiecefield import BigfilePiecefield, BigfilePiecefieldPacked

# 插件加载后执行
@PluginManager.afterLoad
def importPluginnedClasses():
    # 全局变量
    global VerifyError, config
    # 导入内容管理模块的验证错误类
    from Content.ContentManager import VerifyError
    # 导入配置模块
    from Config import config

# 如果upload_nonces不在局部变量中
if "upload_nonces" not in locals():
    # 初始化upload_nonces为空字典
    upload_nonces = {}

# 注册到UiRequest插件管理器
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 判断是否允许跨域请求
    def isCorsAllowed(self, path):
        # 如果路径为"/ZeroNet-Internal/BigfileUpload"
        if path == "/ZeroNet-Internal/BigfileUpload":
            # 允许跨域请求
            return True
        else:
            # 否则调用父类的方法判断是否允许跨域请求
            return super(UiRequestPlugin, self).isCorsAllowed(path)

    # 对actionBigfileUpload方法进行响应编码
    @helper.encodeResponse
    def actionBigfileUpload(self):
        # 获取上传随机数
        nonce = self.get.get("upload_nonce")
        # 如果上传随机数不在upload_nonces中
        if nonce not in upload_nonces:
            # 返回403错误，上传随机数错误
            return self.error403("Upload nonce error.")

        # 获取上传信息
        upload_info = upload_nonces[nonce]
        # 删除上传随机数对应的信息
        del upload_nonces[nonce]

        # 发送响应头
        self.sendHeader(200, "text/html", noscript=True, extra_headers={
            "Access-Control-Allow-Origin": "null",
            "Access-Control-Allow-Credentials": "true"
        })

        # 读取多部分头部
        self.readMultipartHeaders(self.env['wsgi.input'])  # 跳过http头部
        # 处理大文件上传
        result = self.handleBigfileUpload(upload_info, self.env['wsgi.input'].read)
        # 返回处理结果的JSON字符串
        return json.dumps(result)
    # 通过 WebSocket 处理大文件上传的动作
    def actionBigfileUploadWebsocket(self):
        # 获取 WebSocket 对象
        ws = self.env.get("wsgi.websocket")

        # 如果没有 WebSocket 对象，则返回错误响应
        if not ws:
            self.start_response("400 Bad Request", [])
            return [b"Not a websocket request!"]

        # 获取上传随机数
        nonce = self.get.get("upload_nonce")
        # 如果随机数不在上传随机数列表中，则返回 403 错误
        if nonce not in upload_nonces:
            return self.error403("Upload nonce error.")

        # 获取上传信息并从上传随机数列表中删除
        upload_info = upload_nonces[nonce]
        del upload_nonces[nonce]

        # 向客户端发送 "poll" 消息
        ws.send("poll")

        # 定义读取函数，用于从 WebSocket 接收数据
        buffer = b""
        def read(size):
            nonlocal buffer
            while len(buffer) < size:
                buffer += ws.receive()
                ws.send("poll")
            part, buffer = buffer[:size], buffer[size:]
            return part

        # 处理大文件上传，并向客户端发送结果
        result = self.handleBigfileUpload(upload_info, read)
        ws.send(json.dumps(result))

    # 读取多部分头信息
    def readMultipartHeaders(self, wsgi_input):
        found = False
        for i in range(100):
            line = wsgi_input.readline()
            if line == b"\r\n":
                found = True
                break
        if not found:
            raise Exception("No multipart header found")
        return i

    # 处理文件动作
    def actionFile(self, file_path, *args, **kwargs):
        # 如果文件大小大于 1MB 并且路径部分存在，则进行处理
        if kwargs.get("file_size", 0) > 1024 * 1024 and kwargs.get("path_parts"):  
            path_parts = kwargs["path_parts"]
            # 获取站点对象并打开大文件
            site = self.server.site_manager.get(path_parts["address"])
            big_file = site.storage.openBigfile(path_parts["inner_path"], prebuffer=2 * 1024 * 1024)
            # 如果大文件存在，则设置文件对象和文件大小
            if big_file:
                kwargs["file_obj"] = big_file
                kwargs["file_size"] = big_file.size

        # 调用父类的文件处理动作
        return super(UiRequestPlugin, self).actionFile(file_path, *args, **kwargs)
# 将 UiWebsocketPlugin 类注册到 PluginManager 中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # 标记为不支持多用户的方法
    @flag.no_multiuser
    def actionSiteSetAutodownloadBigfileLimit(self, to, limit):
        # 获取指定用户的权限
        permissions = self.getPermissions(to)
        # 如果没有管理员权限，则返回错误信息
        if "ADMIN" not in permissions:
            return self.response(to, "You don't have permission to run this command")

        # 设置站点的自动下载大文件限制
        self.site.settings["autodownload_bigfile_size_limit"] = int(limit)
        # 返回操作成功信息
        self.response(to, "ok")

    # 删除文件的方法
    def actionFileDelete(self, to, inner_path):
        # 拼接得到文件的 piecemap 路径
        piecemap_inner_path = inner_path + ".piecemap.msgpack"
        # 如果有文件权限，并且 piecemap 文件存在
        if self.hasFilePermission(inner_path) and self.site.storage.isFile(piecemap_inner_path):
            # 删除 piecemap 文件
            self.log.debug("Deleting piecemap: %s" % piecemap_inner_path)
            # 获取文件信息
            file_info = self.site.content_manager.getFileInfo(piecemap_inner_path)
            if file_info:
                # 加载文件内容的 JSON 数据
                content_json = self.site.storage.loadJson(file_info["content_inner_path"])
                relative_path = file_info["relative_path"]
                # 如果文件在可选文件列表中，则删除
                if relative_path in content_json.get("files_optional", {}):
                    del content_json["files_optional"][relative_path]
                    self.site.storage.writeJson(file_info["content_inner_path"], content_json)
                    self.site.content_manager.loadContent(file_info["content_inner_path"], add_bad_files=False, force=True)
                    try:
                        # 尝试删除 piecemap 文件
                        self.site.storage.delete(piecemap_inner_path)
                    except Exception as err:
                        self.log.error("File %s delete error: %s" % (piecemap_inner_path, err))

        # 调用父类的 actionFileDelete 方法
        return super(UiWebsocketPlugin, self).actionFileDelete(to, inner_path)


# 将 ContentManagerPlugin 类注册到 PluginManager 中
@PluginManager.registerTo("ContentManager")
class ContentManagerPlugin(object):
    # 如果 inner_path 中不包含 "|"，则调用父类的 getFileInfo 方法
    def getFileInfo(self, inner_path, *args, **kwargs):
        if "|" not in inner_path:
            return super(ContentManagerPlugin, self).getFileInfo(inner_path, *args, **kwargs)

        # 从 inner_path 中分离出文件路径和文件范围
        inner_path, file_range = inner_path.split("|")
        # 将文件范围分割成起始位置和结束位置
        pos_from, pos_to = map(int, file_range.split("-"))
        # 调用父类的 getFileInfo 方法获取文件信息
        file_info = super(ContentManagerPlugin, self).getFileInfo(inner_path, *args, **kwargs)
        # 返回文件信息
        return file_info

    # 读取文件内容并返回生成器
    def readFile(self, read_func, size, buff_size=1024 * 64):
        part_num = 0
        recv_left = size

        # 循环读取文件内容
        while 1:
            part_num += 1
            # 计算本次读取的大小
            read_size = min(buff_size, recv_left)
            # 调用 read_func 方法读取数据
            part = read_func(read_size)

            # 如果没有读取到数据，跳出循环
            if not part:
                break
            # 返回读取的数据
            yield part

            # 每读取 100 次数据，休眠一小段时间，避免阻塞 ZeroNet 执行
            if part_num % 100 == 0:
                time.sleep(0.001)

            # 更新剩余需要读取的数据大小
            recv_left -= read_size
            # 如果剩余需要读取的数据大小小于等于 0，跳出循环
            if recv_left <= 0:
                break
    # 对大文件进行哈希计算
    def hashBigfile(self, read_func, size, piece_size=1024 * 1024, file_out=None):
        # 设置标志位，表示正在处理大文件
        self.site.settings["has_bigfile"] = True

        # 初始化接收数据的变量
        recv = 0
        try:
            # 初始化单个数据块的哈希对象
            piece_hash = CryptHash.sha512t()
            # 初始化存储所有数据块哈希值的列表
            piece_hashes = []
            # 初始化单个数据块接收的数据量
            piece_recv = 0

            # 创建默克尔树对象
            mt = merkletools.MerkleTools()
            # 设置哈希函数为 sha512t
            mt.hash_function = CryptHash.sha512t

            # 初始化数据块
            part = ""
            # 读取文件内容并进行哈希计算
            for part in self.readFile(read_func, size):
                # 如果有输出文件，则将数据写入输出文件
                if file_out:
                    file_out.write(part)

                # 更新接收数据的总量
                recv += len(part)
                # 更新单个数据块接收的数据量
                piece_recv += len(part)
                # 更新单个数据块的哈希值
                piece_hash.update(part)
                # 如果单个数据块接收的数据量达到设定的大小
                if piece_recv >= piece_size:
                    # 计算单个数据块的哈希值
                    piece_digest = piece_hash.digest()
                    # 将单个数据块的哈希值添加到列表中
                    piece_hashes.append(piece_digest)
                    # 将单个数据块的哈希值添加到默克尔树的叶子节点中
                    mt.leaves.append(piece_digest)
                    # 重置单个数据块的哈希对象
                    piece_hash = CryptHash.sha512t()
                    # 重置单个数据块接收的数据量
                    piece_recv = 0

                    # 每处理100个数据块或者接收数据量达到文件总大小时，输出哈希进度信息
                    if len(piece_hashes) % 100 == 0 or recv == size:
                        self.log.info("- [HASHING:%.0f%%] Pieces: %s, %.1fMB/%.1fMB" % (
                            float(recv) / size * 100, len(piece_hashes), recv / 1024 / 1024, size / 1024 / 1024
                        ))
                        part = ""
            # 如果最后一个数据块的大小大于0，则计算其哈希值并添加到列表和默克尔树中
            if len(part) > 0:
                piece_digest = piece_hash.digest()
                piece_hashes.append(piece_digest)
                mt.leaves.append(piece_digest)
        # 捕获并抛出异常
        except Exception as err:
            raise err
        # 无论是否发生异常，都会执行的代码块
        finally:
            # 如果有输出文件，则关闭输出文件
            if file_out:
                file_out.close()

        # 构建默克尔树
        mt.make_tree()
        # 获取默克尔树的根哈希值
        merkle_root = mt.get_merkle_root()
        # 如果根哈希值的类型为字节流，则转换为字符串（适用于 Python <3.5）
        if type(merkle_root) is bytes:  # Python <3.5
            merkle_root = merkle_root.decode()
        # 返回默克尔树的根哈希值、数据块大小和包含所有数据块哈希值的字典
        return merkle_root, piece_size, {
            "sha512_pieces": piece_hashes
        }
    # 获取指定文件的分片映射信息
    def getPiecemap(self, inner_path):
        # 获取指定文件的信息
        file_info = self.site.content_manager.getFileInfo(inner_path)
        # 获取分片映射文件的内部路径
        piecemap_inner_path = helper.getDirname(file_info["content_inner_path"]) + file_info["piecemap"]
        # 请求下载分片映射文件
        self.site.needFile(piecemap_inner_path, priority=20)
        # 读取并解析分片映射文件
        piecemap = Msgpack.unpack(self.site.storage.open(piecemap_inner_path, "rb").read())[helper.getFilename(inner_path)]
        # 将文件的分片大小添加到分片映射信息中
        piecemap["piece_size"] = file_info["piece_size"]
        # 返回分片映射信息
        return piecemap

    # 验证指定位置的分片是否正确
    def verifyPiece(self, inner_path, pos, piece):
        try:
            # 获取指定文件的分片映射信息
            piecemap = self.getPiecemap(inner_path)
        except Exception as err:
            # 抛出验证错误，表示无法下载分片映射文件
            raise VerifyError("Unable to download piecemap: %s" % Debug.formatException(err))
        
        # 计算分片在分片映射中的索引
        piece_i = int(pos / piecemap["piece_size"])
        # 使用哈希函数计算分片的哈希值，并与分片映射中的哈希值进行比较
        if CryptHash.sha512sum(piece, format="digest") != piecemap["sha512_pieces"][piece_i]:
            # 如果哈希值不匹配，则抛出验证错误
            raise VerifyError("Invalid hash")
        # 验证通过，返回True
        return True

    # 验证整个文件的正确性
    def verifyFile(self, inner_path, file, ignore_same=True):
        # 如果内部路径中不包含"|"，则调用父类的验证文件方法
        if "|" not in inner_path:
            return super(ContentManagerPlugin, self).verifyFile(inner_path, file, ignore_same)

        # 解析内部路径和文件范围
        inner_path, file_range = inner_path.split("|")
        pos_from, pos_to = map(int, file_range.split("-"))

        # 调用验证分片方法，验证指定位置的分片
        return self.verifyPiece(inner_path, pos_from, file)
    # 检查是否为可选下载的文件，根据 inner_path、hash_id、size 和 own 参数进行判断
    def optionalDownloaded(self, inner_path, hash_id, size=None, own=False):
        # 如果 inner_path 中包含 "|" 符号
        if "|" in inner_path:
            # 以 "|" 符号为分隔符，将 inner_path 和 file_range 分割开
            inner_path, file_range = inner_path.split("|")
            # 将 file_range 按照 "-" 分割，得到 pos_from 和 pos_to
            pos_from, pos_to = map(int, file_range.split("-"))
            # 获取 inner_path 对应的文件信息
            file_info = self.getFileInfo(inner_path)

            # 标记已下载的片段
            piece_i = int(pos_from / file_info["piece_size"])
            self.site.storage.piecefields[file_info["sha512"]][piece_i] = b"\x01"

            # 只在第一次请求时将文件大小添加到站点大小
            if hash_id in self.hashfield:
                size = 0
        # 如果文件大小大于 1024 * 1024
        elif size > 1024 * 1024:
            # 获取 inner_path 对应的文件信息
            file_info = self.getFileInfo(inner_path)
            # 如果文件信息存在且包含 "sha512" 字段
            if file_info and "sha512" in file_info:  # We already have the file, but not in piecefield
                sha512 = file_info["sha512"]
                # 如果 sha512 不在站点存储的片段字段中
                if sha512 not in self.site.storage.piecefields:
                    # 检查大文件
                    self.site.storage.checkBigfile(inner_path)

        # 调用父类的 optionalDownloaded 方法，并返回结果
        return super(ContentManagerPlugin, self).optionalDownloaded(inner_path, hash_id, size, own)

    # 检查是否为可选移除的文件，根据 inner_path、hash_id 和 size 参数进行判断
    def optionalRemoved(self, inner_path, hash_id, size=None):
        # 如果 size 存在且大于 1024 * 1024
        if size and size > 1024 * 1024:
            # 获取 inner_path 对应的文件信息
            file_info = self.getFileInfo(inner_path)
            sha512 = file_info["sha512"]
            # 如果 sha512 在站点存储的片段字段中
            if sha512 in self.site.storage.piecefields:
                # 从站点存储的片段字段中删除 sha512
                del self.site.storage.piecefields[sha512]

            # 同时从下载队列中移除文件的其他片段
            for key in list(self.site.bad_files.keys()):
                if key.startswith(inner_path + "|"):
                    del self.site.bad_files[key]
            # 移除已解决的文件任务
            self.site.worker_manager.removeSolvedFileTasks()
        # 调用父类的 optionalRemoved 方法，并返回结果
        return super(ContentManagerPlugin, self).optionalRemoved(inner_path, hash_id, size)
# 将 SiteStoragePlugin 类注册到 PluginManager 的 SiteStorage 插件中
@PluginManager.registerTo("SiteStorage")
class SiteStoragePlugin(object):
    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super(SiteStoragePlugin, self).__init__(*args, **kwargs)
        # 创建一个默认值为 BigfilePiecefield 的字典
        self.piecefields = collections.defaultdict(BigfilePiecefield)
        # 如果在站点设置的缓存中存在 "piecefields"，则遍历处理
        if "piecefields" in self.site.settings.get("cache", {}):
            for sha512, piecefield_packed in self.site.settings["cache"].get("piecefields").items():
                # 如果 piecefield_packed 存在，则解码并存入 piecefields 字典
                if piecefield_packed:
                    self.piecefields[sha512].unpack(base64.b64decode(piecefield_packed))
            # 清空站点设置中的 "piecefields"
            self.site.settings["cache"]["piecefields"] = {}

    # 创建稀疏文件的方法
    def createSparseFile(self, inner_path, size, sha512=None):
        # 获取文件路径
        file_path = self.getPath(inner_path)
        # 确保文件所在目录存在
        self.ensureDir(os.path.dirname(inner_path))
        # 以二进制写模式打开文件
        f = open(file_path, 'wb')
        # 截断文件大小，最大为 5MB
        f.truncate(min(1024 * 1024 * 5, size))  # Only pre-allocate up to 5MB
        # 关闭文件
        f.close()
        # 如果操作系统为 Windows
        if os.name == "nt":
            # 设置 subprocess 的启动信息
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            # 调用 fsutil 命令设置文件为稀疏文件
            subprocess.call(["fsutil", "sparse", "setflag", file_path], close_fds=True, startupinfo=startupinfo)
        # 如果存在 sha512 并且在 piecefields 中
        if sha512 and sha512 in self.piecefields:
            # 输出日志，删除 piecefields 中的 sha512
            self.log.debug("%s: File not exists, but has piecefield. Deleting piecefield." % inner_path)
            del self.piecefields[sha512]
    # 定义一个方法，用于向指定位置写入内容
    def write(self, inner_path, content):
        # 如果文件路径中不包含"|"，则调用父类的写入方法
        if "|" not in inner_path:
            return super(SiteStoragePlugin, self).write(inner_path, content)

        # 通过"|"分割文件路径和位置信息
        inner_path, file_range = inner_path.split("|")
        # 将位置信息分割为起始位置和结束位置
        pos_from, pos_to = map(int, file_range.split("-"))
        # 获取文件的完整路径
        file_path = self.getPath(inner_path)

        # 如果文件路径不存在，则创建对应的目录
        self.ensureDir(os.path.dirname(inner_path))

        # 如果文件不存在，则根据文件信息创建稀疏文件
        if not os.path.isfile(file_path):
            file_info = self.site.content_manager.getFileInfo(inner_path)
            self.createSparseFile(inner_path, file_info["size"])

        # 打开文件，定位到指定位置，并写入内容
        with open(file_path, "rb+") as file:
            file.seek(pos_from)
            # 如果内容是类文件对象，则使用shutil.copyfileobj方法将内容写入文件
            if hasattr(content, 'read'):  # File-like object
                shutil.copyfileobj(content, file)  # Write buff to disk
            # 如果内容是简单字符串，则直接写入文件
            else:  # Simple string
                file.write(content)
        # 释放内容对象
        del content
        # 调用更新方法，通知文件已更新
        self.onUpdated(inner_path)
    # 检查指定路径的文件是否为大文件，如果是则进行相应处理，否则返回 False
    def checkBigfile(self, inner_path):
        # 获取文件信息
        file_info = self.site.content_manager.getFileInfo(inner_path)
        # 如果文件信息不存在，或者文件信息中不包含 "piecemap" 字段，则不是大文件，返回 False
        if not file_info or (file_info and "piecemap" not in file_info):  # It's not a big file
            return False

        # 将站点设置中的 "has_bigfile" 置为 True
        self.site.settings["has_bigfile"] = True
        # 获取文件路径
        file_path = self.getPath(inner_path)
        # 获取文件的 sha512 值和分片数量
        sha512 = file_info["sha512"]
        piece_num = int(math.ceil(float(file_info["size"]) / file_info["piece_size"]))
        # 如果文件路径对应的文件存在
        if os.path.isfile(file_path):
            # 如果 sha512 值不在 self.piecefields 中
            if sha512 not in self.piecefields:
                # 如果文件的前 128 字节全为 0，则将 piece_data 设置为 b"\x00"，否则设置为 b"\x01"
                if open(file_path, "rb").read(128) == b"\0" * 128:
                    piece_data = b"\x00"
                else:
                    piece_data = b"\x01"
                # 在日志中记录文件存在但不在 piecefield 中的情况，并填充 piecefield
                self.log.debug("%s: File exists, but not in piecefield. Filling piecefiled with %s * %s." % (inner_path, piece_num, piece_data))
                self.piecefields[sha512].frombytes(piece_data * piece_num)
        # 如果文件路径对应的文件不存在
        else:
            # 在日志中记录创建大文件的情况
            self.log.debug("Creating bigfile: %s" % inner_path)
            # 创建稀疏文件，并填充 piecefield
            self.createSparseFile(inner_path, file_info["size"], sha512)
            self.piecefields[sha512].frombytes(b"\x00" * piece_num)
            # 在日志中记录创建大文件成功的情况
            self.log.debug("Created bigfile: %s" % inner_path)
        # 返回 True
        return True

    # 打开大文件，如果成功打开则返回 BigFile 对象，否则返回 False
    def openBigfile(self, inner_path, prebuffer=0):
        # 如果检查大文件失败，则返回 False
        if not self.checkBigfile(inner_path):
            return False
        # 请求下载 piecemap
        self.site.needFile(inner_path, blocking=False)  # Download piecemap
        # 返回 BigFile 对象
        return BigFile(self.site, inner_path, prebuffer=prebuffer)
# 定义一个名为 BigFile 的类
class BigFile(object):
    # 初始化方法，接受 site、inner_path 和 prebuffer 三个参数
    def __init__(self, site, inner_path, prebuffer=0):
        # 将 site 和 inner_path 分别赋值给对象的 site 和 inner_path 属性
        self.site = site
        self.inner_path = inner_path
        # 通过 site 的 storage 属性获取文件路径
        file_path = site.storage.getPath(inner_path)
        # 通过 site 的 content_manager 属性获取文件信息
        file_info = self.site.content_manager.getFileInfo(inner_path)
        # 从文件信息中获取 piece_size、sha512 和 size，并分别赋值给对象的 piece_size、sha512 和 size 属性
        self.piece_size = file_info["piece_size"]
        self.sha512 = file_info["sha512"]
        self.size = file_info["size"]
        # 将 prebuffer 赋值给对象的 prebuffer 属性
        self.prebuffer = prebuffer
        # 初始化读取的字节数为 0
        self.read_bytes = 0

        # 从 site 的 storage 属性中获取与 sha512 对应的 piecefield，并赋值给对象的 piecefield 属性
        self.piecefield = self.site.storage.piecefields[self.sha512]
        # 以读写模式打开文件，并赋值给对象的 f 属性
        self.f = open(file_path, "rb+")
        # 创建一个读取锁，并赋值给对象的 read_lock 属性
        self.read_lock = gevent.lock.Semaphore()
    # 读取数据的方法，可以指定缓冲区大小，默认为64KB
    def read(self, buff=64 * 1024):
        # 使用读取锁，确保线程安全
        with self.read_lock:
            # 获取当前文件指针位置
            pos = self.f.tell()
            # 计算读取数据的结束位置
            read_until = min(self.size, pos + buff)
            requests = []
            # 请求所有需要的数据块
            while 1:
                # 计算当前数据块的索引
                piece_i = int(pos / self.piece_size)
                # 如果当前数据块的结束位置超过了读取结束位置，则退出循环
                if piece_i * self.piece_size >= read_until:
                    break
                pos_from = piece_i * self.piece_size
                pos_to = pos_from + self.piece_size
                # 如果当前数据块还未下载，则发起数据块请求
                if not self.piecefield[piece_i]:
                    requests.append(self.site.needFile("%s|%s-%s" % (self.inner_path, pos_from, pos_to), blocking=False, update=True, priority=10))
                pos += self.piece_size

            # 如果有未完成的请求，则返回空
            if not all(requests):
                return None

            # 请求预读缓冲区
            if self.prebuffer:
                prebuffer_until = min(self.size, read_until + self.prebuffer)
                priority = 3
                while 1:
                    piece_i = int(pos / self.piece_size)
                    if piece_i * self.piece_size >= prebuffer_until:
                        break
                    pos_from = piece_i * self.piece_size
                    pos_to = pos_from + self.piece_size
                    if not self.piecefield[piece_i]:
                        self.site.needFile("%s|%s-%s" % (self.inner_path, pos_from, pos_to), blocking=False, update=True, priority=max(0, priority))
                    priority -= 1
                    pos += self.piece_size

            # 等待所有请求完成
            gevent.joinall(requests)
            # 更新已读取的字节数
            self.read_bytes += buff

            # 对于长时间的读取，增加预读缓冲区大小
            if self.read_bytes > 7 * 1024 * 1024 and self.prebuffer < 5 * 1024 * 1024:
                self.site.log.debug("%s: Increasing bigfile buffer size to 5MB..." % self.inner_path)
                self.prebuffer = 5 * 1024 * 1024

            # 返回读取的数据
            return self.f.read(buff)
    # 设置文件指针的位置
    def seek(self, pos, whence=0):
        # 使用读取锁，确保线程安全
        with self.read_lock:
            # 如果相对于文件末尾进行定位
            if whence == 2:  
                # 使用真实大小而不是磁盘上的大小
                pos = self.size + pos  
                # 将 whence 设置为 0，表示相对于文件开头进行定位
                whence = 0
            # 调用文件对象的 seek 方法进行定位
            return self.f.seek(pos, whence)
    
    # 返回文件是否支持随机访问
    def seekable(self):
        return self.f.seekable()
    
    # 返回当前文件指针的位置
    def tell(self):
        return self.f.tell()
    
    # 关闭文件
    def close(self):
        self.f.close()
    
    # 进入上下文管理器时的操作
    def __enter__(self):
        return self
    
    # 退出上下文管理器时的操作
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 调用 close 方法关闭文件
        self.close()
# 将 WorkerManagerPlugin 类注册到 PluginManager 的 WorkerManager 插件中
@PluginManager.registerTo("WorkerManager")
class WorkerManagerPlugin(object):
    # 定义 taskAddPeer 方法，用于向任务添加对等节点
    def taskAddPeer(self, task, peer):
        # 检查任务中是否包含 "piece_i" 字段
        if "piece_i" in task:
            # 如果对等节点的 piecefields 中不包含指定的 sha512 和 piece_i，则执行以下操作
            if not peer.piecefields[task["sha512"]][task["piece_i"]]:
                # 如果对等节点的 piecefields 中不包含指定的 sha512，则执行以下操作
                if task["sha512"] not in peer.piecefields:
                    # 异步执行更新对等节点的 piecefields
                    gevent.spawn(peer.updatePiecefields, force=True)
                # 如果任务中没有指定的对等节点，则执行以下操作
                elif not task["peers"]:
                    # 异步执行更新对等节点的 piecefields
                    gevent.spawn(peer.updatePiecefields)
                # 返回 False，拒绝向任务添加对等节点，如果文件不在 piecefield 中
                return False
        # 调用父类的 taskAddPeer 方法
        return super(WorkerManagerPlugin, self).taskAddPeer(task, peer)

# 将 FileRequestPlugin 类注册到 PluginManager 的 FileRequest 插件中
@PluginManager.registerTo("FileRequest")
class FileRequestPlugin(object):
    # 定义 isReadable 方法，用于判断文件是否可读
    def isReadable(self, site, inner_path, file, pos):
        # 读取文件的前 10 个字节，判断是否为空
        if file.read(10) == b"\0" * 10:
            # 如果文件看起来为空，但确保我们没有该片段，则执行以下操作
            file_info = site.content_manager.getFileInfo(inner_path)
            # 如果文件信息中包含 "piece_size" 字段，则执行以下操作
            if "piece_size" in file_info:
                piece_i = int(pos / file_info["piece_size"])
                # 如果站点的存储中不包含指定的 sha512 和 piece_i，则返回 False
                if not site.storage.piecefields[file_info["sha512"]][piece_i]:
                    return False
        # 将文件指针移动到要读取的位置
        file.seek(pos)
        # 调用父类的 isReadable 方法
        return super(FileRequestPlugin, self).isReadable(site, inner_path, file, pos)

    # 定义 actionGetPiecefields 方法，用于获取 piecefields
    def actionGetPiecefields(self, params):
        # 获取指定站点的信息
        site = self.sites.get(params["site"])
        # 如果站点不存在或不在服务中，则返回错误信息
        if not site or not site.isServing():
            self.response({"error": "Unknown site"})
            return False
        # 如果对等节点未添加到站点中，则执行以下操作
        peer = site.addPeer(self.connection.ip, self.connection.port, return_peer=True)
        if not peer.connection:  # 刚刚添加
            peer.connect(self.connection)  # 将当前连接分配给对等节点
        # 将站点存储中的 piecefields 打包，并作为响应返回
        piecefields_packed = {sha512: piecefield.pack() for sha512, piecefield in site.storage.piecefields.items()}
        self.response({"piecefields_packed": piecefields_packed})
    # 定义一个方法，用于设置站点的字段
    def actionSetPiecefields(self, params):
        # 获取指定站点
        site = self.sites.get(params["site"])
        # 如果站点不存在或者不在服务中，则返回错误信息并记录错误次数，然后返回 False
        if not site or not site.isServing():  # Site unknown or not serving
            self.response({"error": "Unknown site"})
            self.connection.badAction(5)
            return False

        # 添加或获取对等节点
        peer = site.addPeer(self.connection.ip, self.connection.port, return_peer=True, connection=self.connection)
        # 如果对等节点没有连接，则连接它
        if not peer.connection:
            peer.connect(self.connection)

        # 初始化对等节点的字段为默认值
        peer.piecefields = collections.defaultdict(BigfilePiecefieldPacked)
        # 遍历参数中的字段数据，解包并存储到对等节点的字段中
        for sha512, piecefield_packed in params["piecefields_packed"].items():
            peer.piecefields[sha512].unpack(piecefield_packed)
        # 设置站点的大文件标志为 True
        site.settings["has_bigfile"] = True

        # 返回更新成功的响应
        self.response({"ok": "Updated"})
# 将 PeerPlugin 类注册到 PluginManager 的 "Peer" 插件中
@PluginManager.registerTo("Peer")
class PeerPlugin(object):
    # 定义 __getattr__ 方法，用于获取属性
    def __getattr__(self, key):
        # 如果 key 为 "piecefields"，则创建一个默认值为 BigfilePiecefieldPacked 的字典并返回
        if key == "piecefields":
            self.piecefields = collections.defaultdict(BigfilePiecefieldPacked)
            return self.piecefields
        # 如果 key 为 "time_piecefields_updated"，则返回 None
        elif key == "time_piecefields_updated":
            self.time_piecefields_updated = None
            return self.time_piecefields_updated
        # 否则调用父类的 __getattr__ 方法
        else:
            return super(PeerPlugin, self).__getattr__(key)

    # 定义 updatePiecefields 方法，用于更新 piecefields
    @util.Noparallel(ignore_args=True)
    def updatePiecefields(self, force=False):
        # 如果连接存在且握手信息中的 "rev" 小于 2190，则返回 False
        if self.connection and self.connection.handshake.get("rev", 0) < 2190:
            return False  # Not supported

        # 如果 time_piecefields_updated 存在且当前时间与其更新时间相差不到 1 分钟，并且不是强制更新，则返回 False
        if self.time_piecefields_updated and time.time() - self.time_piecefields_updated < 60 and not force:
            return False

        # 更新 time_piecefields_updated 为当前时间
        self.time_piecefields_updated = time.time()
        # 发送请求获取 piecefields
        res = self.request("getPiecefields", {"site": self.site.address})
        # 如果 res 不存在或者包含 "error"，则返回 False
        if not res or "error" in res:
            return False

        # 重置 piecefields 为一个默认值为 BigfilePiecefieldPacked 的字典
        self.piecefields = collections.defaultdict(BigfilePiecefieldPacked)
        try:
            # 遍历获取到的 piecefields_packed，解析并存储到 piecefields 中
            for sha512, piecefield_packed in res["piecefields_packed"].items():
                self.piecefields[sha512].unpack(piecefield_packed)
        except Exception as err:
            # 捕获异常并记录日志
            self.log("Invalid updatePiecefields response: %s" % Debug.formatException(err))

        # 返回更新后的 piecefields
        return self.piecefields

    # 定义 sendMyHashfield 方法，用于发送自己的 hashfield
    def sendMyHashfield(self, *args, **kwargs):
        return super(PeerPlugin, self).sendMyHashfield(*args, **kwargs)

    # 定义 updateHashfield 方法，用于更新 hashfield
    def updateHashfield(self, *args, **kwargs):
        # 如果站点设置中包含 "has_bigfile"，则创建一个协程来更新 piecefields，并在后台更新 hashfield
        if self.site.settings.get("has_bigfile"):
            thread = gevent.spawn(self.updatePiecefields, *args, **kwargs)
            back = super(PeerPlugin, self).updateHashfield(*args, **kwargs)
            thread.join()
            return back
        # 否则只更新 hashfield
        else:
            return super(PeerPlugin, self).updateHashfield(*args, **kwargs)
    # 定义一个方法，用于获取文件
    def getFile(self, site, inner_path, *args, **kwargs):
        # 如果文件路径中包含 "|" 符号
        if "|" in inner_path:
            # 将文件路径和文件范围分开
            inner_path, file_range = inner_path.split("|")
            # 将文件范围分割成起始位置和结束位置，并转换为整数
            pos_from, pos_to = map(int, file_range.split("-"))
            # 将起始位置和结束位置添加到关键字参数中
            kwargs["pos_from"] = pos_from
            kwargs["pos_to"] = pos_to
        # 调用父类的 getFile 方法，并传入参数和关键字参数
        return super(PeerPlugin, self).getFile(site, inner_path, *args, **kwargs)
# 将 SitePlugin 类注册到 PluginManager 的 Site 插件中
@PluginManager.registerTo("Site")
class SitePlugin(object):
    # 检查文件是否允许下载
    def isFileDownloadAllowed(self, inner_path, file_info):
        # 如果文件信息中包含 "piecemap"
        if "piecemap" in file_info:
            # 计算文件大小（MB）
            file_size_mb = file_info["size"] / 1024 / 1024
            # 如果配置中设置了大文件大小限制，并且文件大小超过限制
            if config.bigfile_size_limit and file_size_mb > config.bigfile_size_limit:
                # 记录日志，跳过大文件
                self.log.debug(
                    "Bigfile size %s too large: %sMB > %sMB, skipping..." %
                    (inner_path, file_size_mb, config.bigfile_size_limit)
                )
                return False

            # 复制文件信息，将文件大小设置为分片大小
            file_info = file_info.copy()
            file_info["size"] = file_info["piece_size"]
        # 调用父类的 isFileDownloadAllowed 方法
        return super(SitePlugin, self).isFileDownloadAllowed(inner_path, file_info)

    # 获取设置缓存
    def getSettingsCache(self):
        # 调用父类的 getSettingsCache 方法
        back = super(SitePlugin, self).getSettingsCache()
        # 如果存储中包含分片字段
        if self.storage.piecefields:
            # 将分片字段的 SHA512 值和经过 base64 编码的分片字段数据组成字典
            back["piecefields"] = {sha512: base64.b64encode(piecefield.pack()).decode("utf8") for sha512, piecefield in self.storage.piecefields.items()}
        return back
    # 定义一个需要文件的方法，接受内部路径和其他参数
    def needFile(self, inner_path, *args, **kwargs):
        # 如果内部路径以"|all"结尾
        if inner_path.endswith("|all"):
            # 使用装饰器创建一个池化的需要大文件的方法
            @util.Pooled(20)
            def pooledNeedBigfile(inner_path, *args, **kwargs):
                # 如果内部路径不在坏文件列表中
                if inner_path not in self.bad_files:
                    # 记录调试信息，跳过取消的片段
                    self.log.debug("Cancelled piece, skipping %s" % inner_path)
                    return False
                # 调用需要文件的方法
                return self.needFile(inner_path, *args, **kwargs)

            # 替换内部路径中的"|all"为空字符串
            inner_path = inner_path.replace("|all", "")
            # 获取内部路径的文件信息
            file_info = self.needFileInfo(inner_path)

            # 如果文件信息中不包含"piece_size"，则使用默认方法下载非可选文件
            if "piece_size" not in file_info:
                return super(SitePlugin, self).needFile(inner_path, *args, **kwargs)

            # 获取文件大小和片段大小
            file_size = file_info["size"]
            piece_size = file_info["piece_size"]

            # 计算片段数量
            piece_num = int(math.ceil(float(file_size) / piece_size))

            # 创建文件线程列表
            file_threads = []

            # 获取文件的片段信息
            piecefield = self.storage.piecefields.get(file_info["sha512"])

            # 遍历片段数量
            for piece_i in range(piece_num):
                # 计算片段的起始和结束位置
                piece_from = piece_i * piece_size
                piece_to = min(file_size, piece_from + piece_size)
                # 如果片段信息不存在或者当前片段未下载
                if not piecefield or not piecefield[piece_i]:
                    # 构建片段的内部路径
                    inner_path_piece = "%s|%s-%s" % (inner_path, piece_from, piece_to)
                    # 将片段内部路径加入坏文件列表
                    self.bad_files[inner_path_piece] = self.bad_files.get(inner_path_piece, 1)
                    # 调用池化的需要大文件的方法，非阻塞
                    res = pooledNeedBigfile(inner_path_piece, blocking=False)
                    # 如果返回值既不是True也不是False，则加入文件线程列表
                    if res is not True and res is not False:
                        file_threads.append(res)
            # 等待所有文件线程执行完毕
            gevent.joinall(file_threads)
        else:
            # 如果内部路径不以"|all"结尾，则调用父类的需要文件的方法
            return super(SitePlugin, self).needFile(inner_path, *args, **kwargs)
# 将 ConfigPlugin 类注册到 PluginManager 的 ConfigPlugin 插件中
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    # 创建参数
    def createArguments(self):
        # 添加一个参数组到解析器中，用于处理大文件插件
        group = self.parser.add_argument_group("Bigfile plugin")
        # 添加一个参数到参数组中，用于自动下载大小限制，如果帮助分发选项被选中，也下载小于此限制的大文件，默认大小为10MB
        group.add_argument('--autodownload_bigfile_size_limit', help='Also download bigfiles smaller than this limit if help distribute option is checked', default=10, metavar="MB", type=int)
        # 添加一个参数到参数组中，用于下载的最大大文件大小限制，默认为False
        group.add_argument('--bigfile_size_limit', help='Maximum size of downloaded big files', default=False, metavar="MB", type=int)

        # 调用父类的 createArguments 方法
        return super(ConfigPlugin, self).createArguments()
```