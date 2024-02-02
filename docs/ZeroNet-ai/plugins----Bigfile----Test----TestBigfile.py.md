# `ZeroNet\plugins\Bigfile\Test\TestBigfile.py`

```py
# 导入时间模块
import time
# 导入io模块
import io
# 导入二进制转换模块
import binascii

# 导入pytest模块
import pytest
# 导入mock模块
import mock

# 从Connection模块中导入ConnectionServer类
from Connection import ConnectionServer
# 从Content.ContentManager模块中导入VerifyError类
from Content.ContentManager import VerifyError
# 从File模块中导入FileServer类和FileRequest类
from File import FileServer
from File import FileRequest
# 从Worker模块中导入WorkerManager类
from Worker import WorkerManager
# 从Peer模块中导入Peer类
from Peer import Peer
# 从Bigfile模块中导入BigfilePiecefield类和BigfilePiecefieldPacked类
from Bigfile import BigfilePiecefield, BigfilePiecefieldPacked
# 从Test模块中导入Spy类
from Test import Spy
# 从util模块中导入Msgpack类
from util import Msgpack

# 使用pytest的usefixtures装饰器，重置设置
@pytest.mark.usefixtures("resetSettings")
# 使用pytest的usefixtures装饰器，重置临时设置
@pytest.mark.usefixtures("resetTempSettings")
class TestBigfile:
    # 设置私钥
    privatekey = "5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv"
    # 设置分片大小
    piece_size = 1024 * 1024

    # 创建大文件的方法
    def createBigfile(self, site, inner_path="data/optional.any.iso", pieces=10):
        # 打开文件
        f = site.storage.open(inner_path, "w")
        # 写入数据
        for i in range(pieces * 100):
            f.write(("Test%s" % i).ljust(10, "-") * 1000)
        # 关闭文件
        f.close()
        # 确保站点内容管理器对content.json进行签名
        assert site.content_manager.sign("content.json", self.privatekey)
        # 返回内部路径
        return inner_path

    # 测试创建分片映射的方法
    def testPiecemapCreate(self, site):
        # 创建大文件
        inner_path = self.createBigfile(site)
        # 加载content.json文件
        content = site.storage.loadJson("content.json")
        # 断言文件路径在可选文件列表中
        assert "data/optional.any.iso" in content["files_optional"]
        # 获取文件节点
        file_node = content["files_optional"][inner_path]
        # 断言文件大小
        assert file_node["size"] == 10 * 1000 * 1000
        # 断言文件SHA512值
        assert file_node["sha512"] == "47a72cde3be80b4a829e7674f72b7c6878cf6a70b0c58c6aa6c17d7e9948daf6"
        # 断言文件分片映射
        assert file_node["piecemap"] == inner_path + ".piecemap.msgpack"

        # 解包分片映射文件
        piecemap = Msgpack.unpack(site.storage.open(file_node["piecemap"], "rb").read())["optional.any.iso"]
        # 断言SHA512分片数量
        assert len(piecemap["sha512_pieces"]) == 10
        # 断言第一个SHA512分片与第二个不相等
        assert piecemap["sha512_pieces"][0] != piecemap["sha512_pieces"][1]
        # 断言第一个SHA512分片的十六进制值
        assert binascii.hexlify(piecemap["sha512_pieces"][0]) == b"a73abad9992b3d0b672d0c2a292046695d31bebdcb1e150c8410bbe7c972eff3"
    # 测试验证文件的每个片段是否正确
    def testVerifyPiece(self, site):
        # 创建一个大文件
        inner_path = self.createBigfile(site)

        # 打开文件以进行读取操作
        f = site.storage.open(inner_path, "rb")
        # 遍历文件的前10个片段
        for i in range(10):
            # 读取1024*1024大小的片段数据
            piece = io.BytesIO(f.read(1024 * 1024))
            # 将片段指针移动到开头
            piece.seek(0)
            # 验证片段的正确性
            site.content_manager.verifyPiece(inner_path, i * 1024 * 1024, piece)
        # 关闭文件
        f.close()

        # 尝试使用片段1的哈希值验证片段0
        with pytest.raises(VerifyError) as err:
            i = 1
            # 以读取模式打开文件
            f = site.storage.open(inner_path, "rb")
            # 读取1024*1024大小的片段数据
            piece = io.BytesIO(f.read(1024 * 1024))
            # 关闭文件
            f.close()
            # 验证片段的正确性
            site.content_manager.verifyPiece(inner_path, i * 1024 * 1024, piece)
        # 断言捕获到的错误信息中包含"Invalid hash"
        assert "Invalid hash" in str(err.value)

    # 测试稀疏文件的创建和写入操作
    def testSparseFile(self, site):
        # 定义内部路径
        inner_path = "sparsefile"

        # 创建一个100MB的稀疏文件
        site.storage.createSparseFile(inner_path, 100 * 1024 * 1024)

        # 写入文件开头的数据
        s = time.time()
        f = site.storage.write("%s|%s-%s" % (inner_path, 0, 1024 * 1024), b"hellostart" * 1024)
        # 计算写入开头数据的时间
        time_write_start = time.time() - s

        # 写入文件末尾的数据
        s = time.time()
        f = site.storage.write("%s|%s-%s" % (inner_path, 99 * 1024 * 1024, 99 * 1024 * 1024 + 1024 * 1024), b"helloend" * 1024)
        # 计算写入末尾数据的时间
        time_write_end = time.time() - s

        # 验证写入的数据
        f = site.storage.open(inner_path)
        assert f.read(10) == b"hellostart"
        f.seek(99 * 1024 * 1024)
        assert f.read(8) == b"helloend"
        f.close()

        # 删除稀疏文件
        site.storage.delete(inner_path)

        # 断言写入末尾数据的时间不会比写入开头数据的时间长太多
        assert time_write_end <= max(0.1, time_write_start * 1.1)
    # 测试对文件服务器的范围文件请求
    def testRangedFileRequest(self, file_server, site, site_temp):
        # 创建一个大文件，并返回其内部路径
        inner_path = self.createBigfile(site)

        # 将站点添加到文件服务器的站点字典中
        file_server.sites[site.address] = site
        # 创建一个文件服务器客户端
        client = FileServer(file_server.ip, 1545)
        # 将临时站点添加到客户端的站点字典中
        client.sites[site_temp.address] = site_temp
        # 将客户端设置为临时站点的连接服务器
        site_temp.connection_server = client
        # 从客户端获取到文件服务器的连接
        connection = client.getConnection(file_server.ip, 1544)

        # 将文件服务器添加为客户端的对等文件服务器
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)

        # 从对等文件服务器获取指定范围的文件数据
        buff = peer_file_server.getFile(site_temp.address, "%s|%s-%s" % (inner_path, 5 * 1024 * 1024, 6 * 1024 * 1024))

        # 断言获取的数据长度为1MB，即正确的块大小
        assert len(buff.getvalue()) == 1 * 1024 * 1024  
        # 断言获取的数据以"Test524"开头，即正确的数据
        assert buff.getvalue().startswith(b"Test524")  
        # 重置缓冲区位置
        buff.seek(0)
        # 断言站点内容管理器验证指定范围的文件数据的哈希值，即正确的哈希值
        assert site.content_manager.verifyPiece(inner_path, 5 * 1024 * 1024, buff)  

        # 关闭连接
        connection.close()
        # 停止客户端
        client.stop()
    # 测试分段文件下载功能
    def testRangedFileDownload(self, file_server, site, site_temp):
        # 创建一个大文件
        inner_path = self.createBigfile(site)

        # 初始化源服务器
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # 确保文件和可选哈希字段中的分片映射存在
        file_info = site.content_manager.getFileInfo(inner_path)
        assert site.content_manager.hashfield.hasHash(file_info["sha512"])

        # 获取分片映射的哈希值
        piecemap_hash = site.content_manager.getFileInfo(file_info["piecemap"])["sha512"]
        assert site.content_manager.hashfield.hasHash(piecemap_hash)

        # 初始化客户端服务器
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        peer_client = site_temp.addPeer(file_server.ip, 1544)

        # 下载站点
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # 检查是否有坏文件
        bad_files = site_temp.storage.verifyFiles(quick_check=True)["bad_files"]
        assert not bad_files

        # 下载第5和第10块
        site_temp.needFile("%s|%s-%s" % (inner_path, 5 * 1024 * 1024, 6 * 1024 * 1024))
        site_temp.needFile("%s|%s-%s" % (inner_path, 9 * 1024 * 1024, 10 * 1024 * 1024))

        # 验证第0块未下载
        f = site_temp.storage.open(inner_path)
        assert f.read(10) == b"\0" * 10
        # 验证第5和第10块已下载
        f.seek(5 * 1024 * 1024)
        assert f.read(7) == b"Test524"
        f.seek(9 * 1024 * 1024)
        assert f.read(7) == b"943---T"

        # 验证哈希字段
        assert set(site_temp.content_manager.hashfield) == set([18343, 43727])  # 18343: data/optional.any.iso, 43727: data/optional.any.iso.hashmap.msgpack

    @pytest.mark.parametrize("piecefield_obj", [BigfilePiecefield, BigfilePiecefieldPacked])
    # 定义一个测试函数，用于测试给定的 piecefield_obj 对象和 site 参数
    def testPiecefield(self, piecefield_obj, site):
        # 定义测试数据列表
        testdatas = [
            b"\x01" * 100 + b"\x00" * 900 + b"\x01" * 4000 + b"\x00" * 4999 + b"\x01",
            b"\x00\x01\x00\x01\x00\x01" * 10 + b"\x00\x01" * 90 + b"\x01\x00" * 400 + b"\x00" * 4999,
            b"\x01" * 10000,
            b"\x00" * 10000
        ]
        # 遍历测试数据
        for testdata in testdatas:
            # 创建一个新的 piecefield 对象
            piecefield = piecefield_obj()

            # 将测试数据转换为字节流，并赋值给 piecefield 对象
            piecefield.frombytes(testdata)
            # 断言 piecefield 对象转换为字节流后与原始测试数据相等
            assert piecefield.tobytes() == testdata
            # 断言 piecefield 对象的第一个元素与测试数据的第一个元素相等
            assert piecefield[0] == testdata[0]
            # 断言 piecefield 对象的第100个元素与测试数据的第100个元素相等
            assert piecefield[100] == testdata[100]
            # 断言 piecefield 对象的第1000个元素与测试数据的第1000个元素相等
            assert piecefield[1000] == testdata[1000]
            # 断言 piecefield 对象的最后一个元素与测试数据的最后一个元素相等
            assert piecefield[len(testdata) - 1] == testdata[len(testdata) - 1]

            # 将 piecefield 对象打包
            packed = piecefield.pack()
            # 创建一个新的 piecefield 对象
            piecefield_new = piecefield_obj()
            # 解包打包后的数据，并赋值给新的 piecefield 对象
            piecefield_new.unpack(packed)
            # 断言原始 piecefield 对象转换为字节流后与新的 piecefield 对象转换为字节流相等
            assert piecefield.tobytes() == piecefield_new.tobytes()
            # 断言新的 piecefield 对象转换为字节流后与原始测试数据相等
            assert piecefield_new.tobytes() == testdata
    # 测试文件获取函数，传入文件服务器、站点和临时站点参数
    def testFileGet(self, file_server, site, site_temp):
        # 创建一个大文件，并返回其内部路径
        inner_path = self.createBigfile(site)

        # 初始化源服务器
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # 初始化客户端服务器
        site_temp.connection_server = FileServer(file_server.ip, 1545)
        site_temp.connection_server.sites[site_temp.address] = site_temp
        site_temp.addPeer(file_server.ip, 1544)

        # 下载站点
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # 下载第二个数据块
        with site_temp.storage.openBigfile(inner_path) as f:
            f.seek(1024 * 1024)
            assert f.read(1024)[0:1] != b"\0"

        # 确保第一个数据块没有被下载
        with site_temp.storage.open(inner_path) as f:
            assert f.read(1024)[0:1] == b"\0"

        # 添加对文件服务器的第二个对等体
        peer2 = site.addPeer(file_server.ip, 1545, return_peer=True)

        # 应该在第一个数据块请求时出现错误
        assert not peer2.getFile(site.address, "%s|0-%s" % (inner_path, 1024 * 1024 * 1))

        # 对第二个数据块请求不应该出现错误
        assert peer2.getFile(site.address, "%s|%s-%s" % (inner_path, 1024 * 1024 * 1, 1024 * 1024 * 2))

    # 对等体内存基准测试函数，传入站点和文件服务器参数
    def benchmarkPeerMemory(self, site, file_server):
        # 初始化源服务器
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # 导入 psutil 和 os 模块
        import psutil, os
        # 获取当前进程的内存信息
        meminfo = psutil.Process(os.getpid()).memory_info

        # 记录初始内存使用量
        mem_s = meminfo()[0]
        s = time.time()
        # 循环添加对等体
        for i in range(25000):
            site.addPeer(file_server.ip, i)
        # 打印内存使用情况
        print("%.3fs MEM: + %sKB" % (time.time() - s, (meminfo()[0] - mem_s) / 1024))  # 0.082s MEM: + 6800KB
        # 打印站点的第一个对等体的数据块信息
        print(list(site.peers.values())[0].piecefields)
    # 测试更新文件服务器的片段字段
    def testUpdatePiecefield(self, file_server, site, site_temp):
        # 创建一个大文件
        inner_path = self.createBigfile(site)

        # 设置server1为file_server
        server1 = file_server
        server1.sites[site.address] = site
        # 创建一个新的FileServer对象server2
        server2 = FileServer(file_server.ip, 1545)
        server2.sites[site_temp.address] = site_temp
        site_temp.connection_server = server2

        # 将file_server添加为client的对等方
        server2_peer1 = site_temp.addPeer(file_server.ip, 1544)

        # 测试片段字段同步
        assert len(server2_peer1.piecefields) == 0
        assert server2_peer1.updatePiecefields()  # 从对等方查询片段字段
        assert len(server2_peer1.piecefields) > 0

    # 测试WorkerManager的片段字段拒绝
    def testWorkerManagerPiecefieldDeny(self, file_server, site, site_temp):
        # 创建一个大文件
        inner_path = self.createBigfile(site)

        # 设置server1为file_server
        server1 = file_server
        server1.sites[site.address] = site
        # 创建一个新的FileServer对象server2
        server2 = FileServer(file_server.ip, 1545)
        server2.sites[site_temp.address] = site_temp
        site_temp.connection_server = server2

        # 将file_server添加为client的对等方
        server2_peer1 = site_temp.addPeer(file_server.ip, 1544)  # 正在工作

        # 下载content.json文件，但不下载文件
        site_temp.downloadContent("content.json", download_files=False)
        site_temp.needFile("data/optional.any.iso.piecemap.msgpack")

        # 添加带有已下载可选文件的虚假对等方
        for i in range(5):
            fake_peer = site_temp.addPeer("127.0.1.%s" % i, 1544)
            fake_peer.hashfield = site.content_manager.hashfield
            fake_peer.has_hashfield = True

        # 使用Spy监视WorkerManager的addWorker方法调用
        with Spy.Spy(WorkerManager, "addWorker") as requests:
            site_temp.needFile("%s|%s-%s" % (inner_path, 5 * 1024 * 1024, 6 * 1024 * 1024))
            site_temp.needFile("%s|%s-%s" % (inner_path, 6 * 1024 * 1024, 7 * 1024 * 1024))

        # 应该只从peer1请求部分，因为其他对等方的片段字段中没有请求的部分
        assert len([request[1] for request in requests if request[1] != server2_peer1]) == 0
    # 测试 WorkerManagerPiecefieldDownload 方法，传入文件服务器、站点和临时站点参数
    def testWorkerManagerPiecefieldDownload(self, file_server, site, site_temp):
        # 创建一个大文件，并返回其内部路径
        inner_path = self.createBigfile(site)

        # 将 file_server 中的站点地址映射到 site
        server1 = file_server
        server1.sites[site.address] = site
        # 创建一个新的文件服务器对象 server2
        server2 = FileServer(file_server.ip, 1545)
        # 将 site_temp 的站点地址映射到 server2
        server2.sites[site_temp.address] = site_temp
        # 将 site_temp 的连接服务器设置为 server2
        site_temp.connection_server = server2
        # 获取 inner_path 文件的 sha512 值
        sha512 = site.content_manager.getFileInfo(inner_path)["sha512"]

        # 为每个片段创建 10 个虚假对等体
        for i in range(10):
            # 创建一个对等体对象 peer，传入文件服务器 IP、端口、site_temp 和 server2 参数
            peer = Peer(file_server.ip, 1544, site_temp, server2)
            # 将 sha512 对应的第 i 个片段的值设置为 b"\x01"
            peer.piecefields[sha512][i] = b"\x01"
            # 设置 updateHashfield 方法为 mock.MagicMock 对象，返回值为 False
            peer.updateHashfield = mock.MagicMock(return_value=False)
            # 设置 updatePiecefields 方法为 mock.MagicMock 对象，返回值为 False
            peer.updatePiecefields = mock.MagicMock(return_value=False)
            # 设置 findHashIds 方法为 mock.MagicMock 对象，返回值为 {"nope": []}
            peer.findHashIds = mock.MagicMock(return_value={"nope": []})
            # 将 peer 的 hashfield 属性设置为 site.content_manager.hashfield
            peer.hashfield = site.content_manager.hashfield
            # 将 peer 的 has_hashfield 属性设置为 True
            peer.has_hashfield = True
            # 将 peer 的 key 属性设置为 "Peer:%s" % i
            peer.key = "Peer:%s" % i
            # 将 "Peer:%s" % i 作为 key，将 peer 对象添加到 site_temp 的 peers 字典中
            site_temp.peers["Peer:%s" % i] = peer

        # 下载 content.json 文件，不下载文件
        site_temp.downloadContent("content.json", download_files=False)
        # 请求 data/optional.any.iso.piecemap.msgpack 文件
        site_temp.needFile("data/optional.any.iso.piecemap.msgpack")

        # 使用 Spy.Spy 对象监视 Peer 类的 getFile 方法调用
        with Spy.Spy(Peer, "getFile") as requests:
            # 循环 10 次
            for i in range(10):
                # 请求 inner_path 文件的第 i 个片段
                site_temp.needFile("%s|%s-%s" % (inner_path, i * 1024 * 1024, (i + 1) * 1024 * 1024))

        # 断言 requests 的长度为 10
        assert len(requests) == 10
        # 循环 10 次
        for i in range(10):
            # 断言 requests[i][0] 等于 site_temp.peers["Peer:%s" % i]，即每个部分应该从片段所有者对等体请求
            assert requests[i][0] == site_temp.peers["Peer:%s" % i]
    # 测试下载统计信息的方法，传入文件服务器、站点和临时站点对象
    def testDownloadStats(self, file_server, site, site_temp):
        # 创建一个大文件，并获取其内部路径
        inner_path = self.createBigfile(site)

        # 初始化源服务器
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # 初始化客户端服务器
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        site_temp.addPeer(file_server.ip, 1544)

        # 下载站点
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # 打开虚拟文件
        assert not site_temp.storage.isFile(inner_path)

        # 检查下载前的文件大小
        assert site_temp.settings["size"] < 10 * 1024 * 1024
        assert site_temp.settings["optional_downloaded"] == 0
        size_piecemap = site_temp.content_manager.getFileInfo(inner_path + ".piecemap.msgpack")["size"]
        size_bigfile = site_temp.content_manager.getFileInfo(inner_path)["size"]

        with site_temp.storage.openBigfile(inner_path) as f:
            assert b"\0" not in f.read(1024)
            assert site_temp.settings["optional_downloaded"] == size_piecemap + size_bigfile

        with site_temp.storage.openBigfile(inner_path) as f:
            # 不重复计算
            assert b"\0" not in f.read(1024)
            assert site_temp.settings["optional_downloaded"] == size_piecemap + size_bigfile

            # 添加第二个数据块
            assert b"\0" not in f.read(1024 * 1024)
            assert site_temp.settings["optional_downloaded"] == size_piecemap + size_bigfile
    # 测试预缓冲功能
    def testPrebuffer(self, file_server, site, site_temp):
        # 创建一个大文件
        inner_path = self.createBigfile(site)

        # 初始化源服务器
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # 初始化客户端服务器
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        site_temp.addPeer(file_server.ip, 1544)

        # 下载站点
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # 打开虚拟文件
        assert not site_temp.storage.isFile(inner_path)

        # 使用预缓冲打开大文件，预缓冲大小为 1024 * 1024 * 2 字节
        with site_temp.storage.openBigfile(inner_path, prebuffer=1024 * 1024 * 2) as f:
            # 使用 Spy 模块监视 FileRequest 的路由
            with Spy.Spy(FileRequest, "route") as requests:
                # 移动文件指针到 5 * 1024 * 1024 字节处
                f.seek(5 * 1024 * 1024)
                # 读取 7 个字节并断言内容为 b"Test524"
                assert f.read(7) == b"Test524"
            # 断言任务数量为 2
            assert len([task for task in site_temp.worker_manager.tasks if task["inner_path"].startswith(inner_path)]) == 2

            # 等待预缓冲下载完成
            time.sleep(0.5)

            # 获取文件的 SHA512 值
            sha512 = site.content_manager.getFileInfo(inner_path)["sha512"]
            # 断言预缓冲的 piecefields 内容为 "0000011100"
            assert site_temp.storage.piecefields[sha512].tostring() == "0000011100"

            # 文件末尾没有预缓冲
            f.seek(9 * 1024 * 1024)
            # 断言读取的 7 个字节中不包含空字符
            assert b"\0" not in f.read(7)

            # 断言任务数量为 0
            assert len([task for task in site_temp.worker_manager.tasks if task["inner_path"].startswith(inner_path)]) == 0
    # 测试下载所有文件片段的方法
    def testDownloadAllPieces(self, file_server, site, site_temp):
        # 创建一个大文件，并获取其内部路径
        inner_path = self.createBigfile(site)

        # 初始化源服务器
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # 初始化客户端服务器
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        site_temp.addPeer(file_server.ip, 1544)

        # 下载站点
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # 打开虚拟文件
        assert not site_temp.storage.isFile(inner_path)

        # 使用 Spy.Spy 类监视 FileRequest 的路由方法
        with Spy.Spy(FileRequest, "route") as requests:
            # 需要获取指定路径下的所有文件片段
            site_temp.needFile("%s|all" % inner_path)

        # 断言请求的数量为12，包括 piecemap.msgpack, getPiecefields, 10 个片段
        assert len(requests) == 12  

        # 不重新下载已经获取的文件片段
        with Spy.Spy(FileRequest, "route") as requests:
            site_temp.needFile("%s|all" % inner_path)

        # 断言请求的数量为0
        assert len(requests) == 0
    # 测试文件大小的方法，接受文件服务器、站点和临时站点作为参数
    def testFileSize(self, file_server, site, site_temp):
        # 创建一个大文件，并返回其内部路径
        inner_path = self.createBigfile(site)

        # 初始化源服务器
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # 初始化客户端服务器
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        site_temp.addPeer(file_server.ip, 1544)

        # 下载站点
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # 打开虚拟文件
        assert not site_temp.storage.isFile(inner_path)

        # 下载第一个数据块
        site_temp.needFile("%s|%s-%s" % (inner_path, 0 * 1024 * 1024, 1 * 1024 * 1024))
        # 断言站点临时存储的文件大小小于实际大小的 1000 * 1000 * 10
        assert site_temp.storage.getSize(inner_path) < 1000 * 1000 * 10  # Size on the disk should be smaller than the real size

        # 下载第二个数据块
        site_temp.needFile("%s|%s-%s" % (inner_path, 9 * 1024 * 1024, 10 * 1024 * 1024))
        # 断言站点临时存储的文件大小等于站点存储的文件大小
        assert site_temp.storage.getSize(inner_path) == site.storage.getSize(inner_path)

    # 使用不同大小的参数进行测试
    @pytest.mark.parametrize("size", [1024 * 3, 1024 * 1024 * 3, 1024 * 1024 * 30])
    # 定义一个测试空文件读取的方法，传入文件服务器、站点、临时站点和文件大小作为参数
    def testNullFileRead(self, file_server, site, site_temp, size):
        # 内部路径为"data/optional.iso"
        inner_path = "data/optional.iso"

        # 在站点存储中打开内部路径的文件，以写入模式
        f = site.storage.open(inner_path, "w")
        # 向文件中写入指定大小的空字符
        f.write("\0" * size)
        # 关闭文件
        f.close()
        # 确保站点内容管理器对"content.json"进行签名
        assert site.content_manager.sign("content.json", self.privatekey)

        # 初始化源服务器
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # 初始化客户端服务器
        site_temp.connection_server = FileServer(file_server.ip, 1545)
        site_temp.connection_server.sites[site_temp.address] = site_temp
        site_temp.addPeer(file_server.ip, 1544)

        # 下载站点
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # 如果"piecemap"在站点内容管理器的内部路径文件信息中，则为大文件
        if "piecemap" in site.content_manager.getFileInfo(inner_path):  
            site_temp.needFile(inner_path + "|all")
        else:
            site_temp.needFile(inner_path)

        # 确保临时站点存储中内部路径文件的大小为指定大小
        assert site_temp.storage.getSize(inner_path) == size
```