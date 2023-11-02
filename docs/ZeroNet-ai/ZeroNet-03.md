# ZeroNet源码解析 3

# `plugins/Bigfile/__init__.py`

这段代码定义了一个名为BigfilePlugin的类，以及一个名为BigfilePiecefield的类，还有一个名为BigfilePiecefieldPacked的类。

BigfilePlugin是一个用于Bigfile文件的插件，它可以将Bigfile文件中的每一条记录转化成一个 packing 包。 packing 包是一种用于传输数据到另一个 Bigfile文件的数据单位，它由一系列的 packing 包组成。每个 packing 包包含一个文件名、一个数据序列和一个数据大小。

BigfilePiecefield是一个包装类，用于表示一个记录。它包含一个文件名和一个数据序列，文件名用于标识该记录所属的包。

BigfilePiecefieldPacked是一个包装类，用于表示一个 packing 包。它包含一个数据序列、一个数据大小和一个文件名。

这段代码定义了一个用于将Bigfile文件中的数据打包成数据单位的框架，以及用于表示每个记录和每个打包包的类。


```py
from . import BigfilePlugin
from .BigfilePiecefield import BigfilePiecefield, BigfilePiecefieldPacked
```

# `plugins/Bigfile/Test/conftest.py`

这段代码是一个Python源代码文件中的装饰函数，它从另一个名为"src.Test.conftest"的文件中导入了一个名为"*"的函数。这个装饰函数的作用是在导入的函数前后执行一些操作，可能会包括一些日志、记录、配置等。从代码中很难看到具体会执行哪些操作，因为它只是一个简单的导入装饰函数。


```py
from src.Test.conftest import *

```

# `plugins/Bigfile/Test/TestBigfile.py`

这段代码的作用是实现一个简单的并发连接池，用于处理大量的并发连接请求。具体解释如下：

1. 导入time、io、binascii、pytest、mock、Connection、ContentManager、FileServer、FileRequest、Worker、Peer、Bigfile、BigfilePiecefield、BigfilePiecefieldPacked、Spy等模块。

2. 创建一个名为ConnectionServer的类，实现了一个服务器端的连接池，用于处理来自客户端的连接请求。

3. 创建一个名为FileServer的类，实现了一个文件服务器，用于提供文件内容的读写操作。

4. 创建一个名为FileRequest的类，实现了一个文件请求，用于在客户端和文件服务器之间的交互。

5. 创建一个名为WorkerManager的类，实现了一个工作器管理器，用于管理所有工作器的工作状态。

6. 创建一个名为Peer的类，实现了一个用于连接到远程计算机的类，用于与远程服务器建立连接。

7. 创建一个名为BigfilePiecefield的类，实现了一个用于表示大文件中的块的类。

8. 创建一个名为BigfilePiecefieldPacked的类，实现了一个用于在大文件中移动块的类。

9. 导入pytest、mock等模块，用于进行测试和模拟。

10. 在代码中引入了用于模拟连接池中连接的一些模块，包括time.sleep、io.BufferedReader、io.BufferedWriter、io.BytesIO、mock.Mock、pytest. DixitMocker、mock.Spy。

11. 在代码中定义了一些常量和函数，用于进行时间控制、模拟连接池中的连接、模拟文件服务器等。

12. 在代码中通过使用这些函数和常量，实现了以下功能：

13. 通过time.sleep保证每个连接请求都有一个延迟，可以更好地控制延迟时间。

14. 通过io.BufferedReader和io.BufferedWriter实现了文件服务器对客户端的读写操作。

15. 通过io.BytesIO实现了将文件服务器中的内容写入到io.BytesIO对象中，并从该对象中读取文件内容。

16. 通过mock.Mock和pytest.DixitMocker实现了对连接池中连接的模拟，包括模拟连接建立、断开连接、发送数据等。

17. 通过mock.Spy实现了对工作器Manager中工作状态的模拟，包括连接建立、断开连接、执行请求等。

18. 通过这些模拟实现了并发连接池中连接的并发处理，可以更好地控制连接请求的发送和接收。


```py
import time
import io
import binascii

import pytest
import mock

from Connection import ConnectionServer
from Content.ContentManager import VerifyError
from File import FileServer
from File import FileRequest
from Worker import WorkerManager
from Peer import Peer
from Bigfile import BigfilePiecefield, BigfilePiecefieldPacked
from Test import Spy
```

What is the purpose of the `site_temp.storage.getSize` and `site_temp.storage.getOverall負据` functions in the `test_null_file_read.py` file?

It seems that `site_temp.storage` is being used for downloading and caching files, and the `getSize` and `getOverall` functions are used to measure the size of the downloaded file in bytes.

The `getSize` function returns the number of bytes that have been downloaded from the server, while the `getOverall` function returns a tuple with the total size of the downloaded file (including the metadata).

Is there anything else you'd like to know about this code?


```py
from util import Msgpack


@pytest.mark.usefixtures("resetSettings")
@pytest.mark.usefixtures("resetTempSettings")
class TestBigfile:
    privatekey = "5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv"
    piece_size = 1024 * 1024

    def createBigfile(self, site, inner_path="data/optional.any.iso", pieces=10):
        f = site.storage.open(inner_path, "w")
        for i in range(pieces * 100):
            f.write(("Test%s" % i).ljust(10, "-") * 1000)
        f.close()
        assert site.content_manager.sign("content.json", self.privatekey)
        return inner_path

    def testPiecemapCreate(self, site):
        inner_path = self.createBigfile(site)
        content = site.storage.loadJson("content.json")
        assert "data/optional.any.iso" in content["files_optional"]
        file_node = content["files_optional"][inner_path]
        assert file_node["size"] == 10 * 1000 * 1000
        assert file_node["sha512"] == "47a72cde3be80b4a829e7674f72b7c6878cf6a70b0c58c6aa6c17d7e9948daf6"
        assert file_node["piecemap"] == inner_path + ".piecemap.msgpack"

        piecemap = Msgpack.unpack(site.storage.open(file_node["piecemap"], "rb").read())["optional.any.iso"]
        assert len(piecemap["sha512_pieces"]) == 10
        assert piecemap["sha512_pieces"][0] != piecemap["sha512_pieces"][1]
        assert binascii.hexlify(piecemap["sha512_pieces"][0]) == b"a73abad9992b3d0b672d0c2a292046695d31bebdcb1e150c8410bbe7c972eff3"

    def testVerifyPiece(self, site):
        inner_path = self.createBigfile(site)

        # Verify all 10 piece
        f = site.storage.open(inner_path, "rb")
        for i in range(10):
            piece = io.BytesIO(f.read(1024 * 1024))
            piece.seek(0)
            site.content_manager.verifyPiece(inner_path, i * 1024 * 1024, piece)
        f.close()

        # Try to verify piece 0 with piece 1 hash
        with pytest.raises(VerifyError) as err:
            i = 1
            f = site.storage.open(inner_path, "rb")
            piece = io.BytesIO(f.read(1024 * 1024))
            f.close()
            site.content_manager.verifyPiece(inner_path, i * 1024 * 1024, piece)
        assert "Invalid hash" in str(err.value)

    def testSparseFile(self, site):
        inner_path = "sparsefile"

        # Create a 100MB sparse file
        site.storage.createSparseFile(inner_path, 100 * 1024 * 1024)

        # Write to file beginning
        s = time.time()
        f = site.storage.write("%s|%s-%s" % (inner_path, 0, 1024 * 1024), b"hellostart" * 1024)
        time_write_start = time.time() - s

        # Write to file end
        s = time.time()
        f = site.storage.write("%s|%s-%s" % (inner_path, 99 * 1024 * 1024, 99 * 1024 * 1024 + 1024 * 1024), b"helloend" * 1024)
        time_write_end = time.time() - s

        # Verify writes
        f = site.storage.open(inner_path)
        assert f.read(10) == b"hellostart"
        f.seek(99 * 1024 * 1024)
        assert f.read(8) == b"helloend"
        f.close()

        site.storage.delete(inner_path)

        # Writing to end shold not take much longer, than writing to start
        assert time_write_end <= max(0.1, time_write_start * 1.1)

    def testRangedFileRequest(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        file_server.sites[site.address] = site
        client = FileServer(file_server.ip, 1545)
        client.sites[site_temp.address] = site_temp
        site_temp.connection_server = client
        connection = client.getConnection(file_server.ip, 1544)

        # Add file_server as peer to client
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)

        buff = peer_file_server.getFile(site_temp.address, "%s|%s-%s" % (inner_path, 5 * 1024 * 1024, 6 * 1024 * 1024))

        assert len(buff.getvalue()) == 1 * 1024 * 1024  # Correct block size
        assert buff.getvalue().startswith(b"Test524")  # Correct data
        buff.seek(0)
        assert site.content_manager.verifyPiece(inner_path, 5 * 1024 * 1024, buff)  # Correct hash

        connection.close()
        client.stop()

    def testRangedFileDownload(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        # Init source server
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # Make sure the file and the piecemap in the optional hashfield
        file_info = site.content_manager.getFileInfo(inner_path)
        assert site.content_manager.hashfield.hasHash(file_info["sha512"])

        piecemap_hash = site.content_manager.getFileInfo(file_info["piecemap"])["sha512"]
        assert site.content_manager.hashfield.hasHash(piecemap_hash)

        # Init client server
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        peer_client = site_temp.addPeer(file_server.ip, 1544)

        # Download site
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        bad_files = site_temp.storage.verifyFiles(quick_check=True)["bad_files"]
        assert not bad_files

        # client_piecefield = peer_client.piecefields[file_info["sha512"]].tostring()
        # assert client_piecefield == "1" * 10

        # Download 5. and 10. block

        site_temp.needFile("%s|%s-%s" % (inner_path, 5 * 1024 * 1024, 6 * 1024 * 1024))
        site_temp.needFile("%s|%s-%s" % (inner_path, 9 * 1024 * 1024, 10 * 1024 * 1024))

        # Verify 0. block not downloaded
        f = site_temp.storage.open(inner_path)
        assert f.read(10) == b"\0" * 10
        # Verify 5. and 10. block downloaded
        f.seek(5 * 1024 * 1024)
        assert f.read(7) == b"Test524"
        f.seek(9 * 1024 * 1024)
        assert f.read(7) == b"943---T"

        # Verify hashfield
        assert set(site_temp.content_manager.hashfield) == set([18343, 43727])  # 18343: data/optional.any.iso, 43727: data/optional.any.iso.hashmap.msgpack

    def testOpenBigfile(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        # Init source server
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # Init client server
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        site_temp.addPeer(file_server.ip, 1544)

        # Download site
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # Open virtual file
        assert not site_temp.storage.isFile(inner_path)

        with site_temp.storage.openBigfile(inner_path) as f:
            with Spy.Spy(FileRequest, "route") as requests:
                f.seek(5 * 1024 * 1024)
                assert f.read(7) == b"Test524"

                f.seek(9 * 1024 * 1024)
                assert f.read(7) == b"943---T"

            assert len(requests) == 4  # 1x peicemap + 1x getpiecefield + 2x for pieces

            assert set(site_temp.content_manager.hashfield) == set([18343, 43727])

            assert site_temp.storage.piecefields[f.sha512].tostring() == "0000010001"
            assert f.sha512 in site_temp.getSettingsCache()["piecefields"]

            # Test requesting already downloaded
            with Spy.Spy(FileRequest, "route") as requests:
                f.seek(5 * 1024 * 1024)
                assert f.read(7) == b"Test524"

            assert len(requests) == 0

            # Test requesting multi-block overflow reads
            with Spy.Spy(FileRequest, "route") as requests:
                f.seek(5 * 1024 * 1024)  # We already have this block
                data = f.read(1024 * 1024 * 3)  # Our read overflow to 6. and 7. block
                assert data.startswith(b"Test524")
                assert data.endswith(b"Test838-")
                assert b"\0" not in data  # No null bytes allowed

            assert len(requests) == 2  # Two block download

            # Test out of range request
            f.seek(5 * 1024 * 1024)
            data = f.read(1024 * 1024 * 30)
            assert len(data) == 10 * 1000 * 1000 - (5 * 1024 * 1024)

            f.seek(30 * 1024 * 1024)
            data = f.read(1024 * 1024 * 30)
            assert len(data) == 0

    @pytest.mark.parametrize("piecefield_obj", [BigfilePiecefield, BigfilePiecefieldPacked])
    def testPiecefield(self, piecefield_obj, site):
        testdatas = [
            b"\x01" * 100 + b"\x00" * 900 + b"\x01" * 4000 + b"\x00" * 4999 + b"\x01",
            b"\x00\x01\x00\x01\x00\x01" * 10 + b"\x00\x01" * 90 + b"\x01\x00" * 400 + b"\x00" * 4999,
            b"\x01" * 10000,
            b"\x00" * 10000
        ]
        for testdata in testdatas:
            piecefield = piecefield_obj()

            piecefield.frombytes(testdata)
            assert piecefield.tobytes() == testdata
            assert piecefield[0] == testdata[0]
            assert piecefield[100] == testdata[100]
            assert piecefield[1000] == testdata[1000]
            assert piecefield[len(testdata) - 1] == testdata[len(testdata) - 1]

            packed = piecefield.pack()
            piecefield_new = piecefield_obj()
            piecefield_new.unpack(packed)
            assert piecefield.tobytes() == piecefield_new.tobytes()
            assert piecefield_new.tobytes() == testdata

    def testFileGet(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        # Init source server
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # Init client server
        site_temp.connection_server = FileServer(file_server.ip, 1545)
        site_temp.connection_server.sites[site_temp.address] = site_temp
        site_temp.addPeer(file_server.ip, 1544)

        # Download site
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # Download second block
        with site_temp.storage.openBigfile(inner_path) as f:
            f.seek(1024 * 1024)
            assert f.read(1024)[0:1] != b"\0"

        # Make sure first block not download
        with site_temp.storage.open(inner_path) as f:
            assert f.read(1024)[0:1] == b"\0"

        peer2 = site.addPeer(file_server.ip, 1545, return_peer=True)

        # Should drop error on first block request
        assert not peer2.getFile(site.address, "%s|0-%s" % (inner_path, 1024 * 1024 * 1))

        # Should not drop error for second block request
        assert peer2.getFile(site.address, "%s|%s-%s" % (inner_path, 1024 * 1024 * 1, 1024 * 1024 * 2))

    def benchmarkPeerMemory(self, site, file_server):
        # Init source server
        site.connection_server = file_server
        file_server.sites[site.address] = site

        import psutil, os
        meminfo = psutil.Process(os.getpid()).memory_info

        mem_s = meminfo()[0]
        s = time.time()
        for i in range(25000):
            site.addPeer(file_server.ip, i)
        print("%.3fs MEM: + %sKB" % (time.time() - s, (meminfo()[0] - mem_s) / 1024))  # 0.082s MEM: + 6800KB
        print(list(site.peers.values())[0].piecefields)

    def testUpdatePiecefield(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        server1 = file_server
        server1.sites[site.address] = site
        server2 = FileServer(file_server.ip, 1545)
        server2.sites[site_temp.address] = site_temp
        site_temp.connection_server = server2

        # Add file_server as peer to client
        server2_peer1 = site_temp.addPeer(file_server.ip, 1544)

        # Testing piecefield sync
        assert len(server2_peer1.piecefields) == 0
        assert server2_peer1.updatePiecefields()  # Query piecefields from peer
        assert len(server2_peer1.piecefields) > 0

    def testWorkerManagerPiecefieldDeny(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        server1 = file_server
        server1.sites[site.address] = site
        server2 = FileServer(file_server.ip, 1545)
        server2.sites[site_temp.address] = site_temp
        site_temp.connection_server = server2

        # Add file_server as peer to client
        server2_peer1 = site_temp.addPeer(file_server.ip, 1544)  # Working

        site_temp.downloadContent("content.json", download_files=False)
        site_temp.needFile("data/optional.any.iso.piecemap.msgpack")

        # Add fake peers with optional files downloaded
        for i in range(5):
            fake_peer = site_temp.addPeer("127.0.1.%s" % i, 1544)
            fake_peer.hashfield = site.content_manager.hashfield
            fake_peer.has_hashfield = True

        with Spy.Spy(WorkerManager, "addWorker") as requests:
            site_temp.needFile("%s|%s-%s" % (inner_path, 5 * 1024 * 1024, 6 * 1024 * 1024))
            site_temp.needFile("%s|%s-%s" % (inner_path, 6 * 1024 * 1024, 7 * 1024 * 1024))

        # It should only request parts from peer1 as the other peers does not have the requested parts in piecefields
        assert len([request[1] for request in requests if request[1] != server2_peer1]) == 0

    def testWorkerManagerPiecefieldDownload(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        server1 = file_server
        server1.sites[site.address] = site
        server2 = FileServer(file_server.ip, 1545)
        server2.sites[site_temp.address] = site_temp
        site_temp.connection_server = server2
        sha512 = site.content_manager.getFileInfo(inner_path)["sha512"]

        # Create 10 fake peer for each piece
        for i in range(10):
            peer = Peer(file_server.ip, 1544, site_temp, server2)
            peer.piecefields[sha512][i] = b"\x01"
            peer.updateHashfield = mock.MagicMock(return_value=False)
            peer.updatePiecefields = mock.MagicMock(return_value=False)
            peer.findHashIds = mock.MagicMock(return_value={"nope": []})
            peer.hashfield = site.content_manager.hashfield
            peer.has_hashfield = True
            peer.key = "Peer:%s" % i
            site_temp.peers["Peer:%s" % i] = peer

        site_temp.downloadContent("content.json", download_files=False)
        site_temp.needFile("data/optional.any.iso.piecemap.msgpack")

        with Spy.Spy(Peer, "getFile") as requests:
            for i in range(10):
                site_temp.needFile("%s|%s-%s" % (inner_path, i * 1024 * 1024, (i + 1) * 1024 * 1024))

        assert len(requests) == 10
        for i in range(10):
            assert requests[i][0] == site_temp.peers["Peer:%s" % i]  # Every part should be requested from piece owner peer

    def testDownloadStats(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        # Init source server
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # Init client server
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        site_temp.addPeer(file_server.ip, 1544)

        # Download site
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # Open virtual file
        assert not site_temp.storage.isFile(inner_path)

        # Check size before downloads
        assert site_temp.settings["size"] < 10 * 1024 * 1024
        assert site_temp.settings["optional_downloaded"] == 0
        size_piecemap = site_temp.content_manager.getFileInfo(inner_path + ".piecemap.msgpack")["size"]
        size_bigfile = site_temp.content_manager.getFileInfo(inner_path)["size"]

        with site_temp.storage.openBigfile(inner_path) as f:
            assert b"\0" not in f.read(1024)
            assert site_temp.settings["optional_downloaded"] == size_piecemap + size_bigfile

        with site_temp.storage.openBigfile(inner_path) as f:
            # Don't count twice
            assert b"\0" not in f.read(1024)
            assert site_temp.settings["optional_downloaded"] == size_piecemap + size_bigfile

            # Add second block
            assert b"\0" not in f.read(1024 * 1024)
            assert site_temp.settings["optional_downloaded"] == size_piecemap + size_bigfile

    def testPrebuffer(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        # Init source server
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # Init client server
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        site_temp.addPeer(file_server.ip, 1544)

        # Download site
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # Open virtual file
        assert not site_temp.storage.isFile(inner_path)

        with site_temp.storage.openBigfile(inner_path, prebuffer=1024 * 1024 * 2) as f:
            with Spy.Spy(FileRequest, "route") as requests:
                f.seek(5 * 1024 * 1024)
                assert f.read(7) == b"Test524"
            # assert len(requests) == 3  # 1x piecemap + 1x getpiecefield + 1x for pieces
            assert len([task for task in site_temp.worker_manager.tasks if task["inner_path"].startswith(inner_path)]) == 2

            time.sleep(0.5)  # Wait prebuffer download

            sha512 = site.content_manager.getFileInfo(inner_path)["sha512"]
            assert site_temp.storage.piecefields[sha512].tostring() == "0000011100"

            # No prebuffer beyond end of the file
            f.seek(9 * 1024 * 1024)
            assert b"\0" not in f.read(7)

            assert len([task for task in site_temp.worker_manager.tasks if task["inner_path"].startswith(inner_path)]) == 0

    def testDownloadAllPieces(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        # Init source server
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # Init client server
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        site_temp.addPeer(file_server.ip, 1544)

        # Download site
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # Open virtual file
        assert not site_temp.storage.isFile(inner_path)

        with Spy.Spy(FileRequest, "route") as requests:
            site_temp.needFile("%s|all" % inner_path)

        assert len(requests) == 12  # piecemap.msgpack, getPiecefields, 10 x piece

        # Don't re-download already got pieces
        with Spy.Spy(FileRequest, "route") as requests:
            site_temp.needFile("%s|all" % inner_path)

        assert len(requests) == 0

    def testFileSize(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        # Init source server
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # Init client server
        client = ConnectionServer(file_server.ip, 1545)
        site_temp.connection_server = client
        site_temp.addPeer(file_server.ip, 1544)

        # Download site
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        # Open virtual file
        assert not site_temp.storage.isFile(inner_path)

        # Download first block
        site_temp.needFile("%s|%s-%s" % (inner_path, 0 * 1024 * 1024, 1 * 1024 * 1024))
        assert site_temp.storage.getSize(inner_path) < 1000 * 1000 * 10  # Size on the disk should be smaller than the real size

        site_temp.needFile("%s|%s-%s" % (inner_path, 9 * 1024 * 1024, 10 * 1024 * 1024))
        assert site_temp.storage.getSize(inner_path) == site.storage.getSize(inner_path)

    def testFileRename(self, file_server, site, site_temp):
        inner_path = self.createBigfile(site)

        # Init source server
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # Init client server
        site_temp.connection_server = FileServer(file_server.ip, 1545)
        site_temp.connection_server.sites[site_temp.address] = site_temp
        site_temp.addPeer(file_server.ip, 1544)

        # Download site
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        with Spy.Spy(FileRequest, "route") as requests:
            site_temp.needFile("%s|%s-%s" % (inner_path, 0, 1 * self.piece_size))

        assert len([req for req in requests if req[1] == "streamFile"]) == 2  # 1 piece + piecemap

        # Rename the file
        inner_path_new = inner_path.replace(".iso", "-new.iso")
        site.storage.rename(inner_path, inner_path_new)
        site.storage.delete("data/optional.any.iso.piecemap.msgpack")
        assert site.content_manager.sign("content.json", self.privatekey, remove_missing_optional=True)

        files_optional = site.content_manager.contents["content.json"]["files_optional"].keys()

        assert "data/optional.any-new.iso.piecemap.msgpack" in files_optional
        assert "data/optional.any.iso.piecemap.msgpack" not in files_optional
        assert "data/optional.any.iso" not in files_optional

        with Spy.Spy(FileRequest, "route") as requests:
            site.publish()
            time.sleep(0.1)
            site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)  # Wait for download

            assert len([req[1] for req in requests if req[1] == "streamFile"]) == 0

            with site_temp.storage.openBigfile(inner_path_new, prebuffer=0) as f:
                f.read(1024)

                # First piece already downloaded
                assert [req for req in requests if req[1] == "streamFile"] == []

                # Second piece needs to be downloaded + changed piecemap
                f.seek(self.piece_size)
                f.read(1024)
                assert [req[3]["inner_path"] for req in requests if req[1] == "streamFile"] == [inner_path_new + ".piecemap.msgpack", inner_path_new]

    @pytest.mark.parametrize("size", [1024 * 3, 1024 * 1024 * 3, 1024 * 1024 * 30])
    def testNullFileRead(self, file_server, site, site_temp, size):
        inner_path = "data/optional.iso"

        f = site.storage.open(inner_path, "w")
        f.write("\0" * size)
        f.close()
        assert site.content_manager.sign("content.json", self.privatekey)

        # Init source server
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # Init client server
        site_temp.connection_server = FileServer(file_server.ip, 1545)
        site_temp.connection_server.sites[site_temp.address] = site_temp
        site_temp.addPeer(file_server.ip, 1544)

        # Download site
        site_temp.download(blind_includes=True, retry_bad_files=False).join(timeout=10)

        if "piecemap" in site.content_manager.getFileInfo(inner_path):  # Bigfile
            site_temp.needFile(inner_path + "|all")
        else:
            site_temp.needFile(inner_path)


        assert site_temp.storage.getSize(inner_path) == size

```

# `plugins/Chart/ChartCollector.py`



This is a Python class that manages site collection for a website. It uses the site collection process to collect data from different data sources and stores it in the database.

The `collector` method is responsible for collecting data from the different site collectors and inserting it into the `data` table in the database. It takes site collectors, site collectors, and the last values from the database.

The `collectSites` method is responsible for collecting site data from the site collectors and inserting it into the `data` table in the database. It takes site collectors, site collectors, and the last values from the database.

The `getCollectors` method returns a list of site collectors that are configured on the website.

The `getSiteCollectors` method returns a list of site collectors for each site on the website.

The `file_server.sites` property is a dictionary that maps each site on the website to its corresponding site collector.

The `last_values` property is a dictionary that stores the last values for each data type in the database.

The `time.sleep` function is used to pause the execution of the code for 0.001 seconds to ensure that the site collection process runs smoothly.

The `INSERT INTO` statement is used to insert data into the `data` table in the database.


```py
import time
import sys
import collections
import itertools
import logging

import gevent
from util import helper
from Config import config


class ChartCollector(object):
    def __init__(self, db):
        self.db = db
        if config.action == "main":
            gevent.spawn_later(60 * 3, self.collector)
        self.log = logging.getLogger("ChartCollector")
        self.last_values = collections.defaultdict(dict)

    def setInitialLastValues(self, sites):
        # Recover last value of site bytes/sent
        for site in sites:
            self.last_values["site:" + site.address]["site_bytes_recv"] = site.settings.get("bytes_recv", 0)
            self.last_values["site:" + site.address]["site_bytes_sent"] = site.settings.get("bytes_sent", 0)

    def getCollectors(self):
        collectors = {}
        import main
        file_server = main.file_server
        sites = file_server.sites
        if not sites:
            return collectors
        content_db = list(sites.values())[0].content_manager.contents.db

        # Connection stats
        collectors["connection"] = lambda: len(file_server.connections)
        collectors["connection_in"] = (
            lambda: len([1 for connection in file_server.connections if connection.type == "in"])
        )
        collectors["connection_onion"] = (
            lambda: len([1 for connection in file_server.connections if connection.ip.endswith(".onion")])
        )
        collectors["connection_ping_avg"] = (
            lambda: round(1000 * helper.avg(
                [connection.last_ping_delay for connection in file_server.connections if connection.last_ping_delay]
            ))
        )
        collectors["connection_ping_min"] = (
            lambda: round(1000 * min(
                [connection.last_ping_delay for connection in file_server.connections if connection.last_ping_delay]
            ))
        )
        collectors["connection_rev_avg"] = (
            lambda: helper.avg(
                [connection.handshake["rev"] for connection in file_server.connections if connection.handshake]
            )
        )

        # Request stats
        collectors["file_bytes_recv|change"] = lambda: file_server.bytes_recv
        collectors["file_bytes_sent|change"] = lambda: file_server.bytes_sent
        collectors["request_num_recv|change"] = lambda: file_server.num_recv
        collectors["request_num_sent|change"] = lambda: file_server.num_sent

        # Limit
        collectors["optional_limit"] = lambda: content_db.getOptionalLimitBytes()
        collectors["optional_used"] = lambda: content_db.getOptionalUsedBytes()
        collectors["optional_downloaded"] = lambda: sum([site.settings.get("optional_downloaded", 0) for site in sites.values()])

        # Peers
        collectors["peer"] = lambda peers: len(peers)
        collectors["peer_onion"] = lambda peers: len([True for peer in peers if ".onion" in peer])

        # Size
        collectors["size"] = lambda: sum([site.settings.get("size", 0) for site in sites.values()])
        collectors["size_optional"] = lambda: sum([site.settings.get("size_optional", 0) for site in sites.values()])
        collectors["content"] = lambda: sum([len(site.content_manager.contents) for site in sites.values()])

        return collectors

    def getSiteCollectors(self):
        site_collectors = {}

        # Size
        site_collectors["site_size"] = lambda site: site.settings.get("size", 0)
        site_collectors["site_size_optional"] = lambda site: site.settings.get("size_optional", 0)
        site_collectors["site_optional_downloaded"] = lambda site: site.settings.get("optional_downloaded", 0)
        site_collectors["site_content"] = lambda site: len(site.content_manager.contents)

        # Data transfer
        site_collectors["site_bytes_recv|change"] = lambda site: site.settings.get("bytes_recv", 0)
        site_collectors["site_bytes_sent|change"] = lambda site: site.settings.get("bytes_sent", 0)

        # Peers
        site_collectors["site_peer"] = lambda site: len(site.peers)
        site_collectors["site_peer_onion"] = lambda site: len(
            [True for peer in site.peers.values() if peer.ip.endswith(".onion")]
        )
        site_collectors["site_peer_connected"] = lambda site: len([True for peer in site.peers.values() if peer.connection])

        return site_collectors

    def getUniquePeers(self):
        import main
        sites = main.file_server.sites
        return set(itertools.chain.from_iterable(
            [site.peers.keys() for site in sites.values()]
        ))

    def collectDatas(self, collectors, last_values, site=None):
        if site is None:
            peers = self.getUniquePeers()
        datas = {}
        for key, collector in collectors.items():
            try:
                if site:
                    value = collector(site)
                elif key.startswith("peer"):
                    value = collector(peers)
                else:
                    value = collector()
            except ValueError:
                value = None
            except Exception as err:
                self.log.info("Collector %s error: %s" % (key, err))
                value = None

            if "|change" in key:  # Store changes relative to last value
                key = key.replace("|change", "")
                last_value = last_values.get(key, 0)
                last_values[key] = value
                value = value - last_value

            if value is None:
                datas[key] = None
            else:
                datas[key] = round(value, 3)
        return datas

    def collectGlobal(self, collectors, last_values):
        now = int(time.time())
        s = time.time()
        datas = self.collectDatas(collectors, last_values["global"])
        values = []
        for key, value in datas.items():
            values.append((self.db.getTypeId(key), value, now))
        self.log.debug("Global collectors done in %.3fs" % (time.time() - s))

        s = time.time()
        cur = self.db.getCursor()
        cur.executemany("INSERT INTO data (type_id, value, date_added) VALUES (?, ?, ?)", values)
        self.log.debug("Global collectors inserted in %.3fs" % (time.time() - s))

    def collectSites(self, sites, collectors, last_values):
        now = int(time.time())
        s = time.time()
        values = []
        for address, site in list(sites.items()):
            site_datas = self.collectDatas(collectors, last_values["site:%s" % address], site)
            for key, value in site_datas.items():
                values.append((self.db.getTypeId(key), self.db.getSiteId(address), value, now))
            time.sleep(0.001)
        self.log.debug("Site collections done in %.3fs" % (time.time() - s))

        s = time.time()
        cur = self.db.getCursor()
        cur.executemany("INSERT INTO data (type_id, site_id, value, date_added) VALUES (?, ?, ?, ?)", values)
        self.log.debug("Site collectors inserted in %.3fs" % (time.time() - s))

    def collector(self):
        collectors = self.getCollectors()
        site_collectors = self.getSiteCollectors()
        import main
        sites = main.file_server.sites
        i = 0
        while 1:
            self.collectGlobal(collectors, self.last_values)
            if i % 12 == 0:  # Only collect sites data every hour
                self.collectSites(sites, site_collectors, self.last_values)
            time.sleep(60 * 5)
            i += 1

```

# `plugins/Chart/ChartDb.py`

This is a Python class that appears to be a database model for storing data about site statistics. It contains a query that is executed on a database using the `psycopg2` library, as well as code to interact with the database using the `psycopg2` library. The query appears to be for a table called `data` in a database, and it appears to be querying for data that has been archived for more than 6 months and has either a value or a `NULL` value for `site_id`. The code also includes code to filter the data by week, month, and year, as well as code to filter the data by site ID. The class also includes code to limit the number of global statistics to 6 months and to only keep 1 month of site statistics.


```py
from Config import config
from Db.Db import Db
import time


class ChartDb(Db):
    def __init__(self):
        self.version = 2
        super(ChartDb, self).__init__(self.getSchema(), "%s/chart.db" % config.data_dir)
        self.foreign_keys = True
        self.checkTables()
        self.sites = self.loadSites()
        self.types = self.loadTypes()

    def getSchema(self):
        schema = {}
        schema["db_name"] = "Chart"
        schema["tables"] = {}
        schema["tables"]["data"] = {
            "cols": [
                ["data_id", "INTEGER PRIMARY KEY ASC AUTOINCREMENT NOT NULL UNIQUE"],
                ["type_id", "INTEGER NOT NULL"],
                ["site_id", "INTEGER"],
                ["value", "INTEGER"],
                ["date_added", "DATETIME DEFAULT (CURRENT_TIMESTAMP)"]
            ],
            "indexes": [
                "CREATE INDEX site_id ON data (site_id)",
                "CREATE INDEX date_added ON data (date_added)"
            ],
            "schema_changed": 2
        }
        schema["tables"]["type"] = {
            "cols": [
                ["type_id", "INTEGER PRIMARY KEY NOT NULL UNIQUE"],
                ["name", "TEXT"]
            ],
            "schema_changed": 1
        }
        schema["tables"]["site"] = {
            "cols": [
                ["site_id", "INTEGER PRIMARY KEY NOT NULL UNIQUE"],
                ["address", "TEXT"]
            ],
            "schema_changed": 1
        }
        return schema

    def getTypeId(self, name):
        if name not in self.types:
            res = self.execute("INSERT INTO type ?", {"name": name})
            self.types[name] = res.lastrowid

        return self.types[name]

    def getSiteId(self, address):
        if address not in self.sites:
            res = self.execute("INSERT INTO site ?", {"address": address})
            self.sites[address] = res.lastrowid

        return self.sites[address]

    def loadSites(self):
        sites = {}
        for row in self.execute("SELECT * FROM site"):
            sites[row["address"]] = row["site_id"]
        return sites

    def loadTypes(self):
        types = {}
        for row in self.execute("SELECT * FROM type"):
            types[row["name"]] = row["type_id"]
        return types

    def deleteSite(self, address):
        if address in self.sites:
            site_id = self.sites[address]
            del self.sites[address]
            self.execute("DELETE FROM site WHERE ?", {"site_id": site_id})
            self.execute("DELETE FROM data WHERE ?", {"site_id": site_id})

    def archive(self):
        week_back = 1
        while 1:
            s = time.time()
            date_added_from = time.time() - 60 * 60 * 24 * 7 * (week_back + 1)
            date_added_to = date_added_from + 60 * 60 * 24 * 7
            res = self.execute("""
                SELECT
                 MAX(date_added) AS date_added,
                 SUM(value) AS value,
                 GROUP_CONCAT(data_id) AS data_ids,
                 type_id,
                 site_id,
                 COUNT(*) AS num
                FROM data
                WHERE
                 site_id IS NULL AND
                 date_added > :date_added_from AND
                 date_added < :date_added_to
                GROUP BY strftime('%Y-%m-%d %H', date_added, 'unixepoch', 'localtime'), type_id
            """, {"date_added_from": date_added_from, "date_added_to": date_added_to})

            num_archived = 0
            cur = self.getCursor()
            for row in res:
                if row["num"] == 1:
                    continue
                cur.execute("INSERT INTO data ?", {
                    "type_id": row["type_id"],
                    "site_id": row["site_id"],
                    "value": row["value"],
                    "date_added": row["date_added"]
                })
                cur.execute("DELETE FROM data WHERE data_id IN (%s)" % row["data_ids"])
                num_archived += row["num"]
            self.log.debug("Archived %s data from %s weeks ago in %.3fs" % (num_archived, week_back, time.time() - s))
            week_back += 1
            time.sleep(0.1)
            if num_archived == 0:
                break
        # Only keep 6 month of global stats
        self.execute(
            "DELETE FROM data WHERE site_id IS NULL AND date_added < :date_added_limit",
            {"date_added_limit": time.time() - 60 * 60 * 24 * 30 * 6 }
        )
        # Only keep 1 month of site stats
        self.execute(
            "DELETE FROM data WHERE site_id IS NOT NULL AND date_added < :date_added_limit",
            {"date_added_limit": time.time() - 60 * 60 * 24 * 30 }
        )
        if week_back > 1:
            self.execute("VACUUM")

```

# `plugins/Chart/ChartPlugin.py`

这段代码的作用是：

1. 导入time、itertools、gevent、Config、Helper、PluginManager、ChartDb、ChartCollector类。
2. 从Config类中读取一个配置文件，并执行其中的函数。
3. 使用PluginManager类加载一个插件，并将其管理器实例存储在Config中的db属性中。
4. 启动一个定时任务，这个任务每60秒执行一次ChartDb类中的archive函数，将当前时间序列的 ChartDb 对象保存到数据库中。
5. 在同一个定时任务中，使用ChartCollector类收集应用程序中所有 Chart对象的元数据，并将其存储到 ChartDb 对象中。
6. 在每次定时任务结束时，保存已收集的元数据到 ChartDb 中，并继续收集新的 Chart对象。
7. 如果之前没有连接到数据库，那么代码将在第一次加载应用程序时执行这些操作。


```py
import time
import itertools

import gevent

from Config import config
from util import helper
from util.Flag import flag
from Plugin import PluginManager
from .ChartDb import ChartDb
from .ChartCollector import ChartCollector

if "db" not in locals().keys():  # Share on reloads
    db = ChartDb()
    gevent.spawn_later(10 * 60, db.archive)
    helper.timer(60 * 60 * 6, db.archive)
    collector = ChartCollector(db)

```

这段代码定义了两个类，一个是SiteManagerPlugin，另一个是UiWebsocketPlugin。它们都属于一个名为PluginManager的插件框架。

PluginManager是一个用于注册管理器类插件的框架类。注册插件的时候需要提供插件的类名、管理器类名、需要调用的方法名和参数列表等信息，通过这些信息来创建插件的管理器类插件。

SiteManagerPlugin是一个实现了插件接口的类，它重写了父类SiteManagerPlugin的load和delete方法。在load方法中，它实现了将调用父类的load方法的结果保存到site对象中，并在delete方法中实现了删除指定网站的功能。

UiWebsocketPlugin是一个实现了插件接口的类，它重写了父类SiteManagerPlugin的actionChartDbQuery和actionChartGetPeerLocations方法。在actionChartDbQuery方法中，它实现了从数据库中查询数据，并将查询结果返回。在actionChartGetPeerLocations方法中，它实现了获取指定服务器上所有站点的位置，并将结果返回。


```py
@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    def load(self, *args, **kwargs):
        back = super(SiteManagerPlugin, self).load(*args, **kwargs)
        collector.setInitialLastValues(self.sites.values())
        return back

    def delete(self, address, *args, **kwargs):
        db.deleteSite(address)
        return super(SiteManagerPlugin, self).delete(address, *args, **kwargs)

@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    @flag.admin
    def actionChartDbQuery(self, to, query, params=None):
        if config.debug or config.verbose:
            s = time.time()
        rows = []
        try:
            if not query.strip().upper().startswith("SELECT"):
                raise Exception("Only SELECT query supported")
            res = db.execute(query, params)
        except Exception as err:  # Response the error to client
            self.log.error("ChartDbQuery error: %s" % err)
            return {"error": str(err)}
        # Convert result to dict
        for row in res:
            rows.append(dict(row))
        if config.verbose and time.time() - s > 0.1:  # Log slow query
            self.log.debug("Slow query: %s (%.3fs)" % (query, time.time() - s))
        return rows

    @flag.admin
    def actionChartGetPeerLocations(self, to):
        peers = {}
        for site in self.server.sites.values():
            peers.update(site.peers)
        peer_locations = self.getPeerLocations(peers)
        return peer_locations

```

# `plugins/Chart/__init__.py`

这段代码使用了 Python 的动态导入机制，用于导入名为 "ChartPlugin" 的模块。具体来说，它将在运行时读取 Python 脚本时，自动导入该模块中的所有内容，并在程序运行时随时可以使用这些模块中的函数和变量。

ChartPlugin 是一个用于可视化图表的插件，可以用来创建各种不同类型的图表，如股票图、柱状图、折线图等等。通过导入这个插件，用户就可以使用它提供的函数和类来创建自己的图表，更加方便和灵活。


```py
from . import ChartPlugin
```

# `plugins/ContentFilter/ContentFilterPlugin.py`

这段代码使用了多个 Python 模块，包括 `time`、`re`、`html`、`os`、`Plugin`、`Translate` 和 `Config`，以及自定义的 `util.Flag` 模块。它主要实现了以下功能：

1. 导入时针 `time` 和正则表达式 `re`，用于获取和解析时间戳和正则表达式匹配的内容。
2. 导入 HTML 模块，用于将 HTML 内容解析为文档对象模型（Document Object Model，DOM）。
3. 导入操作系统中的 `os` 模块，用于获取和操作文件系统。
4. 从 `Plugin` 模块中导入 `PluginManager`，实现对插件的管理。
5. 从 `Translate` 模块中导入 `Translate`，实现翻译功能。
6. 从 `Config` 模块中导入 `config`，用于设置插件的配置。
7. 从自定义的 `util.Flag` 模块中导入 `flag`，用于设置某些标头的设置。
8. 从 `ContentFilterStorage` 类中导入 `ContentFilterStorage`，实现内容过滤存储的功能。

具体来说，这段代码主要实现了将获取到的网页内容进行正则表达式匹配，然后解析出时间戳，接着将匹配到的内容提交给 `Plugin` 模块进行处理，如获取翻译服务，最后将结果输出到 `ContentFilterStorage` 中。


```py
import time
import re
import html
import os

from Plugin import PluginManager
from Translate import Translate
from Config import config
from util.Flag import flag

from .ContentFilterStorage import ContentFilterStorage


plugin_dir = os.path.dirname(__file__)

```

这段代码是一个 Python 类，名为 SiteManagerPlugin，属于 SiteManager 插件框架。它的作用是处理网站管理人员在添加或编辑某些网站时遇到的屏蔽问题。

首先，代码检查是否在本地环境中定义了_变量，如果没有，就在类中定义一个常量_，值为Translate(plugin_dir + "/languages/")，表示将会调用 plugin 目录下的 translate.py 文件进行翻译。

然后，定义了一个 SiteManagerPlugin 类，这个类继承自 SiteManager 插件框架的类，负责处理所有的添加和编辑操作。

在 load 方法中，除了传递给 SiteManager 类的参数外，还定义了一个 filter_storage 变量，用于记录当前被过滤的网站内容。在 add 方法中，首先会判断是否已经定义了 should_ignore_block 参数，如果是，就表示已经传入了这个参数，会进一步判断是否需要忽略这个网站的访问。如果是这个方法第一次被调用，将会从 site_manager 对象中获取这个网站的管理员，并将这个管理员实例中的 filter_storage 变量赋值给 this。

接下来，在 add 方法中，会首先判断是否应该忽略这个网站。如果 should_ignore_block 参数被传入了，表示这个网站已经被屏蔽了，会尝试从 site_manager 对象中的 filter_storage 变量中获取这个网站的地址对应的 block_details 信息，并检查是否已经被屏蔽。如果这个网站没有被屏蔽，会使用 this 对象中的 filter_storage 变量中存储的地址对应的 block_details 信息，如果没有这个信息，就会输出一个错误信息并返回。如果这个网站已经被屏蔽了，就会输出一个包含网站地址和详细原因信息的错误消息并返回。


```py
if "_" not in locals():
    _ = Translate(plugin_dir + "/languages/")


@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    def load(self, *args, **kwargs):
        global filter_storage
        super(SiteManagerPlugin, self).load(*args, **kwargs)
        filter_storage = ContentFilterStorage(site_manager=self)

    def add(self, address, *args, **kwargs):
        should_ignore_block = kwargs.get("ignore_block") or kwargs.get("settings")
        if should_ignore_block:
            block_details = None
        elif filter_storage.isSiteblocked(address):
            block_details = filter_storage.getSiteblockDetails(address)
        else:
            address_hashed = filter_storage.getSiteAddressHashed(address)
            if filter_storage.isSiteblocked(address_hashed):
                block_details = filter_storage.getSiteblockDetails(address_hashed)
            else:
                block_details = None

        if block_details:
            raise Exception("Site blocked: %s" % html.escape(block_details.get("reason", "unknown reason")))
        else:
            return super(SiteManagerPlugin, self).add(address, *args, **kwargs)


```

This is a Flask-RESTPlus鉴权应用程序的实现。它实现了对filter_storage数据库的filterIncludeAdd、filterIncludeRemove和filterIncludeList方法。filterIncludeAdd用于将一个包含地址的列表添加到filter_storage数据库中，而filterIncludeRemove用于从filter_storage数据库中删除指定的地址的包含。filterIncludeList用于列出指定网站的所有包含文件。

此实现中，如果all_sites标志为真，那么该应用程序可以访问所有的网站，即使没有ADMIN权限。在这种情况下，如果address属性在一个未包含admin列表的客户端中，该应用程序将返回"Forbidden: Only ADMIN sites can manage different site includes"错误消息。

此实现中，如果应用程序拥有ADMIN权限，那么filterIncludeList方法将返回一个包含所有网站包含的列表。在这种情况下，如果address属性在一个已包含admin列表的客户端中，该应用程序将不会返回包含网站的列表。相反，它将返回一个空列表，类似于在filter_storage数据库中没有包含任何网站的包含。


```py
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # Mute
    def cbMuteAdd(self, to, auth_address, cert_user_id, reason):
        filter_storage.file_content["mutes"][auth_address] = {
            "cert_user_id": cert_user_id, "reason": reason, "source": self.site.address, "date_added": time.time()
        }
        filter_storage.save()
        filter_storage.changeDbs(auth_address, "remove")
        self.response(to, "ok")

    @flag.no_multiuser
    def actionMuteAdd(self, to, auth_address, cert_user_id, reason):
        if "ADMIN" in self.getPermissions(to):
            self.cbMuteAdd(to, auth_address, cert_user_id, reason)
        else:
            self.cmd(
                "confirm",
                [_["Hide all content from <b>%s</b>?"] % html.escape(cert_user_id), _["Mute"]],
                lambda res: self.cbMuteAdd(to, auth_address, cert_user_id, reason)
            )

    @flag.no_multiuser
    def cbMuteRemove(self, to, auth_address):
        del filter_storage.file_content["mutes"][auth_address]
        filter_storage.save()
        filter_storage.changeDbs(auth_address, "load")
        self.response(to, "ok")

    @flag.no_multiuser
    def actionMuteRemove(self, to, auth_address):
        if "ADMIN" in self.getPermissions(to):
            self.cbMuteRemove(to, auth_address)
        else:
            cert_user_id = html.escape(filter_storage.file_content["mutes"][auth_address]["cert_user_id"])
            self.cmd(
                "confirm",
                [_["Unmute <b>%s</b>?"] % cert_user_id, _["Unmute"]],
                lambda res: self.cbMuteRemove(to, auth_address)
            )

    @flag.admin
    def actionMuteList(self, to):
        self.response(to, filter_storage.file_content["mutes"])

    # Siteblock
    @flag.no_multiuser
    @flag.admin
    def actionSiteblockIgnoreAddSite(self, to, site_address):
        if site_address in filter_storage.site_manager.sites:
            return {"error": "Site already added"}
        else:
            if filter_storage.site_manager.need(site_address, ignore_block=True):
                return "ok"
            else:
                return {"error": "Invalid address"}

    @flag.no_multiuser
    @flag.admin
    def actionSiteblockAdd(self, to, site_address, reason=None):
        filter_storage.file_content["siteblocks"][site_address] = {"date_added": time.time(), "reason": reason}
        filter_storage.save()
        self.response(to, "ok")

    @flag.no_multiuser
    @flag.admin
    def actionSiteblockRemove(self, to, site_address):
        del filter_storage.file_content["siteblocks"][site_address]
        filter_storage.save()
        self.response(to, "ok")

    @flag.admin
    def actionSiteblockList(self, to):
        self.response(to, filter_storage.file_content["siteblocks"])

    @flag.admin
    def actionSiteblockGet(self, to, site_address):
        if filter_storage.isSiteblocked(site_address):
            res = filter_storage.getSiteblockDetails(site_address)
        else:
            site_address_hashed = filter_storage.getSiteAddressHashed(site_address)
            if filter_storage.isSiteblocked(site_address_hashed):
                res = filter_storage.getSiteblockDetails(site_address_hashed)
            else:
                res = {"error": "Site block not found"}
        self.response(to, res)

    # Include
    @flag.no_multiuser
    def actionFilterIncludeAdd(self, to, inner_path, description=None, address=None):
        if address:
            if "ADMIN" not in self.getPermissions(to):
                return self.response(to, {"error": "Forbidden: Only ADMIN sites can manage different site include"})
            site = self.server.sites[address]
        else:
            address = self.site.address
            site = self.site

        if "ADMIN" in self.getPermissions(to):
            self.cbFilterIncludeAdd(to, True, address, inner_path, description)
        else:
            content = site.storage.loadJson(inner_path)
            title = _["New shared global content filter: <b>%s</b> (%s sites, %s users)"] % (
                html.escape(inner_path), len(content.get("siteblocks", {})), len(content.get("mutes", {}))
            )

            self.cmd(
                "confirm",
                [title, "Add"],
                lambda res: self.cbFilterIncludeAdd(to, res, address, inner_path, description)
            )

    def cbFilterIncludeAdd(self, to, res, address, inner_path, description):
        if not res:
            self.response(to, res)
            return False

        filter_storage.includeAdd(address, inner_path, description)
        self.response(to, "ok")

    @flag.no_multiuser
    def actionFilterIncludeRemove(self, to, inner_path, address=None):
        if address:
            if "ADMIN" not in self.getPermissions(to):
                return self.response(to, {"error": "Forbidden: Only ADMIN sites can manage different site include"})
        else:
            address = self.site.address

        key = "%s/%s" % (address, inner_path)
        if key not in filter_storage.file_content["includes"]:
            self.response(to, {"error": "Include not found"})
        filter_storage.includeRemove(address, inner_path)
        self.response(to, "ok")

    def actionFilterIncludeList(self, to, all_sites=False, filters=False):
        if all_sites and "ADMIN" not in self.getPermissions(to):
            return self.response(to, {"error": "Forbidden: Only ADMIN sites can list all sites includes"})

        back = []
        includes = filter_storage.file_content.get("includes", {}).values()
        for include in includes:
            if not all_sites and include["address"] != self.site.address:
                continue
            if filters:
                include = dict(include)  # Don't modify original file_content
                include_site = filter_storage.site_manager.get(include["address"])
                if not include_site:
                    continue
                content = include_site.storage.loadJson(include["inner_path"])
                include["mutes"] = content.get("mutes", {})
                include["siteblocks"] = content.get("siteblocks", {})
            back.append(include)
        self.response(to, back)


```

这段代码定义了一个名为 SiteStoragePlugin 的类，用于在 SiteStorage 插件中存储和管理数据库文件。

PluginManager.registerTo("SiteStorage") 是一个注册命令，用于将 SiteStorage 插件注册到 SiteStorage 插件集中。

SiteStoragePlugin 类包含两个方法，分别如下：

updateDbFile(inner_path, file=None, cur=None):

这个方法用于更新数据库文件。如果传入的文件路径是空字符串，那么它不会对此做出任何操作。否则，它将读取文件内容并检查其中是否包含比特币地址。如果是比特币地址，那么将记录在日志中并返回 False。否则，它将调用 super(SiteStoragePlugin, self).updateDbFile(inner_path, file=file, cur=cur) 来继续执行 updateDbFile 方法。

onUpdated(inner_path, file=None):

这个方法在插件更新时被调用。如果传入的文件路径是空字符串，那么它不会对此做出任何操作。否则，它将读取文件内容并检查其中是否包含过滤器列表中的地址。如果是地址，那么将记录在日志中并从过滤器列表中移除该地址。否则，它将调用 super(SiteStoragePlugin, self).onUpdated(inner_path, file=file) 来继续执行 onUpdated 方法。

注意，在 onUpdated 方法中，由于 site.address 是通过环境变量来获取的，所以在生产环境中需要添加 SiteStoragePlugin.site.address 变量来访问站点地址。


```py
@PluginManager.registerTo("SiteStorage")
class SiteStoragePlugin(object):
    def updateDbFile(self, inner_path, file=None, cur=None):
        if file is not False:  # File deletion always allowed
            # Find for bitcoin addresses in file path
            matches = re.findall("/(1[A-Za-z0-9]{26,35})/", inner_path)
            # Check if any of the adresses are in the mute list
            for auth_address in matches:
                if filter_storage.isMuted(auth_address):
                    self.log.debug("Mute match: %s, ignoring %s" % (auth_address, inner_path))
                    return False

        return super(SiteStoragePlugin, self).updateDbFile(inner_path, file=file, cur=cur)

    def onUpdated(self, inner_path, file=None):
        file_path = "%s/%s" % (self.site.address, inner_path)
        if file_path in filter_storage.file_content["includes"]:
            self.log.debug("Filter file updated: %s" % inner_path)
            filter_storage.includeUpdateAll()
        return super(SiteStoragePlugin, self).onUpdated(inner_path, file=file)


```

This is a Python class that implements the `UiRequestPlugin` interface for a plugin to handle UiControl's request to display a file or directory.

The `UiRequestPlugin` class includes the following methods:

* `actionWrapper(path, extra_headers=None)`: This method wraps the file or directory selection in the server and returns the result of calling the `actionWrapper` method of the `UiRequestPlugin` object. It reads the file or directory path from the `path` parameter, adds any specified extra headers to the request, and returns the result of calling the `actionWrapper` method of the `UiRequestPlugin` object.
* `actionUiMedia(path, *args, **kwargs)`: This method handles the selection of a file or directory using the `actionUiMedia` method of the `UiRequestPlugin` object. It adds the specified arguments to the `path` parameter, constructs the full file or directory path based on the specified directory path and the file or directory name, and returns the result of calling the `actionUiMedia` method of the `UiRequestPlugin` object.
* `getScriptNonce()`: This method returns a nonce (a one-time use identifier) for the script to use for cross-site request protection. This nonce will be included in the request headers when the file or directory is selected.
* `filter_storage.isSiteblocked(address)`: This method returns a boolean indicating whether the site at the specified address is currently blocked.
* `filter_storage.isSiteblocked(address_hashed)`: This method returns a boolean indicating whether the site at the specified address (with hashed storage) is currently blocked.
* `self.server.site_manager.get(address)`: This method returns a reference to the site object for the site at the specified address.
* `self.server.site_manager.get(config.homepage)`: This method returns the homepage of the server.
* `self.sendHeader(extra_headers, script_nonce)`: This method sends the specified extra headers and the script nonce to the server.
* `renderWrapper(site, path, description, extra_headers=None, show_loadingscreen=False, script_nonce=None)`: This method renders the specified description and description for the selected file or directory and returns the media (in the format of a `media` object) to be included in the request.

Note: This class assumes that the server has the `filter_storage` and `script_nonce` classes defined, as well as the `UiRequestPlugin` interface defined. These classes and the interface may be modified to meet the specific needs of the server and application.


```py
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    def actionWrapper(self, path, extra_headers=None):
        match = re.match(r"/(?P<address>[A-Za-z0-9\._-]+)(?P<inner_path>/.*|$)", path)
        if not match:
            return False
        address = match.group("address")

        if self.server.site_manager.get(address):  # Site already exists
            return super(UiRequestPlugin, self).actionWrapper(path, extra_headers)

        if self.isDomain(address):
            address = self.resolveDomain(address)

        if address:
            address_hashed = filter_storage.getSiteAddressHashed(address)
        else:
            address_hashed = None

        if filter_storage.isSiteblocked(address) or filter_storage.isSiteblocked(address_hashed):
            site = self.server.site_manager.get(config.homepage)
            if not extra_headers:
                extra_headers = {}

            script_nonce = self.getScriptNonce()

            self.sendHeader(extra_headers=extra_headers, script_nonce=script_nonce)
            return iter([super(UiRequestPlugin, self).renderWrapper(
                site, path, "uimedia/plugins/contentfilter/blocklisted.html?address=" + address,
                "Blacklisted site", extra_headers, show_loadingscreen=False, script_nonce=script_nonce
            )])
        else:
            return super(UiRequestPlugin, self).actionWrapper(path, extra_headers)

    def actionUiMedia(self, path, *args, **kwargs):
        if path.startswith("/uimedia/plugins/contentfilter/"):
            file_path = path.replace("/uimedia/plugins/contentfilter/", plugin_dir + "/media/")
            return self.actionFile(file_path)
        else:
            return super(UiRequestPlugin, self).actionUiMedia(path)

```

# `plugins/ContentFilter/ContentFilterStorage.py`

This is a Python class that appears to be a file storage management system for a website. It has methods for site block management, site content management, and file operations.

The `MuteAction` class has a method for mute an action on a user, which is not related to file management.

The `changeDbs` class has methods for changing the database status and file operations. It has a `changeDbs` method that takes an auth_address and an action, and returns information about the action. It has a `listSites` method that returns a list of sites. It has a `getSite` method that retrieves the site content for a given site. It has a `getSiteBlock` method that retrieves the site block content for a given site. It has a `executeSql` method that executes an SQL query on the site content database. It has a `writeFile` method that writes the file content to the site content database. It has a `deleteFile` method that deletes the file from the site content database. It has a `renameFile` method that renames the file.

The `getAuthAddress` method has a method for getting the auth address of a user. The `getSiteAddressHashed` method has a method for hashing a site address for security purposes.


```py
import os
import json
import logging
import collections
import time
import hashlib

from Debug import Debug
from Plugin import PluginManager
from Config import config
from util import helper


class ContentFilterStorage(object):
    def __init__(self, site_manager):
        self.log = logging.getLogger("ContentFilterStorage")
        self.file_path = "%s/filters.json" % config.data_dir
        self.site_manager = site_manager
        self.file_content = self.load()

        # Set default values for filters.json
        if not self.file_content:
            self.file_content = {}

        # Site blacklist renamed to site blocks
        if "site_blacklist" in self.file_content:
            self.file_content["siteblocks"] = self.file_content["site_blacklist"]
            del self.file_content["site_blacklist"]

        for key in ["mutes", "siteblocks", "includes"]:
            if key not in self.file_content:
                self.file_content[key] = {}

        self.include_filters = collections.defaultdict(set)  # Merged list of mutes and blacklists from all include
        self.includeUpdateAll(update_site_dbs=False)

    def load(self):
        # Rename previously used mutes.json -> filters.json
        if os.path.isfile("%s/mutes.json" % config.data_dir):
            self.log.info("Renaming mutes.json to filters.json...")
            os.rename("%s/mutes.json" % config.data_dir, self.file_path)
        if os.path.isfile(self.file_path):
            try:
                return json.load(open(self.file_path))
            except Exception as err:
                self.log.error("Error loading filters.json: %s" % err)
                return None
        else:
            return None

    def includeUpdateAll(self, update_site_dbs=True):
        s = time.time()
        new_include_filters = collections.defaultdict(set)

        # Load all include files data into a merged set
        for include_path in self.file_content["includes"]:
            address, inner_path = include_path.split("/", 1)
            try:
                content = self.site_manager.get(address).storage.loadJson(inner_path)
            except Exception as err:
                self.log.warning(
                    "Error loading include %s: %s" %
                    (include_path, Debug.formatException(err))
                )
                continue

            for key, val in content.items():
                if type(val) is not dict:
                    continue

                new_include_filters[key].update(val.keys())

        mutes_added = new_include_filters["mutes"].difference(self.include_filters["mutes"])
        mutes_removed = self.include_filters["mutes"].difference(new_include_filters["mutes"])

        self.include_filters = new_include_filters

        if update_site_dbs:
            for auth_address in mutes_added:
                self.changeDbs(auth_address, "remove")

            for auth_address in mutes_removed:
                if not self.isMuted(auth_address):
                    self.changeDbs(auth_address, "load")

        num_mutes = len(self.include_filters["mutes"])
        num_siteblocks = len(self.include_filters["siteblocks"])
        self.log.debug(
            "Loaded %s mutes, %s blocked sites from %s includes in %.3fs" %
            (num_mutes, num_siteblocks, len(self.file_content["includes"]), time.time() - s)
        )

    def includeAdd(self, address, inner_path, description=None):
        self.file_content["includes"]["%s/%s" % (address, inner_path)] = {
            "date_added": time.time(),
            "address": address,
            "description": description,
            "inner_path": inner_path
        }
        self.includeUpdateAll()
        self.save()

    def includeRemove(self, address, inner_path):
        del self.file_content["includes"]["%s/%s" % (address, inner_path)]
        self.includeUpdateAll()
        self.save()

    def save(self):
        s = time.time()
        helper.atomicWrite(self.file_path, json.dumps(self.file_content, indent=2, sort_keys=True).encode("utf8"))
        self.log.debug("Saved in %.3fs" % (time.time() - s))

    def isMuted(self, auth_address):
        if auth_address in self.file_content["mutes"] or auth_address in self.include_filters["mutes"]:
            return True
        else:
            return False

    def getSiteAddressHashed(self, address):
        return "0x" + hashlib.sha256(address.encode("ascii")).hexdigest()

    def isSiteblocked(self, address):
        if address in self.file_content["siteblocks"] or address in self.include_filters["siteblocks"]:
            return True
        return False

    def getSiteblockDetails(self, address):
        details = self.file_content["siteblocks"].get(address)
        if not details:
            address_sha256 = self.getSiteAddressHashed(address)
            details = self.file_content["siteblocks"].get(address_sha256)

        if not details:
            includes = self.file_content.get("includes", {}).values()
            for include in includes:
                include_site = self.site_manager.get(include["address"])
                if not include_site:
                    continue
                content = include_site.storage.loadJson(include["inner_path"])
                details = content.get("siteblocks", {}).get(address)
                if details:
                    details["include"] = include
                    break

        return details

    # Search and remove or readd files of an user
    def changeDbs(self, auth_address, action):
        self.log.debug("Mute action %s on user %s" % (action, auth_address))
        res = list(self.site_manager.list().values())[0].content_manager.contents.db.execute(
            "SELECT * FROM content LEFT JOIN site USING (site_id) WHERE inner_path LIKE :inner_path",
            {"inner_path": "%%/%s/%%" % auth_address}
        )
        for row in res:
            site = self.site_manager.sites.get(row["address"])
            if not site:
                continue
            dir_inner_path = helper.getDirname(row["inner_path"])
            for file_name in site.storage.walk(dir_inner_path):
                if action == "remove":
                    site.storage.onUpdated(dir_inner_path + file_name, False)
                else:
                    site.storage.onUpdated(dir_inner_path + file_name)
                site.onFileDone(dir_inner_path + file_name)

```