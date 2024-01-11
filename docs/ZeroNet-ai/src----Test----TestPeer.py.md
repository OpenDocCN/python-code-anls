# `ZeroNet\src\Test\TestPeer.py`

```
# 导入时间模块
import time
# 导入io模块
import io
# 导入pytest模块
import pytest
# 从File模块中导入FileServer和FileRequest类
from File import FileServer
from File import FileRequest
# 从Crypt模块中导入CryptHash类
from Crypt import CryptHash
# 从当前目录下的Spy模块中导入所有内容
from . import Spy

# 使用resetSettings装饰器重置设置
@pytest.mark.usefixtures("resetSettings")
# 使用resetTempSettings装饰器重置临时设置
@pytest.mark.usefixtures("resetTempSettings")
# 定义TestPeer类
class TestPeer:
    # 定义testPing方法，接受file_server, site, site_temp三个参数
    def testPing(self, file_server, site, site_temp):
        # 将site添加到file_server的sites字典中
        file_server.sites[site.address] = site
        # 创建FileServer对象client，连接到file_server的IP和端口1545
        client = FileServer(file_server.ip, 1545)
        # 将site_temp添加到client的sites字典中
        client.sites = {site_temp.address: site_temp}
        # 将client设置为site_temp的connection_server
        site_temp.connection_server = client
        # 通过client连接到file_server的IP和端口1544，返回连接对象connection
        connection = client.getConnection(file_server.ip, 1544)

        # 将file_server添加为client的peer
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)

        # 断言peer_file_server的ping方法返回值不为None
        assert peer_file_server.ping() is not None

        # 断言peer_file_server在site_temp的peers字典中
        assert peer_file_server in site_temp.peers.values()
        # 移除peer_file_server
        peer_file_server.remove()
        # 断言peer_file_server不在site_temp的peers字典中
        assert peer_file_server not in site_temp.peers.values()

        # 关闭连接
        connection.close()
        # 停止client
        client.stop()

    # 定义testDownloadFile方法，接受file_server, site, site_temp三个参数
    def testDownloadFile(self, file_server, site, site_temp):
        # 将site添加到file_server的sites字典中
        file_server.sites[site.address] = site
        # 创建FileServer对象client，连接到file_server的IP和端口1545
        client = FileServer(file_server.ip, 1545)
        # 将site_temp添加到client的sites字典中
        client.sites = {site_temp.address: site_temp}
        # 将client设置为site_temp的connection_server
        site_temp.connection_server = client
        # 通过client连接到file_server的IP和端口1544，返回连接对象connection
        connection = client.getConnection(file_server.ip, 1544)

        # 将file_server添加为client的peer
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)

        # 测试streamFile方法
        buff = peer_file_server.getFile(site_temp.address, "content.json", streaming=True)
        # 断言buff中包含b"sign"
        assert b"sign" in buff.getvalue()

        # 测试getFile方法
        buff = peer_file_server.getFile(site_temp.address, "content.json")
        # 断言buff中包含b"sign"
        assert b"sign" in buff.getvalue()

        # 关闭连接
        connection.close()
        # 停止client
        client.stop()
    # 测试 hashfield 方法，传入 site 对象
    def testHashfield(self, site):
        # 从 site 对象的内容管理器中获取 content.json 文件的第一个可选文件的 sha512 值
        sample_hash = list(site.content_manager.contents["content.json"]["files_optional"].values())[0]["sha512"]

        # 调用存储对象的 verifyFiles 方法，进行快速检查可选文件
        site.storage.verifyFiles(quick_check=True)  # Find what optional files we have

        # 检查 hashfield 是否存在
        assert site.content_manager.hashfield
        # 检查 hashfield 中的文件数量是否大于 0
        assert len(site.content_manager.hashfield) > 0

        # 检查是否存在指定的 hash
        assert site.content_manager.hashfield.getHashId(sample_hash) in site.content_manager.hashfield

        # 添加新的 hash
        new_hash = CryptHash.sha512sum(io.BytesIO(b"hello"))
        # 检查 hashfield 中是否不存在新的 hash
        assert site.content_manager.hashfield.getHashId(new_hash) not in site.content_manager.hashfield
        # 将新的 hash 添加到 hashfield 中
        assert site.content_manager.hashfield.appendHash(new_hash)
        # 再次尝试添加相同的 hash，预期不会成功
        assert not site.content_manager.hashfield.appendHash(new_hash)  # Don't add second time
        # 检查 hashfield 中是否存在新的 hash
        assert site.content_manager.hashfield.getHashId(new_hash) in site.content_manager.hashfield

        # 移除新的 hash
        assert site.content_manager.hashfield.removeHash(new_hash)
        # 检查 hashfield 中是否不存在新的 hash
        assert site.content_manager.hashfield.getHashId(new_hash) not in site.content_manager.hashfield
    # 测试查找哈希值的方法
    def testFindHash(self, file_server, site, site_temp):
        # 将站点对象添加到文件服务器的站点字典中
        file_server.sites[site.address] = site
        # 创建一个文件服务器客户端对象
        client = FileServer(file_server.ip, 1545)
        # 将临时站点对象添加到客户端的站点字典中
        client.sites = {site_temp.address: site_temp}
        # 将客户端对象设置为临时站点对象的连接服务器
        site_temp.connection_server = client

        # 将文件服务器添加为客户端的对等节点
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)

        # 断言对等文件服务器查找哈希值为1234时返回空字典
        assert peer_file_server.findHashIds([1234]) == {}

        # 添加具有所需哈希值的虚假对等节点
        fake_peer_1 = site.addPeer(file_server.ip_external, 1544)
        fake_peer_1.hashfield.append(1234)
        fake_peer_2 = site.addPeer("1.2.3.5", 1545)
        fake_peer_2.hashfield.append(1234)
        fake_peer_2.hashfield.append(1235)
        fake_peer_3 = site.addPeer("1.2.3.6", 1546)
        fake_peer_3.hashfield.append(1235)
        fake_peer_3.hashfield.append(1236)

        # 断言对等文件服务器查找哈希值为1234和1235时返回的结果符合预期
        res = peer_file_server.findHashIds([1234, 1235])
        assert sorted(res[1234]) == sorted([(file_server.ip_external, 1544), ("1.2.3.5", 1545)])
        assert sorted(res[1235]) == sorted([("1.2.3.5", 1545), ("1.2.3.6", 1546)])

        # 测试添加我的地址
        site.content_manager.hashfield.append(1234)

        # 断言对等文件服务器查找哈希值为1234和1235时返回的结果符合预期
        res = peer_file_server.findHashIds([1234, 1235])
        assert sorted(res[1234]) == sorted([(file_server.ip_external, 1544), ("1.2.3.5", 1545), (file_server.ip, 1544)])
        assert sorted(res[1235]) == sorted([("1.2.3.5", 1545), ("1.2.3.6", 1546)])
```