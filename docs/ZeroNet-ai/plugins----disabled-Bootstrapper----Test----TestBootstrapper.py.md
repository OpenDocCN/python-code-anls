# `ZeroNet\plugins\disabled-Bootstrapper\Test\TestBootstrapper.py`

```
# 导入 hashlib 模块
import hashlib
# 导入 os 模块
import os
# 导入 pytest 模块
import pytest
# 从 Bootstrapper 模块中导入 BootstrapperPlugin 类
from Bootstrapper import BootstrapperPlugin
# 从 Bootstrapper.BootstrapperDb 模块中导入 BootstrapperDb 类
from Bootstrapper.BootstrapperDb import BootstrapperDb
# 从 Peer 模块中导入 Peer 类
from Peer import Peer
# 从 Crypt 模块中导入 CryptRsa 类
from Crypt import CryptRsa
# 从 util 模块中导入 helper 函数
from util import helper

# 使用 pytest 的 fixture 装饰器定义 bootstrapper_db 函数
@pytest.fixture()
def bootstrapper_db(request):
    # 关闭 BootstrapperPlugin 的数据库连接
    BootstrapperPlugin.db.close()
    # 创建一个新的 BootstrapperDb 对象，并赋值给 BootstrapperPlugin 的 db 属性
    BootstrapperPlugin.db = BootstrapperDb()
    # 创建 BootstrapperDb 的表
    BootstrapperPlugin.db.createTables()  # Reset db
    # 设置 BootstrapperDb 的日志记录为 True
    BootstrapperPlugin.db.cur.logging = True

    # 定义 cleanup 函数，用于清理数据库和文件
    def cleanup():
        # 关闭 BootstrapperPlugin 的数据库连接
        BootstrapperPlugin.db.close()
        # 删除 BootstrapperPlugin 的数据库路径对应的文件
        os.unlink(BootstrapperPlugin.db.db_path)

    # 将 cleanup 函数添加到 request 的最终化器中
    request.addfinalizer(cleanup)
    # 返回 BootstrapperPlugin 的 db 属性
    return BootstrapperPlugin.db

# 使用 pytest 的 mark.usefixtures 装饰器，指定 resetSettings 作为测试用例的 fixture
@pytest.mark.usefixtures("resetSettings")
# 定义 TestBootstrapper 类
class TestBootstrapper:
    # 定义 testHashCache 方法，接受 file_server 和 bootstrapper_db 两个参数
    def testHashCache(self, file_server, bootstrapper_db):
        # 获取 file_server 的 IP 类型
        ip_type = helper.getIpType(file_server.ip)
        # 创建一个 Peer 对象
        peer = Peer(file_server.ip, 1544, connection_server=file_server)
        # 计算字符串 "site1" 的 SHA256 哈希值
        hash1 = hashlib.sha256(b"site1").digest()
        # 计算字符串 "site2" 的 SHA256 哈希值
        hash2 = hashlib.sha256(b"site2").digest()
        # 计算字符串 "site3" 的 SHA256 哈希值
        hash3 = hashlib.sha256(b"site3").digest()

        # 发送 "announce" 请求到 peer，并获取返回结果
        res = peer.request("announce", {
            "hashes": [hash1, hash2],
            "port": 15441, "need_types": [ip_type], "need_num": 10, "add": [ip_type]
        })

        # 断言返回结果中第一个 peer 的指定 IP 类型的长度为 0
        assert len(res["peers"][0][ip_type]) == 0  # Empty result

        # 复制 bootstrapper_db 的 hash_ids 到 hash_ids_before
        hash_ids_before = bootstrapper_db.hash_ids.copy()

        # 更新 bootstrapper_db 的哈希缓存
        bootstrapper_db.updateHashCache()

        # 断言更新前后的 hash_ids 相等
        assert hash_ids_before == bootstrapper_db.hash_ids

    # 定义 testPassive 方法，接受 file_server 和 bootstrapper_db 两个参数
    def testPassive(self, file_server, bootstrapper_db):
        # 创建一个 Peer 对象
        peer = Peer(file_server.ip, 1544, connection_server=file_server)
        # 获取 file_server 的 IP 类型
        ip_type = helper.getIpType(file_server.ip)
        # 计算字符串 "hash1" 的 SHA256 哈希值
        hash1 = hashlib.sha256(b"hash1").digest()

        # 在 bootstrapper_db 中进行 peerAnnounce 操作
        bootstrapper_db.peerAnnounce(ip_type, address=None, port=15441, hashes=[hash1])
        # 发送 "announce" 请求到 peer，并获取返回结果
        res = peer.request("announce", {
            "hashes": [hash1], "port": 15441, "need_types": [ip_type], "need_num": 10, "add": []
        })

        # 断言返回结果中第一个 peer 的指定 IPv4 类型的长度为 0
        assert len(res["peers"][0]["ipv4"]) == 0  # Empty result
    # 测试请求对等节点
    def testRequestPeers(self, file_server, site, bootstrapper_db, tor_manager):
        # 将站点的连接服务器设置为文件服务器
        site.connection_server = file_server
        # 将文件服务器的 TOR 管理器设置为 TOR 管理器
        file_server.tor_manager = tor_manager
        # 使用站点地址生成 SHA256 哈希值
        hash = hashlib.sha256(site.address.encode()).digest()

        # 从跟踪器请求对等节点
        assert len(site.peers) == 0
        # 向引导数据库通告对等节点的 IP 类型、地址、端口和哈希值
        bootstrapper_db.peerAnnounce(ip_type="ipv4", address="1.2.3.4", port=1234, hashes=[hash])
        # 通告跟踪器站点的地址和端口
        site.announcer.announceTracker("zero://%s:%s" % (file_server.ip, file_server.port))
        assert len(site.peers) == 1

        # 测试洋葱地址存储
        # 向引导数据库通告洋葱地址的对等节点的 IP 类型、地址、端口、哈希值和洋葱签名
        bootstrapper_db.peerAnnounce(ip_type="onion", address="bka4ht2bzxchy44r", port=1234, hashes=[hash], onion_signed=True)
        # 通告跟踪器站点的地址和端口
        site.announcer.announceTracker("zero://%s:%s" % (file_server.ip, file_server.port))
        assert len(site.peers) == 2
        assert "bka4ht2bzxchy44r.onion:1234" in site.peers

    # 标记为慢速测试
    @pytest.mark.slow
    def testAnnounce(self, file_server, tor_manager):
        # 将文件服务器的 TOR 管理器设置为 TOR 管理器
        file_server.tor_manager = tor_manager
        # 使用字符串生成 SHA256 哈希值
        hash1 = hashlib.sha256(b"1Nekos4fiBqfcazyG1bAxdBT5oBvA76Z").digest()
        hash2 = hashlib.sha256(b"1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr").digest()
        # 创建对等节点对象，连接服务器设置为文件服务器
        peer = Peer("zero.booth.moe", 443, connection_server=file_server)
        # 发送 ping 请求给对等节点
        assert peer.request("ping")
        # 创建对等节点对象，连接服务器设置为文件服务器
        peer = Peer("boot3rdez4rzn36x.onion", 15441, connection_server=file_server)
        # 发送 ping 请求给对等节点
        assert peer.request("ping")
        # 发送 announce 请求给对等节点，包括哈希值、端口、需要的类型、需要的数量和附加信息
        res = peer.request("announce", {
            "hashes": [hash1, hash2],
            "port": 15441, "need_types": ["ip4", "onion"], "need_num": 100, "add": [""]
        })

        assert res
    # 测试向后兼容性，传入文件服务器和引导程序数据库
    def testBackwardCompatibility(self, file_server, bootstrapper_db):
        # 创建一个对等体对象，传入文件服务器的 IP 地址和端口号
        peer = Peer(file_server.ip, 1544, connection_server=file_server)
        # 计算字符串 "site1" 的 SHA256 哈希值
        hash1 = hashlib.sha256(b"site1").digest()

        # 向引导程序数据库通告 peer 的 ipv4 地址、外部 IP 地址、端口号、哈希值和删除缺失哈希值的标志
        bootstrapper_db.peerAnnounce("ipv4", file_server.ip_external, port=15441, hashes=[hash1], delete_missing_hashes=True)

        # 使用 ipv4 类型进行测试
        res = peer.request("announce", {
            "hashes": [hash1],
            "port": 15441, "need_types": ["ipv4"], "need_num": 10, "add": []
        })

        # 断言返回的 peers 中的 ipv4 地址数量为 1
        assert len(res["peers"][0]["ipv4"]) == 1

        # 使用 ip4 类型进行测试
        res = peer.request("announce", {
            "hashes": [hash1],
            "port": 15441, "need_types": ["ip4"], "need_num": 10, "add": []
        })

        # 断言返回的 peers 中的 ip4 地址数量为 1
        assert len(res["peers"][0]["ip4"]) == 1
```