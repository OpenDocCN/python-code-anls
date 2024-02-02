# `ZeroNet\src\Test\TestTor.py`

```py
# 导入时间模块
import time

# 导入 pytest 模块
import pytest

# 导入 mock 模块
import mock

# 从 File 模块中导入 FileServer 类
from File import FileServer

# 从 Crypt 模块中导入 CryptRsa 类
from Crypt import CryptRsa

# 从 Config 模块中导入 config 变量
from Config import config

# 使用 resetSettings fixture 重置设置
@pytest.mark.usefixtures("resetSettings")

# 使用 resetTempSettings fixture 重置临时设置
@pytest.mark.usefixtures("resetTempSettings")
class TestTor:
    # 测试下载功能
    def testDownload(self, tor_manager):
        # 循环尝试连接 Tor，最多尝试 15 次
        for retry in range(15):
            time.sleep(1)
            if tor_manager.enabled and tor_manager.conn:
                break
        # 断言 Tor 已启用
        assert tor_manager.enabled

    # 测试管理器连接功能
    def testManagerConnection(self, tor_manager):
        # 发送 GETINFO version 请求，断言返回结果中包含 "250-version"
        assert "250-version" in tor_manager.request("GETINFO version")

    # 测试添加 Onion 功能
    def testAddOnion(self, tor_manager):
        # 添加 Onion 地址
        address = tor_manager.addOnion()
        # 断言地址不为空
        assert address
        # 断言地址在私钥列表中
        assert address in tor_manager.privatekeys

        # 删除 Onion 地址
        assert tor_manager.delOnion(address)
        # 断言地址不在私钥列表中
        assert address not in tor_manager.privatekeys

    # 测试签名 Onion 功能
    def testSignOnion(self, tor_manager):
        # 添加 Onion 地址
        address = tor_manager.addOnion()

        # 对消息 "hello" 进行签名
        sign = CryptRsa.sign(b"hello", tor_manager.getPrivatekey(address))
        # 断言签名长度为 128
        assert len(sign) == 128

        # 验证签名
        publickey = CryptRsa.privatekeyToPublickey(tor_manager.getPrivatekey(address))
        # 断言公钥长度为 140
        assert len(publickey) == 140
        # 断言消息 "hello" 被正确验证
        assert CryptRsa.verify(b"hello", publickey, sign)
        # 断言消息 "not hello" 未被验证
        assert not CryptRsa.verify(b"not hello", publickey, sign)

        # 公钥转换为地址
        assert CryptRsa.publickeyToOnion(publickey) == address

        # 删除 Onion 地址
        tor_manager.delOnion(address)

    # 标记为慢速测试
    @pytest.mark.slow
    # 测试连接是否正常
    def testConnection(self, tor_manager, file_server, site, site_temp):
        # 设置文件服务器的Tor管理器启用onions
        file_server.tor_manager.start_onions = True
        # 获取站点的.onion地址
        address = file_server.tor_manager.getOnion(site.address)
        # 断言地址存在
        assert address
        # 打印连接地址
        print("Connecting to", address)
        # 循环5次，等待隐藏服务创建
        for retry in range(5):
            # 暂停10秒
            time.sleep(10)
            try:
                # 尝试建立连接
                connection = file_server.getConnection(address + ".onion", 1544)
                # 如果成功建立连接，则跳出循环
                if connection:
                    break
            except Exception as err:
                continue
        # 断言连接握手成功
        assert connection.handshake
        # 断言Tor连接没有peer_id
        assert not connection.handshake["peer_id"]

        # 返回相同的连接，不指定站点
        assert file_server.getConnection(address + ".onion", 1544) == connection
        # 不同站点不允许重用连接
        assert file_server.getConnection(address + ".onion", 1544, site=site) != connection
        # 同一站点允许重用连接
        assert file_server.getConnection(address + ".onion", 1544, site=site) == file_server.getConnection(address + ".onion", 1544, site=site)
        # 修改临时站点地址
        site_temp.address = "1OTHERSITE"
        # 不同站点不允许重用连接
        assert file_server.getConnection(address + ".onion", 1544, site=site) != file_server.getConnection(address + ".onion", 1544, site=site_temp)

        # 只允许从锁定的站点查询
        file_server.sites[site.address] = site
        # 获取锁定站点的连接
        connection_locked = file_server.getConnection(address + ".onion", 1544, site=site)
        # 断言连接请求成功
        assert "body" in connection_locked.request("getFile", {"site": site.address, "inner_path": "content.json", "location": 0})
        # 断言从不同站点请求会返回错误
        assert connection_locked.request("getFile", {"site": "1OTHERSITE", "inner_path": "content.json", "location": 0})["error"] == "Invalid site"
    # 测试 Pex 方法
    def testPex(self, file_server, site, site_temp):
        # 将站点注册到当前运行的文件服务器
        site.connection_server = file_server
        file_server.sites[site.address] = site
        # 创建一个新的文件服务器来模拟连接到我们的对等点的新对等点
        file_server_temp = FileServer(file_server.ip, 1545)
        site_temp.connection_server = file_server_temp
        file_server_temp.sites[site_temp.address] = site_temp

        # 从这里请求对等点
        peer_source = site_temp.addPeer(file_server.ip, 1544)

        # 从源站点获取 ip4 对等点
        site.addPeer("1.2.3.4", 1555)  # 将对等点添加到源站点
        assert peer_source.pex(need_num=10) == 1
        assert len(site_temp.peers) == 2
        assert "1.2.3.4:1555" in site_temp.peers

        # 从源站点获取 onion 对等点
        site.addPeer("bka4ht2bzxchy44r.onion", 1555)
        assert "bka4ht2bzxchy44r.onion:1555" not in site_temp.peers

        # 如果不支持，不要添加 onion 对等点
        assert "onion" not in file_server_temp.supported_ip_types
        assert peer_source.pex(need_num=10) == 0

        file_server_temp.supported_ip_types.append("onion")
        assert peer_source.pex(need_num=10) == 1

        assert "bka4ht2bzxchy44r.onion:1555" in site_temp.peers
    # 测试查找哈希值的方法
    def testFindHash(self, tor_manager, file_server, site, site_temp):
        # 重置洪水保护
        file_server.ip_incoming = {}
        # 将站点添加到文件服务器的站点字典中
        file_server.sites[site.address] = site
        # 设置文件服务器的 TOR 管理器
        file_server.tor_manager = tor_manager

        # 创建客户端对象
        client = FileServer(file_server.ip, 1545)
        # 将临时站点添加到客户端的站点字典中
        client.sites = {site_temp.address: site_temp}
        # 将客户端设置为临时站点的连接服务器
        site_temp.connection_server = client

        # 将文件服务器添加为客户端的对等节点
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)

        # 断言对等文件服务器查找哈希值为 1234 时返回空字典
        assert peer_file_server.findHashIds([1234]) == {}

        # 添加具有所需哈希值的虚假对等节点
        fake_peer_1 = site.addPeer("bka4ht2bzxchy44r.onion", 1544)
        fake_peer_1.hashfield.append(1234)
        fake_peer_2 = site.addPeer("1.2.3.5", 1545)
        fake_peer_2.hashfield.append(1234)
        fake_peer_2.hashfield.append(1235)
        fake_peer_3 = site.addPeer("1.2.3.6", 1546)
        fake_peer_3.hashfield.append(1235)
        fake_peer_3.hashfield.append(1236)

        # 断言对等文件服务器查找哈希值为 1234 和 1235 时返回的结果
        res = peer_file_server.findHashIds([1234, 1235])
        assert sorted(res[1234]) == [('1.2.3.5', 1545), ("bka4ht2bzxchy44r.onion", 1544)]
        assert sorted(res[1235]) == [('1.2.3.5', 1545), ('1.2.3.6', 1546)]

        # 测试我的地址添加
        site.content_manager.hashfield.append(1234)

        # 再次断言对等文件服务器查找哈希值为 1234 和 1235 时返回的结果
        res = peer_file_server.findHashIds([1234, 1235])
        assert sorted(res[1234]) == [('1.2.3.5', 1545), (file_server.ip, 1544), ("bka4ht2bzxchy44r.onion", 1544)]
        assert sorted(res[1235]) == [('1.2.3.5', 1545), ('1.2.3.6', 1546)]

    # 测试使用 TOR 管理器获取 .onion 地址
    def testSiteOnion(self, tor_manager):
        # 使用模拟对象设置 TOR 配置为 "always"
        with mock.patch.object(config, "tor", "always"):
            # 断言获取的两个地址不相同
            assert tor_manager.getOnion("address1") != tor_manager.getOnion("address2")
            # 断言两次获取相同地址的结果相同
            assert tor_manager.getOnion("address1") == tor_manager.getOnion("address1")
```