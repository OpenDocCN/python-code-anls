# `ZeroNet\src\Test\TestConnectionServer.py`

```py
# 导入所需的模块
import time
import socket
import gevent

import pytest
import mock

# 导入自定义模块
from Crypt import CryptConnection
from Connection import ConnectionServer
from Config import config

# 使用 pytest 的 usefixtures 装饰器，重置设置
@pytest.mark.usefixtures("resetSettings")
class TestConnection:
    # 测试 IPv6 连接
    def testIpv6(self, file_server6):
        # 断言文件服务器的 IP 地址中包含冒号，即为 IPv6 地址
        assert ":" in file_server6.ip

        # 创建客户端连接对象
        client = ConnectionServer(file_server6.ip, 1545)
        # 获取连接对象
        connection = client.getConnection(file_server6.ip, 1544)
        
        # 断言连接对象的 ping 方法返回 True
        assert connection.ping()

        # 关闭连接
        connection.close()
        client.stop()
        time.sleep(0.01)
        # 断言文件服务器的连接数为 0
        assert len(file_server6.connections) == 0

        # 尝试使用 IPv4 地址连接，断言会抛出 socket.error 异常
        with pytest.raises(socket.error) as err:
            client = ConnectionServer("127.0.0.1", 1545)
            connection = client.getConnection("127.0.0.1", 1544)

    # 测试 SSL 连接
    def testSslConnection(self, file_server):
        # 创建客户端连接对象
        client = ConnectionServer(file_server.ip, 1545)
        # 断言文件服务器对象不等于客户端对象
        assert file_server != client

        # 使用 mock.patch 临时替换本地 IP 地址的配置，SSL 不适用于本地 IP 地址
        with mock.patch('Config.config.ip_local', return_value=[]):
            # 获取连接对象
            connection = client.getConnection(file_server.ip, 1544)

        # 断言文件服务器的连接数为 1
        assert len(file_server.connections) == 1
        # 断言连接对象的握手状态为 True
        assert connection.handshake
        # 断言连接对象的加密状态为 True
        assert connection.crypt

        # 关闭连接
        connection.close("Test ended")
        client.stop()
        time.sleep(0.1)
        # 断言文件服务器的连接数为 0
        assert len(file_server.connections) == 0
        # 断言文件服务器的传入连接数为 2，一个是文件服务器的 fixture，一个是测试用例的连接
        assert file_server.num_incoming == 2
    # 测试原始连接功能
    def testRawConnection(self, file_server):
        # 创建连接服务器对象
        client = ConnectionServer(file_server.ip, 1545)
        # 断言文件服务器对象不等于客户端对象
        assert file_server != client

        # 移除所有支持的加密方式
        crypt_supported_bk = CryptConnection.manager.crypt_supported
        CryptConnection.manager.crypt_supported = []

        # 使用模拟补丁，配置本地 IP 地址不使用 SSL
        with mock.patch('Config.config.ip_local', return_value=[]):  # SSL not used for local ips
            # 获取连接对象
            connection = client.getConnection(file_server.ip, 1544)
        # 断言文件服务器的连接数为1
        assert len(file_server.connections) == 1
        # 断言连接对象不使用加密
        assert not connection.crypt

        # 关闭连接
        connection.close()
        client.stop()
        time.sleep(0.01)
        # 断言文件服务器的连接数为0
        assert len(file_server.connections) == 0

        # 恢复支持的加密方式
        CryptConnection.manager.crypt_supported = crypt_supported_bk

    # 测试 ping 功能
    def testPing(self, file_server, site):
        # 创建连接服务器对象
        client = ConnectionServer(file_server.ip, 1545)
        # 获取连接对象
        connection = client.getConnection(file_server.ip, 1544)

        # 断言连接对象可以 ping 通
        assert connection.ping()

        # 关闭连接
        connection.close()
        client.stop()

    # 测试获取连接功能
    def testGetConnection(self, file_server):
        # 创建连接服务器对象
        client = ConnectionServer(file_server.ip, 1545)
        # 获取连接对象
        connection = client.getConnection(file_server.ip, 1544)

        # 通过 IP/端口获取连接对象
        connection2 = client.getConnection(file_server.ip, 1544)
        # 断言两个连接对象相等
        assert connection == connection2

        # 通过 peerid 获取连接对象，断言不存在的情况
        assert not client.getConnection(file_server.ip, 1544, peer_id="notexists", create=False)
        # 通过 peerid 获取连接对象，断言存在的情况
        connection2 = client.getConnection(file_server.ip, 1544, peer_id=connection.handshake["peer_id"], create=False)
        # 断言两个连接对象相等
        assert connection2 == connection

        # 关闭连接
        connection.close()
        client.stop()
    # 测试防洪保护功能，传入文件服务器对象
    def testFloodProtection(self, file_server):
        # 保存白名单以便重置
        whitelist = file_server.whitelist  
        # 禁用 127.0.0.1 的白名单
        file_server.whitelist = []  
        # 创建与文件服务器的连接
        client = ConnectionServer(file_server.ip, 1545)

        # 限制一分钟内只允许 6 个连接
        for reconnect in range(6):
            # 建立连接
            connection = client.getConnection(file_server.ip, 1544)
            # 断言握手成功
            assert connection.handshake
            # 关闭连接
            connection.close()

        # 第 7 个连接将超时
        with pytest.raises(gevent.Timeout):
            # 设置超时时间为 0.1 秒
            with gevent.Timeout(0.1):
                # 尝试建立连接
                connection = client.getConnection(file_server.ip, 1544)

        # 重置白名单
        file_server.whitelist = whitelist
```