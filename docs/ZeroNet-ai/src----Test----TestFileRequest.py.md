# `ZeroNet\src\Test\TestFileRequest.py`

```
# 导入所需的模块
import io
import pytest
import time
from Connection import ConnectionServer
from Connection import Connection
from File import FileServer

# 使用 pytest 的 usefixtures 重置设置
@pytest.mark.usefixtures("resetSettings")
@pytest.mark.usefixtures("resetTempSettings")
class TestFileRequest:
    # 测试从文件服务器流式传输文件
    def testStreamFile(self, file_server, site):
        # 重置洪水保护
        file_server.ip_incoming = {}
        # 创建到文件服务器的连接
        client = ConnectionServer(file_server.ip, 1545)
        connection = client.getConnection(file_server.ip, 1544)
        # 将站点添加到文件服务器
        file_server.sites[site.address] = site

        # 创建一个字节流缓冲区
        buff = io.BytesIO()
        # 发送请求并将响应数据写入缓冲区
        response = connection.request("streamFile", {"site": site.address, "inner_path": "content.json", "location": 0}, buff)
        # 断言响应中包含流式传输的字节流
        assert "stream_bytes" in response
        # 断言缓冲区中包含签名数据
        assert b"sign" in buff.getvalue()

        # 无效文件
        buff = io.BytesIO()
        response = connection.request("streamFile", {"site": site.address, "inner_path": "invalid.file", "location": 0}, buff)
        # 断言响应中包含文件读取错误信息
        assert "File read error" in response["error"]

        # 位置超出文件大小
        buff = io.BytesIO()
        response = connection.request(
            "streamFile", {"site": site.address, "inner_path": "content.json", "location": 1024 * 1024}, buff
        )
        # 断言响应中包含文件读取错误信息
        assert "File read error" in response["error"]

        # 从父目录流式传输
        buff = io.BytesIO()
        response = connection.request("streamFile", {"site": site.address, "inner_path": "../users.json", "location": 0}, buff)
        # 断言响应中包含文件读取异常信息
        assert "File read exception" in response["error"]

        # 关闭连接
        connection.close()
        # 停止客户端
        client.stop()
    # 测试 Pex 功能
    def testPex(self, file_server, site, site_temp):
        # 将站点添加到文件服务器的站点字典中
        file_server.sites[site.address] = site
        # 创建文件服务器客户端
        client = FileServer(file_server.ip, 1545)
        # 将临时站点添加到客户端的站点字典中
        client.sites = {site_temp.address: site_temp}
        # 为临时站点设置连接服务器
        site_temp.connection_server = client
        # 从客户端获取连接
        connection = client.getConnection(file_server.ip, 1544)

        # 向站点添加新的虚假对等点
        fake_peer = site.addPeer(file_server.ip_external, 11337, return_peer=True)
        # 为其添加虚假连接
        fake_peer.connection = Connection(file_server, file_server.ip_external, 11337)
        fake_peer.connection.last_recv_time = time.time()
        # 断言虚假对等点在可连接对等点列表中
        assert fake_peer in site.getConnectablePeers()

        # 将文件服务器作为对等点添加到客户端
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)

        # 断言文件服务器的外部 IP 和端口不在临时站点的对等点列表中
        assert "%s:11337" % file_server.ip_external not in site_temp.peers
        # 执行 Pex 操作，将文件服务器添加到临时站点的对等点列表中
        assert peer_file_server.pex()
        assert "%s:11337" % file_server.ip_external in site_temp.peers

        # 不应该交换来自本地网络的私有对等点
        fake_peer_private = site.addPeer("192.168.0.1", 11337, return_peer=True)
        # 断言私有对等点不在可连接对等点列表中
        assert fake_peer_private not in site.getConnectablePeers(allow_private=False)
        fake_peer_private.connection = Connection(file_server, "192.168.0.1", 11337)
        fake_peer_private.connection.last_recv_time = time.time()

        # 断言私有对等点不在临时站点的对等点列表中
        assert "192.168.0.1:11337" not in site_temp.peers
        # 执行 Pex 操作，不将私有对等点添加到临时站点的对等点列表中
        assert not peer_file_server.pex()
        assert "192.168.0.1:11337" not in site_temp.peers

        # 关闭连接
        connection.close()
        # 停止客户端
        client.stop()
```