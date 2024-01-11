# `ZeroNet\plugins\AnnounceLocal\Test\TestAnnounce.py`

```
# 导入所需的模块
import time
import copy

import gevent
import pytest
import mock

# 从指定的模块中导入指定的类或函数
from AnnounceLocal import AnnounceLocalPlugin
from File import FileServer
from Test import Spy

# 定义一个用于测试的 fixture，用于创建并配置一个 announcer 对象
@pytest.fixture
def announcer(file_server, site):
    # 将 site 添加到 file_server 的 sites 字典中
    file_server.sites[site.address] = site
    # 创建一个本地 announcer 对象，设置监听端口为 1100
    announcer = AnnounceLocalPlugin.LocalAnnouncer(file_server, listen_port=1100)
    # 将 file_server 的 local_announcer 属性设置为 announcer
    file_server.local_announcer = announcer
    # 设置 announcer 的监听端口为 1100
    announcer.listen_port = 1100
    # 设置 announcer 的 sender_info 字典中的 broadcast_port 为 1100
    announcer.sender_info["broadcast_port"] = 1100
    # 使用 mock.MagicMock 创建一个虚拟的 getMyIps 方法，返回固定的值 ["127.0.0.1"]
    announcer.getMyIps = mock.MagicMock(return_value=["127.0.0.1"])
    # 使用 mock.MagicMock 创建一个虚拟的 discover 方法，返回固定的值 False，不自动发送 discover 请求
    announcer.discover = mock.MagicMock(return_value=False)
    # 使用 gevent.spawn 方法启动 announcer 的 start 方法
    gevent.spawn(announcer.start)
    # 等待 0.5 秒
    time.sleep(0.5)

    # 断言 file_server 的 local_announcer 的 running 属性为 True
    assert file_server.local_announcer.running
    # 返回 file_server 的 local_announcer 对象
    return file_server.local_announcer

# 定义一个用于测试的 fixture，用于创建并配置一个远程的 announcer 对象
@pytest.fixture
def announcer_remote(request, site_temp):
    # 创建一个远程的 file_server 对象，IP 为 "127.0.0.1"，端口为 1545
    file_server_remote = FileServer("127.0.0.1", 1545)
    # 将 site_temp 添加到 file_server_remote 的 sites 字典中
    file_server_remote.sites[site_temp.address] = site_temp
    # 创建一个远程的 announcer 对象，设置监听端口为 1101
    announcer = AnnounceLocalPlugin.LocalAnnouncer(file_server_remote, listen_port=1101)
    # 将 file_server_remote 的 local_announcer 属性设置为 announcer
    file_server_remote.local_announcer = announcer
    # 设置 announcer 的监听端口为 1101
    announcer.listen_port = 1101
    # 设置 announcer 的 sender_info 字典中的 broadcast_port 为 1101
    announcer.sender_info["broadcast_port"] = 1101
    # 使用 mock.MagicMock 创建一个虚拟的 getMyIps 方法，返回固定的值 ["127.0.0.1"]
    announcer.getMyIps = mock.MagicMock(return_value=["127.0.0.1"])
    # 使用 mock.MagicMock 创建一个虚拟的 discover 方法，返回固定的值 False，不自动发送 discover 请求
    announcer.discover = mock.MagicMock(return_value=False)
    # 使用 gevent.spawn 方法启动 announcer 的 start 方法
    gevent.spawn(announcer.start)
    # 等待 0.5 秒
    time.sleep(0.5)

    # 断言 file_server_remote 的 local_announcer 的 running 属性为 True
    assert file_server_remote.local_announcer.running

    # 定义一个清理函数，用于在测试结束时停止 file_server_remote
    def cleanup():
        file_server_remote.stop()
    # 将清理函数添加到 request 的 finalizer 中，确保在测试结束时执行清理函数
    request.addfinalizer(cleanup)

    # 返回 file_server_remote 的 local_announcer 对象
    return file_server_remote.local_announcer

# 使用 fixture 重置设置
@pytest.mark.usefixtures("resetSettings")
# 使用 fixture 重置临时设置
@pytest.mark.usefixtures("resetTempSettings")
# 定义一个测试类 TestAnnounce
class TestAnnounce:
    # 定义一个测试方法 testSenderInfo，接收 announcer 作为参数
    def testSenderInfo(self, announcer):
        # 获取 announcer 的 sender_info 字典
        sender_info = announcer.sender_info
        # 断言 sender_info 中 port 的值大于 0
        assert sender_info["port"] > 0
        # 断言 sender_info 中 peer_id 的长度为 20
        assert len(sender_info["peer_id"]) == 20
        # 断言 sender_info 中 rev 的值大于 0
        assert sender_info["rev"] > 0
    # 测试忽略自身消息的情况
    def testIgnoreSelfMessages(self, announcer):
        # 不响应与服务器相同的对等 ID 的消息
        assert not announcer.handleMessage(("0.0.0.0", 123), {"cmd": "discoverRequest", "sender": announcer.sender_info, "params": {}})[1]

        # 响应与不同对等 ID 的消息
        sender_info = copy.copy(announcer.sender_info)
        sender_info["peer_id"] += "-"
        addr, res = announcer.handleMessage(("0.0.0.0", 123), {"cmd": "discoverRequest", "sender": sender_info, "params": {}})
        assert res["params"]["sites_changed"] > 0

    # 测试发现请求的情况
    def testDiscoverRequest(self, announcer, announcer_remote):
        assert len(announcer_remote.known_peers) == 0
        # 使用 Spy 监控 announcer_remote 的 handleMessage 方法的调用
        with Spy.Spy(announcer_remote, "handleMessage") as responses:
            # 广播发现请求消息
            announcer_remote.broadcast({"cmd": "discoverRequest", "params": {}}, port=announcer.listen_port)
            time.sleep(0.1)

        # 获取响应消息的命令列表
        response_cmds = [response[1]["cmd"] for response in responses]
        assert response_cmds == ["discoverResponse", "siteListResponse"]
        assert len(responses[-1][1]["params"]["sites"]) == 1

        # 只有在 sites_changed 值与上次响应不同时才应请求 siteList
        with Spy.Spy(announcer_remote, "handleMessage") as responses:
            announcer_remote.broadcast({"cmd": "discoverRequest", "params": {}}, port=announcer.listen_port)
            time.sleep(0.1)

        response_cmds = [response[1]["cmd"] for response in responses]
        assert response_cmds == ["discoverResponse"]

    # 测试对等发现的情况
    def testPeerDiscover(self, announcer, announcer_remote, site):
        assert announcer.server.peer_id != announcer_remote.server.peer_id
        assert len(list(announcer.server.sites.values())[0].peers) == 0
        # 广播发现请求消息
        announcer.broadcast({"cmd": "discoverRequest"}, port=announcer_remote.listen_port)
        time.sleep(0.1)
        assert len(list(announcer.server.sites.values())[0].peers) == 1
    # 测试最近对等节点列表的功能
    def testRecentPeerList(self, announcer, announcer_remote, site):
        # 确保最近对等节点列表为空
        assert len(site.peers_recent) == 0
        # 确保对等节点列表为空
        assert len(site.peers) == 0
        # 使用 Spy 类监视 announcer 对象的 handleMessage 方法，并记录响应
        with Spy.Spy(announcer, "handleMessage") as responses:
            # 广播发现请求消息
            announcer.broadcast({"cmd": "discoverRequest", "params": {}}, port=announcer_remote.listen_port)
            # 等待0.1秒
            time.sleep(0.1)
        # 确保响应中包含 "discoverResponse" 和 "siteListResponse" 消息
        assert [response[1]["cmd"] for response in responses] == ["discoverResponse", "siteListResponse"]
        # 确保最近对等节点列表中有一个节点
        assert len(site.peers_recent) == 1
        # 确保对等节点列表中有一个节点
        assert len(site.peers) == 1

        # 如果没有 siteListResponse 消息，应该更新对等节点
        # 记录最后发现的时间
        last_time_found = list(site.peers.values())[0].time_found
        # 清空最近对等节点列表
        site.peers_recent.clear()
        # 使用 Spy 类监视 announcer 对象的 handleMessage 方法，并记录响应
        with Spy.Spy(announcer, "handleMessage") as responses:
            # 广播发现请求消息
            announcer.broadcast({"cmd": "discoverRequest", "params": {}}, port=announcer_remote.listen_port)
            # 等待0.1秒
            time.sleep(0.1)
        # 确保响应中包含 "discoverResponse" 消息
        assert [response[1]["cmd"] for response in responses] == ["discoverResponse"]
        # 确保最近对等节点列表中有一个节点
        assert len(site.peers_recent) == 1
        # 确保最后发现的时间晚于记录的最后发现时间
        assert list(site.peers.values())[0].time_found > last_time_found
```