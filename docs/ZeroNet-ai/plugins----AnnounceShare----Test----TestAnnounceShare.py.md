# `ZeroNet\plugins\AnnounceShare\Test\TestAnnounceShare.py`

```py
# 导入 pytest 模块
import pytest

# 从 AnnounceShare 模块中导入 AnnounceSharePlugin 类
from AnnounceShare import AnnounceSharePlugin
# 从 Peer 模块中导入 Peer 类
from Peer import Peer
# 从 Config 模块中导入 config 对象
from Config import config

# 使用 resetSettings 修饰器重置设置
@pytest.mark.usefixtures("resetSettings")
# 使用 resetTempSettings 修饰器重置临时设置
@pytest.mark.usefixtures("resetTempSettings")
# 定义 TestAnnounceShare 类
class TestAnnounceShare:
    # 定义 testAnnounceList 方法，传入 file_server 参数
    def testAnnounceList(self, file_server):
        # 打开并写入空的 trackers.json 文件
        open("%s/trackers.json" % config.data_dir, "w").write("{}")
        # 获取 AnnounceSharePlugin 类的 tracker_storage 属性
        tracker_storage = AnnounceSharePlugin.tracker_storage
        # 加载 tracker_storage
        tracker_storage.load()
        # 创建 Peer 对象
        peer = Peer(file_server.ip, 1544, connection_server=file_server)
        # 断言 peer 请求 "getTrackers" 返回的 trackers 为空列表
        assert peer.request("getTrackers")["trackers"] == []

        # 当发现 tracker 时，不会改变 trackers 列表
        tracker_storage.onTrackerFound("zero://%s:15441" % file_server.ip)
        assert peer.request("getTrackers")["trackers"] == []

        # 需要至少一个成功的 announce 才能共享给其他对等方
        tracker_storage.onTrackerSuccess("zero://%s:15441" % file_server.ip, 1.0)
        assert peer.request("getTrackers")["trackers"] == ["zero://%s:15441" % file_server.ip]
```