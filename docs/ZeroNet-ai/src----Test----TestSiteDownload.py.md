# `ZeroNet\src\Test\TestSiteDownload.py`

```py
# 导入时间模块
import time
# 导入 pytest 模块
import pytest
# 导入 mock 模块
import mock
# 导入 gevent 模块
import gevent
# 导入 gevent.event 模块
import gevent.event
# 导入 os 模块
import os
# 从 Connection 模块中导入 ConnectionServer 类
from Connection import ConnectionServer
# 从 Config 模块中导入 config 变量
from Config import config
# 从 File 模块中导入 FileRequest 和 FileServer 类
from File import FileRequest
from File import FileServer
# 从 Site.Site 模块中导入 Site 类
from Site.Site import Site
# 从当前目录中导入 Spy 模块
from . import Spy

# 使用 pytest 的 usefixtures 装饰器，重置临时设置
@pytest.mark.usefixtures("resetTempSettings")
# 使用 pytest 的 usefixtures 装饰器，重置设置
@pytest.mark.usefixtures("resetSettings")
# 定义 TestSiteDownload 类
class TestSiteDownload:
    # 测试当连接的对等方具有可选文件时
    # 测试当连接的对等方没有文件时，询问他是否知道谁有这个文件
    # 测试站点的 content.json 大于站点限制时会发生什么
    # 测试文件名中包含 Unicode 字符的情况
    def testUnicodeFilename(self, file_server, site, site_temp):
        # 断言 site.storage.directory 等于 config.data_dir + "/" + site.address
        assert site.storage.directory == config.data_dir + "/" + site.address
        # 断言 site_temp.storage.directory 等于 config.data_dir + "-temp/" + site.address
        assert site_temp.storage.directory == config.data_dir + "-temp/" + site.address

        # 初始化源服务器
        site.connection_server = file_server
        file_server.sites[site.address] = site

        # 初始化客户端服务器
        client = FileServer(file_server.ip, 1545)
        client.sites = {site_temp.address: site_temp}
        site_temp.connection_server = client
        site_temp.announce = mock.MagicMock(return_value=True)  # 不要尝试从网络中查找对等点

        site_temp.addPeer(file_server.ip, 1544)

        # 断言 site_temp.download(blind_includes=True, retry_bad_files=False) 返回结果为真，并设置超时时间为10秒
        assert site_temp.download(blind_includes=True, retry_bad_files=False).get(timeout=10)

        # 在 site.storage 中写入文件 "data/img/árvíztűrő.png"，内容为 b"test"
        site.storage.write("data/img/árvíztűrő.png", b"test")

        # 对 site.content_manager 中的 "content.json" 文件进行签名，使用私钥 "5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv"
        site.content_manager.sign("content.json", privatekey="5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv")

        # 从 site.storage 中加载 "content.json" 文件内容
        content = site.storage.loadJson("content.json")
        # 断言 "data/img/árvíztűrő.png" 在 content["files"] 中
        assert "data/img/árvíztűrő.png" in content["files"]
        # 断言 site_temp.storage 中不存在 "data/img/árvíztűrő.png" 文件
        assert not site_temp.storage.isFile("data/img/árvíztűrő.png")
        # 保存 site_temp.settings 的当前值
        settings_before = site_temp.settings

        # 使用 Spy.Spy(FileRequest, "route") 监听请求，然后发布站点
        with Spy.Spy(FileRequest, "route") as requests:
            site.publish()
            time.sleep(0.1)
            # 断言 site_temp.download(blind_includes=True, retry_bad_files=False) 返回结果为真，并设置超时时间为10秒，等待下载完成
            assert site_temp.download(blind_includes=True, retry_bad_files=False).get(timeout=10)
            # 断言请求中 "streamFile" 的数量为1
            assert len([req[1] for req in requests if req[1] == "streamFile"]) == 1

        # 从 site_temp.storage 中加载 "content.json" 文件内容
        content = site_temp.storage.loadJson("content.json")
        # 断言 "data/img/árvíztűrő.png" 在 content["files"] 中
        assert "data/img/árvíztűrő.png" in content["files"]
        # 断言 site_temp.storage 中存在 "data/img/árvíztűrő.png" 文件
        assert site_temp.storage.isFile("data/img/árvíztűrő.png")

        # 断言 site_temp.settings["size"] 等于 settings_before["size"]
        assert site_temp.settings["size"] == settings_before["size"]
        # 断言 site_temp.settings["size_optional"] 等于 settings_before["size_optional"]
        assert site_temp.settings["size_optional"] == settings_before["size_optional"]

        # 断言 site_temp.storage 中删除所有文件
        assert site_temp.storage.deleteFiles()
        # 关闭所有与 file_server 的连接
        [connection.close() for connection in file_server.connections]
```