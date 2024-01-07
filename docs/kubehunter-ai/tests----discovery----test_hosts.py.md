# `.\kubehunter\tests\discovery\test_hosts.py`

```

# 导入所需的模块
import requests_mock  # 用于模拟 HTTP 请求
import pytest  # 用于编写测试用例

from netaddr import IPNetwork, IPAddress  # 用于处理 IP 地址和网络
from kube_hunter.modules.discovery.hosts import (  # 导入主机发现相关模块
    FromPodHostDiscovery,
    RunningAsPodEvent,
    HostScanEvent,
    AzureMetadataApi,
    HostDiscoveryHelpers,
)
from kube_hunter.core.events.types import NewHostEvent  # 导入事件类型
from kube_hunter.core.events import handler  # 导入事件处理器
from kube_hunter.conf import config  # 导入配置

# 定义测试函数 test_FromPodHostDiscovery
def test_FromPodHostDiscovery():
    # 使用 requests_mock 模拟 HTTP 请求
    with requests_mock.Mocker() as m:
        e = RunningAsPodEvent()  # 创建 RunningAsPodEvent 事件对象

        # 设置配置参数
        config.azure = False
        config.remote = None
        config.cidr = None
        # 模拟 HTTP GET 请求，返回 404 错误
        m.get("http://169.254.169.254/metadata/instance?api-version=2017-08-01", status_code=404)
        f = FromPodHostDiscovery(e)  # 创建 FromPodHostDiscovery 对象
        assert not f.is_azure_pod()  # 断言不是 Azure Pod

        # 设置配置参数
        config.azure = True
        # 模拟 HTTP GET 请求，返回 Azure Metadata API 数据
        m.get(
            "http://169.254.169.254/metadata/instance?api-version=2017-08-01",
            text='{"network":{"interface":[{"ipv4":{"subnet":[{"address": "3.4.5.6", "prefix": "255.255.255.252"}]}}]}}',
        )
        assert f.is_azure_pod()  # 断言是 Azure Pod
        f.execute()  # 执行 FromPodHostDiscovery 对象的 execute 方法

        # 设置配置参数
        config.remote = "1.2.3.4"
        f.execute()  # 执行 FromPodHostDiscovery 对象的 execute 方法

        # 设置配置参数
        config.azure = False
        config.remote = None
        config.cidr = "1.2.3.4/24"
        f.execute()  # 执行 FromPodHostDiscovery 对象的 execute 方法

# 订阅 HostScanEvent 事件
@handler.subscribe(HostScanEvent)
class testHostDiscovery(object):
    def __init__(self, event):
        assert config.remote is not None or config.cidr is not None
        assert config.remote == "1.2.3.4" or config.cidr == "1.2.3.4/24"

# 订阅 NewHostEvent 事件
@handler.subscribe(NewHostEvent)
class testHostDiscoveryEvent(object):
    def __init__(self, event):
        assert config.azure
        assert str(event.host).startswith("3.4.5.")
        assert config.remote is None
        assert config.cidr is None

# 订阅 AzureMetadataApi 事件
@handler.subscribe(AzureMetadataApi)
class testAzureMetadataApi(object):
    def __init__(self, event):
        assert config.azure

# 定义 TestDiscoveryUtils 类
class TestDiscoveryUtils:
    # 测试生成有效 CIDR 的主机
    @staticmethod
    def test_generate_hosts_valid_cidr():
        test_cidr = "192.168.0.0/24"
        expected = set(IPNetwork(test_cidr))  # 期望的结果

        actual = set(HostDiscoveryHelpers.generate_hosts([test_cidr]))  # 实际结果

        assert actual == expected  # 断言实际结果与期望结果相等

    # 测试生成有效忽略的主机
    @staticmethod
    def test_generate_hosts_valid_ignore():
        remove = IPAddress("192.168.1.8")
        scan = "192.168.1.0/24"
        expected = set(ip for ip in IPNetwork(scan) if ip != remove)  # 期望的结果

        actual = set(HostDiscoveryHelpers.generate_hosts([scan, f"!{str(remove)}"]))  # 实际结果

        assert actual == expected  # 断言实际结果与期望结果相等

    # 测试生成无效 CIDR 的主机
    @staticmethod
    def test_generate_hosts_invalid_cidr():
        with pytest.raises(ValueError):
            list(HostDiscoveryHelpers.generate_hosts(["192..2.3/24"]))

    # 测试生成无效忽略的主机
    @staticmethod
    def test_generate_hosts_invalid_ignore():
        with pytest.raises(ValueError):
            list(HostDiscoveryHelpers.generate_hosts(["192.168.1.8", "!29.2..1/24"]))

```