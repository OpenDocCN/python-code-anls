# `kubehunter\tests\discovery\test_hosts.py`

```
# 导入所需的模块
import requests_mock
import pytest
from netaddr import IPNetwork, IPAddress
from kube_hunter.modules.discovery.hosts import (
    FromPodHostDiscovery,
    RunningAsPodEvent,
    HostScanEvent,
    AzureMetadataApi,
    HostDiscoveryHelpers,
)
from kube_hunter.core.events.types import NewHostEvent
from kube_hunter.core.events import handler
from kube_hunter.conf import config


# 测试 FromPodHostDiscovery 类
def test_FromPodHostDiscovery():
    # 使用 requests_mock 创建一个 Mock 对象
    with requests_mock.Mocker() as m:
        # 创建 RunningAsPodEvent 对象
        e = RunningAsPodEvent()

        # 设置配置参数
        config.azure = False
        config.remote = None
        config.cidr = None
        # 模拟请求 Azure Metadata API 返回 404
        m.get(
            "http://169.254.169.254/metadata/instance?api-version=2017-08-01", status_code=404,
        )
        # 创建 FromPodHostDiscovery 对象
        f = FromPodHostDiscovery(e)
        # 断言不是 Azure Pod
        assert not f.is_azure_pod()
        # TODO 暂时不测试 traceroute 发现版本
        # f.execute()

        # 测试是否为 Azure Pod，并执行相应操作
        config.azure = True
        # 模拟请求 Azure Metadata API 返回网络信息
        m.get(
            "http://169.254.169.254/metadata/instance?api-version=2017-08-01",
            text='{"network":{"interface":[{"ipv4":{"subnet":[{"address": "3.4.5.6", "prefix": "255.255.255.252"}]}}]}}',
        )
        assert f.is_azure_pod()
        f.execute()

        # 测试当 config.remote 或 config.cidr 配置时是否触发 HostScanEvent
        config.remote = "1.2.3.4"
        f.execute()

        config.azure = False
        config.remote = None
        config.cidr = "1.2.3.4/24"
        f.execute()


# 在这组测试中，只有在设置了 remote 或 cidr 时才会触发 HostScanEvent
@handler.subscribe(HostScanEvent)
class testHostDiscovery(object):
    def __init__(self, event):
        assert config.remote is not None or config.cidr is not None
        assert config.remote == "1.2.3.4" or config.cidr == "1.2.3.4/24"


# 在这组测试中，只有在 Azure 中找到主机时才会进行下一步操作
# 因为我们没有运行通常由 HostScanEvent 触发的代码
@handler.subscribe(NewHostEvent)
class testHostDiscoveryEvent(object):
    def __init__(self, event):
        # 断言配置中存在 Azure
        assert config.azure
        # 断言事件的主机以 "3.4.5." 开头
        assert str(event.host).startswith("3.4.5.")
        # 断言配置中 remote 为 None
        assert config.remote is None
        # 断言配置中 cidr 为 None
        assert config.cidr is None


# 测试我们只针对 Azure 主机报告此事件
@handler.subscribe(AzureMetadataApi)
class testAzureMetadataApi(object):
    def __init__(self, event):
        # 断言配置中存在 Azure
        assert config.azure


class TestDiscoveryUtils:
    @staticmethod
    def test_generate_hosts_valid_cidr():
        # 设置测试的 CIDR
        test_cidr = "192.168.0.0/24"
        # 期望的结果是一个 IPNetwork 对象的集合
        expected = set(IPNetwork(test_cidr))

        # 实际结果是使用给定的 CIDR 生成的主机集合
        actual = set(HostDiscoveryHelpers.generate_hosts([test_cidr]))

        # 断言实际结果与期望结果相等
        assert actual == expected

    @staticmethod
    def test_generate_hosts_valid_ignore():
        # 设置要移除的 IP 地址
        remove = IPAddress("192.168.1.8")
        # 设置要扫描的 CIDR
        scan = "192.168.1.0/24"
        # 期望的结果是一个不包含要移除 IP 地址的 IPNetwork 对象的集合
        expected = set(ip for ip in IPNetwork(scan) if ip != remove)

        # 实际结果是使用给定的 CIDR 和要忽略的 IP 地址生成的主机集合
        actual = set(HostDiscoveryHelpers.generate_hosts([scan, f"!{str(remove)}"]))

        # 断言实际结果与期望结果相等
        assert actual == expected

    @staticmethod
    def test_generate_hosts_invalid_cidr():
        # 测试生成主机集合时传入无效的 CIDR，预期会引发 ValueError 异常
        with pytest.raises(ValueError):
            list(HostDiscoveryHelpers.generate_hosts(["192..2.3/24"]))

    @staticmethod
    def test_generate_hosts_invalid_ignore():
        # 测试生成主机集合时传入无效的忽略规则，预期会引发 ValueError 异常
        with pytest.raises(ValueError):
            list(HostDiscoveryHelpers.generate_hosts(["192.168.1.8", "!29.2..1/24"]))
```