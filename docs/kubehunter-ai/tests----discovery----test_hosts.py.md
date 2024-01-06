# `kubehunter\tests\discovery\test_hosts.py`

```
# 导入需要的模块
import requests_mock  # 用于模拟 HTTP 请求
import pytest  # 用于编写测试用例

from netaddr import IPNetwork, IPAddress  # 用于处理 IP 地址
from kube_hunter.modules.discovery.hosts import (  # 导入主机发现模块
    FromPodHostDiscovery,  # 从 Pod 中发现主机
    RunningAsPodEvent,  # 当作为 Pod 运行时的事件
    HostScanEvent,  # 主机扫描事件
    AzureMetadataApi,  # Azure 元数据 API
    HostDiscoveryHelpers,  # 主机发现辅助函数
)
from kube_hunter.core.events.types import NewHostEvent  # 导入新主机事件类型
from kube_hunter.core.events import handler  # 导入事件处理器
from kube_hunter.conf import config  # 导入配置文件

# 定义测试函数 test_FromPodHostDiscovery
def test_FromPodHostDiscovery():
    # 使用 requests_mock 创建一个 HTTP 请求的模拟器
    with requests_mock.Mocker() as m:
        # 创建一个 RunningAsPodEvent 对象
        e = RunningAsPodEvent()
# 设置 config 对象的 azure 属性为 False
config.azure = False
# 设置 config 对象的 remote 属性为 None
config.remote = None
# 设置 config 对象的 cidr 属性为 None
config.cidr = None
# 发送 GET 请求到指定 URL，返回状态码为 404
m.get("http://169.254.169.254/metadata/instance?api-version=2017-08-01", status_code=404)
# 创建 FromPodHostDiscovery 对象
f = FromPodHostDiscovery(e)
# 断言 FromPodHostDiscovery 对象不是 Azure Pod
assert not f.is_azure_pod()

# TODO 目前我们不测试 traceroute 发现版本
# 执行 FromPodHostDiscovery 对象的 execute 方法
# f.execute()

# 测试我们是否为 Azure Metadata API 报告的地址生成 NewHostEvent
# 设置 config 对象的 azure 属性为 True
config.azure = True
# 发送 GET 请求到指定 URL，返回的文本为 JSON 格式的网络接口信息
m.get("http://169.254.169.254/metadata/instance?api-version=2017-08-01", text='{"network":{"interface":[{"ipv4":{"subnet":[{"address": "3.4.5.6", "prefix": "255.255.255.252"}]}}]}}')
# 断言 FromPodHostDiscovery 对象是 Azure Pod
assert f.is_azure_pod()
# 执行 FromPodHostDiscovery 对象的 execute 方法
f.execute()
# 测试我们只有在配置了config.remote或config.cidr时才触发HostScanEvent
config.remote = "1.2.3.4"  # 设置远程主机地址
f.execute()  # 执行函数

config.azure = False  # 设置为非Azure环境
config.remote = None  # 清空远程主机地址
config.cidr = "1.2.3.4/24"  # 设置CIDR地址
f.execute()  # 执行函数

# 在这组测试中，只有在设置了remote或cidr时才会触发HostScanEvent
@handler.subscribe(HostScanEvent)
class testHostDiscovery(object):
    def __init__(self, event):
        assert config.remote is not None or config.cidr is not None  # 断言远程主机地址或CIDR地址不为空
        assert config.remote == "1.2.3.4" or config.cidr == "1.2.3.4/24"  # 断言远程主机地址或CIDR地址符合预期

# 在这组测试中，只有在Azure环境下才会找到主机
# 因为我们没有运行通常由 HostScanEvent 触发的代码
@handler.subscribe(NewHostEvent)
class testHostDiscoveryEvent(object):
    # 初始化函数，对事件进行断言检查
    def __init__(self, event):
        # 断言配置为 Azure
        assert config.azure
        # 断言事件的主机以 "3.4.5." 开头
        assert str(event.host).startswith("3.4.5.")
        # 断言配置的远程和 CIDR 为 None
        assert config.remote is None
        assert config.cidr is None


# 测试我们只报告 Azure 主机的事件
@handler.subscribe(AzureMetadataApi)
class testAzureMetadataApi(object):
    # 初始化函数，对事件进行断言检查
    def __init__(self, event):
        # 断言配置为 Azure
        assert config.azure


class TestDiscoveryUtils:
    # 静态方法，测试生成有效 CIDR 的主机
    @staticmethod
    def test_generate_hosts_valid_cidr():
# 设置测试用的CIDR地址
test_cidr = "192.168.0.0/24"
# 期望的结果是将CIDR地址转换成IP地址集合
expected = set(IPNetwork(test_cidr))

# 生成实际的IP地址集合
actual = set(HostDiscoveryHelpers.generate_hosts([test_cidr]))

# 断言实际结果与期望结果相等
assert actual == expected

# 测试生成主机列表时忽略指定IP地址
@staticmethod
def test_generate_hosts_valid_ignore():
    # 设置要忽略的IP地址
    remove = IPAddress("192.168.1.8")
    # 设置要扫描的CIDR地址
    scan = "192.168.1.0/24"
    # 期望的结果是将扫描的CIDR地址转换成IP地址集合，并排除指定的IP地址
    expected = set(ip for ip in IPNetwork(scan) if ip != remove)

    # 生成实际的IP地址集合，忽略指定的IP地址
    actual = set(HostDiscoveryHelpers.generate_hosts([scan, f"!{str(remove)}"]))

    # 断言实际结果与期望结果相等
    assert actual == expected

# 测试生成主机列表时输入无效的CIDR地址
@staticmethod
def test_generate_hosts_invalid_cidr():
    # 使用pytest断言抛出值错误异常
    with pytest.raises(ValueError):
# 调用 HostDiscoveryHelpers 类的 generate_hosts 方法，生成 IP 地址列表
list(HostDiscoveryHelpers.generate_hosts(["192..2.3/24"]))

# 静态方法，用于测试 generate_hosts 方法对于无效的忽略参数的处理
def test_generate_hosts_invalid_ignore():
    # 使用 pytest 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 调用 generate_hosts 方法，传入包含无效忽略参数的列表，生成 IP 地址列表
        list(HostDiscoveryHelpers.generate_hosts(["192.168.1.8", "!29.2..1/24"]))
```